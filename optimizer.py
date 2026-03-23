# optimizer.py - Single GPU Version
import json
import os
from collections import defaultdict
import time
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from compressor import GradientCompressor
import torch.nn.functional as F
from collections import deque
from torch.utils.data import DataLoader
import numpy as np
import copy

class CompressedOptimizer:
    def __init__(self, 
                 model, 
                 optimizer: torch.optim.Optimizer,
                 compression_config: Dict,
                 training_config: Dict,
                 use_compression: bool = True,
                 use_error_feedback: bool = True,
                 use_momentum: bool = True,
                 gradient_clipping: float = None,
                 device: torch.device = None,
                 criterion = None,
                 trainer_ref=None):# << 新增一个对Trainer的引用
        self.model = model
        self.optimizer = optimizer
        self.use_compression = use_compression 

        self.trainer_ref = trainer_ref

        self.ema_grad_norm_vector = None # 用于 EMA 平滑

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if use_compression:
            compression_config = compression_config.copy()
            compression_config['device'] = self.device
            self.compressor = GradientCompressor(**compression_config)
        else:
            self.compressor = None

        self.use_error_feedback = use_error_feedback and use_compression
        self.use_momentum = use_momentum
        self.gradient_clipping = gradient_clipping
        self.momentum_buffer = {}
        self.momentum = 0.9  # 使用与SGD相同的动量系数
        self.mse_history = []
        self.criterion = criterion

        self.grads_t_plus_1 = None # 用于缓存 g_{t+1}
        self.grads_t_minus_1 = None  # 用于缓存 g_{t-1}

        # --- 新增和修改的部分 for 自适应分配间隔方法 ---

        # 迭代计数器
        self.iter_counter = 0

        # --- 自适应分配配置 ---
        self.min_reallocation_interval = training_config.get('min_reallocation_interval')
        self.realloc_sim_threshold = training_config.get('realloc_sim_threshold')
        self.adaptive_metric = training_config.get('adaptive_metric', 'cosine_similarity')

        # 选择触发策略
        self.adaptive_trigger_method = training_config.get('adaptive_trigger_method') # 默认为策略B
        print(f"[INFO] Using adaptive trigger method: {self.adaptive_trigger_method}")
        
        # --- 状态变量 ---
        # 用于策略A (Anchor vs. Current): 保存锚点时刻的梯度范数分布
        self.anchor_grad_norm_vector = None
        # 用于策略B (Iterative): 保存上一步的梯度范数分布
        self.prev_grad_norm_vector = None

        # 用于分配：维护一个历史梯度窗口 (对两种策略都通用)
        self.grad_stats_window_size = training_config.get('grad_stats_batch_size')
        self.grad_history_buffer = deque(maxlen=self.grad_stats_window_size) 
        self.last_reallocation_iter = -self.min_reallocation_interval

        # 压缩时间统计
        self.compressed_time = 0.0

        # 层名映射
        self.layer_names = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.layer_names[param] = name
                
        # 如果使用压缩，设置模型引用
        if self.use_compression and self.compressor:
            current_lr = self.optimizer.param_groups[0]['lr']
            self.compressor.set_model_references(model, criterion, optimizer, current_lr)
                
    def update_bit_allocation(self, bit_allocation: Dict[str, int]):
        """更新当前的比特分配"""
        if self.compressor:
            self.compressor.set_bit_allocation(bit_allocation)
        
    def set_current_batch(self, inputs, targets):
        """设置当前batch的数据，用于计算loss差异"""
        self.current_inputs = inputs
        self.current_targets = targets
        if self.compressor:
            self.compressor.set_current_batch(inputs, targets)
            
    def update_learning_rate(self):
        """更新压缩器中的学习率引用"""
        if self.compressor:
            current_lr = self.optimizer.param_groups[0]['lr']
            self.compressor.current_lr = current_lr

    def _get_random_image_batches(self, num_batches: int, batch_size: int = None):
        """
        为图像任务随机采样batch
        """
        if not self.trainer_ref or not self.trainer_ref.train_loader:
            return []
        
        dataset = self.trainer_ref.train_loader.dataset
        total_samples = len(dataset)
        
        if batch_size is None:
            batch_size = self.trainer_ref.train_loader.batch_size
        
        random_batches = []
        
        for _ in range(num_batches):
            # 随机选择batch_size个样本索引
            random_indices = np.random.choice(total_samples, batch_size, replace=False)
            
            # 构造batch
            batch_inputs = []
            batch_targets = []
            
            for idx in random_indices:
                sample = dataset[idx]
                if isinstance(sample, (tuple, list)) and len(sample) == 2:
                    input_data, target = sample
                    batch_inputs.append(input_data)
                    batch_targets.append(target)
                else:
                    print(f"Warning: Unexpected sample format at index {idx}")
                    continue
            
            if batch_inputs and batch_targets:
                # 将列表转换为tensor
                try:
                    batch_inputs_tensor = torch.stack(batch_inputs).to(self.device)
                    batch_targets_tensor = torch.tensor(batch_targets, dtype=torch.long).to(self.device)
                    random_batches.append((batch_inputs_tensor, batch_targets_tensor))
                except Exception as e:
                    print(f"Warning: Failed to create batch tensor: {e}")
                    continue
        
        return random_batches

    # def step(self, current_batch_for_stats: Tuple[torch.Tensor, torch.Tensor] = None, epoch: int = None):
    #     # 1. 更新学习率引用
    #     self.update_learning_rate()

    #     # 2. [关键] 应用梯度裁剪 (Gradient Clipping)
    #     # NLP 任务中，这一步必须在任何量化操作之前完成，推荐阈值 0.25
    #     if self.gradient_clipping is not None:
    #         torch.nn.utils.clip_grad_norm_(
    #             self.optimizer.param_groups[0]['params'],
    #             self.gradient_clipping
    #         )

    #     # 3. 不压缩情况：直接执行优化器步骤
    #     if not self.use_compression:
    #         self.optimizer.step()
    #         return None

    #     # --- 准备工作 ---
    #     # 获取当前梯度 (已经是被 Clip 过的)
    #     # 过滤掉没有梯度的参数
    #     raw_grads_t = {}
    #     for p in self.model.parameters():
    #         if p.grad is not None:
    #             raw_grads_t[self.layer_names[p]] = p.grad.data.clone()

    #     # 4. [高级功能] 预计算 g_{t+1} (Lookahead Gradient)
    #     # 仅当使用了依赖 g_{t+1} 的 Taylor 指标时才计算
    #     metric_needs_forward_info = self.compressor.allocation_metric in ['taylor1_gt1', 'taylor2', 'taylor1+2']
    #     self.grads_t_plus_1 = None
        
    #     if self.compressor.use_adaptive and metric_needs_forward_info and current_batch_for_stats:
    #         # 创建临时模型模拟更新
    #         temp_model = copy.deepcopy(self.model)
    #         # 使用与主优化器配置相同的临时优化器
    #         temp_optimizer = type(self.optimizer)(temp_model.parameters(), **self.optimizer.defaults)
    #         temp_optimizer.load_state_dict(self.optimizer.state_dict())
            
    #         # 将当前梯度赋值给临时模型
    #         for p_temp, p_orig in zip(temp_model.parameters(), self.model.parameters()):
    #             if p_orig.grad is not None:
    #                 p_temp.grad = p_orig.grad.clone()

    #         # 模拟一步更新 w_{t+1}
    #         temp_optimizer.step()
            
    #         # 计算 g_{t+1}
    #         temp_model.zero_grad()
    #         inputs, targets = current_batch_for_stats
    #         outputs_t_plus_1 = temp_model(inputs)
    #         if isinstance(outputs_t_plus_1, tuple): outputs_t_plus_1 = outputs_t_plus_1[0] # Handle LSTM output
            
    #         # 兼容 NLP 的 shape
    #         if targets.dim() == 1 and outputs_t_plus_1.dim() > 2: # Check for NLP output shape mismatch
    #              loss_t_plus_1 = self.criterion(outputs_t_plus_1.view(-1, outputs_t_plus_1.size(-1)), targets)
    #         else:
    #              loss_t_plus_1 = self.criterion(outputs_t_plus_1, targets)
                 
    #         loss_t_plus_1.backward()
            
    #         self.grads_t_plus_1 = {name: p.grad.data.clone() for name, p in temp_model.named_parameters() if p.grad is not None}
    #         del temp_model, temp_optimizer

    #     stats_to_return = None

    #     # 5. 更新历史梯度窗口
    #     self.grad_history_buffer.append(raw_grads_t)

    #     # ============================================================
    #     # 6. 自适应分配触发逻辑 (Adaptive Trigger) - [包含针对 NLP/MobileNet 的修改]
    #     # ============================================================
    #     if self.compressor.use_adaptive:
    #         trigger_allocation = False
    #         can_reallocate = (self.iter_counter - self.last_reallocation_iter) >= self.min_reallocation_interval

    #         # 首次运行必须分配
    #         if self.compressor._current_bit_allocation is None:
    #              trigger_allocation = True
    #              print(f"\n[INFO] Iter {self.iter_counter}: Triggering initial bit allocation.")
            
    #         elif can_reallocate:
    #             # [新增] 计算梯度范数分布向量
    #             sorted_layer_names = sorted(raw_grads_t.keys())
    #             current_grad_norms = torch.tensor([raw_grads_t[name].norm() for name in sorted_layer_names], device=self.device)

    #             # [新增] EMA 平滑处理 (Fix for Noisy Gradients in NLP/MobileNet)
    #             # 使用 EMA 向量代替瞬时向量进行 Trigger 判断
    #             if not hasattr(self, 'ema_grad_norm_vector') or self.ema_grad_norm_vector is None:
    #                 self.ema_grad_norm_vector = current_grad_norms
    #             else:
    #                 alpha = 0.9 # EMA 平滑系数
    #                 self.ema_grad_norm_vector = alpha * self.ema_grad_norm_vector + (1 - alpha) * current_grad_norms
                
    #             # 使用平滑后的向量进行比较
    #             current_norm_vector_for_check = self.ema_grad_norm_vector

    #             # 选择比较对象 (Old Vector)
    #             if self.adaptive_trigger_method == 'anchor_vs_current':
    #                 old_norm_vector = self.anchor_grad_norm_vector
    #             else: # 'iterative'
    #                 old_norm_vector = self.prev_grad_norm_vector
                
    #             if old_norm_vector is not None:
    #                 # --- 触发条件 A: Cosine Similarity ---
    #                 if self.adaptive_metric == 'cosine_similarity':
    #                     sim = F.cosine_similarity(old_norm_vector, current_norm_vector_for_check, dim=0).item()
    #                     if sim < self.realloc_sim_threshold:
    #                         print(f"\n[INFO] Iter {self.iter_counter}: Triggering (Reason: Cos Sim {sim:.4f} < {self.realloc_sim_threshold})")
    #                         trigger_allocation = True

    #                 # --- 触发条件 B: Relative Change (Fix for Gradient Scale Shift) ---
    #                 elif self.adaptive_metric == 'relative_change':
    #                     # 计算 L2 相对变化率: ||v_new - v_old|| / ||v_old||
    #                     diff_norm = torch.norm(current_norm_vector_for_check - old_norm_vector)
    #                     old_norm_val = torch.norm(old_norm_vector)
    #                     rel_change = (diff_norm / (old_norm_val + 1e-8)).item()
                        
    #                     if rel_change > (1.0 - self.realloc_sim_threshold): # 注意：这里 threshold 含义变为“允许的变化量”
    #                          print(f"\n[INFO] Iter {self.iter_counter}: Triggering (Reason: Rel Change {rel_change:.4f} > {1.0-self.realloc_sim_threshold:.4f})")
    #                          trigger_allocation = True
                
    #             # 更新 iterative 模式的历史向量
    #             if self.adaptive_trigger_method == 'iterative':
    #                 self.prev_grad_norm_vector = current_norm_vector_for_check.clone()

    #         # ============================================================
    #         # 7. 执行比特分配 (如果触发)
    #         # ============================================================
    #         if trigger_allocation:
    #             # 计算历史梯度的平均值（降低噪声）
    #             buffer_sum = {}
    #             for hist_grad_map in self.grad_history_buffer:
    #                 for name, grad in hist_grad_map.items():
    #                     if name not in buffer_sum: buffer_sum[name] = grad.clone()
    #                     else: buffer_sum[name].add_(grad)
    #             avg_hist_grads = {name: g / len(self.grad_history_buffer) for name, g in buffer_sum.items()}

    #             # 准备用于 RD 曲线计算的数据 Batch
    #             data_batches_for_rd = []
    #             if current_batch_for_stats:
    #                 data_batches_for_rd.append(current_batch_for_stats)
                
    #             # 获取额外的 Batch 以稳定统计
    #             num_additional = self.grad_stats_window_size - 1
    #             if num_additional > 0 and self.trainer_ref:
    #                 if self.trainer_ref.is_nlp_task: # NLP 采样逻辑
    #                     source = self.trainer_ref.nlp_data.train_data
    #                     bptt = self.trainer_ref.bptt
    #                     if len(source) > bptt:
    #                         for _ in range(num_additional):
    #                             start_idx = np.random.randint(0, source.size(0) - 1 - bptt)
    #                             data = source[start_idx : start_idx + bptt]
    #                             target = source[start_idx + 1 : start_idx + 1 + bptt].reshape(-1)
    #                             data_batches_for_rd.append((data.to(self.device), target.to(self.device)))
    #                 elif self.trainer_ref.train_loader: # CV 采样逻辑
    #                     try:
    #                         data_batches_for_rd.extend(self._get_random_image_batches(num_additional))
    #                     except: pass

    #             # 获取当前 Epoch
    #             # curr_epoch = self.trainer_ref.scheduler.current_epoch if (self.trainer_ref and hasattr(self.trainer_ref, 'scheduler')) else 0
    #             # 不再从 scheduler 获取，而是直接使用传入的 epoch 参数
    #             curr_epoch = epoch if epoch is not None else 0

    #             # ---> 调用压缩器进行优化 <---
    #             bit_allocation = self.compressor.optimize_bit_allocation(
    #                 avg_hist_grads,
    #                 data_batches_for_rd=data_batches_for_rd,
    #                 current_epoch=curr_epoch,
    #                 grads_t=raw_grads_t,
    #                 grads_t_plus_1=self.grads_t_plus_1
    #             )
                
    #             self.compressor.set_bit_allocation(bit_allocation)
    #             stats_to_return = self.compressor.calculate_compression_stats(raw_grads_t, bit_allocation)
                
    #             # 更新状态
    #             self.last_reallocation_iter = self.iter_counter
    #             # 更新 Anchor 向量 (使用 EMA 后的值)
    #             if hasattr(self, 'ema_grad_norm_vector'):
    #                  self.anchor_grad_norm_vector = self.ema_grad_norm_vector.clone()
    #             else:
    #                  self.anchor_grad_norm_vector = current_grad_norms.clone()

    #     self.iter_counter += 1

    #     # ============================================================
    #     # 8. 梯度压缩与应用 (包含敏感层保护)
    #     # ============================================================
    #     total_mse = 0.0
    #     total_params = 0

    #     for group in self.optimizer.param_groups:
    #         for param in group['params']:
    #             if param.grad is None:
    #                 continue
                    
    #             layer_name = self.layer_names[param]
    #             gradient = param.grad.data.clone()
    #             original_gradient = gradient.clone()

    #             # --- 8.1 敏感层检查 (Fix for Loss Explosion) ---
    #             # 对于 NLP 任务，跳过 Embedding, Bias, 和 LayerNorm/BatchNorm
    #             is_sensitive = False
    #             ln = layer_name.lower()
    #             if 'embedding' in ln or 'token_emb' in ln or 'pos_encoder' in ln:
    #                 is_sensitive = True
    #             elif 'bias' in ln:
    #                 is_sensitive = True
    #             elif 'norm' in ln or 'bn' in ln: # layer_norm, batch_norm
    #                 is_sensitive = True
                
    #             # 如果是敏感层，直接跳过所有压缩逻辑
    #             if is_sensitive:
    #                 decompressed_grad = gradient
    #             else:
    #                 # --- 8.2 正常压缩逻辑 ---
                    
    #                 # 误差反馈 (Error Feedback)
    #                 if self.use_error_feedback:
    #                     error = self.compressor.get_error_feedback(layer_name, gradient)
    #                     if torch.isnan(error).any(): error.zero_() # 安全检查
    #                     gradient = gradient + error

    #                 # 压缩与解压
    #                 start_time = time.time()
    #                 compressed_data, rebuild_info = self.compressor.compress(gradient, layer_name)
                    
    #                 if compressed_data is not None:
    #                     decompressed_grad = self.compressor.decompress(compressed_data, rebuild_info)
    #                 else:
    #                     decompressed_grad = gradient # 失败回退

    #                 self.compressed_time += (time.time() - start_time)

    #                 # 更新误差反馈
    #                 if self.use_error_feedback:
    #                     self.compressor.add_error_feedback(layer_name, gradient, decompressed_grad)

    #             # --- 8.3 统计与应用 ---
    #             if not is_sensitive: # 仅统计被压缩层的 MSE
    #                 curr_mse = F.mse_loss(original_gradient, decompressed_grad, reduction='sum')
    #                 total_mse += curr_mse.item()
    #                 total_params += param.numel()

    #             # 应用动量 (Momentum)
    #             if self.use_momentum:
    #                 if layer_name not in self.momentum_buffer:
    #                     self.momentum_buffer[layer_name] = torch.zeros_like(decompressed_grad)
                    
    #                 # buf = momentum * buf + g
    #                 self.momentum_buffer[layer_name].mul_(self.momentum).add_(decompressed_grad)
    #                 param.grad.data.copy_(self.momentum_buffer[layer_name])
    #             else:
    #                 param.grad.data.copy_(decompressed_grad)

    #     # 9. 记录 MSE 历史
    #     if total_params > 0:
    #         self.mse_history.append(torch.tensor(total_mse / total_params, device=self.device))

    #     # 10. 真正更新参数
    #     self.optimizer.step()

    #     return stats_to_return
    # optimizer.py - step 函数 (无 EMA 版)

    def step(self, current_batch_for_stats: Tuple[torch.Tensor, torch.Tensor] = None, epoch: int = None):
        # 1. 更新学习率引用
        self.update_learning_rate()

        # 2. 应用梯度裁剪
        if self.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(
                self.optimizer.param_groups[0]['params'],
                self.gradient_clipping
            )

        # 3. 不压缩情况
        if not self.use_compression:
            self.optimizer.step()
            return None

        # --- 准备工作 ---
        raw_grads_t = {}
        for p in self.model.parameters():
            if p.grad is not None:
                raw_grads_t[self.layer_names[p]] = p.grad.data.clone()

        # 4. 预计算 g_{t+1} (如果需要)
        metric_needs_forward_info = self.compressor.allocation_metric in ['taylor1_gt1', 'taylor2', 'taylor1+2']
        self.grads_t_plus_1 = None
        
        if self.compressor.use_adaptive and metric_needs_forward_info and current_batch_for_stats:
            temp_model = copy.deepcopy(self.model)
            temp_optimizer = type(self.optimizer)(temp_model.parameters(), **self.optimizer.defaults)
            temp_optimizer.load_state_dict(self.optimizer.state_dict())
            
            for p_temp, p_orig in zip(temp_model.parameters(), self.model.parameters()):
                if p_orig.grad is not None:
                    p_temp.grad = p_orig.grad.clone()

            temp_optimizer.step()
            
            temp_model.zero_grad()
            inputs, targets = current_batch_for_stats
            outputs_t_plus_1 = temp_model(inputs)
            if isinstance(outputs_t_plus_1, tuple): outputs_t_plus_1 = outputs_t_plus_1[0]
            
            if targets.dim() == 1 and outputs_t_plus_1.dim() > 2: 
                 loss_t_plus_1 = self.criterion(outputs_t_plus_1.view(-1, outputs_t_plus_1.size(-1)), targets)
            else:
                 loss_t_plus_1 = self.criterion(outputs_t_plus_1, targets)
                 
            loss_t_plus_1.backward()
            self.grads_t_plus_1 = {name: p.grad.data.clone() for name, p in temp_model.named_parameters() if p.grad is not None}
            del temp_model, temp_optimizer

        stats_to_return = None

        # 5. 更新历史梯度窗口
        self.grad_history_buffer.append(raw_grads_t)

        # ============================================================
        # 6. 自适应分配触发逻辑 (Adaptive Trigger) - [无 EMA 版]
        # ============================================================
        if self.compressor.use_adaptive:
            trigger_allocation = False
            can_reallocate = (self.iter_counter - self.last_reallocation_iter) >= self.min_reallocation_interval

            # === [修改] 直接计算当前梯度范数 ===
            sorted_layer_names = sorted(raw_grads_t.keys())
            current_grad_norms = torch.tensor([raw_grads_t[name].norm() for name in sorted_layer_names], device=self.device)
            
            # 直接使用瞬时范数作为检查对象
            current_norm_vector_for_check = current_grad_norms
            # =================================

            # 首次运行必须分配
            if self.compressor._current_bit_allocation is None:
                 trigger_allocation = True
                 print(f"\n[INFO] Iter {self.iter_counter}: Triggering initial bit allocation.")
            
            elif can_reallocate:
                if self.adaptive_trigger_method == 'anchor_vs_current':
                    old_norm_vector = self.anchor_grad_norm_vector
                else: # 'iterative'
                    old_norm_vector = self.prev_grad_norm_vector
                
                if old_norm_vector is not None:
                    # --- 触发条件 A: Cosine Similarity ---
                    if self.adaptive_metric == 'cosine_similarity':
                        sim = F.cosine_similarity(old_norm_vector, current_norm_vector_for_check, dim=0).item()
                        if sim < self.realloc_sim_threshold:
                            print(f"\n[INFO] Iter {self.iter_counter}: Triggering (Reason: Cos Sim {sim:.4f} < {self.realloc_sim_threshold})")
                            trigger_allocation = True

                    # --- 触发条件 B: Relative Change ---
                    elif self.adaptive_metric == 'relative_change':
                        diff_norm = torch.norm(current_norm_vector_for_check - old_norm_vector)
                        old_norm_val = torch.norm(old_norm_vector)
                        # 添加 epsilon 防止除零
                        rel_change = (diff_norm / (old_norm_val + 1e-8)).item()
                        
                        if rel_change > (1.0 - self.realloc_sim_threshold): 
                             print(f"\n[INFO] Iter {self.iter_counter}: Triggering (Reason: Rel Change {rel_change:.4f} > {1.0-self.realloc_sim_threshold:.4f})")
                             trigger_allocation = True
                
                if self.adaptive_trigger_method == 'iterative':
                    self.prev_grad_norm_vector = current_norm_vector_for_check.clone()

            # ============================================================
            # 7. 执行比特分配
            # ============================================================
            if trigger_allocation:
                buffer_sum = {}
                for hist_grad_map in self.grad_history_buffer:
                    for name, grad in hist_grad_map.items():
                        if name not in buffer_sum: buffer_sum[name] = grad.clone()
                        else: buffer_sum[name].add_(grad)
                avg_hist_grads = {name: g / len(self.grad_history_buffer) for name, g in buffer_sum.items()}

                data_batches_for_rd = []
                if current_batch_for_stats:
                    data_batches_for_rd.append(current_batch_for_stats)
                
                num_additional = self.grad_stats_window_size - 1
                if num_additional > 0 and self.trainer_ref:
                    if self.trainer_ref.is_nlp_task: 
                        source = self.trainer_ref.nlp_data.train_data
                        bptt = self.trainer_ref.bptt
                        if len(source) > bptt:
                            for _ in range(num_additional):
                                start_idx = np.random.randint(0, source.size(0) - 1 - bptt)
                                data = source[start_idx : start_idx + bptt]
                                target = source[start_idx + 1 : start_idx + 1 + bptt].reshape(-1)
                                data_batches_for_rd.append((data.to(self.device), target.to(self.device)))
                    elif self.trainer_ref.train_loader: 
                        try:
                            data_batches_for_rd.extend(self._get_random_image_batches(num_additional))
                        except: pass

                curr_epoch = epoch if epoch is not None else 0

                bit_allocation = self.compressor.optimize_bit_allocation(
                    avg_hist_grads,
                    data_batches_for_rd=data_batches_for_rd,
                    current_epoch=curr_epoch,
                    grads_t=raw_grads_t,
                    grads_t_plus_1=self.grads_t_plus_1,
                    grads_t_minus_1=self.grads_t_minus_1 # <--- 传入 g_{t-1}
                )
                
                self.compressor.set_bit_allocation(bit_allocation)
                stats_to_return = self.compressor.calculate_compression_stats(raw_grads_t, bit_allocation)
                
                self.last_reallocation_iter = self.iter_counter
                
                # === [修改] 直接 clone 当前的瞬时向量作为锚点 ===
                self.anchor_grad_norm_vector = current_norm_vector_for_check.clone()

        self.iter_counter += 1

        # ============================================================
        # 8. 梯度压缩与应用 (保持不变)
        # ============================================================
        total_mse = 0.0
        total_params = 0

        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                    
                layer_name = self.layer_names[param]
                gradient = param.grad.data.clone()
                original_gradient = gradient.clone()

                is_sensitive = False
                ln = layer_name.lower()
                # === 修改开始 ===
                # 1. Embedding 层：通常必须压缩，否则压缩率上不去。
                # 除非你发现 Embedding 压缩导致严重不收敛，否则不要跳过它。
                # if 'embedding' in ln or 'token_emb' in ln or 'pos_encoder' in ln:
                #     is_sensitive = True  <-- 注释掉这一行，让 Embedding 参与压缩
                
                # 2. 偏置项 (Bias)：参数很少，为了稳定通常不压缩
                if 'bias' in ln:
                    is_sensitive = True
                
                # 3. 归一化层 (Norm)：参数很少，为了稳定通常不压缩
                elif 'norm' in ln or 'bn' in ln: # layer_norm, batch_norm
                    is_sensitive = True
                # === 修改结束 ===
                
                if is_sensitive:
                    decompressed_grad = gradient
                else:
                    if self.use_error_feedback:
                        error = self.compressor.get_error_feedback(layer_name, gradient)
                        if torch.isnan(error).any(): error.zero_() 
                        gradient = gradient + error

                    start_time = time.time()
                    compressed_data, rebuild_info = self.compressor.compress(gradient, layer_name)
                    
                    if compressed_data is not None:
                        decompressed_grad = self.compressor.decompress(compressed_data, rebuild_info)
                    else:
                        decompressed_grad = gradient 

                    self.compressed_time += (time.time() - start_time)

                    if self.use_error_feedback:
                        self.compressor.add_error_feedback(layer_name, gradient, decompressed_grad)

                if not is_sensitive:
                    curr_mse = F.mse_loss(original_gradient, decompressed_grad, reduction='sum')
                    total_mse += curr_mse.item()
                    total_params += param.numel()

                if self.use_momentum:
                    if layer_name not in self.momentum_buffer:
                        self.momentum_buffer[layer_name] = torch.zeros_like(decompressed_grad)
                    self.momentum_buffer[layer_name].mul_(self.momentum).add_(decompressed_grad)
                    param.grad.data.copy_(self.momentum_buffer[layer_name])
                else:
                    param.grad.data.copy_(decompressed_grad)

        if total_params > 0:
            self.mse_history.append(torch.tensor(total_mse / total_params, device=self.device))

        self.optimizer.step()

        # === [新增] 在 Step 结束前，将当前的 g_t 存为 g_{t-1}，供下一步使用 ===
        # 注意：这里我们直接引用字典，因为 raw_grads_t 里的 tensor 已经是 clone 过的了
        self.grads_t_minus_1 = raw_grads_t

        return stats_to_return
    
    def get_compression_stats(self) -> Dict:
        """获取当前的压缩统计信息"""
        if not self.use_compression or not self.compressor.use_adaptive:
            return None
            
        gradients = {}
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    layer_name = self.layer_names[param]
                    gradients[layer_name] = param.grad.data
                    
        stats = self.compressor.calculate_compression_stats(gradients, self.compressor._current_bit_allocation)
        return stats