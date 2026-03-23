# trainer.py - Single GPU Version
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import logging
import traceback
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
from optimizer import CompressedOptimizer
import math
from utils import NLPData, LSTMModel, TransformerModel
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

class LRScheduler:
    """统一的学习率调度器"""
    def __init__(self, optimizer, training_config: Dict):
        self.optimizer = optimizer
        self.config = training_config
        self.current_epoch = 0
        
        # 记录初始学习率
        self.initial_lr = training_config['base_lr']
        # 如果使用warmup，调度器实际上在warmup结束后才开始工作
        warmup_epochs = self.config.get('warmup_epochs', 0) if self.config.get('use_warmup') else 0
        
        if training_config['lr_scheduler'] == 'multistep':
            # 将milestones向左平移warmup的长度
            adjusted_milestones = [m - warmup_epochs for m in training_config['milestones']]
            print(f"[LRScheduler] Original milestones: {training_config['milestones']}. Adjusted for warmup: {adjusted_milestones}")
            self.base_scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=adjusted_milestones,
                gamma=training_config['lr_decay']
            )
        elif training_config['lr_scheduler'] == 'cosine':
            # T_max应该是warmup之后剩余的epoch数
            t_max_after_warmup = training_config['num_epochs'] - warmup_epochs
            print(f"[LRScheduler] Cosine T_max adjusted for warmup: {t_max_after_warmup} epochs")
            self.base_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=t_max_after_warmup,
                #eta_min=0.001 # 你可以根据需要调整这个值
                eta_min=1e-6 # vit on imagenet
            )
        else:
            # 对于 'plateau' 或其他类型，不创建基础调度器
            self.base_scheduler = None


    def step(self):
        """更新学习率"""
        if self.base_scheduler is None:
            return

        self.current_epoch += 1
        
        if self.config['use_warmup'] and self.current_epoch <= self.config['warmup_epochs']:
            # Warmup阶段线性增加学习率
            lr = self.config['warmup_lr'] + (self.initial_lr - self.config['warmup_lr']) * \
                 (self.current_epoch / self.config['warmup_epochs'])
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Warmup后使用基础调度器
            self.base_scheduler.step()

    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']

class Trainer:
    def __init__(self, 
                 model: nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 test_loader: torch.utils.data.DataLoader,
                 nlp_data: NLPData,
                 compression_config: Dict,
                 training_config: Dict,
                 experiment_name: str,
                 device: torch.device,
                 ):

        try:
            self.device = device
            self.model = model.to(self.device)
            
            self.train_loader = train_loader
            self.test_loader = test_loader
            self.nlp_data = nlp_data
            self.is_nlp_task = (nlp_data is not None)
            
            self.compression_config = compression_config
            self.training_config = training_config
            self.experiment_name = experiment_name
            self.log_freq = training_config.get('log_freq', 10)
            self.bptt = training_config.get('bptt', 35)

            self.exp_dir = f'experiments/{experiment_name}'
            os.makedirs(self.exp_dir, exist_ok=True)
            os.makedirs(f'{self.exp_dir}/logs', exist_ok=True)

            logging.basicConfig(
                filename=f'{self.exp_dir}/logs/training.log',
                level=logging.INFO,
                format='%(asctime)s - %(message)s'
            )
            
            self.criterion = nn.CrossEntropyLoss()
            
            # --- 解耦逻辑: 1. 选择优化器 ---
            print(f"Optimizer selected: {training_config['optimizer'].upper()}")
            if training_config['optimizer'] == 'adam':
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=training_config['base_lr'],
                    weight_decay=training_config.get('weight_decay', 1e-6)
                )
            elif training_config['optimizer'] == 'adamw': # 新增一个 'adamw' 选项
                print("Using AdamW optimizer.")
                self.optimizer = optim.AdamW(
                    self.model.parameters(),
                    lr=training_config['base_lr'],
                    weight_decay=training_config.get('weight_decay', 0.01) # 给一个更合理的默认值
                )
            else: # 默认为 SGD
                self.optimizer = optim.SGD(
                    self.model.parameters(),
                    lr=training_config['base_lr'],
                    momentum=training_config['momentum'],
                    weight_decay=training_config['weight_decay']
                )
            
            # --- 初始化压缩优化器 ---
            self.compressed_optimizer = CompressedOptimizer(
                model=self.model, 
                optimizer=self.optimizer,
                compression_config=compression_config,
                training_config=training_config, 
                use_compression=training_config.get('use_compression', True), 
                use_error_feedback=training_config['error_feedback'],
                # 仅在SGD时使用外部动量
                use_momentum=training_config['use_momentum'] and (training_config['optimizer'] == 'sgd'),
                gradient_clipping=training_config['grad_clip'],
                device=self.device,
                criterion=self.criterion,
                trainer_ref=self
            )
            
            # --- 3. 初始化学习率调度器 (Scheduler) ---
            # 优先支持 NLP 常用的基于步数 (Step-based) 的 Warmup 调度器
            if training_config.get('warmup_steps', 0) > 0 or training_config['optimizer'] == 'adamw':
                # 计算总训练步数
                if self.is_nlp_task:
                    batches_per_epoch = (self.nlp_data.train_data.size(0) - 1) // self.bptt
                else:
                    batches_per_epoch = len(self.train_loader)
                
                total_steps = training_config['num_epochs'] * batches_per_epoch
                warmup_steps = training_config.get('warmup_steps', int(0.1 * total_steps))
                
                print(f"[Scheduler] Using HuggingFace Cosine Schedule with Warmup.")
                print(f"Total Steps: {total_steps}, Warmup Steps: {warmup_steps}")
                
                self.scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer, 
                    num_warmup_steps=warmup_steps, 
                    num_training_steps=total_steps
                )
                self.scheduler_type = 'step_based' # 标记为每步更新
            else:
                # 传统的基于 Epoch 的调度器 (CV 常用)
                self.scheduler = self._get_epoch_scheduler(self.optimizer, training_config)
                self.scheduler_type = 'epoch_based'
            
            # --- 初始化历史记录 ---
            self.history = {
                'train_loss': {}, 'train_acc': {}, 'train_ppl': {},
                'test_loss': {}, 'test_acc': {}, 'test_ppl': {},
                'lr': {}, 'best_epoch': None, 'best_metric': 0.0 if not self.is_nlp_task else float('inf'),
                'total_time': 0.0, 'mse': {}, 'compressed_time': {},
                'compression_stats': {
                    'iterations': [], 'bit_allocation': {},
                    'compression_ratio': {}, 'avg_bits': {}
                }
            }
            self.current_iteration = 0
            self.best_val_loss = float('inf') # 用于 'adaptive' 调度

            if self.training_config.get('use_compression', False):
                self.history.update({
                    'bit_allocation': {},
                    'compression_ratio': {},
                    'avg_bits': {}
                })

        except Exception as e:
            logging.error(f"Error in Trainer initialization: {str(e)}")
            logging.error(traceback.format_exc())
            raise
    
    # === [新增] 计算 Fisher 信息矩阵的方法 ===
    def calculate_fisher(self, num_batches=5):
        """
        计算 Fisher 信息矩阵的对角线近似。
        使用 num_batches 个批次的数据进行估计。
        """
        self.log_info(f"Calculating Fisher Information using {num_batches} batches...")
        fisher_diag = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_diag[name] = torch.zeros_like(param)
        
        self.model.eval()
        processed_samples = 0
        
        # 创建一个临时的 loader 迭代器
        data_iter = iter(self.train_loader)
        
        for _ in range(num_batches):
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                break
                
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.model.zero_grad()
            
            # 根据任务类型进行前向传播
            if self.is_nlp_task:
                 is_lstm = isinstance(self.model, LSTMModel)
                 is_transformer = isinstance(self.model, TransformerModel)
                 if is_lstm:
                     hidden = self.model.init_hidden(inputs.size(1))
                     outputs, _ = self.model(inputs, hidden)
                 elif is_transformer:
                     src_mask = self.model._generate_square_subsequent_mask(inputs.size(0)).to(self.device)
                     outputs = self.model(inputs, src_mask)
                 else:
                     outputs = self.model(inputs)
                     
                 if is_lstm or is_transformer:
                     loss = self.criterion(outputs.view(-1, self.nlp_data.vocab_size), targets.view(-1))
                 else:
                     loss = self.criterion(outputs, targets)
            else:
                # Image task
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            loss.backward()
            
            # 累积梯度平方
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        # Fisher = E[ (dL/dw)^2 ]
                        # 这里我们用 batch 梯度平方的平均来近似
                        fisher_diag[name] += param.grad.pow(2) * inputs.size(0) 
            
            processed_samples += inputs.size(0)

        # 平均化
        with torch.no_grad():
            for name in fisher_diag:
                fisher_diag[name] /= processed_samples
        
        self.model.train() # 恢复训练模式
        self.log_info("Fisher Information calculation completed.")
        return fisher_diag

    def should_log_epoch(self, epoch: int) -> bool:
        return epoch % self.log_freq == 0 or epoch == self.training_config['num_epochs']

    def log_info(self, message: str):
        logging.info(message)
        print(message)

    # def save_checkpoint(self, epoch: int, is_best: bool = True):
    #     if is_best:
    #         # 获取基础调度器的 state_dict，如果存在的话
    #         scheduler_state_dict = self.scheduler.base_scheduler.state_dict() if self.scheduler and self.scheduler.base_scheduler else None
    #         state = {
    #             'epoch': epoch,
    #             'model_state_dict': self.model.state_dict(),
    #             'optimizer_state_dict': self.optimizer.state_dict(),
    #             'scheduler_state_dict': scheduler_state_dict,
    #             'compression_config': self.compression_config,
    #             'training_config': self.training_config,
    #             'history': self.history,
    #             'best_metric': self.history['best_metric']
    #         }
    #         torch.save(state, f'{self.exp_dir}/best_model.pth')
    def save_checkpoint(self, epoch: int, is_best: bool = True):
        if is_best:
            # === [修改开始] 兼容多种调度器的保存逻辑 ===
            scheduler_state_dict = None
            if self.scheduler is not None:
                # 情况 A: 自定义的 LRScheduler 包装类 (通常用于 CV)
                if hasattr(self.scheduler, 'base_scheduler'):
                    if self.scheduler.base_scheduler is not None:
                        scheduler_state_dict = self.scheduler.base_scheduler.state_dict()
                # 情况 B: 原生 PyTorch 或 HuggingFace 调度器 (通常用于 NLP)
                elif hasattr(self.scheduler, 'state_dict'):
                    scheduler_state_dict = self.scheduler.state_dict()
            # === [修改结束] ===

            state = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': scheduler_state_dict,
                'compression_config': self.compression_config,
                'training_config': self.training_config,
                'history': self.history,
                'best_metric': self.history.get('best_metric', None)
            }
            torch.save(state, f'{self.exp_dir}/best_model.pth')

    def train_epoch_image(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        should_log = self.should_log_epoch(epoch)
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            stats = self.compressed_optimizer.step(current_batch_for_stats=(inputs, targets))
            
            if stats is not None:
                self.history['compression_stats']['iterations'].append(self.current_iteration)
                self.history['compression_stats']['bit_allocation'][self.current_iteration] = stats['bit_allocation']
                self.history['compression_stats']['compression_ratio'][self.current_iteration] = stats['compression_ratio']
                self.history['compression_stats']['avg_bits'][self.current_iteration] = stats['avg_bits']
            
            self.current_iteration += 1
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 50 == 0:
                self.log_info(
                    f'Epoch: {epoch} | Batch: {batch_idx + 1}/{len(self.train_loader)} | '
                    f'Loss: {train_loss/(batch_idx+1):.3f} | '
                    f'Acc: {100.*correct/total:.2f}%'
                )
        
        if hasattr(self.compressed_optimizer, 'mse_history') and self.compressed_optimizer.mse_history:
            mse = torch.mean(torch.stack(self.compressed_optimizer.mse_history)).item()
            self.compressed_optimizer.mse_history = []
        else: mse = 0.0

        compressed_time = getattr(self.compressed_optimizer, 'compressed_time', 0.0)
        self.compressed_optimizer.compressed_time = 0.0
        
        avg_loss = train_loss / len(self.train_loader)
        avg_acc = 100. * correct / total

        if should_log:
            self.history['train_loss'][epoch] = avg_loss
            self.history['train_acc'][epoch] = avg_acc
            self.history['lr'][epoch] = self.optimizer.param_groups[0]['lr']
            self.history['mse'][epoch] = mse
            self.history['compressed_time'][epoch] = compressed_time
        
        return avg_loss, avg_acc
    
    def test_image(self, epoch: int) -> Tuple[float, float]:
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = test_loss / len(self.test_loader)
        avg_acc = 100. * correct / total
        
        if self.should_log_epoch(epoch):
            self.history['test_loss'][epoch] = avg_loss
            self.history['test_acc'][epoch] = avg_acc
        
        return avg_loss, avg_acc

    def train_epoch_nlp(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.
        start_time = time.time()
        
        is_lstm = isinstance(self.model, LSTMModel)
        is_transformer = isinstance(self.model, TransformerModel)
        
        data_source = self.nlp_data.train_data
        batch_size = data_source.size(1)
        
        if is_lstm:
            hidden = self.model.init_hidden(batch_size)
        if is_transformer:
            src_mask = self.model._generate_square_subsequent_mask(self.bptt).to(self.device)

        # 遍历数据集
        nbatches = (data_source.size(0) - 1) // self.bptt
        for batch, i in enumerate(range(0, data_source.size(0) - 1, self.bptt)):
            data, targets = self.get_batch_nlp(data_source, i)
            data, targets = data.to(self.device), targets.to(self.device) # 确保数据在GPU
            
            self.optimizer.zero_grad()
            
            if is_lstm:
                hidden = repackage_hidden(hidden)
                output, hidden = self.model(data, hidden)
            elif is_transformer:
                if src_mask.size(0) != len(data):
                    src_mask = self.model._generate_square_subsequent_mask(len(data)).to(self.device)
                output = self.model(data, src_mask)
            
            # 计算 Loss
            loss = self.criterion(output.view(-1, self.nlp_data.vocab_size), targets)
            loss.backward()
            
            # === [关键修改] 调用 CompressedOptimizer.step() ===
            # 这里已经包含了梯度裁剪 (在 CompressedOptimizer 内部)
            stats = self.compressed_optimizer.step(current_batch_for_stats=(data, targets), epoch=epoch)
            
            # === [关键修改] 学习率调度 (Per-Step) ===
            if self.scheduler_type == 'step_based':
                self.scheduler.step()

            if stats is not None:
                self.history['compression_stats']['iterations'].append(self.current_iteration)
                self.history['compression_stats']['bit_allocation'][self.current_iteration] = stats['bit_allocation']
                self.history['compression_stats']['compression_ratio'][self.current_iteration] = stats['compression_ratio']
                self.history['compression_stats']['avg_bits'][self.current_iteration] = stats['avg_bits']
            
            self.current_iteration += 1
            total_loss += loss.item()

            if batch % 100 == 0 and batch > 0:
                cur_loss = total_loss / (batch + 1)
                ppl = math.exp(cur_loss) if cur_loss < 20 else float('inf')
                elapsed = time.time() - start_time
                self.log_info(f'| epoch {epoch:3d} | {batch:5d}/{data_source.size(0) // self.bptt:5d} batches | '
                              f'lr {self.optimizer.param_groups[0]["lr"]:02.5f} | ms/batch {elapsed * 1000 / (batch+1):5.2f} | '
                              f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
       
        # ### 新增：在NLP的epoch训练结束后，处理MSE历史记录 ###
        if hasattr(self.compressed_optimizer, 'mse_history') and self.compressed_optimizer.mse_history:
            # 计算这个 epoch 内所有迭代的平均 MSE
            mse_tensor = torch.stack(self.compressed_optimizer.mse_history)
            mse_for_epoch = torch.mean(mse_tensor).item()
            # 清空历史，为下一个 epoch 做准备
            self.compressed_optimizer.mse_history = [] 
        else:
            mse_for_epoch = 0.0

        compressed_time = getattr(self.compressed_optimizer, 'compressed_time', 0.0)
        self.compressed_optimizer.compressed_time = 0.0

        avg_loss = total_loss / (batch + 1)
        avg_ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')

        if self.should_log_epoch(epoch):
            self.history['train_loss'][epoch] = avg_loss
            self.history['train_ppl'][epoch] = avg_ppl
            self.history['lr'][epoch] = self.optimizer.param_groups[0]['lr']
            self.history['mse'][epoch] = mse_for_epoch
            self.history['compressed_time'][epoch] = compressed_time

        return avg_loss, avg_ppl

    def test_nlp(self, epoch: int) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.
        
        is_lstm = isinstance(self.model, LSTMModel)
        is_transformer = isinstance(self.model, TransformerModel)
        
        data_source = self.nlp_data.val_data
        batch_size = data_source.size(1)
        
        if is_lstm:
            hidden = self.model.init_hidden(batch_size)
        if is_transformer:
             src_mask = self.model._generate_square_subsequent_mask(self.bptt).to(self.device)
        
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, self.bptt):
                data, targets = self.get_batch_nlp(data_source, i)
                if is_lstm:
                    output, hidden = self.model(data, hidden)
                    hidden = repackage_hidden(hidden)
                elif is_transformer:
                    if src_mask.size(0) != len(data):
                         src_mask = self.model._generate_square_subsequent_mask(len(data)).to(self.device)
                    output = self.model(data, src_mask)
                total_loss += len(data) * self.criterion(output.view(-1, self.nlp_data.vocab_size), targets).item()

        avg_loss = total_loss / (len(data_source) - 1)
        avg_ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')

        if self.should_log_epoch(epoch):
            self.history['test_loss'][epoch] = avg_loss
            self.history['test_ppl'][epoch] = avg_ppl

        return avg_loss, avg_ppl

    def get_batch_nlp(self, source: torch.Tensor, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = min(self.bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].reshape(-1)
        return data, target

    def train(self) -> Dict:
        self.log_info(f"Starting training: {self.experiment_name}")
        self.log_info(f"Training config: {self.training_config}")
        self.log_info(f"Compression config: {self.compression_config}")
        
        start_time = time.time()
        best_metric = self.history['best_metric']
        current_lr = self.training_config['base_lr']

        # --- 新增: 用于学习率衰减的计数器 ---
        patience_counter = 0
        lr_decay_patience = 5 # 如果连续3个epoch没有提升，就衰减LR

        for epoch in range(1, self.training_config['num_epochs']+1):
            epoch_start_time = time.time()
            
            # --- 训练和评估 ---
            if self.is_nlp_task:
                train_loss, train_ppl = self.train_epoch_nlp(epoch)
                val_loss, val_ppl = self.test_nlp(epoch)
                current_metric = val_ppl
                is_better = current_metric < best_metric
            else:
                train_loss, train_acc = self.train_epoch_image(epoch)
                test_loss, test_acc = self.test_image(epoch)
                current_metric = test_acc
                is_better = current_metric > best_metric
            
            epoch_duration = time.time() - epoch_start_time

            # --- 日志输出 ---
            if self.should_log_epoch(epoch):
                if self.is_nlp_task:
                    self.log_info('-' * 89)
                    self.log_info(f'| end of epoch {epoch:3d} | time: {epoch_duration:5.2f}s | valid loss {val_loss:5.2f} | '
                                  f'valid ppl {val_ppl:8.2f}')
                    self.log_info('-' * 89)
                else:
                    self.log_info(
                        f"\nEpoch {epoch} Summary:\n"
                        f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}\n"
                        f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%\n"
                        f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%"
                    )

            # --- 保存最佳模型 ---
            if is_better:
                best_metric = current_metric
                self.history['best_epoch'] = epoch
                self.history['best_metric'] = best_metric
                self.save_checkpoint(epoch, is_best=True)
                # 重置耐心计数器
                patience_counter = 0 
                if self.is_nlp_task:
                    self.log_info(f"New best model saved with perplexity: {best_metric:.2f}")
                else:
                    self.log_info(f"New best model saved with accuracy: {best_metric:.2f}%")

            # --- 学习率调度 ---
            if self.scheduler is not None:
                # 策略 1 & 2: 使用 LRScheduler 类 (处理 'multistep' 和 'cosine')
                self.scheduler.step()
            elif self.training_config['lr_scheduler'] == 'plateau':
                # 策略 3: 手动处理 'plateau' 逻辑
                if is_better:
                    # 如果性能提升，重置耐心计数器
                    patience_counter = 0
                else:
                    # 如果性能没有提升，增加计数器
                    patience_counter += 1
                
                # 检查耐心是否耗尽
                if patience_counter >= lr_decay_patience:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    # 使用一个合理的衰减因子
                    new_lr = current_lr * self.training_config.get('lr_decay', 0.25) 
                    
                    # 防止学习率过低
                    if new_lr > 1e-6:
                        self.log_info(f"Validation metric did not improve for {patience_counter} epochs. "
                                      f"Decaying learning rate from {current_lr:.6f} to {new_lr:.6f}")
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        patience_counter = 0 # 衰减后重置计数器
                    else:
                        self.log_info("Learning rate is too low. Stopping training might be beneficial.")
                        # 可以在这里选择 break 提前终止训练
                        break
        
        # --- 训练结束后的统计和绘图 ---
        self.history['total_time'] = time.time() - start_time
        
        if (self.training_config.get('use_compression', False) and 
            self.compression_config.get('use_adaptive', False) and
            hasattr(self.compressed_optimizer, 'compressor')):
            
            final_stats = self.compressed_optimizer.compressor.get_bit_allocation_stats()
            self.history['bit_allocation_time_stats'] = final_stats
            
            self.log_info(
                f"\nFinal Bit Allocation Statistics:\n"
                f"Total bit allocation time: {final_stats['total_time']:.3f} seconds\n"
                f"Average allocation time: {final_stats['average_time']*1000:.3f} ms\n"
                f"Total number of allocations: {final_stats['num_allocations']}"
            )

        overall_avg_bits = 0.0
        overall_avg_compression_ratio = 0.0
        if self.training_config.get('use_compression', False) and self.history['compression_stats']['avg_bits']:
            avg_bits_values = list(self.history['compression_stats']['avg_bits'].values())
            compression_ratio_values = list(self.history['compression_stats']['compression_ratio'].values())
            if avg_bits_values: overall_avg_bits = np.mean(avg_bits_values)
            if compression_ratio_values: overall_avg_compression_ratio = np.mean(compression_ratio_values)

        self.history['overall_avg_bits'] = overall_avg_bits
        self.history['overall_avg_compression_ratio'] = overall_avg_compression_ratio

        # 计算平均MSE
        if self.history['mse']:
            mse_values = list(self.history['mse'].values())
            avg_mse_all_epochs = torch.mean(torch.tensor(mse_values)).item() if mse_values else 0.0
        else:
            avg_mse_all_epochs = 0.0
        self.history['avg_mse'] = avg_mse_all_epochs

        np.save(f'{self.exp_dir}/history.npy', self.history)
        self.plot_training_history()

        # --- START OF MODIFICATION ---
        # 生成并绘制 R-D 曲线 (新版：直接使用缓存)
        # 只有在使用压缩和拉格朗日自适应方法时，这个缓存才会有意义
        if (self.training_config.get('use_compression', False) and
            self.compression_config.get('use_adaptive', False) and
            self.compression_config.get('adaptive_method') == 'lagrangian'):
            
            print("\n--- Generating Post-Training Analysis: R-D Curves (from cache) ---")
            
            # 确保压缩器和缓存都存在
            if (hasattr(self.compressed_optimizer, 'compressor') and
                self.compressed_optimizer.compressor is not None and
                self.compressed_optimizer.compressor.rd_data_cache is not None):
                
                # 直接从缓存获取数据
                rd_data_from_cache = self.compressed_optimizer.compressor.rd_data_cache
                self.plot_rd_curves(rd_data_from_cache)
            else:
                print("R-D curve cache not found. This might be because Lagrangian allocation was never triggered.")
        # --- END OF MODIFICATION ---

        
        final_message = f"\nTraining completed in {self.history['total_time']/3600:.2f} hours\n"
        if self.is_nlp_task:
             final_message += f"Best validation perplexity: {best_metric:.2f} at epoch {self.history['best_epoch']}"
        else:
            final_message += f"Best accuracy: {best_metric:.2f}% at epoch {self.history['best_epoch']}"
        
        final_message += (f"\nOverall Avg Bits  : {self.history['overall_avg_bits']:.2f}\n"
                          f"Overall Avg Ratio : {self.history['overall_avg_compression_ratio']:.2f}x\n")
        final_message += f"Average MSE over all epochs: {avg_mse_all_epochs:.12f}\n"
        self.log_info(final_message)
        
        return self.history

    def plot_training_history(self):
        fig = plt.figure(figsize=(20, 12))
        gs = plt.GridSpec(2, 2, height_ratios=[1, 1.2])
        
        # 1. 主指标图 (准确率或困惑度)
        ax1 = fig.add_subplot(gs[0, 0])
        if self.is_nlp_task:
            train_epochs = sorted(list(self.history['train_ppl'].keys()))
            test_epochs = sorted(list(self.history['test_ppl'].keys()))
            train_metric = [self.history['train_ppl'][e] for e in train_epochs]
            test_metric = [self.history['test_ppl'][e] for e in test_epochs]
            ax1.set_title('Model Perplexity', fontsize=14, pad=15)
            ax1.set_ylabel('Perplexity (PPL)', fontsize=12)
            ax1.set_yscale('log') # PPL 常用对数坐标
        else:
            train_epochs = sorted(list(self.history['train_acc'].keys()))
            test_epochs = sorted(list(self.history['test_acc'].keys()))
            train_metric = [self.history['train_acc'][e] for e in train_epochs]
            test_metric = [self.history['test_acc'][e] for e in test_epochs]
            ax1.set_title('Model Accuracy', fontsize=14, pad=15)
            ax1.set_ylabel('Accuracy (%)', fontsize=12)
        
        ax1.plot(train_epochs, train_metric, color='#2E86C1', linewidth=2, marker='o', markersize=4, label='Train')
        ax1.plot(test_epochs, test_metric, color='#E74C3C', linewidth=2, marker='o', markersize=4, label='Validation/Test')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 2. 损失图
        ax2 = fig.add_subplot(gs[0, 1])
        train_loss_epochs = sorted(list(self.history['train_loss'].keys()))
        test_loss_epochs = sorted(list(self.history['test_loss'].keys()))
        train_loss = [self.history['train_loss'][e] for e in train_loss_epochs]
        test_loss = [self.history['test_loss'][e] for e in test_loss_epochs]
        
        ax2.plot(train_loss_epochs, train_loss, color='#2E86C1', linewidth=2, marker='o', markersize=4, label='Train')
        ax2.plot(test_loss_epochs, test_loss, color='#E74C3C', linewidth=2, marker='o', markersize=4, label='Validation/Test')
        ax2.set_title('Model Loss', fontsize=14, pad=15)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # 3. 压缩率图
        if self.training_config.get('use_compression', False):
            ax3 = fig.add_subplot(gs[1, :])
            iterations = self.history['compression_stats']['iterations']
            if iterations:
                compression_ratios = [self.history['compression_stats']['compression_ratio'][i] for i in iterations]
                ax3.plot(iterations, compression_ratios, color='#3498DB', linewidth=2, label='Compression Ratio')
                ax3.set_xlabel('Iteration', fontsize=12)
                ax3.set_ylabel('Compression Ratio', fontsize=12)
                
                # 添加平滑曲线
                # window = len(compression_ratios) // 50 if len(compression_ratios) > 100 else 5
                # if window > 1:
                #     smooth_ratios = np.convolve(compression_ratios, np.ones(window)/window, mode='valid')
                #     smooth_x = iterations[window-1:]
                #     ax3.plot(smooth_x, smooth_ratios, color='#2980B9', linewidth=1.5, linestyle='--', alpha=0.6, label='Smoothed Ratio')

                ax3.legend(fontsize=10)
                ax3.grid(True, linestyle='--', alpha=0.7)
                ax3.set_title('Compression Ratio over Iterations', fontsize=14, pad=15)
        
        plt.tight_layout()
        plt.savefig(f'{self.exp_dir}/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

    # trainer.py -> 在 Trainer 类中

    # trainer.py -> 在 Trainer 类中

    def plot_rd_curves(self, rd_data: Dict[str, List[Dict]]):
        """
        Plots the Rate-Distortion curves for ALL layers, where Distortion is
        measured as Loss Difference. This version is designed to handle a single
        set of R-D data, potentially splitting it into multiple image files if
        the number of layers is large.
        
        Args:
            rd_data: The R-D data generated by the compressor. The 'dist' key
                     in the data points corresponds to loss difference.
        """
        if not rd_data:
            print("No R-D data to plot.")
            return

        layer_names = sorted(list(rd_data.keys()))
        total_layers = len(layer_names)
        
        # --- 参数准备 ---
        plots_per_figure = 12
        num_cols = 4
        num_rows = 3
        num_figures = (total_layers + plots_per_figure - 1) // plots_per_figure
        
        print(f"Total layers to plot: {total_layers}. This will generate {num_figures} image file(s).")
        
        layer_params = {name: param.numel() for name, param in self.model.named_parameters() if name in layer_names}
        
        for fig_idx in range(num_figures):
            start_idx = fig_idx * plots_per_figure
            end_idx = min(start_idx + plots_per_figure, total_layers)
            layers_to_plot_on_this_figure = layer_names[start_idx:end_idx]
            
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(22, 12), squeeze=False)
            
            # --- START OF MODIFICATION: 更新标题以反映内容 ---
            fig.suptitle(f'Rate-Distortion (Bits vs. Loss Difference) Curves (Part {fig_idx + 1}/{num_figures})', 
                         fontsize=20, y=1.0)
            # --- END OF MODIFICATION ---
            
            axes = axes.flatten()

            for i, layer_name in enumerate(layers_to_plot_on_this_figure):
                ax = axes[i]
                points = rd_data[layer_name]
                
                if not points:
                    ax.set_title(f"{layer_name}\n(No R-D data)", fontsize=10)
                    ax.set_visible(False)
                    continue

                # --- START OF MODIFICATION: 从 'dist' 键获取 loss_diff ---
                # 'dist' 键现在存储的是 loss_diff
                bits = [p['bits'] for p in points]
                loss_diffs = [p['dist'] for p in points] 
                
                # 绘制 loss difference，它通常是正值，但我们仍然用 semilogy 以防数值范围大
                # 如果 loss_diffs 可能为0或负数，用常规的 plot 更安全
                # 检查是否存在非正值
                if any(ld <= 0 for ld in loss_diffs):
                    ax.plot(bits, loss_diffs, marker='o', linestyle='-', color='#D9534F', markersize=5)
                else:
                    ax.semilogy(bits, loss_diffs, marker='o', linestyle='-', color='#D9534F', markersize=5)
                
                param_count_str = f"{layer_params.get(layer_name, 0):,}"
                ax.set_title(f"{layer_name}\n(Params: {param_count_str})", fontsize=10, wrap=True)
                ax.set_xlabel('Rate (Bits)', fontsize=9)
                # 更新 Y 轴标签
                ax.set_ylabel('Distortion (Loss Diff)', fontsize=9) 
                # --- END OF MODIFICATION ---

                ax.grid(True, which="both", ls="--", alpha=0.6)
                
                if bits:
                    ax.set_xticks(bits)
                    ax.tick_params(axis='x', labelsize=8)

            for j in range(len(layers_to_plot_on_this_figure), len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # --- START OF MODIFICATION: 更新保存文件名 ---
            save_path = f'{self.exp_dir}/rd_curves_loss_diff_part_{fig_idx + 1}.png'
            # --- END OF MODIFICATION ---
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"R-D curve (loss diff) plot saved to '{save_path}'")