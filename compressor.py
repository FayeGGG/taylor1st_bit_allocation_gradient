# compressor.py - Transformer/WikiText2 (AdamW Optimized)
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Callable, Any, Optional
import numpy as np
import time
import copy
import random
import math
from utils import LSTMModel, TransformerModel
import torch.nn.functional as F
from metrics import (
    calculate_adamw_taylor_1st_g_t_norm,      
    calculate_adamw_taylor_1st_g_t_plus_1_norm 
)

class GradientCompressor:
    def __init__(self, 
                 method: str = 'uniform',  
                 bits: int = 4,            
                 bit_options: List[int] = [1,2,3,4,5,6,7,8],  
                 drop_ratio: float = 0.9,  
                 alq_k: float = 3.0, # 为ALQ添加一个超参数k
                 use_adaptive: bool = False,
                 adaptive_method: str = 'greedy',
                 device: torch.device = None,
                 kimad_d_factor: int = 1000, # 添加 kimad+ 的 D 因子
                 allocation_metric: str = 'mse', 
                 total_epochs: int = 200
                 ):
        self.method = method
        self._target_bits = bits  
        self.bit_options = sorted(bit_options)
        self.drop_ratio = drop_ratio
        self.use_adaptive = use_adaptive
        self.adaptive_method = adaptive_method

        self.kimad_d_factor = kimad_d_factor

        self.alq_k = alq_k

        self.allocation_metric = allocation_metric
        self.total_epochs = total_epochs
        self.fisher_info = None # 存储 Fisher 信息
        # === [新增] 用于平滑Fisher信息的EMA缓冲区 ===
        self.fisher_ema_buffer = None
        self.fisher_ema_alpha = 0.9 # 平滑系数，越接近1越平滑

        self.layer_params_map = {} # 映射层名到参数列表
        self.name_to_param_map = {} # [新增] 用于 AdamW 状态查找的映射

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            self.device = torch.device("cpu")

        self.error_dict = {}  
        self._total_params = None
        self._total_bit_budget = None
        self._current_bit_allocation = None
        # 添加时间统计相关属性
        self.bit_allocation_times = []
        self.total_bit_allocation_time = 0.0

        self.rd_data_cache = None
        
        # 添加用于loss计算的引用
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.current_lr = None

    def set_model_references(self, model, criterion, optimizer, current_lr):
        """设置用于计算loss和泰勒项的模型引用"""
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.current_lr = current_lr

        # << 新增：构建层名到 Parameter 对象的映射 >>
        # 因为 AdamW 的 state 字典是用 parameter 对象作为 key 的，而不是层名
        self.name_to_param_map = {} 
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.name_to_param_map[name] = param
        
        # << 新增：构建层名到参数的映射，HVP计算需要 >>
        if not self.layer_params_map:
            print("[INFO] Building layer name to parameter map for HVP.")
            # 使用一个更简单的逻辑，直接将参数分组到它们所属的模块名下
            # 这与 optimizer.py 中的 layer_names 逻辑更一致
            temp_map = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    # 'layer1.0.conv1.weight' -> 'layer1.0.conv1'
                    module_name = '.'.join(name.split('.')[:-1])
                    if module_name not in temp_map:
                        temp_map[module_name] = []
                    temp_map[module_name].append(param)
            self.layer_params_map = temp_map

    @property
    def target_bits(self):
        """只读属性，确保目标比特数不被修改"""
        return self._target_bits
    
    def _initialize_stats(self, gradients: Dict[str, torch.Tensor]):
        """初始化和缓存统计信息"""
        if self._total_params is None:
            self._total_params = sum(grad.numel() for grad in gradients.values())
            self._total_bit_budget = self._total_params * self.target_bits
            print(f"Initializing stats: total_params={self._total_params}, "
                  f"target_bits={self.target_bits}, bit_budget={self._total_bit_budget}")
        
    def get_error_feedback(self, name: str, gradient: torch.Tensor) -> torch.Tensor:
        """获取累积误差"""
        if name in self.error_dict:
            error = self.error_dict[name]
            if torch.isnan(error).any() or torch.isinf(error).any():
                print(f"Warning: NaN or Inf detected in error feedback for {name}")
                error = torch.zeros_like(gradient)
                self.error_dict[name] = error
            return error
        return torch.zeros_like(gradient, device=self.device)
        
    def add_error_feedback(self, name: str, original: torch.Tensor, compressed: torch.Tensor):
        """添加误差反馈"""
        error = original - compressed
        if torch.isnan(error).any() or torch.isinf(error).any():
            print(f"Warning: NaN or Inf detected when computing error for {name}")
            error = torch.zeros_like(original)
            
        if name not in self.error_dict:
            self.error_dict[name] = error
        else:
            self.error_dict[name] = self.error_dict[name] + error
            
        # 限制误差大小
        max_norm = 1000.0
        error_norm = torch.norm(self.error_dict[name])
        if error_norm > max_norm:
            self.error_dict[name] = self.error_dict[name] * max_norm / error_norm
            
    def clear_error_feedback(self):
        """清除所有误差"""
        self.error_dict.clear()

    def set_bit_allocation(self, bit_allocation: Dict[str, int]):
        """设置当前的比特分配"""
        self._current_bit_allocation = bit_allocation

    def get_layer_bits(self, layer_name: str) -> int:
        """获取指定层的比特数"""
        if self.use_adaptive and self._current_bit_allocation is not None:
            if layer_name not in self._current_bit_allocation:
                raise KeyError(f"No bit allocation found for layer: {layer_name}")
            return self._current_bit_allocation[layer_name]
        return self._target_bits

    def compress(self, gradient: torch.Tensor, layer_name: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """统一的压缩接口，增加layer_name参数"""
        if gradient is None:
            return None, None
            
        # 获取当前层应使用的比特数
        bits = self.get_layer_bits(layer_name)
            
        if self.method == 'uniform':
            result = self._uniform_quantization(gradient, bits)
        elif self.method == 'qsgd':
            result = self._qsgd_quantization(gradient, bits)
        elif self.method == 'nc':
            result = self._natural_compression(gradient, bits)
        elif self.method == 'alq':
            result = self._alq_quantization(gradient, bits)
        else:
            raise ValueError(f"Unsupported compression method: {self.method}")
        
        return result
    
    def decompress(self, compressed_data: Dict[str, torch.Tensor], rebuild_info: Dict[str, torch.Tensor]) -> torch.Tensor:
        """统一的解压缩接口"""
        if compressed_data is None or rebuild_info is None:
            return None
            
        if self.method == 'uniform':
            return self._uniform_dequantization(compressed_data, rebuild_info)
        elif self.method == 'qsgd':
            return self._qsgd_dequantization(compressed_data, rebuild_info)
        elif self.method == 'nc':
            return self._natural_decompression(compressed_data, rebuild_info)
        elif self.method == 'alq':
            return self._alq_dequantization(compressed_data, rebuild_info)
        else:
             raise ValueError(f"Unsupported decompression for method: {self.method}")
            
            
    def _uniform_quantization(self, gradient: torch.Tensor, bits: int) -> Tuple[torch.Tensor, Dict]:
        """均匀量化，返回量化索引和重建信息"""
        if gradient is None or torch.all(gradient == 0):
            return gradient, None
        
        # 计算量化参数
        max_val = torch.max(gradient)
        min_val = torch.min(gradient)
        range_val = max_val - min_val
        levels = 2**bits - 1
        delta = range_val / levels if levels > 0 else 0
        
        if delta == 0:
            return gradient, None
            
        # 量化为整数索引
        quantized_indices = torch.round((gradient - min_val) / delta)
        quantized_indices = torch.clamp(quantized_indices, 0, levels)
    
        # 打包压缩数据和重建信息
        compressed_data = {
            "indices": quantized_indices
        }
        
        rebuild_info = {
            "min_val": min_val.to(self.device),
            "delta": torch.tensor(delta, device=self.device, dtype=torch.float32)
        }
        
        return compressed_data, rebuild_info
    
    def _uniform_dequantization(self, compressed_data: Dict[str, torch.Tensor], 
                            rebuild_info: Dict[str, torch.Tensor]) -> torch.Tensor:
        """均匀量化的解压缩"""
        if compressed_data is None or rebuild_info is None:
            return None
            
        # 重建原始值
        decompressed = rebuild_info["min_val"] + compressed_data["indices"].float() * rebuild_info["delta"]
        return decompressed

    def _qsgd_quantization(self, gradient: torch.Tensor, bits: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """QSGD量化，使用bits-1位表示量级，1位表示符号"""
        if gradient is None or torch.all(gradient == 0):
            return gradient, None
            
        norm = torch.norm(gradient)
        if norm == 0:
            return gradient, None
            
        levels = 2 << bits - 1
        normalized_grad = gradient / norm
        
        # 符号信息压缩到位级别
        signs = (normalized_grad >= 0).to(torch.uint8)
        
        # 量化绝对值
        abs_grad = torch.abs(normalized_grad)
        level_float = levels * abs_grad
        rand_mask = torch.rand_like(abs_grad) < (level_float - level_float.floor())
        quantized_indices = level_float.floor() + rand_mask.float()
        
        # 将量化索引转换为整数类型
        quantized_indices = quantized_indices.to(torch.uint8)
        
        # 打包符号位和量化索引
        compressed_data = {
            "values": quantized_indices,
            "signs": signs
        }
        
        rebuild_info = {
            "norm": norm,
            "levels": levels
        }
        
        return compressed_data, rebuild_info

    def _qsgd_dequantization(self, compressed_data: Dict, rebuild_info: Dict) -> torch.Tensor:
        if compressed_data is None or rebuild_info is None:
            return None
        
        # 解包数据
        quantized_indices = compressed_data["values"].float() / rebuild_info["levels"]
        signs = (2 * compressed_data["signs"].float() - 1)
        
        # 重建原始值
        decompressed = signs * quantized_indices * rebuild_info["norm"]
        return decompressed

    def _natural_compression(self, gradient: torch.Tensor, bits: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Natural compression实现"""
        if gradient is None or torch.all(gradient == 0):
            return None, None
            
        # 1. 计算范数
        norm = torch.norm(gradient, p=2)
        if norm == 0:
            return None, None
            
        # 2. 归一化
        normalized_grad = gradient / norm
        signs = (normalized_grad >= 0).to(torch.uint8)
        abs_normalized = torch.abs(normalized_grad)
        
        # 3. 生成量化点
        num_levels = 2 ** bits
        exponents = torch.arange(2 - num_levels, 1, device=self.device)
        quantization_points = torch.cat([torch.tensor([0.0], device=self.device), 
                                    torch.pow(2.0, exponents)])
        
        # 4. 找到量化区间
        lower_idx = torch.searchsorted(quantization_points, abs_normalized, right=False) - 1
        upper_idx = torch.clamp(lower_idx + 1, max=len(quantization_points) - 1)
        
        # 5. 随机量化
        lower_bound = quantization_points[lower_idx]
        upper_bound = quantization_points[upper_idx]
        prob = (abs_normalized - lower_bound) / (upper_bound - lower_bound + 1e-12)
        rand_mask = torch.rand_like(prob, device=self.device) < prob
        selected_idx = torch.where(rand_mask, upper_idx, lower_idx).to(torch.uint8)
        
        # 6. 打包压缩数据和重建信息
        compressed_data = {
            "indices": selected_idx,
            "signs": signs
        }
        
        rebuild_info = {
            "norm": norm.to(self.device),
            "quantization_points": quantization_points
        }
        
        return compressed_data, rebuild_info

    def _natural_decompression(self, compressed_data: Dict[str, torch.Tensor], 
                            rebuild_info: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Natural compression解压缩"""
        if compressed_data is None or rebuild_info is None:
            return None
            
        # 1. 获取量化值
        quantized_values = rebuild_info["quantization_points"][compressed_data["indices"].long()]
        
        # 2. 恢复符号
        signs = (2.0 * compressed_data["signs"].float() - 1.0)
        
        # 3. 重建梯度
        decompressed = signs * quantized_values * rebuild_info["norm"]
        return decompressed
    
    def _calculate_alq_alpha(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        为给定的梯度张量动态计算最优的clipping范围 alpha。
        这保留了ALQ的核心思想，即自适应范围，但使其适用于梯度压缩场景。
        """
        if torch.all(gradient == 0):
            return torch.tensor(1e-9, device=self.device) # 避免全零梯度导致alpha为0
            
        abs_grad = torch.abs(gradient)
        mu = torch.mean(abs_grad)
        std = torch.std(abs_grad)
        alpha = mu + self.alq_k * std
        
        # 确保alpha是一个有效的正数
        if alpha.item() <= 1e-9:
            alpha = torch.max(abs_grad)
            if alpha.item() <= 1e-9: # 如果最大值也是0，给一个极小值
                return torch.tensor(1e-9, device=self.device)
        
        return alpha.to(self.device)


    def _alq_quantization(self, gradient: torch.Tensor, bits: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        ALQ 风格的量化，但使用动态计算的alpha，并且分离压缩和解压。
        """
        if gradient is None or torch.all(gradient == 0):
            return None, None # 返回None, None表示压缩失败或无需压缩

        # 1. 动态计算最优 clipping 范围 alpha
        alpha = self._calculate_alq_alpha(gradient)

        # 2. 对梯度进行 clipping
        clipped_grad = torch.clamp(gradient, -alpha, alpha)

        # 3. 计算量化步长 (delta)，为符号位留出1 bit
        # 总共有 2**bits 个等级，对称分布在0两侧，所以正半轴有 2**(bits-1) 个等级
        num_levels_per_side = 2**(bits - 1)
        # 正半轴的量化台阶数是 num_levels_per_side - 1
        delta = alpha / (num_levels_per_side - 1) if num_levels_per_side > 1 else alpha

        if delta.item() < 1e-12: # 防止delta过小导致除零错误
            return None, None
            
        # 4. 量化到整数索引
        # 将 clipped_grad / delta 的结果四舍五入到最近的整数
        quantized_indices = torch.round(clipped_grad / delta)
        quantized_indices = torch.clamp(quantized_indices, 
                                        -(num_levels_per_side - 1), 
                                        (num_levels_per_side - 1)).to(torch.int8) # 使用int8节省空间

        # 5. 打包压缩数据和重建信息
        # 压缩数据只需要传输整数索引
        compressed_data = {
            "indices": quantized_indices 
        }
        
        # 重建信息需要传输步长 delta
        rebuild_info = {
            "delta": delta
        }
        
        return compressed_data, rebuild_info

    def _alq_dequantization(self, compressed_data: Dict[str, torch.Tensor], 
                            rebuild_info: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        ALQ 风格的解压缩。
        """
        if compressed_data is None or rebuild_info is None:
            return None
            
        # 1. 解包数据
        quantized_indices = compressed_data["indices"]
        delta = rebuild_info["delta"]

        # 2. 重建梯度：整数索引 * 步长
        decompressed = quantized_indices.float() * delta
        return decompressed
            
    # def log_bit_allocation(self, result: Dict[str, int], original_grads: Dict[str, torch.Tensor], 
    #                     iteration: int, epoch: int):
    #     """记录每次迭代的比特分配结果"""
    #     print(f"\nEpoch {epoch}, Iteration {iteration} Bit Allocation Details:")
    #     print("=" * 80)
    #     print(f"{'Layer Name':<50} {'Parameters':<12} {'Bits':<6} {'Total Bits':<12} {'Percentage':<10}")
    #     print("-" * 80)
        
    #     total_bits = sum(result[name] * original_grads[name].numel() for name in result.keys())
        
    #     # 按层名称排序以保持输出顺序一致
    #     sorted_names = sorted(result.keys())
        
    #     for name in sorted_names:
    #         params = original_grads[name].numel()
    #         bits = result[name]
    #         layer_bits = params * bits
    #         percentage = (layer_bits / total_bits) * 100
            
    #         print(f"{name:<50} {params:<12} {bits:<6} {layer_bits:<12} {percentage:>8.2f}%")
        
    #     print("-" * 80)
    #     print(f"Total parameters: {sum(original_grads[name].numel() for name in result.keys())}")
    #     print(f"Total bits used: {total_bits}")
    #     print(f"Average bits per parameter: {total_bits/sum(original_grads[name].numel() for name in result.keys()):.2f}")
    #     print("=" * 80)

    def calculate_compression_stats(self, gradients: Dict[str, torch.Tensor], 
                                  bit_allocation: Dict[str, int] = None) -> Dict:
        """计算压缩统计信息"""
        # 确保使用相同的预算
        self._initialize_stats(gradients)
        
        if not self.use_adaptive or bit_allocation is None:
            total_bits = self._total_params * self.target_bits
        else:
            total_bits = sum(bit_allocation[name] * grad.numel() 
                           for name, grad in gradients.items())
        
        assert total_bits <= self._total_bit_budget, \
            f"Total bits ({total_bits}) exceeds budget ({self._total_bit_budget})"
            
        return {
            'total_params': self._total_params,
            'total_bits': total_bits,
            'total_bit_budget': self._total_bit_budget,
            'avg_bits': total_bits / self._total_params,
            'compression_ratio': 32 / (total_bits / self._total_params),
            'bit_allocation': bit_allocation if bit_allocation else {
                name: self.target_bits for name in gradients.keys()
            }
        }

    def _quantize(self, gradient: torch.Tensor, bits: int) -> torch.Tensor:
        """统一的量化接口，返回解压缩后的梯度"""
        if gradient is None:
            return None
                
        if self.method == 'uniform':
            compressed_data, rebuild_info = self._uniform_quantization(gradient, bits)
        elif self.method == 'qsgd':
            compressed_data, rebuild_info = self._qsgd_quantization(gradient, bits)
        elif self.method == 'nc':
            compressed_data, rebuild_info = self._natural_compression(gradient, bits)
        elif self.method == 'alq':
            compressed_data, rebuild_info = self._alq_quantization(gradient, bits)
        else:
            raise ValueError(f"Unsupported compression method: {self.method}")

        # 如果压缩失败返回原始梯度
        if compressed_data is None or rebuild_info is None:
            raise RuntimeError(f"Compression failed with {bits} bits")

        # 解压缩返回重建梯度
        return self.decompress(compressed_data, rebuild_info)

    def _direct_quantize(self, gradient: torch.Tensor, bits: int) -> torch.Tensor:
        """直接量化梯度，不经过传输过程"""
        if gradient is None:
            return None
            
        # 均匀量化的直接实现
        if self.method == 'uniform':
            max_val = torch.max(gradient)
            min_val = torch.min(gradient)
            range_val = max_val - min_val
            levels = 2**bits - 1
            delta = range_val / levels if levels > 0 else 0
            
            if delta == 0:
                return gradient
                
            # 直接量化和反量化
            quantized_indices = torch.round((gradient - min_val) / delta)
            quantized_indices = torch.clamp(quantized_indices, 0, levels)
            quantized_grad = min_val + quantized_indices * delta
            
            return quantized_grad
            
        # QSGD的直接实现
        elif self.method == 'qsgd':
            norm = torch.norm(gradient)
            if norm == 0:
                return gradient
                
            levels = 2 << bits - 1
            normalized_grad = gradient / norm
            
            # 保存符号
            signs = torch.sign(normalized_grad)
            
            # 量化绝对值
            abs_grad = torch.abs(normalized_grad)
            level_float = levels * abs_grad
            rand_mask = torch.rand_like(abs_grad) < (level_float - level_float.floor())
            quantized_indices = level_float.floor() + rand_mask.float()
            
            # 直接重建
            quantized_grad = signs * (quantized_indices / levels) * norm
            
            return quantized_grad
            
        # Natural compression的直接实现
        elif self.method == 'nc':
            norm = torch.norm(gradient, p=2)
            if norm == 0:
                return gradient
                
            normalized_grad = gradient / norm
            signs = torch.sign(normalized_grad)
            abs_normalized = torch.abs(normalized_grad)
            
            # 生成量化点
            num_levels = 2 ** bits
            exponents = torch.arange(2 - num_levels, 1, device=self.device)
            quantization_points = torch.cat([torch.tensor([0.0], device=self.device), 
                                        torch.pow(2.0, exponents)])
            
            # 找到量化区间
            lower_idx = torch.searchsorted(quantization_points, abs_normalized, right=False) - 1
            upper_idx = torch.clamp(lower_idx + 1, max=len(quantization_points) - 1)
            
            # 随机量化
            lower_bound = quantization_points[lower_idx]
            upper_bound = quantization_points[upper_idx]
            prob = (abs_normalized - lower_bound) / (upper_bound - lower_bound + 1e-12)
            rand_mask = torch.rand_like(prob, device=self.device) < prob
            selected_idx = torch.where(rand_mask, upper_idx, lower_idx)
            
            # 直接重建
            # quantized_values = torch.gather(quantization_points, 0, selected_idx.long())
            quantized_values = quantization_points[selected_idx.long()]
            quantized_grad = signs * quantized_values * norm
            
            return quantized_grad
        
        elif self.method == 'alq':
            if torch.all(gradient == 0):
                return gradient

            # 1. 动态计算最优 clipping 范围 alpha
            alpha = self._calculate_alq_alpha(gradient)

            # 2. 对梯度进行 clipping
            clipped_grad = torch.clamp(gradient, -alpha, alpha)

            # 3. 计算量化步长 (delta)
            num_levels_per_side = 2**(bits - 1)
            delta = alpha / (num_levels_per_side - 1) if num_levels_per_side > 1 else alpha

            if delta.item() < 1e-12:
                return gradient # 如果步长过小，则不量化，返回原始梯度

            # 4. 量化到整数索引并立即反量化
            quantized_indices = torch.round(clipped_grad / delta)
            # 注意：这里不需要clamp，因为clipped_grad/delta的范围天然就在[-(levels-1), levels-1]
            quantized_grad = quantized_indices * delta
            
            return quantized_grad

        else:
            raise ValueError(f"Unsupported compression method: {self.method}")

    # === [新增] 设置 Fisher 信息的方法 ===
    def set_fisher_info(self, fisher_info: Dict[str, torch.Tensor]):
        """设置 Fisher 信息矩阵（对角线）"""
        self.fisher_info = fisher_info


    # === [新增] NewMetric 计算方法 ===
    def _calculate_new_metric(self, g_orig: torch.Tensor, g_quant: torch.Tensor, beta: float = 0.5) -> float:
        """计算 NewMetric (方向 + 大小)，修正了类型问题"""
        epsilon = 1e-12
        error_tensor = g_orig - g_quant
        g_orig_flat, g_quant_flat = g_orig.flatten(), g_quant.flatten() 
        norm_g_orig, norm_g_quant = torch.norm(g_orig_flat), torch.norm(g_quant_flat)
        
        cos_theta = 0.0
        if norm_g_orig > epsilon and norm_g_quant > epsilon:
            # 结果是一个 0-dim Tensor
            cos_theta = torch.clamp(torch.dot(g_orig_flat, g_quant_flat) / (norm_g_orig * norm_g_quant), -1.0, 1.0)
        
        # 无论 cos_theta 是 float 还是 Tensor，direction_term 都会是 float 或 Tensor
        direction_term = 1 - cos_theta

        magnitude_term = 0.0
        if torch.max(norm_g_orig, norm_g_quant) > epsilon:
            # 结果是一个 0-dim Tensor
            magnitude_term = torch.norm(error_tensor) / torch.max(norm_g_orig, norm_g_quant)

        # 这里的 result 可能是 float 或 Tensor
        result = beta * direction_term + (1 - beta) * magnitude_term
        
        # === [关键修正] ===
        # 检查 result 是否是 Tensor，如果是，才调用 .item()
        if isinstance(result, torch.Tensor):
            return result.item()
        else:
            # 如果已经是 float，直接返回
            return result
    
    def set_fisher_info(self, fisher_info: Dict[str, torch.Tensor]):
        """
        接收新的Fisher信息，并使用EMA进行平滑。
        """
        if self.fisher_ema_buffer is None:
            # 第一次接收时，直接初始化EMA缓冲区
            print("Initializing Fisher EMA buffer.")
            self.fisher_ema_buffer = fisher_info
        else:
            print("Updating Fisher EMA buffer.")
            # 使用EMA更新
            with torch.no_grad():
                for name in self.fisher_ema_buffer.keys():
                    if name in fisher_info:
                        # ema_new = alpha * ema_old + (1 - alpha) * new_value
                        self.fisher_ema_buffer[name].mul_(self.fisher_ema_alpha).add_(
                            fisher_info[name], alpha=1 - self.fisher_ema_alpha
                        )
        
        # 将 self.fisher_info 指向平滑后的结果
        self.fisher_info = self.fisher_ema_buffer

    # === Fisher Weighted Error 计算方法 ===
    def _calculate_fisher_error(self, g_orig: torch.Tensor, g_quant: torch.Tensor, layer_name: str) -> float:
        """计算 Fisher 加权误差，并进行归一化处理以增强稳定性"""
        if self.fisher_info is None or layer_name not in self.fisher_info:
            return torch.norm(g_orig - g_quant).pow(2).item()
            
        error = g_orig - g_quant
        fisher = self.fisher_info[layer_name].to(g_orig.device)
        
        # === [关键修改] 对 Fisher 信息进行归一化 ===
        # 1. 避免零值，添加一个小的 epsilon
        epsilon = 1e-12
        fisher_norm = fisher + epsilon
        
        # 2. 将 Fisher 值归一化到 [0, 1] 区间，或者使用 log 变换
        # 方法A: 简单的最大值归一化 (推荐)
        # 这保留了层内的相对重要性，同时限制了其绝对大小
        if fisher_norm.max() > 0:
            fisher_norm = fisher_norm / fisher_norm.max()
            
        # 方法B: Log变换 (更激进的平滑)
        # fisher_norm = torch.log1p(fisher / fisher.mean()) # log1p(x) = log(1+x)
        
        # 使用归一化后的 Fisher 信息
        return torch.sum(fisher_norm * (error**2)).item()

    def _calculate_taylor2_error(self, grad_ref: torch.Tensor, quantized_grad: torch.Tensor, layer_name: str, data_batches: list) -> float:
        """计算二阶泰勒项(HVP)失真"""
        if not data_batches:
            print(f"Warning: No data batches for HVP calculation on layer {layer_name}. Returning 0.")
            return 0.0
        if layer_name not in self.layer_params_map:
            print(f"Warning: Cannot find parameters for layer {layer_name} for HVP. Returning 0.")
            return 0.0

        delta_g = (quantized_grad - grad_ref).flatten()
        
        total_error = 0.0
        # 在多个batch上平均HVP结果以获得更稳定的估计
        for inputs, targets in data_batches:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # 调用 metrics.py 中的函数
            error = calculate_taylor_2nd_error(
                model=self.model,
                loss_fn=self.criterion,
                inputs=inputs,
                targets=targets,
                layer_params_list=self.layer_params_map[layer_name],
                delta_g_layer=delta_g,
                learning_rate=self.current_lr
            )
            total_error += error.item()
        
        # 返回平均值，失真应为正
        return abs(total_error / len(data_batches))

    # === 统一的失真计算接口 ===
    def _calculate_distortion(self, grad: torch.Tensor, bits: int, layer_name: str, 
                              data_batches: List[Tuple[torch.Tensor, torch.Tensor]], 
                              current_epoch: int = None,
                              grads_t: Dict[str, torch.Tensor] = None,
                              grads_t_plus_1: Dict[str, torch.Tensor] = None,
                              grads_t_minus_1: Dict[str, torch.Tensor] = None) -> float:
        """
        根据 self.allocation_metric 计算指定层的量化失真。
        返回的是该层的 **总失真** (Total Distortion)。
        支持 AdamW 的准确泰勒展开模拟。
        """
        # 1. 首先对梯度进行量化
        quantized_grad = self._direct_quantize(grad, bits)

        # 2. 获取对应的 Parameter 对象 (AdamW 状态查找必需)
        current_param = self.name_to_param_map.get(layer_name, None)
        
        # 3. 根据选定的度量计算失真
        if self.allocation_metric == 'mse':
             # Total MSE = ||grad - quantized_grad||^2
             mse_total = torch.norm(grad - quantized_grad).pow(2).item()
             return mse_total
             
        elif self.allocation_metric == 'loss_diff':
             # loss_diff 本身就是一个标量，代表整个 batch loss 的变化
             # _get_avg_loss_difference 返回的就是平均 loss difference
             return self._get_avg_loss_difference({layer_name: grad}, bits, layer_name, data_batches)
             
        
        elif self.allocation_metric == 'taylor1':
            # 基础一阶泰勒: g_t^T * Delta_Step(g_t)
            if grads_t is None: raise ValueError("`grads_t` is required.")
            g_t = grads_t[layer_name]
            g_quant = self._direct_quantize(g_t, bits)
            
            # AdamW 模拟
            if self.optimizer and current_param is not None:
                return calculate_adamw_taylor_1st_g_t_norm(
                    g_t=g_t, g_t_quant=g_quant, optimizer=self.optimizer, 
                    param=current_param, num_params=1
                )

        elif self.allocation_metric == 'taylor1_gt1':
            # 一阶泰勒: g_{t+1}^T * Delta_Step(g_t)
            if grads_t_plus_1 is None: raise ValueError("`grads_t_plus_1` is required.")
            g_t = grads_t[layer_name]
            g_t1 = grads_t_plus_1[layer_name]
            g_quant = self._direct_quantize(g_t, bits)
            
            # AdamW 模拟
            if self.optimizer and current_param is not None:
                return calculate_adamw_taylor_1st_g_t_plus_1_norm(
                    g_t=g_t, g_t_quant=g_quant, g_t_plus_1=g_t1,
                    optimizer=self.optimizer, param=current_param, num_params=1
                )
            
        # === 处理两种新的历史指标 ===
        
        # 【指标 1】taylor1_gt_minus_1: 量化当前 g_t，投影到 g_{t-1}
        if self.allocation_metric == 'taylor1_gt_minus_1':
            if grads_t is None: raise ValueError("`grads_t` required.")
            
            # 如果是第一步，没有 t-1，回退到普通的 taylor1 (使用 g_t)
            if grads_t_minus_1 is None or layer_name not in grads_t_minus_1:
                if grads_t is None: raise ValueError("`grads_t` is required.")
                g_t = grads_t[layer_name]
                g_quant = self._direct_quantize(g_t, bits)
                if self.optimizer and current_param is not None:
                    return calculate_adamw_taylor_1st_g_t_norm(
                        g_t=g_t, g_t_quant=g_quant, optimizer=self.optimizer, param=current_param, num_params=1
                    )
                else:
                    return abs(calculate_taylor_1st_error(g_t, g_quant - g_t, self.current_lr).item())


            g_t = grads_t[layer_name]        # 当前梯度
            g_prev = grads_t_minus_1[layer_name] # 历史梯度
            
            g_quant = self._direct_quantize(g_t, bits) # 量化的是 g_t
            
            # 计算 g_{t-1}^T * (Step(g_q) - Step(g_t))
            # 我们可以复用 calculate_adamw_taylor_1st_g_t_plus_1_norm
            # 因为数学形式是一样的：Ref_Vector^T * Delta_Step
            # 这里 Ref_Vector 是 g_prev
            error = calculate_adamw_taylor_1st_g_t_plus_1_norm(
                g_t=g_t,
                g_t_quant=g_quant,
                g_t_plus_1=g_prev, # <--- 将 g_{t-1} 作为投影目标传入
                optimizer=self.optimizer,
                param=current_param,
                num_params=1
            )
            return error

        # 【指标 2】taylor1_fully_historical: 量化 g_{t-1}，投影到 g_{t-1}
        # 这是“延迟指标”：假设 t 时刻的最佳比特分配应该和 t-1 时刻的一样
        elif self.allocation_metric == 'taylor1_fully_historical':
            # 如果是第一步，没有 t-1，回退到普通的 taylor1 (使用 g_t)
            if grads_t_minus_1 is None or layer_name not in grads_t_minus_1:
                # [修复递归错误] 第一步没有历史，直接执行 taylor1 逻辑，不要递归调用
                if grads_t is None: raise ValueError("`grads_t` is required.")
                g_t = grads_t[layer_name]
                g_quant = self._direct_quantize(g_t, bits)
                if self.optimizer and current_param is not None:
                    return calculate_adamw_taylor_1st_g_t_norm(
                        g_t=g_t, g_t_quant=g_quant, optimizer=self.optimizer, param=current_param, num_params=1
                    )
                else:
                    return abs(calculate_taylor_1st_error(g_t, g_quant - g_t, self.current_lr).item())
                    
            # 完全使用历史梯度
            g_prev = grads_t_minus_1[layer_name]
            
            # 注意：这里量化的是 g_prev！不是传入的 grad (g_t)
            g_prev_quant = self._direct_quantize(g_prev, bits)
            
            # 计算 g_{t-1}^T * (Step(Q(g_{t-1})) - Step(g_{t-1}))
            # 这就是标准的 taylor1 逻辑，只是输入全是 g_prev
            error = calculate_adamw_taylor_1st_g_t_norm(
                g_t=g_prev,     # <--- 输入 g_{t-1}
                g_t_quant=g_prev_quant, # <--- 输入 Q(g_{t-1})
                optimizer=self.optimizer,
                param=current_param,
                num_params=1
            )
            return error


        elif self.allocation_metric == 'taylor2':
            if grads_t is None or layer_name not in grads_t:
                raise ValueError("`grads_t` is required for 'taylor2' metric but not provided.")
            g_t = grads_t[layer_name]
            return self._calculate_taylor2_error(g_t, self._direct_quantize(g_t, bits), layer_name, data_batches)

        elif self.allocation_metric == 'taylor1+2':
            if grads_t is None or layer_name not in grads_t:
                raise ValueError("`grads_t` is required for 'taylor1+2' metric but not provided.")
            g_t = grads_t[layer_name]
            quantized_g_t = self._direct_quantize(g_t, bits)

            # 计算一阶和二阶项
            t1_error = self._calculate_taylor1_error(g_t, quantized_g_t, layer_name)
            t2_error = self._calculate_taylor2_error(g_t, quantized_g_t, layer_name, data_batches)
            
            # 返回它们的和，因为两者都已是正的失真值
            return t1_error + t2_error

        elif self.allocation_metric == 'dynamic':
            # 动态切换
            if current_epoch is None or self.total_epochs is None:
                 print("Warning: Epoch info missing for dynamic metric, defaulting to Fisher.")
                 return self._calculate_fisher_error(grad, quantized_grad, layer_name)

            stage_ratio = current_epoch / self.total_epochs if self.total_epochs > 0 else 0
            # 定义 "Early" 阶段，例如前 40%
            if stage_ratio < 0.4: 
                  # Early stage: Use NewMetric to scale the total MSE
                  # This keeps the distortion sensitive to the absolute gradient scale
                  total_mse = torch.norm(grad - quantized_grad).pow(2).item()
                  new_metric_val = self._calculate_new_metric(grad, quantized_grad, beta=0.5)
                  # return (1 + new_metric_val) * total_mse
                  return new_metric_val
            else:
                # 后期阶段: 使用 Fisher Error
                total_fisher_error = self._calculate_fisher_error(grad, quantized_grad, layer_name)
                return total_fisher_error
        else:
             raise ValueError(f"Unknown allocation metric: {self.allocation_metric}")
    
    def log_bit_allocation(self, bit_allocation: Dict[str, int], 
                      original_grads: Dict[str, torch.Tensor]):
        """优化后的比特分配结果显示"""
        # 1. 计算关键指标
        total_params = sum(grad.numel() for grad in original_grads.values())
        bit_budget = total_params * self._target_bits
        used_bits = sum(bit_allocation[name] * original_grads[name].numel() 
                    for name in bit_allocation)

        # 3. 显示总体统计
        print("-" * 70)
        print(f"Bit Budget     : {bit_budget:,}")
        print(f"Actually Used  : {used_bits:,}")
        print(f"Budget Usage   : {used_bits/bit_budget*100:.2f}%")
        
        # 4. 验证是否超出预算
        if used_bits > bit_budget:
            print("\nWARNING: Used bits exceed budget!")

    def optimize_bit_allocation(self, original_grads: Dict[str, torch.Tensor],
                                data_batches_for_rd: List[Tuple[torch.Tensor, torch.Tensor]],
                                current_epoch: int = None, current_iter: int = None,
                                grads_t: Dict[str, torch.Tensor] = None,
                                grads_t_plus_1: Dict[str, torch.Tensor] = None,
                                grads_t_minus_1: Dict[str, torch.Tensor] = None) -> Dict[str, int]:
        """各进程独立进行比特分配优化"""
        if not self.use_adaptive:
            return {name: self._target_bits for name in original_grads.keys()}

        if self._current_bit_allocation is None:
            self._current_bit_allocation = {name: self._target_bits for name in original_grads.keys()}
        
        # 基于梯度独立计算最优分配
        if self.adaptive_method == 'lagrangian':
            bit_allocation = self.optimize_bit_allocation_lagrangian(original_grads, data_batches_for_rd, current_epoch, current_iter, grads_t, grads_t_plus_1, grads_t_minus_1)
        elif self.adaptive_method == 'kimad_dp':
            # Kimad+ 使用L2误差，不需要数据batch
            bit_allocation = self.optimize_bit_allocation_kimad_dp(original_grads, current_epoch, current_iter)
        else:
            # bit_allocation = self.optimize_bit_allocation_greedy(original_grads, current_epoch, current_iter)
            bit_allocation = self.optimize_bit_allocation_greedy(original_grads, data_batches_for_rd, current_epoch, current_iter, grads_t, grads_t_plus_1, grads_t_minus_1)
        
        self._current_bit_allocation = bit_allocation
        return bit_allocation

    def optimize_bit_allocation_greedy(self, original_grads: Dict[str, torch.Tensor],
                                    data_batches_for_rd: List[Tuple[torch.Tensor, torch.Tensor]],
                                    current_epoch: int = None, 
                                    current_iter: int = None,
                                    grads_t: Dict[str, torch.Tensor] = None,
                                    grads_t_plus_1: Dict[str, torch.Tensor] = None,
                                    grads_t_minus_1: Dict[str, torch.Tensor] = None) -> Dict[str, int]:
        """确保总比特数不超过阈值的基于贪婪算法的比特分配算法"""
        if not self.use_adaptive:
            return {name: self._target_bits for name in original_grads.keys()}

        # 1. 初始化
        layer_names = list(original_grads.keys())
        total_params = sum(grad.numel() for grad in original_grads.values())
        bit_budget = total_params * self._target_bits

        # 开始计时
        start_time = time.time()

        # 2. 计算每层的压缩误差
        layer_errors = {}
        for name, grad in original_grads.items():
            min_bits = min(self.bit_options)

             # === 使用统一接口计算总失真 ===
            total_distortion = self._calculate_distortion(
                    grad, min_bits, name, data_batches_for_rd, current_epoch,
                    grads_t, grads_t_plus_1, grads_t_minus_1 # << 传递额外参数
            )
            # 归一化误差
            num_params = grad.numel()
            average_error = total_distortion / num_params if num_params > 0 else 0
            layer_errors[name] = average_error
            
            # print(f"[Greedy] Layer {name}: {num_params} params, loss_diff={avg_loss_diff:.6f}, normalized={average_error:.8f}")

        # 3. 初始分配最小比特数
        current_allocation = {name: min(self.bit_options) for name in layer_names}
        used_bits = sum(current_allocation[name] * original_grads[name].numel() 
                    for name in layer_names)
        remaining_budget = bit_budget - used_bits
        
        iteration = 0
        # 4. 基于误差大小分配剩余比特
        while remaining_budget > 0:
            iteration += 1
            # 找出当前误差最大且未达到最大比特数的层
            candidates = [(name, layer_errors[name]) 
                        for name in layer_names 
                        if current_allocation[name] < max(self.bit_options)]
            
            if not candidates:
                print(f"[Greedy] No more candidates, stopping at iteration {iteration}")
                break
                
            # 选择误差最大的层
            name, current_error = max(candidates, key=lambda x: x[1])
            params = original_grads[name].numel()
            
            # 计算可以增加的比特数
            current_bits = current_allocation[name]
            next_bits = min([b for b in self.bit_options if b > current_bits], 
                        default=max(self.bit_options))
            bits_needed = (next_bits - current_bits) * params
            
            # 确保不超过预算
            if bits_needed <= remaining_budget:
                current_allocation[name] = next_bits
                remaining_budget -= bits_needed

                # print(f"[Greedy] Iter {iteration}: Upgraded {name} from {current_bits} to {next_bits} bits (error={current_error:.8f})")

                # === 更新该层的误差，使用统一接口 ===
                total_distortion = self._calculate_distortion(
                    original_grads[name], next_bits, name, data_batches_for_rd, current_epoch,
                    grads_t, grads_t_plus_1, grads_t_minus_1 # << 传递额外参数
                )
                layer_errors[name] = total_distortion / params if params > 0 else 0
            else:
                # print(f"[Greedy] Cannot upgrade {name}: need {bits_needed} bits but only {remaining_budget} remaining")
                break

        allocation_time = time.time() - start_time
        self.bit_allocation_times.append(allocation_time)
        self.total_bit_allocation_time += allocation_time

        final_used_bits = sum(current_allocation[name] * original_grads[name].numel() 
                         for name in layer_names)
    
        print(f"[Greedy] Allocation completed in {allocation_time:.2f}s")
        print(f"[Greedy] Final allocation: {final_used_bits}/{bit_budget} bits used ({final_used_bits/bit_budget*100:.1f}%)")
        
        assert sum(current_allocation[name] * original_grads[name].numel() 
                for name in layer_names) <= bit_budget, "Exceeded bit budget!"
        
        self._current_bit_allocation = current_allocation
        return current_allocation
    
    
    def optimize_bit_allocation_kimad_dp(self, original_grads: Dict[str, torch.Tensor],
                                         current_epoch: int = None, 
                                         current_iter: int = None) -> Dict[str, int]:
        """
        [DEFINITIVE FIX] 使用标准的二维动态规划和正确的回溯逻辑，
        忠实复现 Kimad+ 的分组背包问题解法。
        """
        print("\n=== [DEFINITIVE FIX] Kimad+ Style Bit Allocation with 2D-DP ===")
        start_time = time.time()
        
        # 1. 初始化
        # 过滤掉不需要梯度的参数（虽然理论上都有，但为了健壮性）
        layer_names = [name for name, grad in original_grads.items() if grad.numel() > 0]
        num_layers = len(layer_names)
        
        valid_grads = {name: original_grads[name] for name in layer_names}
        total_params = sum(grad.numel() for grad in valid_grads.values())
        bit_budget_real = total_params * self._target_bits

        # 预算离散化：Kimad 论文通常设置 d_factor=1000 左右，将背包容量离散化
        # 防止 bit_step 为 0
        bit_step = max(1, int(total_params / getattr(self, 'kimad_d_factor', 1000)))
        discrete_budget = int(bit_budget_real / bit_step)
        
        print(f"Target Bit Budget: {int(bit_budget_real):,}. DP Discrete Budget={discrete_budget} (step={bit_step})")

        # 2. 预计算成本和误差
        print("Step 1: Pre-computing costs and errors...")
        # costs[i][j]: 第 i 层的第 j 个比特选项的离散成本 (向上取整以保守估计)
        costs = [[math.ceil((bits * valid_grads[name].numel()) / bit_step) 
                  for bits in self.bit_options] 
                 for name in layer_names]
        
        # errors[i][j]: 第 i 层的第 j 个比特选项的 L2 误差平方
        errors = np.zeros((num_layers, len(self.bit_options)))
        with torch.no_grad():
            for i, name in enumerate(layer_names):
                grad = valid_grads[name]
                for j, bits in enumerate(self.bit_options):
                    # 注意：这里必须用 L2 误差平方 (Sum of Squared Errors)
                    quant_grad = self._direct_quantize(grad, bits)
                    # Kimad 使用的是 ||g - Q(g)||^2
                    errors[i, j] = torch.sum((grad - quant_grad) ** 2).item()

        # 3. 动态规划求解 (分组背包问题)
        print("Step 2: Solving with Group Knapsack DP...")
        
        # dp[i][b] = 考虑前 i 组（层），恰好/累计使用预算 b 的最小误差
        dp = np.full((num_layers + 1, discrete_budget + 1), np.inf)
        
        # backtrack[i][b] = 记录第 i 组在预算 b 时选择了哪个比特选项索引
        backtrack = np.zeros((num_layers + 1, discrete_budget + 1), dtype=np.int8)
        
        # === [CRITICAL FIX] 初始化 ===
        # 只有 "0层、0预算" 是合法初始状态。
        # 0层消耗非0预算是不可能的，所以保持 inf。
        dp[0, 0] = 0

        for i in range(1, num_layers + 1): # i 表示第 i 层 (1-based in DP table)
            layer_idx = i - 1
            # 优化：不需要遍历所有预算，只遍历可能达到的范围
            # 但为了代码简单清晰，遍历全量预算通常也可以（numpy很快）
            for j, bits in enumerate(self.bit_options):
                cost = costs[layer_idx][j]
                error = errors[layer_idx][j]
                
                # 状态转移：dp[i, b] = min(dp[i-1, b-cost] + error)
                # 使用 numpy 切片加速：同时更新所有可能的 b
                # valid_indices 是那些上一层状态不为 inf 的索引
                
                # Python 循环写法 (易于理解):
                # for b in range(cost, discrete_budget + 1):
                #     if dp[i-1, b - cost] != np.inf:
                #         new_err = dp[i-1, b - cost] + error
                #         if new_err < dp[i, b]:
                #             dp[i, b] = new_err
                #             backtrack[i, b] = j
                
                # Numpy 向量化写法 (加速):
                # 找到上一层所有的有效状态
                prev_layer_costs = dp[i-1, :]
                valid_mask = prev_layer_costs != np.inf
                
                # 计算当前状态的位置：上一层的位置 + cost
                # 我们只需要考虑那些 (prev_idx + cost) <= discrete_budget 的情况
                # 这是一个稍微复杂的向量化，为了保证正确性，这里建议用半向量化或上面的循环
                # 为了稳健性，这里保留上面的显式循环逻辑的优化版：
                
                lower_bound = cost
                # 只有当 b >= cost 且 dp[i-1, b-cost] 有值时才更新
                # 我们可以遍历 budget b
                for b in range(lower_bound, discrete_budget + 1):
                    prev_val = dp[i-1, b - cost]
                    if prev_val != np.inf:
                         new_val = prev_val + error
                         if new_val < dp[i, b]:
                             dp[i, b] = new_val
                             backtrack[i, b] = j

        # 4. 回溯找到最优解
        print("Step 3: Backtracking to find the optimal allocation...")
        
        # 在最后一行 (考虑了所有层) 找到最小误差
        # 注意：我们必须检查是否真的找到了解 (即 min value 不是 inf)
        min_error = np.min(dp[num_layers])
        
        if min_error == np.inf:
            print("[WARNING] DP failed to find a feasible solution within budget!")
            # 降级策略：全部分配最小比特
            return {name: min(self.bit_options) for name in original_grads}

        # 找到最小误差对应的预算索引
        # np.argmin 会返回第一个出现的最小值索引。
        # 在误差相同的情况下，我们倾向于使用较小的预算吗？是的。
        # 如果误差随着预算增加而单调递减，最小值通常出现在预算较大处。
        final_best_budget = np.argmin(dp[num_layers])
        
        final_allocation = {}
        current_budget = int(final_best_budget)

        for i in range(num_layers, 0, -1):
            layer_idx = i - 1
            layer_name = layer_names[layer_idx]
            
            # 获取选择
            bit_option_idx = backtrack[i, current_budget]
            final_allocation[layer_name] = self.bit_options[bit_option_idx]
            
            # 更新剩余预算
            cost = costs[layer_idx][bit_option_idx]
            current_budget -= cost
            
            # 安全检查：预算不应小于0 (如果逻辑正确，不会发生)
            if current_budget < 0:
                print(f"[ERROR] Backtracking logic error at layer {layer_name}")
                current_budget = 0

        # 补充遗漏的层（如果有）
        min_bits = min(self.bit_options)
        for name in original_grads:
            if name not in final_allocation:
                final_allocation[name] = min_bits

        # 5. 结果展示与验证
        allocation_time = time.time() - start_time
        self.bit_allocation_times.append(allocation_time)
        self.total_bit_allocation_time += allocation_time

        final_used_bits = sum(final_allocation.get(name, 0) * original_grads[name].numel() for name in original_grads)
        
        print(f"DP search completed in {allocation_time:.3f}s.")
        print(f"Final minimum error (L2 sum): {min_error:.4f}")
        print(f"Total Bits Used: {int(final_used_bits):,} / {int(bit_budget_real):,} ({final_used_bits / bit_budget_real * 100:.2f}%)")

        # 允许微小的误差 (由于 ceil 离散化)
        if final_used_bits > bit_budget_real:
             print(f"[WARNING] Allocation slightly exceeds budget due to discretization: {final_used_bits} > {bit_budget_real}")

        self._current_bit_allocation = final_allocation
        self.log_bit_allocation(final_allocation, original_grads)
        
        return final_allocation


    def _compute_loss_difference_for_single_batch(self, original_grad_subset: Dict[str, torch.Tensor], bits: int, layer_name: str, 
                                                inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        [私有辅助函数] 为单个batch计算loss difference。
        这个函数不依赖 self.current_inputs 或 self.current_targets。
        """
        # (这部分代码基本就是原来的 compute_loss_difference_simplified 函数)
        original_model_state = copy.deepcopy(self.model.state_dict())
        
        try:
            temp_model = copy.deepcopy(self.model)
            temp_model.eval()

            is_lstm = isinstance(self.model, LSTMModel)
            is_transformer = isinstance(self.model, TransformerModel)
            
            # --- 计算应用“原始”梯度更新后的loss ---
            with torch.no_grad():
                for name, param in temp_model.named_parameters():
                    if name == layer_name:
                        # 使用传入的梯度子集
                        grad = original_grad_subset[name].clone()
                        param.data.add_(grad, alpha=-self.current_lr)
                        break
                
                if is_lstm:
                    hidden = temp_model.init_hidden(inputs.size(1))
                    original_outputs, _ = temp_model(inputs, hidden)
                elif is_transformer:
                    src_mask = temp_model._generate_square_subsequent_mask(inputs.size(0)).to(self.device)
                    original_outputs = temp_model(inputs, src_mask)
                else:
                    original_outputs = temp_model(inputs)
                
                # NLP任务的 target 形状可能需要调整
                if is_lstm or is_transformer:
                    original_updated_loss = self.criterion(original_outputs.view(-1, self.model.decoder.out_features), targets.view(-1)).item()
                else:
                    original_updated_loss = self.criterion(original_outputs, targets).item()

            # 恢复模型状态
            temp_model.load_state_dict(original_model_state)

            # --- 计算应用“量化”梯度更新后的loss ---
            with torch.no_grad():
                for name, param in temp_model.named_parameters():
                    if name == layer_name:
                        grad = original_grad_subset[name].clone()
                        quant_grad = self._direct_quantize(grad, bits)
                        param.data.add_(quant_grad, alpha=-self.current_lr)
                        break
                
                if is_lstm:
                    hidden = temp_model.init_hidden(inputs.size(1))
                    quantized_outputs, _ = temp_model(inputs, hidden)
                elif is_transformer:
                    src_mask = temp_model._generate_square_subsequent_mask(inputs.size(0)).to(self.device)
                    quantized_outputs = temp_model(inputs, src_mask)
                else:
                    quantized_outputs = temp_model(inputs)

                if is_lstm or is_transformer:
                    quantized_updated_loss = self.criterion(quantized_outputs.view(-1, self.model.decoder.out_features), targets.view(-1)).item()
                else:
                    quantized_updated_loss = self.criterion(quantized_outputs, targets).item()

            return abs(quantized_updated_loss - original_updated_loss)
                
        finally:
            pass

    def _get_avg_loss_difference(
        self, 
        original_grad_subset: Dict[str, torch.Tensor],
        bits: int, 
        layer_name: str, 
        data_batches: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> float:
        """
        计算在多个数据batch上的平均loss difference。
        """
        if not data_batches:
            print("Warning: No data batches provided for loss difference calculation.")
            return 0.0

        total_loss_diff = 0.0
        for inputs, targets in data_batches:
            # 确保数据在正确的设备上
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            loss_diff = self._compute_loss_difference_for_single_batch(
                original_grad_subset, bits, layer_name, inputs, targets
            )
            total_loss_diff += loss_diff
        
        return total_loss_diff / len(data_batches)

    def enforce_monotonicity(self, rd_points: List[Dict]) -> List[Dict]:
        # 1. 按rate排序
        rd_points.sort(key=lambda p: p['rate'])
        
        monotonic_points = []
        if not rd_points:
            return []
            
        # 2. 总是接受第一个点（最低比特）
        monotonic_points.append(rd_points[0])
        
        # 3. 遍历剩余的点
        for i in range(1, len(rd_points)):
            current_point = rd_points[i]
            last_monotonic_point = monotonic_points[-1]
            
            # 只有在失真更低时才接受
            if current_point['dist'] < last_monotonic_point['dist']:
                # 确保rate也增加了，避免重复rate
                if current_point['rate'] > last_monotonic_point['rate']:
                    monotonic_points.append(current_point)
        
        return monotonic_points


    def optimize_bit_allocation_lagrangian(self, original_grads: Dict[str, torch.Tensor],
                                           data_batches_for_rd: List[Tuple[torch.Tensor, torch.Tensor]], # << 新增参数
                                           current_epoch: int = None, current_iter: int = None,
                                           grads_t: Dict[str, torch.Tensor] = None,
                                           grads_t_plus_1: Dict[str, torch.Tensor] = None,
                                           grads_t_minus_1: Dict[str, torch.Tensor] = None) -> Dict[str, int]:
        """
        使用二分搜索和更精确的λ范围估算的拉格朗日乘数法。
        """
        if not self.use_adaptive:
            return {name: self._target_bits for name in original_grads.keys()}

        print("\n=== Lagrangian Bit Allocation with Bisection Search (using Loss Difference) ===")
        start_time = time.time()
        
        layer_names = list(original_grads.keys())
        total_params = sum(grad.numel() for grad in original_grads.values())
        bit_budget = total_params * self._target_bits
        param_counts = {name: grad.numel() for name, grad in original_grads.items()}

        print(f"Target Bit Budget: {bit_budget:,.0f}. This may take a while...")

        # 1. 为每个层计算所有比特选项的率(R)和失真(D)
        # D: 归一化的 loss difference (per parameter)
        # R: 比特数 (per parameter)
        rd_data = {}
        min_lambdas, max_lambdas = [], []

        print("Step 1: Calculating Rate-Distortion points for each layer...")
        for idx, (name, grad) in enumerate(original_grads.items()):
            if grad.numel() == 0:
                continue
            
            # 打印进度，因为这一步会很慢
            # print(f"  Processing layer {idx+1}/{len(layer_names)}: {name} ...", end='', flush=True)
            # layer_start_time = time.time()
            
            rd_points = []
            for bits in self.bit_options:

                # === 使用统一接口计算总失真 ===
                total_distortion = self._calculate_distortion(
                    grad, bits, name, data_batches_for_rd, current_epoch,
                    grads_t, grads_t_plus_1, grads_t_minus_1 # << 传递额外参数
                )      
                
                # 归一化失真和比特率
                distortion_per_param = total_distortion / grad.numel() if grad.numel() > 0 else 0
                rate_per_param = bits
                
                rd_points.append({'rate': rate_per_param, 'dist': distortion_per_param, 'bits': bits})
            
            # 按比特率排序，确保单调性
            rd_points.sort(key=lambda p: p['rate'])
            rd_data[name] = rd_points

            # 强制rd曲线单调
            monotonic_rd_points = self.enforce_monotonicity(rd_points)
            rd_data[name] = monotonic_rd_points
            
            # layer_duration = time.time() - layer_start_time
            # print(f" done in {layer_duration:.2f}s.")

            # 2. 估算λ的搜索范围
            # λ ≈ |-ΔD/ΔR|。我们计算相邻点之间的斜率来确定范围。
            for i in range(len(monotonic_rd_points) - 1):
                d1, r1 = monotonic_rd_points[i]['dist'], monotonic_rd_points[i]['rate']
                d2, r2 = monotonic_rd_points[i+1]['dist'], monotonic_rd_points[i+1]['rate']
                
                # ΔR > 0 and ΔD < 0 (失真随比特增加而减少)
                if r2 > r1 and d1 > d2:
                    # 斜率的绝对值
                    slope = abs((d2 - d1) / (r2 - r1))
                    if slope > 1e-25:  # 避免无效或极小的斜率
                        min_lambdas.append(slope)
                        max_lambdas.append(slope)
        # 仅在缓存为空（即第一次调用此方法）时，填充 R-D 数据缓存。
        # 这样可以确保我们保存的是训练最开始时的 R-D 曲线。
        if self.rd_data_cache is None:
            print("Caching R-D data from the initial allocation.")
            self.rd_data_cache = rd_data

        # 缓存最后一次计算出的 R-D 数据，以供后续可视化使用，避免重复计算
        # self.rd_data_cache = rd_data

        if not min_lambdas or not max_lambdas:
            print("Warning: Could not determine a valid lambda range. Using default allocation.")
            return {name: self._target_bits for name in original_grads.keys()}

        # 确定一个健壮的搜索范围
        lambda_low = min(min_lambdas) * 1e-10
        lambda_high = max(max_lambdas) * 1000.0
        
        print("\nStep 2: Performing Bisection Search for optimal Lambda.")
        print(f"Estimated Lambda Search Range: [{lambda_low:.2e}, {lambda_high:.2e}]")

        # 3. 使用二分搜索寻找最优 λ
        best_allocation = None
        best_total_bits = 0
        
        # 迭代次数可以根据需要的精度调整，30-50次通常足够
        for i in range(100):
            lambda_mid = (lambda_low + lambda_high) / 2
            if lambda_mid < 1e-25: break # 防止lambda过小

            current_allocation = {}
            current_total_bits = 0
            
            # 对于给定的λ，为每层找到成本J = D + λR最小的最优比特率
            for name in layer_names:
                if name not in rd_data: continue
                
                costs = [p['dist'] + lambda_mid * p['rate'] for p in rd_data[name]]
                best_idx = np.argmin(costs)
                chosen_bits = rd_data[name][best_idx]['rate']
                
                current_allocation[name] = chosen_bits
                current_total_bits += chosen_bits * param_counts[name]

            # 根据总比特数调整λ的搜索范围
            if current_total_bits > bit_budget:
                # 比特用超了，说明λ太小，对R的惩罚不够，需要增大λ
                lambda_low = lambda_mid
            else:
                # 比特没用完，这是一个可行的解。我们记录下来，
                # 然后尝试用更小的λ来使用更多比特，以期获得更低的失真。
                lambda_high = lambda_mid
                best_allocation = current_allocation
                best_total_bits = current_total_bits

            # 检查收敛
            if (lambda_high - lambda_low) / lambda_low < 1e-4:
                print(f"  Bisection search converged at iteration {i+1}.")
                break
        
        print("\nStep 3: Finalizing allocation.")
        # 如果二分搜索结束后没有找到任何可行解（这在lambda_low设置合理时不太可能发生）
        if best_allocation is None:
            print("Warning: No feasible solution found during search. Allocating minimum bits.")
            best_allocation = {name: min(self.bit_options) for name in layer_names}
            best_total_bits = sum(min(self.bit_options) * p for p in param_counts.values())

        # 4. 结果展示
        allocation_time = time.time() - start_time
        self.bit_allocation_times.append(allocation_time)
        self.total_bit_allocation_time += allocation_time

        print(f"Search completed in {allocation_time:.3f}s. Final Lambda ≈ {lambda_high:.2e}")
        print(f"Total Bits Used: {best_total_bits:,.0f} ({best_total_bits / bit_budget * 100:.2f}% of budget)")
        
        final_bits = sum(best_allocation[name] * param_counts[name] for name in layer_names)
        assert final_bits <= bit_budget, f"FATAL: Final allocation exceeds budget: {final_bits} > {bit_budget}"

        self._current_bit_allocation = best_allocation
        self.log_bit_allocation(best_allocation, original_grads) # 使用你已有的日志函数

        return best_allocation


    def get_bit_allocation_stats(self) -> Dict:
        """获取比特分配时间统计信息"""
        if not self.bit_allocation_times:
            return {
                'total_time': 0.0,
                'average_time': 0.0,
                'min_time': 0.0,
                'max_time': 0.0,
                'num_allocations': 0
            }
            
        return {
            'total_time': self.total_bit_allocation_time,
            'average_time': np.mean(self.bit_allocation_times),
            'min_time': np.min(self.bit_allocation_times),
            'max_time': np.max(self.bit_allocation_times),
            'num_allocations': len(self.bit_allocation_times)
        }