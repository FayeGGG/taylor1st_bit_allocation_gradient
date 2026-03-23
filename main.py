# main.py - Single GPU Version
import os
import torch
import time
import argparse
import numpy as np
from torch.utils.data import DataLoader
from trainer import Trainer
from utils import (
    get_image_dataset, get_nlp_dataset, get_model # ### 修改 ###
)


def main():
    parser = argparse.ArgumentParser(description='Gradient Compression Experiments - Single GPU')
    
    # 基础设置
    parser.add_argument('--model', type=str, choices=['resnet18', 'resnet34', 'resnet50', 'mobilenetv2', 'densenet121' 'vit_small_patch16_224', 'lstm', 'transformer'], default='resnet18',
                       help='Model architecture to use')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet', 'ptb', 'wikitext103', 'wikitext2'], default='cifar10',
                       help='Dataset to use')
    parser.add_argument('--data-path', type=str, default='./data',
                       help='Path to the root of the dataset directory.')
    parser.add_argument('--method', type=str, choices=['uniform', 'qsgd', 'nc', 'alq', 'all', 'full'], 
                       default='qsgd',
                       help='Method to use. Use "full" for full-precision training')
    parser.add_argument('--run-adaptive', action='store_true',
                       help='Whether to run adaptive version')
    parser.add_argument('--adaptive-method', type=str, choices=['greedy', 'lagrangian', 'kimad_dp'], 
                        default='lagrangian',
                        help='Method for adaptive bit allocation (greedy or lagrangian)')
    parser.add_argument('--kimad-d-factor', type=int, default=1000,
                        help='Error discretization factor D for the Kimad+ DP solver.')
    
    # 压缩设置
    parser.add_argument('--alq-k', type=float, default=3.0,
                   help='The hyperparameter k for calculating alpha in ALQ (alpha = mu + k * sigma)')
    parser.add_argument('--bits', type=int, default=4,
                       help='Default number of bits for base method quantization')
    parser.add_argument('--bit-options', type=int, nargs='+', default=[1,2,3,4,5,6,7,8],
                       help='Bit options for adaptive quantization')
    parser.add_argument('--reallocation-interval', type=int, default=20,
                       help='Minimum interval (in iterations) between bit reallocations.')
    parser.add_argument('--realloc-sim-threshold', type=float, default=0.9,
                       help='Trigger reallocation if cosine similarity of consecutive gradients is below this.')
    parser.add_argument('--grad-stats-batch-size', type=int, default=5,
                       help='Number of batches to accumulate for gradient statistics for allocation.')
    parser.add_argument('--adaptive-trigger', type=str, choices=['iterative', 'anchor_vs_current'], default='anchor_vs_current',
                       help='Trigger method for adaptive bit allocation.')
    parser.add_argument('--adaptive-metric', type=str, choices=['cosine_similarity', 'relative_change'], default='cosine_similarity',
                       help='Metric to measure the change for triggering allocation.')
    # === 比特分配时使用的误差度量 ===
    parser.add_argument('--allocation-metric', type=str, 
                        choices=['mse', 'loss_diff', 'fisher', 'dynamic', 
                                 'taylor1', 'taylor1_gt1', 'taylor1_gt_minus_1', 'taylor2', 'taylor1+2'
                                 'taylor1_gt_minus_1',       # 用 g_{t-1} 做投影方向
                                 'taylor1_fully_historical'  # 完全用 g_{t-1} 代理计算 (量化对象和投影方向都是 t-1)
                                 ],
                        default='mse',
                        help='Metric used to calculate distortion for bit allocation. '
                             'mse: L2 norm square; '
                             'loss_diff: Actual loss difference (expensive); '
                             'fisher: Fisher Weighted Error; '
                             'dynamic: NewMetric (Early) -> Fisher (Late);'
                             'taylor1: 1st-order Taylor approx using g_t;'
                             'taylor1_gt1: 1st-order Taylor approx using g_{t+1};'
                             'taylor1_gt_minus_1: 1st-order Taylor approx using g_{t-1}'
                             'taylor2: 2nd-order Taylor approx (HVP);'
                             'taylor1+2: Combined 1st and 2nd order Taylor approx.')


    # 训练超参数
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--base-lr', type=float, default=0.1,
                       help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                       help='Weight decay factor')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum factor')
    
    # 明确的优化器选择 ###
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam', 'adamw'], default='sgd',
                        help='Optimizer to use.')

    # NLP任务特定参数
    parser.add_argument('--bptt', type=int, default=35,
                        help='Sequence length for BPTT (for NLP tasks)')

    # 学习率调度策略
    parser.add_argument('--lr-scheduler', type=str, choices=['multistep', 'cosine', 'plateau'], default='multistep',
                        help="Learning rate scheduler type. 'plateau' reduces LR on validation metric stagnation.")
    parser.add_argument('--milestones', type=int, nargs='+', default=[80, 120],
                        help='Epochs at which to reduce learning rate (for multistep)')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='Learning rate decay factor (e.g., 0.1 for multistep, 0.25 for adaptive PTB)')
    
    # Warmup 设置
    parser.add_argument('--warmup', action='store_true',
                        help='Use learning rate warmup')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--warmup-lr', type=float, default=0.001,
                        help='Initial learning rate for warmup')
    parser.add_argument('--warmup-steps', type=int, default=0,
                        help='Number of warmup steps for Linear Warmup (override warmup-epochs)')
    
    # === Transformer 模型结构参数 ===
    parser.add_argument('--ninp', type=int, default=200, 
                       help='Embedding dimension (e.g., 200 for WT2, 512 for WT103)')
    parser.add_argument('--nhid', type=int, default=200, 
                       help='Feedforward dimension (e.g., 200 for WT2, 2048 for WT103)')
    parser.add_argument('--nlayers', type=int, default=2, 
                       help='Number of Transformer layers')
    parser.add_argument('--nhead', type=int, default=2, 
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2, 
                       help='Dropout rate')
    
    # 优化器设置
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping threshold')
    parser.add_argument('--error-feedback', action='store_true',
                        help='Use error feedback in compression')
    parser.add_argument('--use-momentum', action='store_true', 
                        help='Use momentum in compression')
    
    # GPU设置
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device to use (default: 0)')
    
    # 其他设置
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of data loading workers')
    parser.add_argument('--log-freq', type=int, default=10,
                       help='Frequency of logging training statistics (epochs)')
    
    args = parser.parse_args()
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # ### 新增：任务类型判断 ###
    is_nlp_task = args.dataset in ['ptb', 'wikitext103', 'wikitext2']
    
    # ### 修改：根据任务类型加载数据和模型 ###
    train_loader, test_loader, nlp_data = None, None, None
    model = None

    if is_nlp_task: # nlp
        print(f"Preparing NLP task: {args.model} on {args.dataset}")
        nlp_data = get_nlp_dataset(args.dataset, args.batch_size, args.bptt, device)
        model = get_model(
            args.model, 
            vocab_size=nlp_data.vocab_size,
            ninp=args.ninp,
            nhid=args.nhid,
            nlayers=args.nlayers,
            nhead=args.nhead,
            dropout=args.dropout
        ).to(device)

    else: # image classification
        print(f"Preparing Image task: {args.model} on {args.dataset}")
        print(f"Using data path: {args.data_path}")
        # ImageNet 的特殊处理：utils.py 期望的路径是包含 train/ 和 val/ 的根目录
        if args.dataset == 'imagenet':
            dataset_root_path = args.data_path
        else:
            # 对于 CIFAR 等，data_path 是它们将被下载到的地方
            dataset_root_path = args.data_path
        
        train_dataset, test_dataset, num_classes = get_image_dataset(
            dataset_name=args.dataset,
            data_path=dataset_root_path # 将路径传递给函数
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        model = get_model(args.model, num_classes=num_classes).to(device)
    
    # 创建压缩配置
    compression_config = {
        'method': args.method,
        'bits': args.bits,
        'bit_options': args.bit_options,
        'use_adaptive': args.run_adaptive,
        'adaptive_method': args.adaptive_method,
        'device': device,
        'alq_k': args.alq_k,
        'allocation_metric': args.allocation_metric,
        'total_epochs': args.epochs,
        'kimad_d_factor': args.kimad_d_factor
    }

    experiment_name = f"{args.model}_{args.dataset}_{args.method}_{args.bits}"
    if args.run_adaptive:
        experiment_name += f"_adaptive_{args.adaptive_method}"
        experiment_name += f"_{args.adaptive_trigger}_{args.adaptive_metric}"
        experiment_name += f"_{args.allocation_metric}" 
    else:
        experiment_name += "_base"
    
    # 添加时间戳避免冲突
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{experiment_name}_{timestamp}"
    
    # 创建基础目录
    base_dirs = [
        'experiments',
        'checkpoints',
        'results',
        os.path.join('experiments', experiment_name),
        os.path.join('experiments', experiment_name, 'logs')
    ]
    for d in base_dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created directory: {d}")
    
    # 创建训练配置
    training_config = {
        'num_epochs': args.epochs,
        'base_lr': args.base_lr,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'lr_scheduler': args.lr_scheduler,
        'milestones': args.milestones,
        'lr_decay': args.lr_decay,
        'use_warmup': args.warmup,
        'warmup_epochs': args.warmup_epochs,
        'warmup_steps': args.warmup_steps,
        'warmup_lr': args.warmup_lr,
        'grad_clip': args.grad_clip,
        'error_feedback': args.error_feedback,
        'use_momentum': args.use_momentum,
        'log_freq': args.log_freq,
        'use_compression': (args.method != 'full'),
        'min_reallocation_interval': args.reallocation_interval, # 复用这个参数作为最小间隔
        'realloc_sim_threshold': args.realloc_sim_threshold,
        'grad_stats_batch_size': args.grad_stats_batch_size,
        'adaptive_trigger_method': args.adaptive_trigger,
        'adaptive_metric': args.adaptive_metric,
        'optimizer': args.optimizer, # 新增
        'lr_scheduler': args.lr_scheduler, # 新增
        'bptt': args.bptt, # ### 新增 ###
    }
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        nlp_data=nlp_data,
        compression_config=compression_config,
        training_config=training_config,
        experiment_name=experiment_name,
        device=device
    )
    
    results = trainer.train()
    return results

if __name__ == '__main__':
    main()