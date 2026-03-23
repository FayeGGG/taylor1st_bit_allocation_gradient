# utils.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet34, resnet50, mobilenet_v2, densenet121
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import os 
from torch.utils.data import Dataset 
from PIL import Image 

### 新增：导入timm和torchtext
import timm
from torchtext.datasets import PennTreebank, WikiText103
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import math
import torchtext
torchtext.disable_torchtext_deprecation_warning()

from datasets import load_dataset
from tqdm import tqdm


def get_image_dataset(dataset_name: str, data_path: str = './data') -> tuple:
    """准备图像数据集"""
    if dataset_name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
        num_classes = 10
    elif dataset_name == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)
        num_classes = 100
    ### 新增：ImageNet数据集处理
    elif dataset_name == 'imagenet':
        traindir = os.path.join(data_path, 'train')
        valdir = os.path.join(data_path, 'val') # 验证集目录现在可以直接用
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        
        # 训练集和验证集现在都使用 ImageFolder
        trainset = torchvision.datasets.ImageFolder(traindir, transform=transform_train)
        testset = torchvision.datasets.ImageFolder(valdir, transform=transform_test)
        num_classes = 1000
    else:
        raise ValueError(f"Unknown image dataset: {dataset_name}")
    
    return trainset, testset, num_classes

### ----------------------------------------------------------------
### NLP 任务
### ----------------------------------------------------------------

class NLPData:
    """一个简单的包装器，用于NLP数据和词汇表"""
    def __init__(self, train_data, val_data, test_data, vocab):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.vocab = vocab
        self.vocab_size = len(vocab)

def get_nlp_dataset(dataset_name: str, batch_size: int, bptt_len: int, device: torch.device) -> NLPData:

    tokenizer = get_tokenizer('basic_english')
    
    """准备NLP数据集"""
    if dataset_name == 'ptb':
       # --- PennTreebank: 使用 torchtext 内置迭代器的逻辑 ---
        print("Loading PennTreebank...")
        train_iter, val_iter, test_iter = PennTreebank()
        
        # 1. 构建词汇表 (逻辑移入此块)
        vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
        vocab.set_default_index(vocab['<unk>'])
        
        # 重新创建迭代器，因为上面的操作已经消耗了它
        train_iter, val_iter, test_iter = PennTreebank()
        
        # 2. 处理数据 (逻辑移入此块)
        def data_process_from_iter(raw_text_iter):
            data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
            return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

        train_data = data_process_from_iter(train_iter)
        val_data = data_process_from_iter(val_iter)
        test_data = data_process_from_iter(test_iter)

    elif dataset_name == 'wikitext103':
        print("Loading WikiText-103 from Hugging Face Datasets...")
        try:
            # 尝试从缓存加载，如果失败则下载
            dataset = load_dataset("wikitext", "wikitext-103-v1", trust_remote_code=True)
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            print("Please ensure you have an internet connection and the `datasets` library is installed.")
            raise
            
        print("Dataset loaded. Preparing iterators and building vocabulary...")
        
        # 1. 创建迭代器，过滤空行
        def create_iter_from_dataset(split):
            return (line for line in dataset[split]['text'] if line.strip())

        # 2. 构建词汇表
        train_iter_for_vocab = create_iter_from_dataset('train')
        vocab = build_vocab_from_iterator(map(tokenizer, train_iter_for_vocab), specials=['<unk>'], max_tokens=50000)
        vocab.set_default_index(vocab['<unk>'])
        print(f"Vocabulary built. Size: {len(vocab)}")

        # 3. 处理数据 (将文本转换为ID张量)
        def data_process_from_iter(raw_text_iter, desc):
            # 使用tqdm来显示进度
            tokenized_iter = tqdm(map(tokenizer, raw_text_iter), desc=f"Tokenizing {desc}")
            data = [torch.tensor(vocab(tokens), dtype=torch.long) for tokens in tokenized_iter]
            return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

        train_data = data_process_from_iter(create_iter_from_dataset('train'), "train")
        val_data = data_process_from_iter(create_iter_from_dataset('validation'), "validation")
        test_data = data_process_from_iter(create_iter_from_dataset('test'), "test")
        print("\n[Data Preprocessing] All data processed for WikiText-103.")

    elif dataset_name == 'wikitext2':
        # --- WikiText-2: 直接从文件加载的逻辑 ---
        data_path = './data/wikitext-2/'
        if not os.path.exists(os.path.join(data_path, 'wiki.train.tokens')):
            raise FileNotFoundError(
                f"WikiText-2 files not found in '{data_path}'. "
                f"Please download and place wiki.train.tokens, wiki.valid.tokens, "
                f"and wiki.test.tokens in that directory."
            )

        # 1. 构建词汇表 (逻辑移入此块)
        def yield_tokens(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    yield tokenizer(line)

        print("\n[Data Preprocessing] Step 1: Building vocabulary from training set...")
        train_filepath = os.path.join(data_path, 'wiki.train.tokens')
        vocab = build_vocab_from_iterator(
            tqdm(yield_tokens(train_filepath), desc="Building Vocab"),
            specials=['<unk>']
        )
        vocab.set_default_index(vocab['<unk>'])
        print(f"Vocabulary built. Size: {len(vocab)}")

        # 2. 处理数据：将 token 转换为 ID (逻辑移入此块)
        def data_process_from_file(file_path, desc: str):
            print(f"\n[Data Preprocessing] Step 2: Processing {desc} data...")
            with open(file_path, 'r', encoding='utf-8') as f:
                ids = [vocab[token] for line in tqdm(f, desc=f"Tokenizing {desc}") for token in tokenizer(line.strip()) if token]
            return torch.tensor(ids, dtype=torch.long)
        
        train_data = data_process_from_file(os.path.join(data_path, 'wiki.train.tokens'), "train")
        val_data = data_process_from_file(os.path.join(data_path, 'wiki.valid.tokens'), "validation")
        test_data = data_process_from_file(os.path.join(data_path, 'wiki.test.tokens'), "test")
        print("\n[Data Preprocessing] All data processed.")

    else:
        raise ValueError(f"Unsupported NLP dataset: {dataset_name}. Please use 'wikitext2' or 'ptb'.")



    def batchify(data, bsz):
        # 确保数据在CPU上进行操作，以防原始数据在其他设备上
        data = data.to(torch.device("cpu"))
        seq_len = data.size(0) // bsz
        data = data[:seq_len * bsz]
        data = data.view(bsz, seq_len).t().contiguous()
        return data.to(device) # 最后再移动到目标设备

    train_data = batchify(train_data, batch_size)
    val_data = batchify(val_data, batch_size)
    test_data = batchify(test_data, batch_size)
    
    return NLPData(train_data, val_data, test_data, vocab)

### ----------------------------------------------------------------
### 模型定义
### ----------------------------------------------------------------

def get_model(model_name: str, num_classes: int = None, vocab_size: int = None, ninp: int = 256, nhid: int = 256, nlayers: int = 2, nhead: int = 2, dropout: float = 0.5) -> nn.Module:
    """获取模型"""
    
    # === ResNet 系列 ===
    if model_name in ['resnet18', 'resnet34', 'resnet50']:
        # 1. 加载标准模型结构
        if model_name == 'resnet18':
            model = resnet18(num_classes=num_classes)
        elif model_name == 'resnet34':
            model = resnet34(num_classes=num_classes)
        elif model_name == 'resnet50':
            model = resnet50(num_classes=num_classes)
            
        # 2. 针对 CIFAR-10/100 (32x32图片) 修改第一层结构
        # 标准 ResNet 适用于 224x224，第一层下采样太厉害，必须修改
        if num_classes in [10, 100]: 
            # 将 7x7, stride 2 改为 3x3, stride 1
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # 移除 maxpool (用 Identity 替换)
            model.maxpool = nn.Identity()      
        return model
    
    # === MobileNetV2 (新增: 强烈推荐用于期刊实验) ===
    elif model_name == 'mobilenetv2':
        model = mobilenet_v2(num_classes=num_classes)
        if num_classes in [10, 100]:
            # MobileNetV2 第一层默认是 stride=2，对于CIFAR会导致信息过早丢失
            # 修改 features[0][0] (即第一个 ConvBNReLU)
            model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        return model

    # === DenseNet121 (新增: 可选用于增加多样性) ===
    elif model_name == 'densenet121':
        model = densenet121(num_classes=num_classes)
        if num_classes in [10, 100]:
            # DenseNet 第一层结构与 ResNet 类似 (7x7 conv + maxpool)
            model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.features.pool0 = nn.Identity()
        return model

    ### ViT-small
    elif model_name == 'vit_small_patch16_224':
        # 使用timm创建模型
        return timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=num_classes)
    ### LSTM
    elif model_name == 'lstm':
        return LSTMModel(vocab_size, ninp, nhid, nlayers, dropout) # 2层LSTM
    ### Transformer
    elif model_name == 'transformer':
        # ninp: embedding dimension
        # nhead: a hyperparameter for multi-head attention 
        return TransformerModel(vocab_size, ninp, nhead, nhid, nlayers, dropout)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

### LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, hidden):
        emb = self.drop(self.encoder(src))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, decoded.size(2))
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))

### 新增：Transformer 模型
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

       # === [新增] Weight Tying (权重绑定) ===
        # 将 Embedding 的权重共享给输出层
        # 这对于降低 PPL 至关重要
        self.decoder.weight = self.encoder.weight 
        # ====================================


    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

        # === [新增] 重新初始化 Transformer Encoder 内部参数 ===
        # PyTorch 的 Transformer 默认初始化已经不错，但为了保险：
        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        # return output.view(-1, output.size(2))
        return output # <--- 直接返回原始形状的输出