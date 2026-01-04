#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import random
import numpy as np
import torch
import json
from typing import List, Dict, Any, Tuple, Optional, Union
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """
    设置随机种子，确保结果可复现
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"设置随机种子: {seed}")

def load_json(file_path: str) -> Union[Dict, List]:
    """
    加载JSON文件
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        JSON数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json(data: Union[Dict, List], file_path: str):
    """
    保存数据到JSON文件
    
    Args:
        data: 要保存的数据
        file_path: 保存路径
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"数据已保存到: {file_path}")

class RerankDataset(Dataset):
    """重排序数据集类"""
    
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        """
        初始化数据集
        
        Args:
            data: 数据列表
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取数据项"""
        item = self.data[idx]
        
        # 获取查询和文档
        query = item.get("query", "")
        candidates = item.get("candidates", [])
        
        # 随机选择一个候选文档
        if candidates:
            candidate = random.choice(candidates)
            doc_text = candidate.get("text", "")
            doc_score = candidate.get("score", 0.0)
        else:
            doc_text = ""
            doc_score = 0.0
        
        # 对查询和文档进行分词
        query_encoding = self.tokenizer(
            query,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        doc_encoding = self.tokenizer(
            doc_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "query_input_ids": query_encoding["input_ids"].squeeze(0),
            "query_attention_mask": query_encoding["attention_mask"].squeeze(0),
            "doc_input_ids": doc_encoding["input_ids"].squeeze(0),
            "doc_attention_mask": doc_encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(doc_score, dtype=torch.float)
        }

class SequenceRerankDataset(Dataset):
    """序列重排序数据集类"""
    
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512, max_docs: int = 10):
        """
        初始化数据集
        
        Args:
            data: 数据列表
            tokenizer: 分词器
            max_length: 最大序列长度
            max_docs: 最大文档数量
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_docs = max_docs
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取数据项"""
        item = self.data[idx]
        
        # 获取查询和文档序列
        query = item.get("query", "")
        candidates = item.get("candidates", [])
        
        # 按分数排序候选文档
        sorted_candidates = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
        
        # 取前N个文档
        doc_sequence = sorted_candidates[:self.max_docs]
        
        # 填充序列
        while len(doc_sequence) < self.max_docs:
            doc_sequence.append({
                "text": "",
                "score": 0.0
            })
        
        # 对查询进行分词
        query_encoding = self.tokenizer(
            query,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 对文档序列进行分词
        doc_input_ids = []
        doc_attention_mask = []
        labels = []
        
        for doc in doc_sequence:
            doc_text = doc.get("text", "")
            doc_score = doc.get("score", 0.0)
            
            doc_encoding = self.tokenizer(
                doc_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            doc_input_ids.append(doc_encoding["input_ids"].squeeze(0))
            doc_attention_mask.append(doc_encoding["attention_mask"].squeeze(0))
            labels.append(doc_score)
        
        return {
            "query_input_ids": query_encoding["input_ids"].squeeze(0),
            "query_attention_mask": query_encoding["attention_mask"].squeeze(0),
            "doc_input_ids": torch.stack(doc_input_ids),
            "doc_attention_mask": torch.stack(doc_attention_mask),
            "labels": torch.tensor(labels, dtype=torch.float)
        }

def create_dataloader(dataset: Dataset, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批处理大小
        shuffle: 是否打乱数据
        
    Returns:
        数据加载器
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )

def get_device() -> torch.device:
    """
    获取设备
    
    Returns:
        设备
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("使用CPU")
    
    return device

def save_model(model: torch.nn.Module, path: str):
    """
    保存模型
    
    Args:
        model: 模型
        path: 保存路径
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"模型已保存到: {path}")

def load_model(model: torch.nn.Module, path: str):
    """
    加载模型
    
    Args:
        model: 模型
        path: 模型路径
    """
    model.load_state_dict(torch.load(path))
    logger.info(f"模型已从 {path} 加载")

def compute_loss(predictions: torch.Tensor, labels: torch.Tensor, loss_fn: torch.nn.Module) -> torch.Tensor:
    """
    计算损失
    
    Args:
        predictions: 预测值
        labels: 标签
        loss_fn: 损失函数
        
    Returns:
        损失值
    """
    return loss_fn(predictions, labels)

def compute_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    计算评估指标
    
    Args:
        predictions: 预测值
        labels: 标签
        
    Returns:
        评估指标字典
    """
    predictions = predictions.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    # 计算MSE
    mse = np.mean((predictions - labels) ** 2)
    
    # 计算MAE
    mae = np.mean(np.abs(predictions - labels))
    
    # 计算相关系数
    correlation = np.corrcoef(predictions, labels)[0, 1]
    
    return {
        "mse": mse,
        "mae": mae,
        "correlation": correlation
    } 