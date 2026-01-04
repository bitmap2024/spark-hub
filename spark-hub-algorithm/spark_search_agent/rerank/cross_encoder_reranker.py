#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from utils import set_seed, get_device, save_model, load_model, compute_loss, compute_metrics

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """基于BERT的交叉编码器重排序类"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        device: str = "cuda",
        seed: int = 42
    ):
        """
        初始化交叉编码器重排序器
        
        Args:
            model_path: 预训练模型路径
            batch_size: 批处理大小
            learning_rate: 学习率
            device: 设备类型
            seed: 随机种子
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = get_device() if device == "cuda" else torch.device("cpu")
        self.seed = seed
        
        # 设置随机种子
        set_seed(seed)
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path if model_path else "bert-base-chinese"
        )
        
        # 初始化模型
        self.model = CrossEncoderModel(
            model_path if model_path else "bert-base-chinese"
        ).to(self.device)
        
        # 初始化优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )
        
        # 初始化损失函数
        self.loss_fn = nn.MSELoss()
        
        logger.info(f"初始化交叉编码器重排序器: model_path={model_path}, device={self.device}")
    
    def train(self, data: List[Dict[str, Any]], num_epochs: int = 10):
        """
        训练模型
        
        Args:
            data: 训练数据
            num_epochs: 训练轮数
        """
        logger.info(f"开始训练交叉编码器重排序模型，共 {num_epochs} 轮")
        
        # 创建数据加载器
        from utils import RerankDataset, create_dataloader
        dataset = RerankDataset(data, self.tokenizer)
        dataloader = create_dataloader(dataset, self.batch_size)
        
        # 训练循环
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            total_samples = 0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in progress_bar:
                # 将数据移到设备
                query_input_ids = batch["query_input_ids"].to(self.device)
                query_attention_mask = batch["query_attention_mask"].to(self.device)
                doc_input_ids = batch["doc_input_ids"].to(self.device)
                doc_attention_mask = batch["doc_attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                predictions = self.model(
                    query_input_ids,
                    query_attention_mask,
                    doc_input_ids,
                    doc_attention_mask
                )
                
                # 计算损失
                loss = compute_loss(predictions, labels, self.loss_fn)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                # 更新统计信息
                total_loss += loss.item() * len(labels)
                total_samples += len(labels)
                
                # 更新进度条
                progress_bar.set_postfix({"loss": loss.item()})
            
            # 计算平均损失
            avg_loss = total_loss / total_samples
            logger.info(f"Epoch {epoch+1}/{num_epochs}, 平均损失: {avg_loss:.4f}")
    
    def rerank(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        重排序
        
        Args:
            data: 输入数据
            
        Returns:
            重排序后的数据
        """
        logger.info("开始重排序")
        
        self.model.eval()
        reranked_data = []
        
        with torch.no_grad():
            for item in data:
                query = item.get("query", "")
                candidates = item.get("candidates", [])
                
                if not query or not candidates:
                    reranked_data.append(item)
                    continue
                
                # 对查询进行分词
                query_tokens = self.tokenizer(
                    query,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
                
                # 对每个候选文档进行评分
                scores = []
                for candidate in candidates:
                    doc_text = candidate.get("text", "")
                    
                    # 对文档进行分词
                    doc_tokens = self.tokenizer(
                        doc_text,
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    )
                    
                    # 将数据移到设备
                    query_input_ids = query_tokens["input_ids"].to(self.device)
                    query_attention_mask = query_tokens["attention_mask"].to(self.device)
                    doc_input_ids = doc_tokens["input_ids"].to(self.device)
                    doc_attention_mask = doc_tokens["attention_mask"].to(self.device)
                    
                    # 预测分数
                    score = self.model(
                        query_input_ids,
                        query_attention_mask,
                        doc_input_ids,
                        doc_attention_mask
                    ).item()
                    
                    scores.append(score)
                
                # 按分数排序候选文档
                sorted_indices = torch.argsort(torch.tensor(scores), descending=True)
                sorted_candidates = [candidates[i] for i in sorted_indices]
                
                # 更新分数
                for i, idx in enumerate(sorted_indices):
                    sorted_candidates[i]["score"] = scores[idx]
                
                # 添加到结果
                reranked_data.append({
                    "query": query,
                    "candidates": sorted_candidates
                })
        
        logger.info(f"重排序完成，共处理 {len(reranked_data)} 个查询")
        return reranked_data
    
    def save(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        save_model(self.model, path)
    
    def load(self, path: str):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        load_model(self.model, path)

class CrossEncoderModel(nn.Module):
    """交叉编码器模型"""
    
    def __init__(self, model_path: str):
        """
        初始化模型
        
        Args:
            model_path: 预训练模型路径
        """
        super(CrossEncoderModel, self).__init__()
        
        # 加载预训练模型
        self.bert = AutoModel.from_pretrained(model_path)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            query_input_ids: 查询输入ID
            query_attention_mask: 查询注意力掩码
            doc_input_ids: 文档输入ID
            doc_attention_mask: 文档注意力掩码
            
        Returns:
            预测分数
        """
        # 编码查询
        query_outputs = self.bert(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask
        )
        query_embedding = query_outputs.last_hidden_state[:, 0, :]  # [CLS]标记的表示
        
        # 编码文档
        doc_outputs = self.bert(
            input_ids=doc_input_ids,
            attention_mask=doc_attention_mask
        )
        doc_embedding = doc_outputs.last_hidden_state[:, 0, :]  # [CLS]标记的表示
        
        # 计算相似度
        similarity = torch.sum(query_embedding * doc_embedding, dim=1, keepdim=True)
        
        # 分类
        score = self.classifier(similarity)
        
        return score.squeeze(-1) 