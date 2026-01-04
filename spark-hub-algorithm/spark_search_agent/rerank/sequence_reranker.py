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

class SequenceReranker:
    """基于Transformer的序列重排序类"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        device: str = "cuda",
        seed: int = 42
    ):
        """
        初始化序列重排序器
        
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
        self.model = SequenceRerankModel(
            model_path if model_path else "bert-base-chinese"
        ).to(self.device)
        
        # 初始化优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )
        
        # 初始化损失函数
        self.loss_fn = nn.MSELoss()
        
        logger.info(f"初始化序列重排序器: model_path={model_path}, device={self.device}")
    
    def train(self, data: List[Dict[str, Any]], num_epochs: int = 10):
        """
        训练模型
        
        Args:
            data: 训练数据
            num_epochs: 训练轮数
        """
        logger.info(f"开始训练序列重排序模型，共 {num_epochs} 轮")
        
        # 创建数据加载器
        from utils import SequenceRerankDataset, create_dataloader
        dataset = SequenceRerankDataset(data, self.tokenizer)
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
                labels = batch["labels"].to(self.device)
                
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
                total_loss += loss.item() * labels.numel()
                total_samples += labels.numel()
                
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
                
                # 准备文档序列
                doc_sequences = []
                for candidate in candidates:
                    doc_text = candidate.get("text", "")
                    
                    # 对文档进行分词
                    doc_tokens = self.tokenizer(
                        doc_text,
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    )
                    
                    doc_sequences.append({
                        "input_ids": doc_tokens["input_ids"],
                        "attention_mask": doc_tokens["attention_mask"]
                    })
                
                # 填充序列
                max_docs = 10
                while len(doc_sequences) < max_docs:
                    doc_sequences.append({
                        "input_ids": torch.zeros_like(doc_sequences[0]["input_ids"]),
                        "attention_mask": torch.zeros_like(doc_sequences[0]["attention_mask"])
                    })
                
                # 将数据移到设备
                query_input_ids = query_tokens["input_ids"].to(self.device)
                query_attention_mask = query_tokens["attention_mask"].to(self.device)
                
                doc_input_ids = torch.stack([d["input_ids"] for d in doc_sequences]).to(self.device)
                doc_attention_mask = torch.stack([d["attention_mask"] for d in doc_sequences]).to(self.device)
                
                # 预测分数
                scores = self.model(
                    query_input_ids,
                    query_attention_mask,
                    doc_input_ids,
                    doc_attention_mask
                ).cpu().numpy()
                
                # 按分数排序候选文档
                sorted_indices = torch.argsort(torch.tensor(scores), descending=True)
                sorted_candidates = [candidates[i] for i in sorted_indices if i < len(candidates)]
                
                # 更新分数
                for i, idx in enumerate(sorted_indices):
                    if idx < len(candidates):
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

class SequenceRerankModel(nn.Module):
    """序列重排序模型"""
    
    def __init__(self, model_path: str):
        """
        初始化模型
        
        Args:
            model_path: 预训练模型路径
        """
        super(SequenceRerankModel, self).__init__()
        
        # 加载预训练模型
        self.bert = AutoModel.from_pretrained(model_path)
        
        # 查询编码器
        self.query_encoder = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 文档编码器
        self.doc_encoder = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 序列编码器
        self.sequence_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=512,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1
            ),
            num_layers=2
        )
        
        # 分类头
        self.classifier = nn.Sequential(
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
        batch_size = query_input_ids.size(0)
        num_docs = doc_input_ids.size(1)
        
        # 编码查询
        query_outputs = self.bert(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask
        )
        query_embedding = self.query_encoder(query_outputs.last_hidden_state[:, 0, :])  # [CLS]标记的表示
        
        # 编码文档序列
        doc_embeddings = []
        for i in range(num_docs):
            doc_outputs = self.bert(
                input_ids=doc_input_ids[:, i, :],
                attention_mask=doc_attention_mask[:, i, :]
            )
            doc_embedding = self.doc_encoder(doc_outputs.last_hidden_state[:, 0, :])  # [CLS]标记的表示
            doc_embeddings.append(doc_embedding)
        
        # 堆叠文档嵌入
        doc_sequence = torch.stack(doc_embeddings, dim=1)  # [batch_size, num_docs, 512]
        
        # 序列编码
        sequence_output = self.sequence_encoder(doc_sequence)  # [batch_size, num_docs, 512]
        
        # 计算查询和文档序列的相似度
        query_embedding = query_embedding.unsqueeze(1).expand(-1, num_docs, -1)  # [batch_size, num_docs, 512]
        similarity = torch.sum(query_embedding * sequence_output, dim=2, keepdim=True)  # [batch_size, num_docs, 1]
        
        # 分类
        scores = self.classifier(similarity.squeeze(-1))  # [batch_size, num_docs]
        
        return scores 