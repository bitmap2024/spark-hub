#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ColBERT检索算法实现
基于后期交互的精确检索模型
保留token级别的表示，通过最大相似度匹配计算相关性
"""

import os
import torch
import numpy as np
import faiss
import json
import time
from typing import List, Dict, Union, Tuple, Optional, Any
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ColBERTConfig:
    """ColBERT模型配置类"""
    
    model_name: str = "bert-base-chinese"  # 预训练模型名称
    query_maxlen: int = 32  # 查询最大长度
    doc_maxlen: int = 180  # 文档最大长度
    dim: int = 128  # 编码维度，ColBERT通常用较小的维度
    similarity_metric: str = "cosine"  # 相似度计算方式
    mask_punctuation: bool = True  # 是否屏蔽标点符号
    use_gpu: bool = True  # 是否使用GPU
    device: str = None  # 设备
    
    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() and self.use_gpu else "cpu"

class ColBERTModel(nn.Module):
    """ColBERT模型核心实现"""
    
    def __init__(self, config: ColBERTConfig):
        super().__init__()
        self.config = config
        
        # 加载预训练模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.bert = AutoModel.from_pretrained(config.model_name)
        
        # 线性映射层，将BERT的输出降至指定维度
        self.linear = nn.Linear(self.bert.config.hidden_size, config.dim, bias=False)
        
        # 标点符号屏蔽
        if config.mask_punctuation:
            self.skiplist = {
                self.tokenizer.encode('，', add_special_tokens=False)[0],
                self.tokenizer.encode('。', add_special_tokens=False)[0],
                self.tokenizer.encode('？', add_special_tokens=False)[0],
                self.tokenizer.encode('！', add_special_tokens=False)[0],
                self.tokenizer.encode('：', add_special_tokens=False)[0],
                self.tokenizer.encode('"', add_special_tokens=False)[0],
                self.tokenizer.encode('"', add_special_tokens=False)[0],
                self.tokenizer.encode('；', add_special_tokens=False)[0],
                self.tokenizer.encode('（', add_special_tokens=False)[0],
                self.tokenizer.encode('）', add_special_tokens=False)[0],
                self.tokenizer.encode('、', add_special_tokens=False)[0],
            }
        else:
            self.skiplist = {}
    
    def forward(self, input_ids, attention_mask, is_query=False):
        """
        模型前向传播过程
        
        Args:
            input_ids: 输入token ID
            attention_mask: 注意力掩码
            is_query: 是否为查询（查询和文档处理略有不同）
            
        Returns:
            token级别的编码表示
        """
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 获取所有token的表示 [batch_size, seq_len, hidden_size]
        token_embeddings = outputs.last_hidden_state
        
        # 降维到指定维度
        token_embeddings = self.linear(token_embeddings)
        
        # L2归一化
        token_embeddings = F.normalize(token_embeddings, p=2, dim=2)
        
        if is_query:
            # 对于查询，我们总是保留[MASK] (通常是最后一个token)
            return token_embeddings
        else:
            # 对于文档，我们屏蔽标点符号（如果配置了）
            mask = torch.ones_like(input_ids, dtype=torch.bool)
            
            # 屏蔽[CLS]和[SEP]
            mask[:, 0] = False
            
            # 屏蔽[PAD]
            zero_mask = input_ids != 0
            mask = mask & zero_mask
            
            # 屏蔽标点符号
            for skipid in self.skiplist:
                mask = mask & (input_ids != skipid)
            
            # 应用屏蔽
            masked_embeddings = token_embeddings * mask.unsqueeze(-1)
            return masked_embeddings, mask

class ColBERT:
    """ColBERT检索系统封装类"""
    
    def __init__(self, config: ColBERTConfig = None):
        """初始化ColBERT检索模型
        
        Args:
            config: 模型配置
        """
        self.config = config or ColBERTConfig()
        self.device = torch.device(self.config.device)
        
        # 初始化模型
        self.model = ColBERTModel(self.config)
        self.model.to(self.device)
        
        # 初始化分词器
        self.tokenizer = self.model.tokenizer
        
        # 索引相关
        self.index = None
        self.doc_embeddings = None
        self.doc_masks = None
        self.passages = None
        self.passage_ids = None
    
    def train(
        self, 
        train_dataset,  # 训练数据集
        eval_dataset=None,  # 评估数据集 
        output_path="colbert_model",  # 模型保存路径
        batch_size=8,  # 批次大小
        epochs=3,  # 训练轮数
        learning_rate=2e-5,  # 学习率
        warmup_ratio=0.1,  # 预热比例
        max_grad_norm=1.0,  # 梯度裁剪
    ):
        """训练ColBERT模型
        
        使用对比学习方法训练模型，将相关文档拉近，不相关文档推远
        
        Args:
            train_dataset: 训练数据集
            eval_dataset: 评估数据集
            output_path: 模型保存路径
            batch_size: 批次大小
            epochs: 训练轮数
            learning_rate: 学习率
            warmup_ratio: 预热比例
            max_grad_norm: 梯度裁剪
        """
        # 训练代码涉及更复杂的批处理和损失计算
        # 这里只提供基本框架，实际训练还需要更详细的实现
        pass
        
    def _tokenize(self, texts, max_length, add_special_tokens=True):
        """
        对文本进行分词处理
        
        Args:
            texts: 输入文本列表
            max_length: 最大长度
            add_special_tokens: 是否添加特殊token
            
        Returns:
            分词后的输入字典
        """
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=add_special_tokens
        )
    
    def encode_queries(self, queries: List[str], batch_size: int = 16) -> torch.Tensor:
        """
        编码查询文本
        
        Args:
            queries: 查询文本列表
            batch_size: 批次大小
            
        Returns:
            查询的token级别向量表示 [num_queries, query_maxlen, dim]
        """
        self.model.eval()
        all_query_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                batch_queries = queries[i:i+batch_size]
                
                # 分词
                inputs = self._tokenize(batch_queries, self.config.query_maxlen)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 编码
                query_embeddings = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    is_query=True
                )
                
                all_query_embeddings.append(query_embeddings.cpu())
        
        return torch.cat(all_query_embeddings, dim=0)
    
    def encode_passages(self, passages: List[str], batch_size: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码文档文本
        
        Args:
            passages: 文档文本列表
            batch_size: 批次大小
            
        Returns:
            文档的token级别向量表示 [num_docs, doc_maxlen, dim] 和 对应的mask
        """
        self.model.eval()
        all_doc_embeddings = []
        all_doc_masks = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(passages), batch_size), desc="编码文档"):
                batch_docs = passages[i:i+batch_size]
                
                # 分词
                inputs = self._tokenize(batch_docs, self.config.doc_maxlen)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 编码
                doc_embeddings, doc_masks = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    is_query=False
                )
                
                all_doc_embeddings.append(doc_embeddings.cpu())
                all_doc_masks.append(doc_masks.cpu())
        
        return torch.cat(all_doc_embeddings, dim=0), torch.cat(all_doc_masks, dim=0)
    
    def build_index(
        self,
        passages: List[str],
        passage_ids: List[Union[str, int]] = None,
        index_path: str = None,
        batch_size: int = 8
    ):
        """
        构建检索索引
        
        由于ColBERT使用token级别的交互，需要存储所有token的表示
        
        Args:
            passages: 文档文本列表
            passage_ids: 文档ID列表，如果为None则使用索引位置作为ID
            index_path: 索引保存路径
            batch_size: 批次大小
        """
        print(f"为 {len(passages)} 个文档构建ColBERT索引...")
        
        # 编码所有文档
        doc_embeddings, doc_masks = self.encode_passages(passages, batch_size=batch_size)
        
        # 保存编码结果
        self.doc_embeddings = doc_embeddings  # [num_docs, doc_maxlen, dim]
        self.doc_masks = doc_masks  # [num_docs, doc_maxlen]
        self.passages = passages
        self.passage_ids = passage_ids if passage_ids else list(range(len(passages)))
        
        # 如果提供了索引路径，保存索引内容
        if index_path:
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            
            # 保存模型编码的tensor
            torch.save({
                "doc_embeddings": self.doc_embeddings,
                "doc_masks": self.doc_masks,
                "passage_ids": self.passage_ids,
            }, index_path)
            
            # 保存文档文本和元数据
            metadata = {
                "passages": self.passages,
                "config": {
                    "model_name": self.config.model_name,
                    "query_maxlen": self.config.query_maxlen,
                    "doc_maxlen": self.config.doc_maxlen,
                    "dim": self.config.dim,
                    "similarity_metric": self.config.similarity_metric,
                    "mask_punctuation": self.config.mask_punctuation,
                }
            }
            
            metadata_path = os.path.join(os.path.dirname(index_path), "colbert_metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                
    def load_index(self, index_path: str, metadata_path: str = None):
        """
        加载已保存的索引
        
        Args:
            index_path: 索引文件路径
            metadata_path: 元数据文件路径，默认为index_path同目录下的colbert_metadata.json
        """
        # 加载模型编码的tensor
        checkpoint = torch.load(index_path, map_location=self.device)
        self.doc_embeddings = checkpoint["doc_embeddings"]
        self.doc_masks = checkpoint["doc_masks"]
        self.passage_ids = checkpoint["passage_ids"]
        
        # 加载元数据
        if metadata_path is None:
            metadata_path = os.path.join(os.path.dirname(index_path), "colbert_metadata.json")
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
        self.passages = metadata["passages"]
        
        # 更新配置
        if "config" in metadata:
            for key, value in metadata["config"].items():
                setattr(self.config, key, value)
    
    def _compute_max_similarity(self, query_embedding: torch.Tensor, doc_embeddings: torch.Tensor, doc_mask: torch.Tensor) -> torch.Tensor:
        """
        计算查询和文档之间的最大相似度
        
        Args:
            query_embedding: 查询编码 [query_maxlen, dim]
            doc_embeddings: 文档编码 [doc_maxlen, dim]
            doc_mask: 文档掩码 [doc_maxlen]
            
        Returns:
            相似度分数
        """
        # 计算所有query tokens和doc tokens之间的相似度
        # [query_maxlen, doc_maxlen]
        similarity = torch.matmul(query_embedding, doc_embeddings.transpose(0, 1))
        
        # 应用掩码（将无效位置的相似度设为一个很小的值）
        similarity = similarity * doc_mask.unsqueeze(0) + (~doc_mask.unsqueeze(0)) * -1000.0
        
        # 对于每个查询token，找到最大的文档token相似度
        # [query_maxlen]
        max_similarity = similarity.max(dim=1).values
        
        # 计算总体相似度（所有查询token的最大相似度之和）
        return max_similarity.sum()
    
    def search(
        self,
        queries: Union[str, List[str]],
        top_k: int = 10,
        batch_size: int = 16
    ) -> Union[List[Dict], List[List[Dict]]]:
        """
        检索最相关的文档
        
        Args:
            queries: 查询文本或查询文本列表
            top_k: 返回的文档数量
            batch_size: 批次大小
            
        Returns:
            检索结果，包含文档ID、文档文本和相似度分数
        """
        if self.doc_embeddings is None:
            raise ValueError("请先构建或加载索引")
        
        # 处理单个查询的情况
        single_query = False
        if isinstance(queries, str):
            queries = [queries]
            single_query = True
        
        # 编码查询
        query_embeddings = self.encode_queries(queries, batch_size=batch_size)
        
        results = []
        for query_idx, query_embedding in enumerate(query_embeddings):
            # 计算当前查询和所有文档的相似度
            scores = []
            
            # 实际应用中应该使用更高效的检索方法，这里简化为线性扫描
            for doc_idx in tqdm(range(len(self.doc_embeddings)), desc=f"检索查询 {query_idx+1}/{len(queries)}", disable=len(queries)==1):
                doc_embedding = self.doc_embeddings[doc_idx]
                doc_mask = self.doc_masks[doc_idx]
                
                # 计算相似度分数
                score = self._compute_max_similarity(
                    query_embedding, 
                    doc_embedding,
                    doc_mask
                )
                
                scores.append((doc_idx, score.item()))
            
            # 按分数排序，选择top-k
            scores.sort(key=lambda x: x[1], reverse=True)
            top_scores = scores[:top_k]
            
            # 整理结果
            query_results = []
            for doc_idx, score in top_scores:
                query_results.append({
                    "id": self.passage_ids[doc_idx],
                    "text": self.passages[doc_idx],
                    "score": score
                })
                
            results.append(query_results)
        
        # 返回单个查询的结果或所有查询的结果
        return results[0] if single_query else results

# 使用示例
def example_usage():
    # 配置模型
    config = ColBERTConfig(
        model_name="bert-base-chinese",
        query_maxlen=32,
        doc_maxlen=180,
        dim=128
    )
    
    # 初始化模型
    model = ColBERT(config)
    
    # 示例文档
    passages = [
        "北京是中国的首都，拥有悠久的历史和丰富的文化遗产。",
        "上海是中国最大的城市，是重要的经济、金融、贸易和航运中心。",
        "广州是广东省的省会，是中国南方的经济中心。",
        "深圳是中国改革开放的前沿，是中国的科技创新中心。",
        "杭州是浙江省的省会，以西湖和电子商务而闻名。"
    ]
    
    # 构建索引
    model.build_index(passages)
    
    # 执行检索
    query = "中国最大的城市是哪里？"
    results = model.search(query, top_k=2)
    
    # 打印结果
    print(f"查询: {query}")
    for i, result in enumerate(results):
        print(f"结果 {i+1}: {result['text']} (分数: {result['score']:.4f})")

if __name__ == "__main__":
    example_usage() 