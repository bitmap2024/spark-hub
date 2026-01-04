#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双塔模型检索算法实现
基于预训练语言模型的双塔编码框架
"""

import os
import torch
import numpy as np
import faiss
import json
from typing import List, Dict, Union, Tuple, Optional
from sentence_transformers import SentenceTransformer, models, losses, util
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AdamW
from sklearn.preprocessing import normalize

class BiEncoderConfig:
    """双塔模型的配置类"""
    
    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        max_seq_length: int = 128,
        embedding_dim: int = 768,
        pooling_method: str = "mean",
        distance_metric: str = "cosine",
        device: Optional[str] = None,
        use_fp16: bool = False
    ):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.pooling_method = pooling_method
        self.distance_metric = distance_metric
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and self.device == "cuda"

class BiEncoderDataset(Dataset):
    """双塔模型的训练数据集"""
    
    def __init__(
        self,
        queries: List[str],
        positive_passages: List[str],
        negative_passages: List[str] = None,
        hard_negatives: List[List[str]] = None,
        num_negatives: int = 1
    ):
        self.queries = queries
        self.positive_passages = positive_passages
        self.negative_passages = negative_passages
        self.hard_negatives = hard_negatives
        self.num_negatives = num_negatives
        
        if len(queries) != len(positive_passages):
            raise ValueError("查询数量必须与正例文档数量相同")
        
        if hard_negatives and len(hard_negatives) != len(queries):
            raise ValueError("硬负例列表长度必须与查询数量相同")
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query = self.queries[idx]
        positive = self.positive_passages[idx]
        
        negatives = []
        if self.hard_negatives and idx < len(self.hard_negatives):
            hard_negs = self.hard_negatives[idx]
            if hard_negs:
                negatives.extend(hard_negs[:self.num_negatives])
        
        # 如果硬负例不够，从全局负例中随机选取
        if self.negative_passages and len(negatives) < self.num_negatives:
            candidates = [neg for neg in self.negative_passages if neg != positive]
            if candidates:
                needed = self.num_negatives - len(negatives)
                random_negs = np.random.choice(candidates, min(needed, len(candidates)), replace=False)
                negatives.extend(random_negs)
        
        return {
            "query": query,
            "positive": positive,
            "negatives": negatives
        }

class BiEncoder:
    """双塔检索模型实现类"""
    
    def __init__(self, config: BiEncoderConfig = None):
        """初始化双塔检索模型
        
        Args:
            config: 模型配置对象
        """
        self.config = config or BiEncoderConfig()
        self.device = self.config.device
        
        # 使用sentence-transformers构建双塔编码器
        self.model = SentenceTransformer(self.config.model_name, device=self.device)
        
        # 索引存储
        self.index = None
        self.passage_ids = None
        self.passages = None
    
    def train(
        self,
        train_dataset: BiEncoderDataset,
        eval_dataset: BiEncoderDataset = None,
        output_path: str = "bi_encoder_model",
        batch_size: int = 32,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        warmup_steps: int = 100,
        evaluation_steps: int = 1000,
    ):
        """训练双塔模型
        
        Args:
            train_dataset: 训练数据集
            eval_dataset: 评估数据集
            output_path: 模型保存路径
            batch_size: 批次大小
            epochs: 训练轮数
            learning_rate: 学习率
            warmup_steps: 预热步数
            evaluation_steps: 评估间隔步数
        """
        # 构建训练数据加载器
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 使用多负例对比损失函数
        train_loss = losses.MultipleNegativesRankingLoss(self.model)
        
        # 设置评估器
        evaluator = None
        if eval_dataset:
            evaluator = losses.InformationRetrievalEvaluator(
                queries={str(i): eval_dataset.queries[i] for i in range(len(eval_dataset.queries))},
                corpus={str(i): eval_dataset.positive_passages[i] for i in range(len(eval_dataset.positive_passages))},
                relevant_docs={str(i): {str(i)} for i in range(len(eval_dataset.queries))},
                batch_size=batch_size,
                name="bi-encoder-eval"
            )
        
        # 训练模型
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=epochs,
            steps_per_epoch=len(train_dataloader),
            evaluation_steps=evaluation_steps,
            warmup_steps=warmup_steps,
            output_path=output_path,
            optimizer_params={'lr': learning_rate},
            use_amp=self.config.use_fp16  # 混合精度训练
        )
        
        # 加载最佳模型
        self.model = SentenceTransformer(output_path)
        
    def encode_queries(self, queries: List[str], batch_size: int = 32) -> np.ndarray:
        """编码查询文本
        
        Args:
            queries: 查询文本列表
            batch_size: 批次大小
            
        Returns:
            查询的向量表示
        """
        return self.model.encode(
            queries,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True if self.config.distance_metric == "cosine" else False
        )
    
    def encode_passages(self, passages: List[str], batch_size: int = 32) -> np.ndarray:
        """编码文档文本
        
        Args:
            passages: 文档文本列表
            batch_size: 批次大小
            
        Returns:
            文档的向量表示
        """
        return self.model.encode(
            passages,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True if self.config.distance_metric == "cosine" else False
        )
    
    def build_index(
        self,
        passages: List[str],
        passage_ids: List[Union[str, int]] = None,
        index_path: str = None,
        batch_size: int = 32
    ):
        """构建检索索引
        
        Args:
            passages: 文档文本列表
            passage_ids: 文档ID列表，如果为None则使用索引位置作为ID
            index_path: 索引保存路径
            batch_size: 批次大小
        """
        print(f"为 {len(passages)} 个文档构建索引...")
        
        # 编码所有文档
        passage_embeddings = self.encode_passages(passages, batch_size=batch_size)
        dim = passage_embeddings.shape[1]
        
        # 根据距离度量选择索引类型
        if self.config.distance_metric == "cosine":
            # 对于余弦相似度，使用内积索引并标准化向量
            if not np.allclose(np.linalg.norm(passage_embeddings, axis=1), 1.0, atol=1e-5):
                passage_embeddings = normalize(passage_embeddings, axis=1, norm='l2')
            index = faiss.IndexFlatIP(dim)
        else:
            # 对于欧氏距离，使用L2索引
            index = faiss.IndexFlatL2(dim)
        
        # 添加文档向量到索引
        index.add(passage_embeddings)
        
        # 保存索引和文档信息
        self.index = index
        self.passages = passages
        self.passage_ids = passage_ids if passage_ids else list(range(len(passages)))
        
        # 如果提供了索引路径，保存索引
        if index_path:
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            faiss.write_index(index, index_path)
            
            # 保存文档ID和文本
            metadata = {
                "passage_ids": self.passage_ids,
                "passages": self.passages,
                "distance_metric": self.config.distance_metric
            }
            with open(os.path.join(os.path.dirname(index_path), "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def load_index(self, index_path: str, metadata_path: str = None):
        """加载已保存的索引
        
        Args:
            index_path: 索引文件路径
            metadata_path: 元数据文件路径，默认为index_path同目录下的metadata.json
        """
        # 加载FAISS索引
        self.index = faiss.read_index(index_path)
        
        # 加载元数据
        if metadata_path is None:
            metadata_path = os.path.join(os.path.dirname(index_path), "metadata.json")
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
        self.passage_ids = metadata["passage_ids"]
        self.passages = metadata["passages"]
        self.config.distance_metric = metadata.get("distance_metric", "cosine")
    
    def search(
        self,
        queries: Union[str, List[str]],
        top_k: int = 10,
        batch_size: int = 32
    ) -> Union[List[Dict], List[List[Dict]]]:
        """检索最相关的文档
        
        Args:
            queries: 查询文本或查询文本列表
            top_k: 返回的文档数量
            batch_size: 批次大小
            
        Returns:
            检索结果，包含文档ID、文档文本和相似度分数
        """
        if self.index is None:
            raise ValueError("请先构建或加载索引")
        
        # 处理单个查询的情况
        single_query = False
        if isinstance(queries, str):
            queries = [queries]
            single_query = True
        
        # 编码查询
        query_embeddings = self.encode_queries(queries, batch_size=batch_size)
        
        # 执行检索
        scores, indices = self.index.search(query_embeddings, top_k)
        
        # 整理结果
        results = []
        for i, (query_scores, query_indices) in enumerate(zip(scores, indices)):
            query_results = []
            for score, idx in zip(query_scores, query_indices):
                if idx < 0 or idx >= len(self.passage_ids):
                    continue  # 跳过无效索引
                    
                query_results.append({
                    "id": self.passage_ids[idx],
                    "text": self.passages[idx],
                    "score": float(score)
                })
            results.append(query_results)
        
        # 返回单个查询的结果或所有查询的结果
        return results[0] if single_query else results

# 使用示例
def example_usage():
    # 配置模型
    config = BiEncoderConfig(
        model_name="bert-base-chinese",
        max_seq_length=128,
        distance_metric="cosine"
    )
    
    # 初始化模型
    model = BiEncoder(config)
    
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