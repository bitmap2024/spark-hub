#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
混合检索算法实现
结合稀疏检索(BM25)和密集检索的混合方法
"""

import os
import numpy as np
import json
from typing import List, Dict, Union, Tuple, Optional, Any
from tqdm import tqdm
from dataclasses import dataclass, field
import torch
from rank_bm25 import BM25Okapi
import nltk
import re
from copy import deepcopy

from bi_encoder import BiEncoder, BiEncoderConfig

@dataclass
class HybridRetrievalConfig:
    """混合检索配置类"""
    
    # 基础配置
    use_sparse: bool = True  # 是否使用稀疏检索
    use_dense: bool = True  # 是否使用密集检索
    
    # 稀疏检索相关配置
    tokenizer: str = "jieba"  # 分词器类型: jieba, char, bert
    use_stopwords: bool = True  # 是否使用停用词
    
    # 密集检索相关配置
    dense_config: Optional[BiEncoderConfig] = None  # 密集检索配置
    
    # 融合权重配置
    sparse_weight: float = 0.35  # 稀疏检索权重
    dense_weight: float = 0.65  # 密集检索权重
    
    # 检索配置
    sparse_top_k: int = 100  # 稀疏检索返回的候选数量
    dense_top_k: int = 100  # 密集检索返回的候选数量
    final_top_k: int = 10  # 最终返回的结果数量
    
    # 缓存设置
    use_cache: bool = True  # 是否使用内存缓存
    cache_dir: Optional[str] = None  # 缓存目录

    def __post_init__(self):
        if not self.use_sparse and not self.use_dense:
            raise ValueError("至少需要使用一种检索方法(稀疏或密集)")
        
        if self.sparse_weight + self.dense_weight != 1.0:
            # 自动归一化权重
            total = self.sparse_weight + self.dense_weight
            self.sparse_weight /= total
            self.dense_weight /= total
            
        if self.dense_config is None and self.use_dense:
            self.dense_config = BiEncoderConfig()

class HybridRetrieval:
    """混合检索系统实现类"""
    
    def __init__(self, config: HybridRetrievalConfig = None):
        """初始化混合检索模型
        
        Args:
            config: 混合检索配置
        """
        self.config = config or HybridRetrievalConfig()
        
        # 初始化稀疏检索
        if self.config.use_sparse:
            self._init_sparse_retrieval()
        
        # 初始化密集检索
        if self.config.use_dense:
            self._init_dense_retrieval()
        
        # 文档数据
        self.passages = None
        self.passage_ids = None
        
        # 缓存
        self.query_cache = {}
    
    def _init_sparse_retrieval(self):
        """初始化稀疏检索模型"""
        # 加载停用词
        if self.config.use_stopwords:
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')
            
            try:
                from nltk.corpus import stopwords
                self.stopwords = set(stopwords.words('english'))
                # 添加中文停用词
                cn_stopwords_path = os.path.join(os.path.dirname(__file__), 'data', 'cn_stopwords.txt')
                if os.path.exists(cn_stopwords_path):
                    with open(cn_stopwords_path, 'r', encoding='utf-8') as f:
                        self.stopwords.update([line.strip() for line in f])
            except:
                print("无法加载停用词，将不使用停用词过滤")
                self.stopwords = set()
        else:
            self.stopwords = set()
        
        # 稀疏检索模型
        self.bm25 = None
        self.tokenized_corpus = None
    
    def _init_dense_retrieval(self):
        """初始化密集检索模型"""
        self.dense_retriever = BiEncoder(self.config.dense_config)
    
    def _tokenize(self, text: str) -> List[str]:
        """分词处理
        
        Args:
            text: 输入文本
            
        Returns:
            分词后的token列表
        """
        if self.config.tokenizer == "jieba":
            import jieba
            tokens = jieba.lcut(text)
            if self.config.use_stopwords:
                tokens = [t for t in tokens if t not in self.stopwords and len(t.strip()) > 0]
            return tokens
        elif self.config.tokenizer == "char":
            # 字符级分词
            tokens = list(text)
            if self.config.use_stopwords:
                tokens = [t for t in tokens if t not in self.stopwords and len(t.strip()) > 0]
            return tokens
        elif self.config.tokenizer == "bert":
            # 使用BERT的tokenizer
            if hasattr(self, 'dense_retriever') and hasattr(self.dense_retriever, 'model'):
                tokenizer = self.dense_retriever.model.tokenizer
                tokens = tokenizer.tokenize(text)
                if self.config.use_stopwords:
                    tokens = [t for t in tokens if t not in self.stopwords and len(t.strip()) > 0]
                return tokens
            else:
                # 降级为jieba分词
                import jieba
                return jieba.lcut(text)
        else:
            # 默认使用空格分词
            tokens = text.split()
            if self.config.use_stopwords:
                tokens = [t for t in tokens if t not in self.stopwords and len(t.strip()) > 0]
            return tokens
    
    def index(
        self,
        passages: List[str],
        passage_ids: List[Union[str, int]] = None,
        index_path: Optional[str] = None,
        batch_size: int = 32
    ):
        """构建检索索引
        
        Args:
            passages: 文档文本列表
            passage_ids: 文档ID列表，如果为None则使用索引位置作为ID
            index_path: 索引保存路径
            batch_size: 批次大小
        """
        print(f"为 {len(passages)} 个文档构建混合索引...")
        
        self.passages = passages
        self.passage_ids = passage_ids if passage_ids else list(range(len(passages)))
        
        # 构建稀疏索引
        if self.config.use_sparse:
            print("构建稀疏检索索引 (BM25)...")
            self.tokenized_corpus = [self._tokenize(p) for p in tqdm(passages, desc="分词处理")]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # 构建密集索引
        if self.config.use_dense:
            print("构建密集检索索引...")
            self.dense_retriever.build_index(
                passages=passages,
                passage_ids=self.passage_ids,
                index_path=os.path.join(index_path, "dense_index") if index_path else None,
                batch_size=batch_size
            )
        
        # 保存索引
        if index_path:
            os.makedirs(index_path, exist_ok=True)
            
            # 保存配置和文档
            metadata = {
                "passages": self.passages,
                "passage_ids": self.passage_ids,
                "config": {
                    "use_sparse": self.config.use_sparse,
                    "use_dense": self.config.use_dense,
                    "tokenizer": self.config.tokenizer,
                    "use_stopwords": self.config.use_stopwords,
                    "sparse_weight": self.config.sparse_weight,
                    "dense_weight": self.config.dense_weight,
                    "sparse_top_k": self.config.sparse_top_k,
                    "dense_top_k": self.config.dense_top_k,
                    "final_top_k": self.config.final_top_k,
                }
            }
            
            with open(os.path.join(index_path, "hybrid_metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # 保存分词结果（用于稀疏检索）
            if self.config.use_sparse:
                tokenized_corpus_data = {
                    "tokenized_corpus": self.tokenized_corpus
                }
                with open(os.path.join(index_path, "tokenized_corpus.json"), "w", encoding="utf-8") as f:
                    json.dump(tokenized_corpus_data, f, ensure_ascii=False)
    
    def load_index(self, index_path: str):
        """加载已保存的索引
        
        Args:
            index_path: 索引文件路径
        """
        # 加载元数据
        with open(os.path.join(index_path, "hybrid_metadata.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
        self.passages = metadata["passages"]
        self.passage_ids = metadata["passage_ids"]
        
        # 更新配置
        if "config" in metadata:
            for key, value in metadata["config"].items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        # 加载稀疏索引
        if self.config.use_sparse:
            with open(os.path.join(index_path, "tokenized_corpus.json"), "r", encoding="utf-8") as f:
                tokenized_corpus_data = json.load(f)
            
            self.tokenized_corpus = tokenized_corpus_data["tokenized_corpus"]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # 加载密集索引
        if self.config.use_dense:
            self.dense_retriever.load_index(
                index_path=os.path.join(index_path, "dense_index"),
                metadata_path=os.path.join(index_path, "dense_metadata.json")
            )
    
    def _sparse_search(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """执行稀疏检索
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            
        Returns:
            检索结果，包含文档索引和BM25分数
        """
        # 分词
        tokenized_query = self._tokenize(query)
        
        # 检索
        if len(tokenized_query) == 0:
            # 如果查询分词为空，返回随机结果
            indices = np.random.choice(len(self.passages), min(top_k, len(self.passages)), replace=False)
            scores = np.ones(len(indices)) * 0.01
            return list(zip(indices, scores))
        
        # 计算BM25分数
        scores = self.bm25.get_scores(tokenized_query)
        
        # 构建索引和分数的元组列表
        results = [(idx, score) for idx, score in enumerate(scores)]
        
        # 按分数降序排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 截取top-k结果
        return results[:top_k]
    
    def _dense_search(self, query: str, top_k: int = 100) -> List[Dict]:
        """执行密集检索
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            
        Returns:
            检索结果
        """
        return self.dense_retriever.search(query, top_k=top_k)
    
    def search(
        self,
        query: str,
        top_k: int = None,
        use_sparse: bool = None,
        use_dense: bool = None,
        sparse_weight: float = None,
        dense_weight: float = None,
    ) -> List[Dict]:
        """执行混合检索
        
        Args:
            query: 查询文本
            top_k: 返回的结果数量
            use_sparse: 是否使用稀疏检索，覆盖配置
            use_dense: 是否使用密集检索，覆盖配置
            sparse_weight: 稀疏检索权重，覆盖配置
            dense_weight: 密集检索权重，覆盖配置
            
        Returns:
            检索结果，包含文档ID、文档文本和得分
        """
        # 检查索引是否已构建
        if self.passages is None:
            raise ValueError("请先构建或加载索引")
        
        # 应用参数覆盖
        top_k = top_k if top_k is not None else self.config.final_top_k
        use_sparse = use_sparse if use_sparse is not None else self.config.use_sparse
        use_dense = use_dense if use_dense is not None else self.config.use_dense
        
        # 检查缓存
        cache_key = f"{query}_{top_k}_{use_sparse}_{use_dense}"
        if self.config.use_cache and cache_key in self.query_cache:
            return deepcopy(self.query_cache[cache_key])
        
        # 稀疏检索
        sparse_results = []
        if use_sparse and self.config.use_sparse:
            sparse_results = self._sparse_search(query, top_k=self.config.sparse_top_k)
        
        # 密集检索
        dense_results = []
        if use_dense and self.config.use_dense:
            dense_results = self._dense_search(query, top_k=self.config.dense_top_k)
            # 转换成统一格式：(idx, score)
            dense_results = [(self.passage_ids.index(item["id"]), item["score"]) for item in dense_results]
        
        # 如果只使用了一种检索方法，直接返回结果
        if use_sparse and not use_dense:
            final_scores = sparse_results
        elif use_dense and not use_sparse:
            final_scores = dense_results
        else:
            # 融合两种检索结果
            # 设置权重
            s_weight = sparse_weight if sparse_weight is not None else self.config.sparse_weight
            d_weight = dense_weight if dense_weight is not None else self.config.dense_weight
            
            # 归一化稀疏检索分数
            if sparse_results:
                max_sparse_score = max(score for _, score in sparse_results)
                min_sparse_score = min(score for _, score in sparse_results)
                score_range = max(max_sparse_score - min_sparse_score, 1e-6)  # 避免除零
                sparse_dict = {idx: (score - min_sparse_score) / score_range for idx, score in sparse_results}
            else:
                sparse_dict = {}
            
            # 归一化密集检索分数
            if dense_results:
                max_dense_score = max(score for _, score in dense_results)
                min_dense_score = min(score for _, score in dense_results)
                score_range = max(max_dense_score - min_dense_score, 1e-6)  # 避免除零
                dense_dict = {idx: (score - min_dense_score) / score_range for idx, score in dense_results}
            else:
                dense_dict = {}
            
            # 合并结果
            combined_indices = set(sparse_dict.keys()) | set(dense_dict.keys())
            final_scores = []
            
            for idx in combined_indices:
                sparse_score = sparse_dict.get(idx, 0.0)
                dense_score = dense_dict.get(idx, 0.0)
                
                # 加权组合
                final_score = s_weight * sparse_score + d_weight * dense_score
                final_scores.append((idx, final_score))
        
        # 排序并截取结果
        final_scores.sort(key=lambda x: x[1], reverse=True)
        final_scores = final_scores[:top_k]
        
        # 构建最终结果
        results = []
        for idx, score in final_scores:
            results.append({
                "id": self.passage_ids[idx],
                "text": self.passages[idx],
                "score": float(score)
            })
        
        # 缓存结果
        if self.config.use_cache:
            self.query_cache[cache_key] = deepcopy(results)
        
        return results

# 使用示例
def example_usage():
    # 配置
    config = HybridRetrievalConfig(
        use_sparse=True,
        use_dense=True,
        tokenizer="jieba",
        sparse_weight=0.3,
        dense_weight=0.7,
        final_top_k=3
    )
    
    # 初始化混合检索系统
    retriever = HybridRetrieval(config)
    
    # 示例文档
    passages = [
        "北京是中国的首都，拥有悠久的历史和丰富的文化遗产。",
        "上海是中国最大的城市，是重要的经济、金融、贸易和航运中心。",
        "广州是广东省的省会，是中国南方的经济中心。",
        "深圳是中国改革开放的前沿，是中国的科技创新中心。",
        "杭州是浙江省的省会，以西湖和电子商务而闻名。"
    ]
    
    # 构建索引
    retriever.index(passages)
    
    # 执行检索
    query = "中国最大的城市是哪里？"
    results = retriever.search(query)
    
    # 打印结果
    print(f"查询: {query}")
    for i, result in enumerate(results):
        print(f"结果 {i+1}: {result['text']} (分数: {result['score']:.4f})")

if __name__ == "__main__":
    example_usage() 