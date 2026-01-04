#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import random
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class DataProcessor:
    """数据处理类，用于处理输入数据"""
    
    def __init__(self, max_length: int = 512, tokenizer_name: str = "bert-base-chinese"):
        """
        初始化数据处理器
        
        Args:
            max_length: 最大序列长度
            tokenizer_name: 分词器名称
        """
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        logger.info(f"使用分词器: {tokenizer_name}")
    
    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        处理输入数据
        
        Args:
            data: 输入数据列表，每个元素包含查询和候选文档
            
        Returns:
            处理后的数据列表
        """
        processed_data = []
        
        for item in data:
            query = item.get("query", "")
            candidates = item.get("candidates", [])
            
            if not query or not candidates:
                logger.warning(f"跳过无效数据项: {item}")
                continue
            
            # 处理查询
            query_tokens = self.tokenizer(
                query,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # 处理候选文档
            processed_candidates = []
            for candidate in candidates:
                doc_text = candidate.get("text", "")
                doc_id = candidate.get("id", "")
                doc_score = candidate.get("score", 0.0)
                
                if not doc_text:
                    logger.warning(f"跳过无效候选文档: {candidate}")
                    continue
                
                # 对文档进行分词
                doc_tokens = self.tokenizer(
                    doc_text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                processed_candidates.append({
                    "id": doc_id,
                    "text": doc_text,
                    "score": doc_score,
                    "input_ids": doc_tokens["input_ids"].squeeze(0),
                    "attention_mask": doc_tokens["attention_mask"].squeeze(0)
                })
            
            if not processed_candidates:
                logger.warning(f"跳过无有效候选文档的数据项: {item}")
                continue
            
            processed_data.append({
                "query": query,
                "query_input_ids": query_tokens["input_ids"].squeeze(0),
                "query_attention_mask": query_tokens["attention_mask"].squeeze(0),
                "candidates": processed_candidates
            })
        
        logger.info(f"处理了 {len(processed_data)} 个有效数据项")
        return processed_data
    
    def create_cross_encoder_batch(self, data: List[Dict[str, Any]], batch_size: int = 32) -> List[Dict[str, torch.Tensor]]:
        """
        创建交叉编码器的批处理数据
        
        Args:
            data: 处理后的数据列表
            batch_size: 批处理大小
            
        Returns:
            批处理数据列表
        """
        batches = []
        
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            
            # 收集批处理中的所有查询和文档
            query_input_ids = []
            query_attention_mask = []
            doc_input_ids = []
            doc_attention_mask = []
            labels = []
            
            for item in batch_data:
                query_input_ids.append(item["query_input_ids"])
                query_attention_mask.append(item["query_attention_mask"])
                
                # 对于每个查询，随机选择一个正例和一个负例
                candidates = item["candidates"]
                positive_candidates = [c for c in candidates if c["score"] > 0.5]
                negative_candidates = [c for c in candidates if c["score"] <= 0.5]
                
                if positive_candidates and negative_candidates:
                    positive = random.choice(positive_candidates)
                    negative = random.choice(negative_candidates)
                    
                    doc_input_ids.extend([positive["input_ids"], negative["input_ids"]])
                    doc_attention_mask.extend([positive["attention_mask"], negative["attention_mask"]])
                    labels.extend([1, 0])
            
            if not doc_input_ids:
                continue
            
            # 转换为张量
            batch = {
                "query_input_ids": torch.stack(query_input_ids),
                "query_attention_mask": torch.stack(query_attention_mask),
                "doc_input_ids": torch.stack(doc_input_ids),
                "doc_attention_mask": torch.stack(doc_attention_mask),
                "labels": torch.tensor(labels, dtype=torch.float)
            }
            
            batches.append(batch)
        
        return batches
    
    def create_sequence_batch(self, data: List[Dict[str, Any]], batch_size: int = 32) -> List[Dict[str, torch.Tensor]]:
        """
        创建序列重排序的批处理数据
        
        Args:
            data: 处理后的数据列表
            batch_size: 批处理大小
            
        Returns:
            批处理数据列表
        """
        batches = []
        
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            
            # 收集批处理中的所有查询和文档序列
            query_input_ids = []
            query_attention_mask = []
            doc_sequences = []
            labels = []
            
            for item in batch_data:
                query_input_ids.append(item["query_input_ids"])
                query_attention_mask.append(item["query_attention_mask"])
                
                # 按分数排序候选文档
                sorted_candidates = sorted(item["candidates"], key=lambda x: x["score"], reverse=True)
                
                # 取前N个文档作为序列
                max_docs = 10
                doc_sequence = sorted_candidates[:max_docs]
                
                # 填充序列
                while len(doc_sequence) < max_docs:
                    doc_sequence.append({
                        "input_ids": torch.zeros_like(sorted_candidates[0]["input_ids"]),
                        "attention_mask": torch.zeros_like(sorted_candidates[0]["attention_mask"]),
                        "score": 0.0
                    })
                
                doc_sequences.append(doc_sequence)
                labels.append([c["score"] for c in doc_sequence])
            
            # 转换为张量
            batch = {
                "query_input_ids": torch.stack(query_input_ids),
                "query_attention_mask": torch.stack(query_attention_mask),
                "doc_sequences": doc_sequences,
                "labels": torch.tensor(labels, dtype=torch.float)
            }
            
            batches.append(batch)
        
        return batches
    
    def create_contrastive_batch(self, data: List[Dict[str, Any]], batch_size: int = 32) -> List[Dict[str, torch.Tensor]]:
        """
        创建对比学习的批处理数据
        
        Args:
            data: 处理后的数据列表
            batch_size: 批处理大小
            
        Returns:
            批处理数据列表
        """
        batches = []
        
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            
            # 收集批处理中的所有查询和文档
            query_input_ids = []
            query_attention_mask = []
            doc_input_ids = []
            doc_attention_mask = []
            
            for item in batch_data:
                query_input_ids.append(item["query_input_ids"])
                query_attention_mask.append(item["query_attention_mask"])
                
                # 对于每个查询，选择所有候选文档
                for candidate in item["candidates"]:
                    doc_input_ids.append(candidate["input_ids"])
                    doc_attention_mask.append(candidate["attention_mask"])
            
            if not doc_input_ids:
                continue
            
            # 转换为张量
            batch = {
                "query_input_ids": torch.stack(query_input_ids),
                "query_attention_mask": torch.stack(query_attention_mask),
                "doc_input_ids": torch.stack(doc_input_ids),
                "doc_attention_mask": torch.stack(doc_attention_mask)
            }
            
            batches.append(batch)
        
        return batches 