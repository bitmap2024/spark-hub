#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics import ndcg_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

class Evaluator:
    """评估类，用于评估重排序结果"""
    
    def __init__(self):
        """初始化评估器"""
        pass
    
    def evaluate(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        评估重排序结果
        
        Args:
            results: 重排序结果列表
            
        Returns:
            评估指标字典
        """
        metrics = {}
        
        # 计算NDCG@K
        for k in [1, 3, 5, 10]:
            ndcg = self._calculate_ndcg(results, k)
            metrics[f"ndcg@{k}"] = ndcg
        
        # 计算Precision@K
        for k in [1, 3, 5, 10]:
            precision = self._calculate_precision(results, k)
            metrics[f"precision@{k}"] = precision
        
        # 计算Recall@K
        for k in [1, 3, 5, 10]:
            recall = self._calculate_recall(results, k)
            metrics[f"recall@{k}"] = recall
        
        # 计算MRR (Mean Reciprocal Rank)
        mrr = self._calculate_mrr(results)
        metrics["mrr"] = mrr
        
        # 计算MAP (Mean Average Precision)
        map_score = self._calculate_map(results)
        metrics["map"] = map_score
        
        logger.info(f"评估结果: {metrics}")
        return metrics
    
    def _calculate_ndcg(self, results: List[Dict[str, Any]], k: int) -> float:
        """
        计算NDCG@K
        
        Args:
            results: 重排序结果列表
            k: 截断位置
            
        Returns:
            NDCG@K分数
        """
        ndcg_scores = []
        
        for item in results:
            query = item.get("query", "")
            candidates = item.get("candidates", [])
            
            if not candidates:
                continue
            
            # 获取真实相关性分数
            true_scores = np.array([c.get("true_score", 0.0) for c in candidates])
            
            # 获取预测分数
            pred_scores = np.array([c.get("score", 0.0) for c in candidates])
            
            # 计算NDCG
            try:
                ndcg = ndcg_score([true_scores], [pred_scores], k=k)
                ndcg_scores.append(ndcg)
            except Exception as e:
                logger.warning(f"计算NDCG时出错: {e}")
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    def _calculate_precision(self, results: List[Dict[str, Any]], k: int) -> float:
        """
        计算Precision@K
        
        Args:
            results: 重排序结果列表
            k: 截断位置
            
        Returns:
            Precision@K分数
        """
        precision_scores = []
        
        for item in results:
            query = item.get("query", "")
            candidates = item.get("candidates", [])
            
            if not candidates:
                continue
            
            # 获取前K个候选文档
            top_k = candidates[:k]
            
            # 获取真实标签
            true_labels = np.array([1 if c.get("true_score", 0.0) > 0.5 else 0 for c in top_k])
            
            # 获取预测标签
            pred_labels = np.array([1 if c.get("score", 0.0) > 0.5 else 0 for c in top_k])
            
            # 计算Precision
            precision = precision_score(true_labels, pred_labels, zero_division=0)
            precision_scores.append(precision)
        
        return np.mean(precision_scores) if precision_scores else 0.0
    
    def _calculate_recall(self, results: List[Dict[str, Any]], k: int) -> float:
        """
        计算Recall@K
        
        Args:
            results: 重排序结果列表
            k: 截断位置
            
        Returns:
            Recall@K分数
        """
        recall_scores = []
        
        for item in results:
            query = item.get("query", "")
            candidates = item.get("candidates", [])
            
            if not candidates:
                continue
            
            # 获取前K个候选文档
            top_k = candidates[:k]
            
            # 获取真实标签
            true_labels = np.array([1 if c.get("true_score", 0.0) > 0.5 else 0 for c in top_k])
            
            # 获取预测标签
            pred_labels = np.array([1 if c.get("score", 0.0) > 0.5 else 0 for c in top_k])
            
            # 计算Recall
            recall = recall_score(true_labels, pred_labels, zero_division=0)
            recall_scores.append(recall)
        
        return np.mean(recall_scores) if recall_scores else 0.0
    
    def _calculate_mrr(self, results: List[Dict[str, Any]]) -> float:
        """
        计算MRR (Mean Reciprocal Rank)
        
        Args:
            results: 重排序结果列表
            
        Returns:
            MRR分数
        """
        mrr_scores = []
        
        for item in results:
            query = item.get("query", "")
            candidates = item.get("candidates", [])
            
            if not candidates:
                continue
            
            # 找到第一个相关文档的位置
            for i, candidate in enumerate(candidates):
                if candidate.get("true_score", 0.0) > 0.5:
                    mrr_scores.append(1.0 / (i + 1))
                    break
        
        return np.mean(mrr_scores) if mrr_scores else 0.0
    
    def _calculate_map(self, results: List[Dict[str, Any]]) -> float:
        """
        计算MAP (Mean Average Precision)
        
        Args:
            results: 重排序结果列表
            
        Returns:
            MAP分数
        """
        map_scores = []
        
        for item in results:
            query = item.get("query", "")
            candidates = item.get("candidates", [])
            
            if not candidates:
                continue
            
            # 获取真实标签和预测标签
            true_labels = np.array([1 if c.get("true_score", 0.0) > 0.5 else 0 for c in candidates])
            pred_labels = np.array([1 if c.get("score", 0.0) > 0.5 else 0 for c in candidates])
            
            # 计算AP
            ap = self._average_precision(true_labels, pred_labels)
            map_scores.append(ap)
        
        return np.mean(map_scores) if map_scores else 0.0
    
    def _average_precision(self, true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        """
        计算Average Precision
        
        Args:
            true_labels: 真实标签
            pred_labels: 预测标签
            
        Returns:
            Average Precision分数
        """
        if np.sum(true_labels) == 0:
            return 0.0
        
        # 按预测分数排序
        sorted_indices = np.argsort(pred_labels)[::-1]
        true_labels = true_labels[sorted_indices]
        
        # 计算累积和
        cumsum = np.cumsum(true_labels)
        
        # 计算精度
        precision = cumsum / np.arange(1, len(true_labels) + 1)
        
        # 计算AP
        ap = np.sum(precision * true_labels) / np.sum(true_labels)
        
        return ap 