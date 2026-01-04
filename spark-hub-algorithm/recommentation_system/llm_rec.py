import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
import os
from typing import List, Dict, Any, Tuple, Optional

class LLMRecommender:
    """
    基于大语言模型的推荐系统
    
    利用预训练语言模型理解用户兴趣和物品特征，提供个性化推荐
    """
    def __init__(self, 
                 model_name: str = "bert-base-uncased",
                 embedding_dim: int = 768,
                 max_seq_length: int = 128,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化基于大模型的推荐系统
        
        参数:
            model_name (str): 预训练模型名称
            embedding_dim (int): 嵌入维度
            max_seq_length (int): 最大序列长度
            device (str): 计算设备
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.device = device
        
        # 加载预训练模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        
        # 冻结预训练模型参数
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 用户和物品嵌入
        self.user_embeddings = {}
        self.item_embeddings = {}
        
        # 相似度计算
        self.similarity_fn = nn.CosineSimilarity(dim=1)
        
    def _get_embedding(self, text: str) -> torch.Tensor:
        """
        获取文本的嵌入表示
        
        参数:
            text (str): 输入文本
            
        返回:
            torch.Tensor: 文本嵌入向量
        """
        # 对文本进行编码
        inputs = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # 获取嵌入
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用[CLS]标记的输出作为文本表示
            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
            
        return embedding
    
    def _get_item_embedding(self, item_id: int, item_features: Dict[str, Any]) -> torch.Tensor:
        """
        获取物品的嵌入表示
        
        参数:
            item_id (int): 物品ID
            item_features (Dict[str, Any]): 物品特征
            
        返回:
            torch.Tensor: 物品嵌入向量
        """
        # 如果已经计算过，直接返回
        if item_id in self.item_embeddings:
            return self.item_embeddings[item_id]
        
        # 构建物品描述文本
        item_text = f"Title: {item_features.get('title', '')}. "
        item_text += f"Category: {item_features.get('category', '')}. "
        item_text += f"Description: {item_features.get('description', '')}"
        
        # 获取嵌入
        embedding = self._get_embedding(item_text)
        
        # 缓存嵌入
        self.item_embeddings[item_id] = embedding
        
        return embedding
    
    def _get_user_embedding(self, user_id: int, user_history: List[Dict[str, Any]], 
                           item_features: Dict[int, Dict[str, Any]]) -> torch.Tensor:
        """
        获取用户的嵌入表示
        
        参数:
            user_id (int): 用户ID
            user_history (List[Dict[str, Any]]): 用户历史行为
            item_features (Dict[int, Dict[str, Any]]): 物品特征字典
            
        返回:
            torch.Tensor: 用户嵌入向量
        """
        # 如果已经计算过，直接返回
        if user_id in self.user_embeddings:
            return self.user_embeddings[user_id]
        
        # 构建用户兴趣描述
        user_text = f"User {user_id} has interacted with: "
        
        # 添加历史交互物品的描述
        for interaction in user_history:
            item_id = interaction.get('item_id')
            if item_id in item_features:
                item_text = f"Title: {item_features[item_id].get('title', '')}. "
                item_text += f"Category: {item_features[item_id].get('category', '')}. "
                user_text += item_text
        
        # 获取嵌入
        embedding = self._get_embedding(user_text)
        
        # 缓存嵌入
        self.user_embeddings[user_id] = embedding
        
        return embedding
    
    def fit(self, user_history: Dict[int, List[Dict[str, Any]]], 
            item_features: Dict[int, Dict[str, Any]]):
        """
        训练模型（预计算用户和物品嵌入）
        
        参数:
            user_history (Dict[int, List[Dict[str, Any]]]): 用户历史行为
            item_features (Dict[int, Dict[str, Any]]): 物品特征
        """
        print("预计算用户和物品嵌入...")
        
        # 预计算所有物品嵌入
        for item_id in item_features:
            self._get_item_embedding(item_id, item_features[item_id])
            
        # 预计算所有用户嵌入
        for user_id in user_history:
            self._get_user_embedding(user_id, user_history[user_id], item_features)
            
        print(f"预计算完成: {len(self.user_embeddings)} 用户, {len(self.item_embeddings)} 物品")
    
    def recommend(self, user_id: int, user_history: List[Dict[str, Any]], 
                 item_features: Dict[int, Dict[str, Any]], k: int = 5) -> List[int]:
        """
        为用户推荐物品
        
        参数:
            user_id (int): 用户ID
            user_history (List[Dict[str, Any]]): 用户历史行为
            item_features (Dict[int, Dict[str, Any]]): 物品特征
            k (int): 推荐物品数量
            
        返回:
            List[int]: 推荐的物品ID列表
        """
        # 获取用户嵌入
        user_embedding = self._get_user_embedding(user_id, user_history, item_features)
        
        # 计算用户与所有物品的相似度
        similarities = {}
        for item_id, item_feat in item_features.items():
            # 跳过用户已经交互过的物品
            if any(interaction.get('item_id') == item_id for interaction in user_history):
                continue
                
            # 获取物品嵌入
            item_embedding = self._get_item_embedding(item_id, item_feat)
            
            # 计算相似度
            similarity = self.similarity_fn(
                user_embedding.unsqueeze(0), 
                item_embedding.unsqueeze(0)
            ).item()
            
            similarities[item_id] = similarity
        
        # 选择相似度最高的k个物品
        recommended_items = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
        
        return [item_id for item_id, _ in recommended_items]
    
    def explain_recommendation(self, user_id: int, item_id: int, 
                             user_history: List[Dict[str, Any]], 
                             item_features: Dict[int, Dict[str, Any]]) -> str:
        """
        解释推荐原因
        
        参数:
            user_id (int): 用户ID
            item_id (int): 物品ID
            user_history (List[Dict[str, Any]]): 用户历史行为
            item_features (Dict[int, Dict[str, Any]]): 物品特征
            
        返回:
            str: 推荐解释
        """
        # 获取用户和物品嵌入
        user_embedding = self._get_user_embedding(user_id, user_history, item_features)
        item_embedding = self._get_item_embedding(item_id, item_features[item_id])
        
        # 计算相似度
        similarity = self.similarity_fn(
            user_embedding.unsqueeze(0), 
            item_embedding.unsqueeze(0)
        ).item()
        
        # 构建解释文本
        item_title = item_features[item_id].get('title', 'Unknown Item')
        item_category = item_features[item_id].get('category', 'Unknown Category')
        
        # 找出用户历史中最相似的物品
        history_similarities = {}
        for interaction in user_history:
            hist_item_id = interaction.get('item_id')
            if hist_item_id in item_features:
                hist_item_embedding = self._get_item_embedding(hist_item_id, item_features[hist_item_id])
                hist_similarity = self.similarity_fn(
                    item_embedding.unsqueeze(0), 
                    hist_item_embedding.unsqueeze(0)
                ).item()
                history_similarities[hist_item_id] = hist_similarity
        
        # 选择最相似的历史物品
        most_similar_item_id = max(history_similarities.items(), key=lambda x: x[1])[0]
        most_similar_item_title = item_features[most_similar_item_id].get('title', 'Unknown Item')
        
        # 生成解释
        explanation = f"我们推荐 '{item_title}' (类别: {item_category}) 给您，因为:"
        explanation += f"\n1. 它与您喜欢的 '{most_similar_item_title}' 非常相似 (相似度: {history_similarities[most_similar_item_id]:.2f})"
        explanation += f"\n2. 它与您的兴趣偏好匹配度很高 (匹配度: {similarity:.2f})"
        
        return explanation
    
    def generate_personalized_description(self, user_id: int, item_id: int, 
                                        user_history: List[Dict[str, Any]], 
                                        item_features: Dict[int, Dict[str, Any]]) -> str:
        """
        生成个性化物品描述
        
        参数:
            user_id (int): 用户ID
            item_id (int): 物品ID
            user_history (List[Dict[str, Any]]): 用户历史行为
            item_features (Dict[int, Dict[str, Any]]): 物品特征
            
        返回:
            str: 个性化描述
        """
        # 获取物品特征
        item_title = item_features[item_id].get('title', 'Unknown Item')
        item_category = item_features[item_id].get('category', 'Unknown Category')
        item_description = item_features[item_id].get('description', 'No description available')
        
        # 获取用户历史物品类别
        user_categories = set()
        for interaction in user_history:
            hist_item_id = interaction.get('item_id')
            if hist_item_id in item_features:
                user_categories.add(item_features[hist_item_id].get('category', ''))
        
        # 生成个性化描述
        description = f"亲爱的用户 {user_id}，我们为您推荐 '{item_title}'。"
        
        if item_category in user_categories:
            description += f"\n\n您经常浏览 {item_category} 类别的物品，这个推荐应该符合您的兴趣。"
        else:
            description += f"\n\n这是一个 {item_category} 类别的物品，与您之前浏览的类别不同，但可能也会引起您的兴趣。"
        
        description += f"\n\n{item_description}"
        
        return description


class LLMHybridRecommender:
    """
    混合大模型推荐系统
    
    结合大模型推荐和传统推荐方法
    """
    def __init__(self, llm_weight: float = 0.7):
        """
        初始化混合推荐系统
        
        参数:
            llm_weight (float): 大模型推荐权重
        """
        self.llm_weight = llm_weight
        self.llm_recommender = LLMRecommender()
        self.traditional_recommender = None  # 可以是协同过滤、矩阵分解等
        
    def set_traditional_recommender(self, recommender):
        """
        设置传统推荐器
        
        参数:
            recommender: 传统推荐器实例
        """
        self.traditional_recommender = recommender
        
    def fit(self, user_history: Dict[int, List[Dict[str, Any]]], 
            item_features: Dict[int, Dict[str, Any]], ratings_matrix=None):
        """
        训练混合推荐系统
        
        参数:
            user_history (Dict[int, List[Dict[str, Any]]]): 用户历史行为
            item_features (Dict[int, Dict[str, Any]]): 物品特征
            ratings_matrix: 评分矩阵（用于传统推荐器）
        """
        # 训练大模型推荐器
        self.llm_recommender.fit(user_history, item_features)
        
        # 训练传统推荐器
        if self.traditional_recommender is not None and ratings_matrix is not None:
            self.traditional_recommender.fit(ratings_matrix)
            
    def recommend(self, user_id: int, user_history: List[Dict[str, Any]], 
                 item_features: Dict[int, Dict[str, Any]], k: int = 5) -> List[int]:
        """
        混合推荐
        
        参数:
            user_id (int): 用户ID
            user_history (List[Dict[str, Any]]): 用户历史行为
            item_features (Dict[int, Dict[str, Any]]): 物品特征
            k (int): 推荐物品数量
            
        返回:
            List[int]: 推荐的物品ID列表
        """
        # 获取大模型推荐
        llm_recommendations = self.llm_recommender.recommend(user_id, user_history, item_features, k=k)
        
        # 如果没有传统推荐器，直接返回大模型推荐
        if self.traditional_recommender is None:
            return llm_recommendations
            
        # 获取传统推荐
        traditional_recommendations = self.traditional_recommender.recommend(user_id, k=k)
        
        # 合并推荐结果
        all_recommendations = list(set(llm_recommendations + traditional_recommendations))
        
        # 计算每个推荐物品的混合分数
        scores = {}
        for item_id in all_recommendations:
            llm_score = 1.0 if item_id in llm_recommendations else 0.0
            trad_score = 1.0 if item_id in traditional_recommendations else 0.0
            
            # 加权平均
            mixed_score = self.llm_weight * llm_score + (1 - self.llm_weight) * trad_score
            scores[item_id] = mixed_score
            
        # 选择分数最高的k个物品
        recommended_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        return [item_id for item_id, _ in recommended_items] 