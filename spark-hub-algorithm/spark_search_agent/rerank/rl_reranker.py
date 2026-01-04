#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from utils import set_seed, get_device, save_model, load_model

logger = logging.getLogger(__name__)

class RerankEnv(gym.Env):
    """重排序环境"""
    
    def __init__(self, data: List[Dict[str, Any]], max_steps: int = 10):
        """
        初始化环境
        
        Args:
            data: 数据列表
            max_steps: 最大步数
        """
        super(RerankEnv, self).__init__()
        
        self.data = data
        self.max_steps = max_steps
        self.current_idx = 0
        self.current_step = 0
        self.current_query = None
        self.current_candidates = None
        self.current_scores = None
        self.current_order = None
        
        # 定义动作空间和观察空间
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(len(data[0]["candidates"]),),
            dtype=np.float32
        )
        
        # 观察空间包含查询和候选文档的表示
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(768 * (1 + len(data[0]["candidates"])),),  # BERT隐藏大小 * (查询 + 文档数量)
            dtype=np.float32
        )
    
    def reset(self):
        """
        重置环境
        
        Returns:
            初始观察
        """
        self.current_idx = np.random.randint(0, len(self.data))
        self.current_step = 0
        self.current_query = self.data[self.current_idx]["query"]
        self.current_candidates = self.data[self.current_idx]["candidates"]
        self.current_scores = np.array([c["score"] for c in self.current_candidates])
        
        # 随机初始化顺序
        self.current_order = np.arange(len(self.current_candidates))
        np.random.shuffle(self.current_order)
        
        # 返回初始观察
        return self._get_observation()
    
    def step(self, action):
        """
        执行一步
        
        Args:
            action: 动作
            
        Returns:
            观察, 奖励, 是否结束, 信息
        """
        self.current_step += 1
        
        # 根据动作重新排序
        self.current_order = np.argsort(-action)
        
        # 计算奖励
        reward = self._calculate_reward()
        
        # 检查是否结束
        done = self.current_step >= self.max_steps
        
        # 获取观察
        observation = self._get_observation()
        
        # 信息
        info = {
            "query": self.current_query,
            "candidates": [self.current_candidates[i] for i in self.current_order],
            "scores": self.current_scores[self.current_order]
        }
        
        return observation, reward, done, info
    
    def _get_observation(self):
        """
        获取观察
        
        Returns:
            观察
        """
        # 这里应该使用BERT等模型获取查询和文档的表示
        # 为了简化，我们使用随机向量
        query_embedding = np.random.randn(768)
        doc_embeddings = np.random.randn(len(self.current_candidates), 768)
        
        # 按当前顺序排列文档嵌入
        ordered_doc_embeddings = doc_embeddings[self.current_order]
        
        # 拼接查询和文档嵌入
        observation = np.concatenate([query_embedding] + [ordered_doc_embeddings[i] for i in range(len(self.current_candidates))])
        
        return observation
    
    def _calculate_reward(self):
        """
        计算奖励
        
        Returns:
            奖励
        """
        # 使用NDCG作为奖励
        ideal_scores = np.sort(self.current_scores)[::-1]
        current_scores = self.current_scores[self.current_order]
        
        # 计算NDCG@10
        dcg = 0
        idcg = 0
        
        for i in range(min(10, len(self.current_scores))):
            dcg += (2 ** current_scores[i] - 1) / np.log2(i + 2)
            idcg += (2 ** ideal_scores[i] - 1) / np.log2(i + 2)
        
        ndcg = dcg / idcg if idcg > 0 else 0
        
        return ndcg

class RLReranker:
    """基于强化学习的重排序类"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        batch_size: int = 32,
        learning_rate: float = 3e-4,
        device: str = "cuda",
        seed: int = 42
    ):
        """
        初始化强化学习重排序器
        
        Args:
            model_path: 预训练模型路径
            batch_size: 批处理大小
            learning_rate: 学习率
            device: 设备类型
            seed: 随机种子
        """
        self.model_path = model_path
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
        self.model = None
        
        logger.info(f"初始化强化学习重排序器: model_path={model_path}, device={self.device}")
    
    def train(self, data: List[Dict[str, Any]], num_epochs: int = 10):
        """
        训练模型
        
        Args:
            data: 训练数据
            num_epochs: 训练轮数
        """
        logger.info(f"开始训练强化学习重排序模型，共 {num_epochs} 轮")
        
        # 创建环境
        env = RerankEnv(data)
        env = DummyVecEnv([lambda: env])
        
        # 初始化模型
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=self.learning_rate,
            n_steps=self.batch_size,
            batch_size=self.batch_size,
            n_epochs=num_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1
        )
        
        # 训练模型
        self.model.learn(total_timesteps=len(data) * num_epochs)
        
        logger.info("强化学习重排序模型训练完成")
    
    def rerank(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        重排序
        
        Args:
            data: 输入数据
            
        Returns:
            重排序后的数据
        """
        logger.info("开始重排序")
        
        if self.model is None:
            logger.error("模型未训练，无法进行重排序")
            return data
        
        reranked_data = []
        
        # 创建环境
        env = RerankEnv(data)
        
        # 对每个查询进行重排序
        for item in data:
            query = item.get("query", "")
            candidates = item.get("candidates", [])
            
            if not query or not candidates:
                reranked_data.append(item)
                continue
            
            # 重置环境
            env.current_query = query
            env.current_candidates = candidates
            env.current_scores = np.array([c["score"] for c in candidates])
            env.current_order = np.arange(len(candidates))
            
            # 获取观察
            observation = env._get_observation()
            
            # 预测动作
            action, _ = self.model.predict(observation)
            
            # 根据动作重新排序
            sorted_indices = np.argsort(-action)
            sorted_candidates = [candidates[i] for i in sorted_indices]
            
            # 更新分数
            for i, idx in enumerate(sorted_indices):
                sorted_candidates[i]["score"] = action[idx]
            
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
        if self.model is not None:
            self.model.save(path)
            logger.info(f"模型已保存到: {path}")
        else:
            logger.warning("模型未训练，无法保存")
    
    def load(self, path: str):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        self.model = PPO.load(path)
        logger.info(f"模型已从 {path} 加载") 