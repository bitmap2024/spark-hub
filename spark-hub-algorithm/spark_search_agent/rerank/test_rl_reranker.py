#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import numpy as np
from typing import List, Dict, Any

from rl_reranker import RLReranker, RerankEnv

class TestRLReranker(unittest.TestCase):
    """测试强化学习重排序器"""
    
    def setUp(self):
        """初始化测试数据"""
        self.test_data = [
            {
                "query": "测试查询1",
                "candidates": [
                    {"text": "文档1", "score": 0.9},
                    {"text": "文档2", "score": 0.7},
                    {"text": "文档3", "score": 0.5}
                ]
            },
            {
                "query": "测试查询2",
                "candidates": [
                    {"text": "文档4", "score": 0.8},
                    {"text": "文档5", "score": 0.6},
                    {"text": "文档6", "score": 0.4}
                ]
            }
        ]
    
    def test_rerank_env(self):
        """测试重排序环境"""
        env = RerankEnv(self.test_data)
        
        # 测试重置
        observation = env.reset()
        self.assertIsNotNone(observation)
        self.assertEqual(env.current_step, 0)
        
        # 测试步进
        action = np.array([0.8, 0.6, 0.4])
        observation, reward, done, info = env.step(action)
        
        self.assertIsNotNone(observation)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        
        # 测试奖励计算
        self.assertGreaterEqual(reward, 0.0)
        self.assertLessEqual(reward, 1.0)
    
    def test_rl_reranker(self):
        """测试强化学习重排序器"""
        reranker = RLReranker(
            model_path=None,
            batch_size=2,
            learning_rate=1e-4,
            device="cpu",
            seed=42
        )
        
        # 测试训练
        reranker.train(self.test_data, num_epochs=1)
        self.assertIsNotNone(reranker.model)
        
        # 测试重排序
        reranked_results = reranker.rerank(self.test_data)
        self.assertEqual(len(reranked_results), len(self.test_data))
        
        # 验证结果格式
        for result in reranked_results:
            self.assertIn("query", result)
            self.assertIn("candidates", result)
            self.assertEqual(len(result["candidates"]), 3)
            
            # 验证分数是否已更新
            scores = [c["score"] for c in result["candidates"]]
            self.assertTrue(all(isinstance(score, float) for score in scores))
            self.assertTrue(all(0 <= score <= 1 for score in scores))
        
        # 测试保存和加载
        import tempfile
        with tempfile.NamedTemporaryFile() as tmp:
            reranker.save(tmp.name)
            new_reranker = RLReranker(device="cpu")
            new_reranker.load(tmp.name)
            self.assertIsNotNone(new_reranker.model)

if __name__ == '__main__':
    unittest.main() 

# python main.py --data_path input.json --output_path output.json --method rl --model_path bert-base-chinese