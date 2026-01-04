#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import json
from typing import Dict, List, Any, Optional

from cross_encoder_reranker import CrossEncoderReranker
from sequence_reranker import SequenceReranker
from rl_reranker import RLReranker
from multitask_reranker import MultitaskReranker
from contrastive_reranker import ContrastiveReranker
from data_processor import DataProcessor
from evaluator import Evaluator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='先进重排序系统')
    parser.add_argument('--method', type=str, required=True, 
                        choices=['cross_encoder', 'sequence', 'rl', 'multitask', 'contrastive'],
                        help='选择重排序方法')
    parser.add_argument('--data_path', type=str, required=True,
                        help='输入数据路径')
    parser.add_argument('--output_path', type=str, default='output.json',
                        help='输出结果路径')
    parser.add_argument('--model_path', type=str, default=None,
                        help='预训练模型路径')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批处理大小')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='学习率')
    parser.add_argument('--max_length', type=int, default=512,
                        help='最大序列长度')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备类型 (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    return parser.parse_args()

def load_data(data_path: str) -> List[Dict[str, Any]]:
    """加载数据"""
    logger.info(f"从 {data_path} 加载数据")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_results(results: List[Dict[str, Any]], output_path: str):
    """保存结果"""
    logger.info(f"保存结果到 {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def main():
    """主函数"""
    args = parse_args()
    
    # 加载数据
    data = load_data(args.data_path)
    
    # 数据处理
    data_processor = DataProcessor(max_length=args.max_length)
    processed_data = data_processor.process(data)
    
    # 选择重排序方法
    if args.method == 'cross_encoder':
        reranker = CrossEncoderReranker(
            model_path=args.model_path,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            seed=args.seed
        )
    elif args.method == 'sequence':
        reranker = SequenceReranker(
            model_path=args.model_path,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            seed=args.seed
        )
    elif args.method == 'rl':
        reranker = RLReranker(
            model_path=args.model_path,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            seed=args.seed
        )
    elif args.method == 'multitask':
        reranker = MultitaskReranker(
            model_path=args.model_path,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            seed=args.seed
        )
    elif args.method == 'contrastive':
        reranker = ContrastiveReranker(
            model_path=args.model_path,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            seed=args.seed
        )
    else:
        raise ValueError(f"不支持的重排序方法: {args.method}")
    
    # 训练模型
    logger.info(f"开始训练 {args.method} 重排序模型")
    reranker.train(processed_data, num_epochs=args.num_epochs)
    
    # 重排序
    logger.info("开始重排序")
    reranked_results = reranker.rerank(processed_data)
    
    # 评估
    evaluator = Evaluator()
    metrics = evaluator.evaluate(reranked_results)
    logger.info(f"评估结果: {metrics}")
    
    # 保存结果
    save_results(reranked_results, args.output_path)
    
    logger.info("重排序完成")

if __name__ == "__main__":
    main()