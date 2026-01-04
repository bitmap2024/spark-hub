 # 先进重排序系统

这个项目实现了几种先进的重排序方法，用于在召回和排序阶段之后进一步优化搜索结果。

## 项目结构

- `main.py`: 主入口文件，用于运行不同的重排序方法
- `cross_encoder_reranker.py`: 基于BERT的交叉编码器重排序
- `sequence_reranker.py`: 基于Transformer的序列重排序
- `rl_reranker.py`: 基于强化学习的重排序
- `multitask_reranker.py`: 基于多任务学习的重排序
- `contrastive_reranker.py`: 基于对比学习的重排序
- `utils.py`: 通用工具函数
- `data_processor.py`: 数据处理模块
- `evaluator.py`: 评估模块

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
python main.py --method cross_encoder --data_path data/sample.json
```

## 支持的重排序方法

1. **交叉编码器重排序**: 使用BERT等预训练模型对查询和文档对进行联合编码
2. **序列重排序**: 考虑文档序列的上下文信息进行重排序
3. **强化学习重排序**: 使用强化学习优化重排序策略
4. **多任务学习重排序**: 同时学习多个相关任务以提高重排序性能
5. **对比学习重排序**: 通过对比学习学习更好的文档表示