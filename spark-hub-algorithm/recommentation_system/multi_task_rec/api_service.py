from fastapi import FastAPI, HTTPException, Depends, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import torch
import uvicorn
import json
import os
from multi_objective_rec import MultiObjectiveRecommender

app = FastAPI(title="多目标推荐系统API", description="为应用提供多目标推荐服务")

# 模型实例缓存
model_cache = {}

class FeatureItem(BaseModel):
    """数据特征项"""
    id: str
    features: Dict[str, float]
    
class RecommendRequest(BaseModel):
    """推荐请求"""
    user_id: str
    context: Dict[str, Any] = Field(default_factory=dict)
    items: List[FeatureItem]
    
class ObjectiveWeight(BaseModel):
    """目标权重配置"""
    objective_name: str
    weight: float = 1.0
    
class RecommendConfig(BaseModel):
    """推荐配置"""
    objectives: List[ObjectiveWeight]
    diversity_factor: float = 0.2
    unexpectedness_factor: float = 0.1
    max_items: int = 20
    
class RecommendResponse(BaseModel):
    """推荐响应"""
    recommended_items: List[str]
    objective_scores: Dict[str, Dict[str, float]]
    
def get_model(model_name: str = "multi_objective") -> MultiObjectiveRecommender:
    """获取或加载推荐模型"""
    if model_name in model_cache:
        return model_cache[model_name]
    
    # 模型保存目录
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, f"{model_name}.pt")
    
    if os.path.exists(model_path):
        # 加载已有模型
        model = MultiObjectiveRecommender.load_model(model_path)
    else:
        # 如果没有保存的模型，创建一个默认模型
        # 默认任务：点击率(CTR)、转化率(CVR)、用户满意度(Satisfaction)
        default_tasks = ["ctr", "cvr", "satisfaction"]
        
        # 示例：用户特征 + 物品特征 + 上下文特征
        input_dim = 30  # 根据实际情况调整
        
        model = MultiObjectiveRecommender(
            input_dim=input_dim,
            task_names=default_tasks
        )
        
    model_cache[model_name] = model
    return model

def extract_features(request: RecommendRequest) -> pd.DataFrame:
    """从请求中提取特征"""
    features_list = []
    
    for item in request.items:
        # 基础特征
        item_features = item.features.copy()
        
        # 添加用户特征（通常应该从用户服务获取）
        # 这里简化处理，使用用户ID的哈希值作为简单特征
        user_id_hash = hash(request.user_id) % 10000
        item_features["user_id_hash"] = user_id_hash / 10000.0
        
        # 添加上下文特征
        for context_key, context_value in request.context.items():
            if isinstance(context_value, (int, float)):
                item_features[f"context_{context_key}"] = context_value
            elif isinstance(context_value, str):
                # 字符串特征编码
                item_features[f"context_{context_key}_hash"] = hash(context_value) % 1000 / 1000.0
        
        # 添加物品ID作为元数据
        item_features["item_id"] = item.id
        
        features_list.append(item_features)
    
    # 创建特征数据框
    df = pd.DataFrame(features_list)
    
    # 提取物品ID用于后续映射
    item_ids = df["item_id"].copy()
    
    # 删除非特征列
    df = df.drop(columns=["item_id"])
    
    # 处理缺失值
    df = df.fillna(0)
    
    return df, item_ids

def pareto_efficient_items(scores: Dict[str, np.ndarray], weights: Dict[str, float], 
                          diversity_factor: float, max_items: int) -> List[int]:
    """
    选择帕累托最优的推荐项
    
    参数:
        scores: 每个目标的预测分数
        weights: 每个目标的权重
        diversity_factor: 多样性因子
        max_items: 最大推荐数量
    
    返回:
        selected_indices: 选择的项目索引列表
    """
    # 计算加权评分
    weighted_scores = np.zeros(len(next(iter(scores.values()))))
    for objective_name, objective_scores in scores.items():
        weight = weights.get(objective_name, 1.0)
        weighted_scores += weight * objective_scores.flatten()
    
    # 基于加权评分的初始排序
    ranked_indices = np.argsort(-weighted_scores)
    
    # 帕累托前沿选择与多样性平衡
    selected_indices = []
    selected_scores = []
    
    for idx in ranked_indices:
        item_scores = [scores[obj][idx] for obj in scores.keys()]
        
        # 如果是空列表，直接添加第一个项目
        if not selected_indices:
            selected_indices.append(idx)
            selected_scores.append(item_scores)
            continue
        
        # 计算与已选项目的相似度
        diversity_score = 0
        for existing_scores in selected_scores:
            # 使用余弦相似度的简化版本
            similarity = sum(a * b for a, b in zip(item_scores, existing_scores))
            diversity_score += similarity
        
        diversity_score /= len(selected_scores)
        
        # 结合原始得分和多样性得分
        final_score = (1 - diversity_factor) * weighted_scores[idx] - diversity_factor * diversity_score
        
        # 如果最终分数足够好，则添加此项目
        if final_score > 0 or len(selected_indices) < 3:  # 确保至少有3个推荐
            selected_indices.append(idx)
            selected_scores.append(item_scores)
            
        # 达到最大推荐数量时停止
        if len(selected_indices) >= max_items:
            break
    
    return selected_indices

@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest, config: RecommendConfig = Body(...)):
    """
    生成个性化推荐
    """
    try:
        # 加载模型
        model = get_model()
        
        # 提取特征
        features_df, item_ids = extract_features(request)
        
        # 如果没有物品，返回空结果
        if len(features_df) == 0:
            return RecommendResponse(
                recommended_items=[],
                objective_scores={}
            )
        
        # 预测多个目标
        predictions = model.predict(features_df)
        
        # 构建目标权重字典
        objective_weights = {obj.objective_name: obj.weight for obj in config.objectives}
        
        # 选择帕累托最优项目
        selected_indices = pareto_efficient_items(
            predictions, 
            objective_weights, 
            config.diversity_factor, 
            config.max_items
        )
        
        # 构建推荐响应
        recommended_item_ids = [item_ids.iloc[idx] for idx in selected_indices]
        
        # 构建每个物品的各目标得分
        objective_scores = {}
        for idx, item_id in enumerate(recommended_item_ids):
            item_scores = {}
            for objective_name, objective_preds in predictions.items():
                original_idx = selected_indices[idx]
                item_scores[objective_name] = float(objective_preds[original_idx][0])
            objective_scores[item_id] = item_scores
        
        return RecommendResponse(
            recommended_items=recommended_item_ids,
            objective_scores=objective_scores
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推荐生成失败: {str(e)}")

@app.post("/train")
def train_model(features: List[Dict[str, Any]] = Body(...),
                targets: Dict[str, List[float]] = Body(...),
                task_weights: Optional[List[float]] = None):
    """
    训练或更新推荐模型
    
    参数:
        features: 特征列表
        targets: 各任务的目标值
        task_weights: 任务权重
    """
    try:
        # 加载模型
        model = get_model()
        
        # 将特征和目标转换为DataFrame和字典
        features_df = pd.DataFrame(features)
        targets_dict = {k: np.array(v) for k, v in targets.items()}
        
        # 训练模型
        history = model.train(
            features_df=features_df,
            targets_dict=targets_dict,
            task_weights=task_weights,
            epochs=5  # 可以通过参数配置
        )
        
        # 保存模型
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "multi_objective.pt")
        model.save_model(model_path)
        
        return {"status": "success", "message": "模型训练完成", "history": history}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型训练失败: {str(e)}")

@app.get("/objectives")
def get_objectives():
    """获取当前支持的推荐目标"""
    model = get_model()
    return {"objectives": model.task_names}

@app.get("/health")
def health_check():
    """健康检查"""
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("api_service:app", host="0.0.0.0", port=8000, reload=True) 