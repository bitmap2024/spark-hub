import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFiltering:
    """
    协同过滤推荐算法
    
    支持基于用户和基于物品的协同过滤方法
    """
    def __init__(self, method='user'):
        """
        初始化协同过滤推荐器
        
        参数:
            method (str): 推荐方法，'user'表示基于用户的协同过滤，'item'表示基于物品的协同过滤
        """
        self.method = method
        self.similarity_matrix = None
        
    def fit(self, ratings_matrix):
        """
        训练模型，计算相似度矩阵
        
        参数:
            ratings_matrix (numpy.ndarray): 用户-物品评分矩阵，形状为 (n_users, n_items)
            
        返回:
            self: 返回模型自身，支持链式调用
        """
        if self.method == 'user':
            self.similarity_matrix = cosine_similarity(ratings_matrix)
        else:  # item-based
            self.similarity_matrix = cosine_similarity(ratings_matrix.T)
        return self
    
    def predict(self, ratings_matrix, user_id, item_id, n_neighbors=5):
        """
        预测用户对物品的评分
        
        参数:
            ratings_matrix (numpy.ndarray): 用户-物品评分矩阵
            user_id (int): 用户ID
            item_id (int): 物品ID
            n_neighbors (int): 使用的邻居数量
            
        返回:
            float: 预测的评分
        """
        if self.method == 'user':
            user_similarities = self.similarity_matrix[user_id]
            similar_users = np.argsort(user_similarities)[-n_neighbors-1:-1]
            prediction = np.mean(ratings_matrix[similar_users, item_id])
        else:
            item_similarities = self.similarity_matrix[item_id]
            similar_items = np.argsort(item_similarities)[-n_neighbors-1:-1]
            prediction = np.mean(ratings_matrix[user_id, similar_items])
        return prediction
    
    def recommend(self, ratings_matrix, user_id, n_recommendations=5):
        """
        为用户推荐物品
        
        参数:
            ratings_matrix (numpy.ndarray): 用户-物品评分矩阵
            user_id (int): 用户ID
            n_recommendations (int): 推荐物品数量
            
        返回:
            list: 推荐的物品ID列表
        """
        if self.method == 'user':
            # 基于用户的协同过滤推荐
            user_similarities = self.similarity_matrix[user_id]
            similar_users = np.argsort(user_similarities)[-n_recommendations-1:-1]
            
            # 获取相似用户评分过的物品
            user_rated_items = set(np.where(ratings_matrix[user_id] > 0)[0])
            recommendations = []
            
            for similar_user in similar_users:
                similar_user_rated_items = set(np.where(ratings_matrix[similar_user] > 0)[0])
                # 找出用户未评分的物品
                candidate_items = similar_user_rated_items - user_rated_items
                recommendations.extend(list(candidate_items))
            
            # 去重并限制推荐数量
            recommendations = list(set(recommendations))[:n_recommendations]
            return recommendations
        else:
            # 基于物品的协同过滤推荐
            # 找出用户已评分的物品
            user_rated_items = np.where(ratings_matrix[user_id] > 0)[0]
            
            if len(user_rated_items) == 0:
                return []
            
            # 计算用户已评分物品与其他物品的相似度
            item_similarities = np.zeros(ratings_matrix.shape[1])
            for rated_item in user_rated_items:
                item_similarities += self.similarity_matrix[rated_item]
            
            # 排除用户已评分的物品
            item_similarities[user_rated_items] = -1
            
            # 返回最相似的物品
            recommendations = np.argsort(item_similarities)[-n_recommendations:]
            return recommendations.tolist() 