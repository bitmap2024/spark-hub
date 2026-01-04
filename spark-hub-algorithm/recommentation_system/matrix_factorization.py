import numpy as np
from scipy.sparse.linalg import svds

class MatrixFactorization:
    """
    矩阵分解推荐算法
    
    使用奇异值分解(SVD)进行矩阵分解，适用于稀疏矩阵
    """
    def __init__(self, n_factors=100):
        """
        初始化矩阵分解推荐器
        
        参数:
            n_factors (int): 潜在因子数量
        """
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        
    def fit(self, ratings_matrix):
        """
        训练模型，进行矩阵分解
        
        参数:
            ratings_matrix (numpy.ndarray): 用户-物品评分矩阵，形状为 (n_users, n_items)
            
        返回:
            self: 返回模型自身，支持链式调用
        """
        # 使用SVD进行矩阵分解
        U, sigma, Vt = svds(ratings_matrix, k=self.n_factors)
        
        # 将sigma转换为对角矩阵
        sigma = np.diag(sigma)
        
        # 计算用户和物品的潜在因子
        self.user_factors = U
        self.item_factors = Vt.T
        
        return self
    
    def predict(self, user_id, item_id):
        """
        预测用户对物品的评分
        
        参数:
            user_id (int): 用户ID
            item_id (int): 物品ID
            
        返回:
            float: 预测的评分
        """
        return np.dot(self.user_factors[user_id, :], self.item_factors[item_id, :])
    
    def recommend(self, user_id, n_recommendations=5):
        """
        为用户推荐物品
        
        参数:
            user_id (int): 用户ID
            n_recommendations (int): 推荐物品数量
            
        返回:
            list: 推荐的物品ID列表
        """
        # 计算用户对所有物品的预测评分
        user_predictions = np.dot(self.user_factors[user_id, :], self.item_factors.T)
        
        # 返回评分最高的物品
        recommendations = np.argsort(user_predictions)[-n_recommendations:]
        return recommendations.tolist()
    
    def get_similar_items(self, item_id, n_similar=5):
        """
        获取与给定物品最相似的物品
        
        参数:
            item_id (int): 物品ID
            n_similar (int): 相似物品数量
            
        返回:
            list: 相似物品ID列表
        """
        # 计算物品之间的相似度
        item_similarities = np.dot(self.item_factors, self.item_factors[item_id, :])
        
        # 排除自身
        item_similarities[item_id] = -1
        
        # 返回最相似的物品
        similar_items = np.argsort(item_similarities)[-n_similar:]
        return similar_items.tolist() 