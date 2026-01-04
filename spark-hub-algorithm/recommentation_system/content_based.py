import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class ContentBasedRecommender:
    """
    基于内容的推荐算法
    
    使用物品的特征信息进行推荐
    """
    def __init__(self):
        """
        初始化基于内容的推荐器
        """
        self.tfidf = TfidfVectorizer()
        self.similarity_matrix = None
        self.item_features = None
        
    def fit(self, items_features):
        """
        训练模型，计算物品相似度矩阵
        
        参数:
            items_features (numpy.ndarray): 物品特征矩阵，形状为 (n_items, n_features)
            
        返回:
            self: 返回模型自身，支持链式调用
        """
        self.item_features = items_features
        
        # 如果特征是文本，使用TF-IDF向量化
        if isinstance(items_features[0], str):
            self.similarity_matrix = cosine_similarity(self.tfidf.fit_transform(items_features))
        else:
            # 如果特征是数值，直接计算余弦相似度
            self.similarity_matrix = cosine_similarity(items_features)
            
        return self
    
    def recommend(self, item_id, n_recommendations=5):
        """
        基于物品相似度推荐物品
        
        参数:
            item_id (int): 物品ID
            n_recommendations (int): 推荐物品数量
            
        返回:
            list: 推荐的物品ID列表
        """
        # 获取与给定物品最相似的物品
        similar_items = np.argsort(self.similarity_matrix[item_id])[-n_recommendations-1:-1]
        return similar_items.tolist()
    
    def recommend_by_features(self, user_features, n_recommendations=5):
        """
        基于用户特征推荐物品
        
        参数:
            user_features (numpy.ndarray): 用户特征向量
            n_recommendations (int): 推荐物品数量
            
        返回:
            list: 推荐的物品ID列表
        """
        # 如果特征是文本，使用TF-IDF向量化
        if isinstance(user_features, str):
            user_features = self.tfidf.transform([user_features]).toarray()[0]
        
        # 计算用户特征与所有物品特征的相似度
        similarities = cosine_similarity([user_features], self.item_features)[0]
        
        # 返回最相似的物品
        recommendations = np.argsort(similarities)[-n_recommendations:]
        return recommendations.tolist()
    
    def get_similarity_score(self, item_id1, item_id2):
        """
        获取两个物品之间的相似度
        
        参数:
            item_id1 (int): 第一个物品ID
            item_id2 (int): 第二个物品ID
            
        返回:
            float: 相似度分数
        """
        return self.similarity_matrix[item_id1, item_id2]
    
    def get_item_features(self, item_id):
        """
        获取物品的特征
        
        参数:
            item_id (int): 物品ID
            
        返回:
            numpy.ndarray: 物品特征向量
        """
        return self.item_features[item_id] 