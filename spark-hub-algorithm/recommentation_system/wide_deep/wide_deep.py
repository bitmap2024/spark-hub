import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class WideAndDeep(nn.Module):
    """
    Wide & Deep 推荐算法
    
    结合了线性模型（wide部分）和深度神经网络（deep部分）
    Wide部分用于记忆，Deep部分用于泛化
    """
    def __init__(self, num_users, num_items, num_features=0, embedding_dim=64, 
                 hidden_layers=[100, 50], dropout_rate=0.2):
        """
        初始化Wide & Deep模型
        
        参数:
            num_users (int): 用户数量
            num_items (int): 物品数量
            num_features (int): 特征数量（用于wide部分）
            embedding_dim (int): 嵌入维度
            hidden_layers (list): 深度部分的隐藏层大小
            dropout_rate (float): Dropout比率
        """
        super().__init__()
        
        # Wide部分（线性模型）
        self.wide_user_embedding = nn.Embedding(num_users, 1)
        self.wide_item_embedding = nn.Embedding(num_items, 1)
        self.wide_feature_weights = nn.Linear(num_features, 1) if num_features > 0 else None
        
        # Deep部分（深度神经网络）
        # 用户和物品的嵌入层
        self.deep_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.deep_item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 定义深度网络层
        self.deep_layers = nn.ModuleList()
        input_dim = embedding_dim * 2 + num_features
        for layer_size in hidden_layers:
            self.deep_layers.append(nn.Linear(input_dim, layer_size))
            self.deep_layers.append(nn.ReLU())
            self.deep_layers.append(nn.Dropout(dropout_rate))
            input_dim = layer_size
        
        # 深度网络输出层
        self.deep_output = nn.Linear(hidden_layers[-1], 1)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """
        初始化模型权重
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
    
    def forward(self, user_indices, item_indices, features=None):
        """
        前向传播
        
        参数:
            user_indices (torch.LongTensor): 用户索引
            item_indices (torch.LongTensor): 物品索引
            features (torch.FloatTensor, optional): 额外特征
            
        返回:
            torch.Tensor: 预测的评分
        """
        # Wide部分
        wide_user = self.wide_user_embedding(user_indices).squeeze()
        wide_item = self.wide_item_embedding(item_indices).squeeze()
        wide_output = wide_user + wide_item
        
        if features is not None and self.wide_feature_weights is not None:
            wide_features = self.wide_feature_weights(features).squeeze()
            wide_output = wide_output + wide_features
        
        # Deep部分
        deep_user = self.deep_user_embedding(user_indices)
        deep_item = self.deep_item_embedding(item_indices)
        
        if features is not None:
            deep_input = torch.cat([deep_user, deep_item, features], dim=1)
        else:
            deep_input = torch.cat([deep_user, deep_item], dim=1)
            
        for layer in self.deep_layers:
            deep_input = layer(deep_input)
            
        deep_output = self.deep_output(deep_input).squeeze()
        
        # 结合Wide和Deep的输出
        output = wide_output + deep_output
        
        return torch.sigmoid(output)
    
    def predict(self, user_indices, item_indices, features=None):
        """
        预测用户对物品的评分
        
        参数:
            user_indices (torch.LongTensor): 用户索引
            item_indices (torch.LongTensor): 物品索引
            features (torch.FloatTensor, optional): 额外特征
            
        返回:
            numpy.ndarray: 预测的评分
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(user_indices, item_indices, features)
        return predictions.cpu().numpy()
    
    def train_model(self, train_data, epochs=10, batch_size=256, learning_rate=0.001):
        """
        训练模型
        
        参数:
            train_data (tuple): 训练数据，格式为 (user_indices, item_indices, ratings, [features])
            epochs (int): 训练轮数
            batch_size (int): 批次大小
            learning_rate (float): 学习率
            
        返回:
            list: 训练损失历史
        """
        if len(train_data) == 3:
            user_indices, item_indices, ratings = train_data
            features = None
        else:
            user_indices, item_indices, ratings, features = train_data
        
        # 转换为PyTorch张量
        user_indices = torch.LongTensor(user_indices)
        item_indices = torch.LongTensor(item_indices)
        ratings = torch.FloatTensor(ratings)
        
        if features is not None:
            features = torch.FloatTensor(features)
        
        # 定义优化器和损失函数
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.BCELoss()
        
        # 训练历史
        history = []
        
        # 训练循环
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            
            # 批次训练
            for i in range(0, len(user_indices), batch_size):
                batch_users = user_indices[i:i+batch_size]
                batch_items = item_indices[i:i+batch_size]
                batch_ratings = ratings[i:i+batch_size]
                
                batch_features = None
                if features is not None:
                    batch_features = features[i:i+batch_size]
                
                # 前向传播
                predictions = self.forward(batch_users, batch_items, batch_features)
                
                # 计算损失
                loss = criterion(predictions, batch_ratings)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # 记录平均损失
            avg_loss = total_loss / (len(user_indices) / batch_size)
            history.append(avg_loss)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return history
    
    def recommend(self, user_id, n_recommendations=5, features=None):
        """
        为用户推荐物品
        
        参数:
            user_id (int): 用户ID
            n_recommendations (int): 推荐物品数量
            features (numpy.ndarray, optional): 特征矩阵，每行对应一个物品的特征
            
        返回:
            list: 推荐的物品ID列表
        """
        self.eval()
        with torch.no_grad():
            # 创建用户索引和所有物品索引
            num_items = self.deep_item_embedding.num_embeddings
            user_indices = torch.LongTensor([user_id] * num_items)
            item_indices = torch.LongTensor(range(num_items))
            
            # 处理特征（如果有）
            batch_features = None
            if features is not None:
                batch_features = torch.FloatTensor(features)
            
            # 预测所有物品的评分
            predictions = self.forward(user_indices, item_indices, batch_features)
            
            # 获取评分最高的物品
            _, indices = torch.topk(predictions, n_recommendations)
            
        return indices.cpu().numpy().tolist()
    
    def save_model(self, path):
        """
        保存模型
        
        参数:
            path (str): 保存路径
        """
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        """
        加载模型
        
        参数:
            path (str): 模型路径
        """
        self.load_state_dict(torch.load(path))
        self.eval() 