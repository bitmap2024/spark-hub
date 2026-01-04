import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class NeuralCollaborativeFiltering(nn.Module):
    """
    神经协同过滤推荐算法
    
    使用深度神经网络学习用户和物品的潜在表示
    """
    def __init__(self, num_users, num_items, embedding_dim=50, layers=[100, 50, 20]):
        """
        初始化神经协同过滤模型
        
        参数:
            num_users (int): 用户数量
            num_items (int): 物品数量
            embedding_dim (int): 嵌入维度
            layers (list): 全连接层的神经元数量列表
        """
        super().__init__()
        
        # 用户和物品的嵌入层
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 全连接层
        self.fc_layers = nn.ModuleList()
        input_dim = embedding_dim * 2
        for layer_size in layers:
            self.fc_layers.append(nn.Linear(input_dim, layer_size))
            input_dim = layer_size
            
        # 输出层
        self.output_layer = nn.Linear(layers[-1], 1)
        
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
                
    def forward(self, user_indices, item_indices):
        """
        前向传播
        
        参数:
            user_indices (torch.LongTensor): 用户索引
            item_indices (torch.LongTensor): 物品索引
            
        返回:
            torch.Tensor: 预测的评分
        """
        # 获取用户和物品的嵌入
        user_embedding = self.user_embedding(user_indices)
        item_embedding = self.item_embedding(item_indices)
        
        # 连接用户和物品嵌入
        vector = torch.cat([user_embedding, item_embedding], dim=-1)
        
        # 通过全连接层
        for layer in self.fc_layers:
            vector = torch.relu(layer(vector))
            
        # 输出层
        return torch.sigmoid(self.output_layer(vector))
    
    def predict(self, user_indices, item_indices):
        """
        预测用户对物品的评分
        
        参数:
            user_indices (torch.LongTensor): 用户索引
            item_indices (torch.LongTensor): 物品索引
            
        返回:
            numpy.ndarray: 预测的评分
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(user_indices, item_indices)
        return predictions.cpu().numpy()
    
    def train_model(self, train_data, epochs=10, batch_size=256, learning_rate=0.001):
        """
        训练模型
        
        参数:
            train_data (tuple): 训练数据，格式为 (user_indices, item_indices, ratings)
            epochs (int): 训练轮数
            batch_size (int): 批次大小
            learning_rate (float): 学习率
            
        返回:
            list: 训练损失历史
        """
        user_indices, item_indices, ratings = train_data
        
        # 转换为PyTorch张量
        user_indices = torch.LongTensor(user_indices)
        item_indices = torch.LongTensor(item_indices)
        ratings = torch.FloatTensor(ratings).view(-1, 1)
        
        # 定义优化器和损失函数
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
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
                
                # 前向传播
                predictions = self.forward(batch_users, batch_items)
                
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
    
    def recommend(self, user_id, n_recommendations=5):
        """
        为用户推荐物品
        
        参数:
            user_id (int): 用户ID
            n_recommendations (int): 推荐物品数量
            
        返回:
            list: 推荐的物品ID列表
        """
        self.eval()
        with torch.no_grad():
            # 创建用户索引和所有物品索引
            user_indices = torch.LongTensor([user_id] * self.item_embedding.num_embeddings)
            item_indices = torch.LongTensor(range(self.item_embedding.num_embeddings))
            
            # 预测所有物品的评分
            predictions = self.forward(user_indices, item_indices)
            
            # 获取评分最高的物品
            _, indices = torch.topk(predictions.squeeze(), n_recommendations)
            
        return indices.cpu().numpy().tolist() 