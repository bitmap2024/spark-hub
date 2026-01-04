import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class ActivationUnit(nn.Module):
    """
    DIN中的注意力单元，用于计算候选物品与历史行为的注意力权重
    """
    def __init__(self, embedding_dim, attention_dim=64):
        """
        初始化激活单元
        
        参数:
            embedding_dim (int): 嵌入维度
            attention_dim (int): 注意力层维度
        """
        super().__init__()
        self.attention_layers = nn.Sequential(
            nn.Linear(embedding_dim * 4, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1),
        )
        
    def forward(self, query, keys, keys_length):
        """
        前向传播
        
        参数:
            query (torch.Tensor): 候选物品嵌入，形状为 (batch_size, embedding_dim)
            keys (torch.Tensor): 历史行为物品嵌入，形状为 (batch_size, max_hist_len, embedding_dim)
            keys_length (torch.Tensor): 历史行为序列的实际长度
            
        返回:
            torch.Tensor: 注意力加权后的表示
        """
        batch_size, max_hist_len, embedding_dim = keys.size()
        
        # 将query扩展为与keys相同的形状
        query = query.unsqueeze(1).expand(-1, max_hist_len, -1)  # (batch_size, max_hist_len, embedding_dim)
        
        # 计算query和keys的交互特征
        # [q, k, q-k, q*k]
        q_k = torch.cat([
            query,
            keys,
            query - keys,
            query * keys
        ], dim=2)  # (batch_size, max_hist_len, embedding_dim * 4)
        
        # 计算注意力分数
        attention_scores = self.attention_layers(q_k).squeeze(-1)  # (batch_size, max_hist_len)
        
        # 创建掩码
        mask = torch.arange(max_hist_len, device=keys_length.device).expand(batch_size, max_hist_len)
        mask = (mask < keys_length.unsqueeze(1)).float()
        
        # 使用softmax将分数归一化
        paddings = torch.ones_like(attention_scores) * (-2**32 + 1)
        attention_scores = torch.where(mask.bool(), attention_scores, paddings)
        attention_scores = F.softmax(attention_scores, dim=1)  # (batch_size, max_hist_len)
        
        # 计算加权和
        output = torch.bmm(attention_scores.unsqueeze(1), keys).squeeze(1)  # (batch_size, embedding_dim)
        
        return output

class DIN(nn.Module):
    """
    Deep Interest Network (DIN) 推荐算法
    
    通过局部激活单元对用户的历史行为序列进行建模，从而捕获用户的动态兴趣
    论文: Deep Interest Network for Click-Through Rate Prediction
    """
    def __init__(self, num_users, num_items, embedding_dim=64, mlp_layers=[128, 64], dropout_rate=0.2, max_hist_len=50):
        """
        初始化DIN模型
        
        参数:
            num_users (int): 用户数量
            num_items (int): 物品数量
            embedding_dim (int): 嵌入维度
            mlp_layers (list): MLP层隐藏单元数量
            dropout_rate (float): Dropout比率
            max_hist_len (int): 最大历史序列长度
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_hist_len = max_hist_len
        
        # 嵌入层
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 注意力单元
        self.attention_unit = ActivationUnit(embedding_dim)
        
        # 构建MLP层
        self.mlp = nn.ModuleList()
        input_dim = embedding_dim * 3  # 用户嵌入 + 候选物品嵌入 + 历史交互的注意力
        
        for layer_size in mlp_layers:
            self.mlp.append(nn.Linear(input_dim, layer_size))
            self.mlp.append(nn.BatchNorm1d(layer_size))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Dropout(dropout_rate))
            input_dim = layer_size
            
        # 输出层
        self.output_layer = nn.Linear(mlp_layers[-1], 1)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
    
    def forward(self, user_indices, item_indices, history_indices, history_length):
        """
        前向传播
        
        参数:
            user_indices (torch.LongTensor): 用户索引，形状为 (batch_size,)
            item_indices (torch.LongTensor): 候选物品索引，形状为 (batch_size,)
            history_indices (torch.LongTensor): 历史行为物品索引，形状为 (batch_size, max_hist_len)
            history_length (torch.LongTensor): 历史行为序列的实际长度，形状为 (batch_size,)
            
        返回:
            torch.Tensor: 预测的点击/转化概率
        """
        # 获取嵌入表示
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)
        hist_emb = self.item_embedding(history_indices)
        
        # 计算注意力加权的历史行为表示
        hist_attention = self.attention_unit(item_emb, hist_emb, history_length)
        
        # 拼接特征
        concat_feature = torch.cat([user_emb, item_emb, hist_attention], dim=1)
        
        # 通过MLP层
        for layer in self.mlp:
            concat_feature = layer(concat_feature)
            
        # 输出层
        output = self.output_layer(concat_feature)
        
        return torch.sigmoid(output).squeeze(1)
    
    def predict(self, user_indices, item_indices, history_indices, history_length):
        """
        预测用户对物品的点击/转化概率
        
        参数:
            user_indices (torch.LongTensor): 用户索引
            item_indices (torch.LongTensor): 物品索引
            history_indices (torch.LongTensor): 历史行为物品索引
            history_length (torch.LongTensor): 历史行为序列的实际长度
            
        返回:
            numpy.ndarray: 预测的概率
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(user_indices, item_indices, history_indices, history_length)
        return predictions.cpu().numpy()
    
    def recommend(self, user_id, history_indices, history_length, n_recommendations=10):
        """
        为用户推荐物品
        
        参数:
            user_id (int): 用户ID
            history_indices (torch.LongTensor): 历史行为物品索引，形状为 (max_hist_len,)
            history_length (int): 历史行为序列的实际长度
            n_recommendations (int): 推荐物品数量
            
        返回:
            list: 推荐的物品ID列表
        """
        self.eval()
        with torch.no_grad():
            # 创建用户索引和所有物品索引
            num_items = self.item_embedding.num_embeddings
            user_indices = torch.LongTensor([user_id]).repeat(num_items)
            item_indices = torch.LongTensor(range(num_items))
            
            # 扩展历史序列以匹配batch_size
            history_indices = history_indices.unsqueeze(0).repeat(num_items, 1)
            history_length = torch.LongTensor([history_length]).repeat(num_items)
            
            # 预测所有物品的评分（可以分批次处理以节省内存）
            batch_size = 128
            predictions = []
            
            for i in range(0, num_items, batch_size):
                end = min(i + batch_size, num_items)
                batch_predictions = self.forward(
                    user_indices[i:end],
                    item_indices[i:end],
                    history_indices[i:end],
                    history_length[i:end]
                )
                predictions.append(batch_predictions)
                
            predictions = torch.cat(predictions, dim=0)
            
            # 获取评分最高的物品
            _, indices = torch.topk(predictions, n_recommendations)
            
        return indices.cpu().numpy().tolist()
    
    def train_model(self, train_data, epochs=10, batch_size=256, learning_rate=0.001):
        """
        训练模型
        
        参数:
            train_data (tuple): 训练数据，格式为 (user_indices, item_indices, history_indices, history_length, labels)
            epochs (int): 训练轮数
            batch_size (int): 批次大小
            learning_rate (float): 学习率
            
        返回:
            list: 训练损失历史
        """
        user_indices, item_indices, history_indices, history_length, labels = train_data
        
        # 转换为PyTorch张量
        user_indices = torch.LongTensor(user_indices)
        item_indices = torch.LongTensor(item_indices)
        history_indices = torch.LongTensor(history_indices)
        history_length = torch.LongTensor(history_length)
        labels = torch.FloatTensor(labels)
        
        # 定义优化器和损失函数
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.BCELoss()
        
        # 训练历史
        history = []
        
        # 开始训练
        self.train()
        n_samples = len(user_indices)
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            
            # 按批次训练
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_user_indices = user_indices[batch_indices]
                batch_item_indices = item_indices[batch_indices]
                batch_history_indices = history_indices[batch_indices]
                batch_history_length = history_length[batch_indices]
                batch_labels = labels[batch_indices]
                
                # 前向传播
                outputs = self.forward(
                    batch_user_indices,
                    batch_item_indices,
                    batch_history_indices,
                    batch_history_length
                )
                
                # 计算损失
                loss = criterion(outputs, batch_labels)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * (end_idx - start_idx)
            
            # 计算平均损失
            epoch_loss /= n_samples
            history.append(epoch_loss)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")
            
        return history
    
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