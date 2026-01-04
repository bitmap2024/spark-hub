import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class FM(nn.Module):
    """
    因子分解机部分，用于建模交叉特征
    """
    def __init__(self, input_dim, latent_dim):
        """
        初始化因子分解机层
        
        参数:
            input_dim (int): 输入特征维度
            latent_dim (int): 隐向量维度
        """
        super().__init__()
        # 一阶特征线性变换
        self.first_order = nn.Embedding(input_dim, 1)
        # 二阶特征的隐向量
        self.second_order = nn.Embedding(input_dim, latent_dim)
        
        # 初始化权重
        nn.init.xavier_normal_(self.first_order.weight)
        nn.init.xavier_normal_(self.second_order.weight)
        
    def forward(self, feature_idx, feature_values=None):
        """
        前向传播
        
        参数:
            feature_idx (torch.LongTensor): 特征索引，形状为 (batch_size, field_size)
            feature_values (torch.FloatTensor, optional): 特征值，形状为 (batch_size, field_size)
            
        返回:
            torch.Tensor: FM部分的输出
        """
        # 获取batch_size和field_size
        batch_size, field_size = feature_idx.size()
        
        # 获取一阶特征
        first_order_weight = self.first_order(feature_idx)  # (batch_size, field_size, 1)
        first_order_weight = first_order_weight.squeeze(-1)  # (batch_size, field_size)
        
        # 如果有特征值，需要乘以特征值
        if feature_values is not None:
            first_order_weight = first_order_weight * feature_values
            
        # 一阶特征求和
        first_order_output = torch.sum(first_order_weight, dim=1)  # (batch_size,)
        
        # 获取二阶特征
        second_order_weight = self.second_order(feature_idx)  # (batch_size, field_size, latent_dim)
        
        # 如果有特征值，需要乘以特征值
        if feature_values is not None:
            second_order_weight = second_order_weight * feature_values.unsqueeze(-1)
            
        # 计算二阶交叉项 sum_i(sum_j(v_i * v_j * x_i * x_j))
        sum_square = torch.sum(second_order_weight, dim=1).pow(2)  # (batch_size, latent_dim)
        square_sum = torch.sum(second_order_weight.pow(2), dim=1)  # (batch_size, latent_dim)
        second_order_output = 0.5 * torch.sum(sum_square - square_sum, dim=1)  # (batch_size,)
        
        return first_order_output, second_order_output

class DNN(nn.Module):
    """
    深度神经网络部分，用于建模高阶特征交互
    """
    def __init__(self, input_dim, hidden_layers=[400, 400, 400], dropout_rate=0.5, batch_norm=True):
        """
        初始化深度神经网络
        
        参数:
            input_dim (int): 输入维度
            hidden_layers (list): 隐藏层神经元数量
            dropout_rate (float): Dropout比率
            batch_norm (bool): 是否使用批归一化
        """
        super().__init__()
        
        self.layers = nn.ModuleList()
        layer_sizes = [input_dim] + hidden_layers
        
        for i in range(len(layer_sizes) - 1):
            layer = []
            layer.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            
            if batch_norm:
                layer.append(nn.BatchNorm1d(layer_sizes[i+1]))
                
            layer.append(nn.ReLU())
            
            if dropout_rate > 0:
                layer.append(nn.Dropout(dropout_rate))
                
            self.layers.append(nn.Sequential(*layer))
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入特征，形状为 (batch_size, input_dim)
            
        返回:
            torch.Tensor: DNN部分的输出
        """
        for layer in self.layers:
            x = layer(x)
        return x

class DeepFM(nn.Module):
    """
    DeepFM 推荐算法
    
    结合了因子分解机(FM)和深度神经网络(DNN)
    论文: DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
    """
    def __init__(self, feature_size, field_size, embedding_dim=64, 
                 hidden_layers=[400, 400, 400], dropout_rate=0.5, batch_norm=True):
        """
        初始化DeepFM模型
        
        参数:
            feature_size (int): 特征数量
            field_size (int): 特征域数量
            embedding_dim (int): 嵌入维度
            hidden_layers (list): DNN隐藏层神经元数量
            dropout_rate (float): Dropout比率
            batch_norm (bool): 是否使用批归一化
        """
        super().__init__()
        
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_dim = embedding_dim
        
        # FM部分
        self.fm = FM(feature_size, embedding_dim)
        
        # 获取所有特征的嵌入向量，用于DNN部分
        self.embedding = nn.Embedding(feature_size, embedding_dim)
        nn.init.xavier_normal_(self.embedding.weight)
        
        # DNN部分
        self.dnn = DNN(field_size * embedding_dim, hidden_layers, dropout_rate, batch_norm)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_layers[-1], 1)
        nn.init.xavier_normal_(self.output_layer.weight)
        
    def forward(self, feature_idx, feature_values=None):
        """
        前向传播
        
        参数:
            feature_idx (torch.LongTensor): 特征索引，形状为 (batch_size, field_size)
            feature_values (torch.FloatTensor, optional): 特征值，形状为 (batch_size, field_size)
            
        返回:
            torch.Tensor: 预测的点击/转化概率
        """
        # FM部分
        fm_first_order, fm_second_order = self.fm(feature_idx, feature_values)
        
        # DNN部分
        feature_emb = self.embedding(feature_idx)  # (batch_size, field_size, embedding_dim)
        
        if feature_values is not None:
            feature_emb = feature_emb * feature_values.unsqueeze(-1)
            
        dnn_input = feature_emb.view(-1, self.field_size * self.embedding_dim)
        dnn_output = self.dnn(dnn_input)
        dnn_output = self.output_layer(dnn_output).squeeze(1)
        
        # 组合FM和DNN的输出
        output = fm_first_order + fm_second_order + dnn_output
        
        return torch.sigmoid(output)
    
    def predict(self, feature_idx, feature_values=None):
        """
        预测用户对物品的点击/转化概率
        
        参数:
            feature_idx (torch.LongTensor): 特征索引
            feature_values (torch.FloatTensor, optional): 特征值
            
        返回:
            numpy.ndarray: 预测的概率
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(feature_idx, feature_values)
        return predictions.cpu().numpy()
    
    def train_model(self, train_data, epochs=10, batch_size=256, learning_rate=0.001):
        """
        训练模型
        
        参数:
            train_data (tuple): 训练数据，格式为 (feature_idx, feature_values, labels)
            epochs (int): 训练轮数
            batch_size (int): 批次大小
            learning_rate (float): 学习率
            
        返回:
            list: 训练损失历史
        """
        if len(train_data) == 3:
            feature_idx, feature_values, labels = train_data
        else:
            feature_idx, labels = train_data
            feature_values = None
        
        # 转换为PyTorch张量
        feature_idx = torch.LongTensor(feature_idx)
        labels = torch.FloatTensor(labels)
        
        if feature_values is not None:
            feature_values = torch.FloatTensor(feature_values)
        
        # 定义优化器和损失函数
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.BCELoss()
        
        # 训练历史
        history = []
        
        # 开始训练
        self.train()
        n_samples = len(feature_idx)
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            
            # 按批次训练
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_feature_idx = feature_idx[batch_indices]
                batch_labels = labels[batch_indices]
                
                if feature_values is not None:
                    batch_feature_values = feature_values[batch_indices]
                else:
                    batch_feature_values = None
                
                # 前向传播
                outputs = self.forward(batch_feature_idx, batch_feature_values)
                
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