import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class LinearLayer(nn.Module):
    """
    xDeepFM的线性部分，相当于一阶交互
    """
    def __init__(self, feature_size):
        """
        初始化线性层
        
        参数:
            feature_size (int): 特征数量
        """
        super().__init__()
        self.linear = nn.Embedding(feature_size, 1)
        nn.init.xavier_normal_(self.linear.weight)
        
    def forward(self, feature_idx, feature_values=None):
        """
        前向传播
        
        参数:
            feature_idx (torch.LongTensor): 特征索引，形状为 (batch_size, field_size)
            feature_values (torch.FloatTensor, optional): 特征值，形状为 (batch_size, field_size)
            
        返回:
            torch.Tensor: 线性部分的输出
        """
        weights = self.linear(feature_idx)  # (batch_size, field_size, 1)
        weights = weights.squeeze(-1)  # (batch_size, field_size)
        
        if feature_values is not None:
            weights = weights * feature_values
            
        return torch.sum(weights, dim=1)  # (batch_size,)

class CIN(nn.Module):
    """
    压缩交互网络 (Compressed Interaction Network)
    
    用于xDeepFM中显式特征交互的部分
    """
    def __init__(self, field_size, embedding_dim, cin_layer_sizes=[128, 128], direct=False):
        """
        初始化CIN层
        
        参数:
            field_size (int): 特征域数量
            embedding_dim (int): 嵌入维度
            cin_layer_sizes (list): CIN层神经元数量
            direct (bool): 是否直接连接到输出
        """
        super().__init__()
        self.field_size = field_size
        self.embedding_dim = embedding_dim
        self.cin_layer_sizes = cin_layer_sizes
        self.direct = direct
        
        # CIN层的卷积核
        self.conv_layers = nn.ModuleList()
        # 第k层的特征图大小 H_k = |X_k|, 其中X_0是原始特征
        self.cin_feature_maps = [field_size]
        
        for i, layer_size in enumerate(self.cin_layer_sizes):
            # 对于每一层，我们需要m个卷积核，每个卷积核大小为H_k * H_0
            # 这里m就是layer_size
            kernel_size = self.cin_feature_maps[i] * field_size
            self.conv_layers.append(nn.Conv1d(kernel_size, layer_size, 1))
            self.cin_feature_maps.append(layer_size)
            
        # 如果使用直接连接，则使用所有中间层的输出；否则只使用最后一层的输出
        if self.direct:
            self.output_dim = sum(self.cin_layer_sizes)
        else:
            self.output_dim = self.cin_layer_sizes[-1]
            
        self.linear = nn.Linear(self.output_dim, 1)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 嵌入特征，形状为 (batch_size, field_size, embedding_dim)
            
        返回:
            torch.Tensor: CIN部分的输出
        """
        batch_size = x.size(0)
        
        # X_0, shape: (batch_size, field_size, embedding_dim)
        x_0 = x
        
        # 记录每一层的输出
        xs = []
        for i in range(len(self.cin_layer_sizes)):
            if i == 0:
                x_k_prev = x_0
            else:
                x_k_prev = xs[-1]
                
            # 执行特征交互
            # 首先，我们需要计算X_k和X_0的外积
            # 将X_k从 (batch_size, H_k, embedding_dim) 变形为 (batch_size * embedding_dim, H_k)
            x_k_prev_reshape = x_k_prev.transpose(1, 2).reshape(batch_size * self.embedding_dim, -1)
            # 将X_0从 (batch_size, H_0, embedding_dim) 变形为 (batch_size * embedding_dim, H_0)
            x_0_reshape = x_0.transpose(1, 2).reshape(batch_size * self.embedding_dim, -1)
            
            # 计算外积，得到 (batch_size * embedding_dim, H_k * H_0)
            outer_product = torch.bmm(
                x_k_prev_reshape.unsqueeze(-1),
                x_0_reshape.unsqueeze(1)
            ).reshape(batch_size * self.embedding_dim, -1)
            
            # 将结果重塑为合适的形状以便于卷积
            outer_product = outer_product.reshape(batch_size, self.embedding_dim, -1)
            
            # 应用卷积
            x_k = self.conv_layers[i](outer_product.transpose(1, 2)).transpose(1, 2)
            
            xs.append(x_k)
            
        # 整合输出
        if self.direct:
            # 聚合所有层的输出
            outputs = torch.cat([x.sum(2) for x in xs], dim=1)
        else:
            # 只使用最后一层的输出
            outputs = xs[-1].sum(2)
            
        return self.linear(outputs).squeeze(1)

class MLP(nn.Module):
    """
    多层感知机，用于xDeepFM的深度部分
    """
    def __init__(self, input_dim, hidden_layers=[400, 400, 400], dropout_rate=0.5, batch_norm=True):
        """
        初始化多层感知机
        
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
            
        self.output_layer = nn.Linear(hidden_layers[-1], 1)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入特征，形状为 (batch_size, input_dim)
            
        返回:
            torch.Tensor: MLP部分的输出
        """
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x).squeeze(1)

class xDeepFM(nn.Module):
    """
    xDeepFM 推荐算法
    
    结合了线性模型、压缩交互网络(CIN)和深度神经网络(DNN)
    论文: xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems
    """
    def __init__(self, feature_size, field_size, embedding_dim=64, 
                 cin_layer_sizes=[128, 128], mlp_hidden_layers=[400, 400, 400], 
                 dropout_rate=0.5, batch_norm=True, direct=False):
        """
        初始化xDeepFM模型
        
        参数:
            feature_size (int): 特征数量
            field_size (int): 特征域数量
            embedding_dim (int): 嵌入维度
            cin_layer_sizes (list): CIN层神经元数量
            mlp_hidden_layers (list): MLP隐藏层神经元数量
            dropout_rate (float): Dropout比率
            batch_norm (bool): 是否使用批归一化
            direct (bool): CIN是否直接连接到输出
        """
        super().__init__()
        
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_dim = embedding_dim
        
        # 创建嵌入层
        self.embedding = nn.Embedding(feature_size, embedding_dim)
        nn.init.xavier_normal_(self.embedding.weight)
        
        # 线性部分（一阶特征）
        self.linear = LinearLayer(feature_size)
        
        # CIN部分（显式高阶特征交互）
        self.cin = CIN(field_size, embedding_dim, cin_layer_sizes, direct)
        
        # DNN部分（隐式高阶特征交互）
        self.mlp = MLP(field_size * embedding_dim, mlp_hidden_layers, dropout_rate, batch_norm)
        
    def forward(self, feature_idx, feature_values=None):
        """
        前向传播
        
        参数:
            feature_idx (torch.LongTensor): 特征索引，形状为 (batch_size, field_size)
            feature_values (torch.FloatTensor, optional): 特征值，形状为 (batch_size, field_size)
            
        返回:
            torch.Tensor: 预测的点击/转化概率
        """
        # 线性部分
        linear_output = self.linear(feature_idx, feature_values)
        
        # 获取嵌入特征
        embedding_output = self.embedding(feature_idx)  # (batch_size, field_size, embedding_dim)
        
        if feature_values is not None:
            embedding_output = embedding_output * feature_values.unsqueeze(-1)
            
        # CIN部分
        cin_output = self.cin(embedding_output)
        
        # MLP部分
        mlp_input = embedding_output.view(-1, self.field_size * self.embedding_dim)
        mlp_output = self.mlp(mlp_input)
        
        # 组合所有输出
        output = linear_output + cin_output + mlp_output
        
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