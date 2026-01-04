import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class AttentionUnit(nn.Module):
    """
    DIEN中的注意力单元，用于计算兴趣与候选物品的匹配程度
    """
    def __init__(self, input_dim, attention_dim=64):
        """
        初始化注意力单元
        
        参数:
            input_dim (int): 输入维度
            attention_dim (int): 注意力层维度
        """
        super().__init__()
        self.attention_layers = nn.Sequential(
            nn.Linear(input_dim * 4, attention_dim),
            nn.PReLU(),
            nn.Linear(attention_dim, attention_dim),
            nn.PReLU(),
            nn.Linear(attention_dim, 1),
        )
        
    def forward(self, query, keys, keys_length):
        """
        前向传播
        
        参数:
            query (torch.Tensor): 候选物品嵌入，形状为 (batch_size, embedding_dim)
            keys (torch.Tensor): 兴趣表示序列，形状为 (batch_size, max_hist_len, embedding_dim)
            keys_length (torch.Tensor): 兴趣序列的实际长度，形状为 (batch_size,)
            
        返回:
            torch.Tensor: 注意力加权后的兴趣表示
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
        
        return output, attention_scores

class GRULayer(nn.Module):
    """
    DIEN中使用的GRU层
    """
    def __init__(self, input_dim, hidden_size, bidirectional=False):
        """
        初始化GRU层
        
        参数:
            input_dim (int): 输入维度
            hidden_size (int): 隐藏层大小
            bidirectional (bool): 是否使用双向GRU
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )
    
    def forward(self, inputs, lengths):
        """
        前向传播
        
        参数:
            inputs (torch.Tensor): 输入序列，形状为 (batch_size, seq_len, input_dim)
            lengths (torch.Tensor): 序列的实际长度，形状为 (batch_size,)
            
        返回:
            tuple: (outputs, final_state)
                outputs 形状为 (batch_size, seq_len, hidden_size * num_directions)
                final_state 形状为 (num_layers * num_directions, batch_size, hidden_size)
        """
        # 按序列长度排序输入
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        
        # 创建PackedSequence
        lengths_cpu = lengths.cpu()
        _, indices = torch.sort(lengths_cpu, descending=True)
        _, reverse_indices = torch.sort(indices)
        
        sorted_inputs = inputs[indices]
        sorted_lengths = lengths_cpu[indices]
        
        # 处理长度为0的序列
        sorted_lengths = sorted_lengths.clamp(min=1)
        
        # 打包序列
        packed_inputs = nn.utils.rnn.pack_padded_sequence(
            sorted_inputs, sorted_lengths, batch_first=True
        )
        
        # 通过GRU
        packed_outputs, final_state = self.gru(packed_inputs)
        
        # 解包序列
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, batch_first=True, total_length=seq_len
        )
        
        # 恢复原始顺序
        outputs = outputs[reverse_indices]
        
        # 处理final_state形状
        if self.bidirectional:
            final_state = torch.cat([final_state[0], final_state[1]], dim=1)
        else:
            final_state = final_state.squeeze(0)
            
        final_state = final_state[reverse_indices]
        
        return outputs, final_state

class AuxiliaryNet(nn.Module):
    """
    DIEN中的辅助网络，用于兴趣抽取层的训练监督
    """
    def __init__(self, input_dim, hidden_size=100):
        """
        初始化辅助网络
        
        参数:
            input_dim (int): 输入维度
            hidden_size (int): 隐藏层大小
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, states, click_seq, noclick_seq):
        """
        前向传播
        
        参数:
            states (torch.Tensor): 兴趣抽取层的隐藏状态，形状为 (batch_size, seq_len, hidden_size)
            click_seq (torch.Tensor): 点击物品序列，形状为 (batch_size, seq_len, embedding_dim)
            noclick_seq (torch.Tensor): 未点击物品序列，形状为 (batch_size, seq_len, embedding_dim)
            
        返回:
            tuple: (click_prob, noclick_prob) 点击和未点击的预测概率
        """
        # 提取序列中非填充部分
        batch_size, seq_len = states.size(0), states.size(1)
        
        # [h, pos_item, h * pos_item]
        click_input = torch.cat([
            states,
            click_seq,
            states * click_seq
        ], dim=2)
        
        # [h, neg_item, h * neg_item]
        noclick_input = torch.cat([
            states,
            noclick_seq,
            states * noclick_seq
        ], dim=2)
        
        # 预测点击概率
        click_prob = self.mlp(click_input.view(-1, click_input.size(2)))
        noclick_prob = self.mlp(noclick_input.view(-1, noclick_input.size(2)))
        
        return click_prob.view(batch_size, seq_len), noclick_prob.view(batch_size, seq_len)

class DIEN(nn.Module):
    """
    Deep Interest Evolution Network (DIEN) 推荐算法
    
    通过兴趣抽取和兴趣演化层捕获用户兴趣的动态变化，提高推荐精度
    论文: Deep Interest Evolution Network for Click-Through Rate Prediction
    """
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_size=64, 
                 attention_dim=64, mlp_layers=[200, 80], dropout_rate=0.2, max_hist_len=50):
        """
        初始化DIEN模型
        
        参数:
            num_users (int): 用户数量
            num_items (int): 物品数量
            embedding_dim (int): 嵌入维度
            hidden_size (int): GRU隐藏层大小
            attention_dim (int): 注意力层维度
            mlp_layers (list): MLP层隐藏单元数量
            dropout_rate (float): Dropout比率
            max_hist_len (int): 最大历史序列长度
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.max_hist_len = max_hist_len
        
        # 嵌入层
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 兴趣抽取层（GRU）
        self.interest_extractor = GRULayer(embedding_dim, hidden_size, bidirectional=False)
        
        # 辅助网络
        self.auxiliary_net = AuxiliaryNet(hidden_size + embedding_dim + hidden_size * embedding_dim)
        
        # 注意力层
        self.attention_layer = AttentionUnit(hidden_size)
        
        # 兴趣演化层（GRU with Attentional Update Gate）
        self.interest_evolution = GRULayer(hidden_size, hidden_size, bidirectional=False)
        
        # 构建MLP层
        self.mlp = nn.ModuleList()
        input_dim = embedding_dim + hidden_size + embedding_dim  # 用户嵌入 + 兴趣表示 + 候选物品嵌入
        
        for layer_size in mlp_layers:
            self.mlp.append(nn.Linear(input_dim, layer_size))
            self.mlp.append(nn.BatchNorm1d(layer_size))
            self.mlp.append(nn.PReLU())
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
    
    def forward(self, user_indices, item_indices, history_indices, history_length,
                click_indices=None, noclick_indices=None, return_aux_loss=False):
        """
        前向传播
        
        参数:
            user_indices (torch.LongTensor): 用户索引，形状为 (batch_size,)
            item_indices (torch.LongTensor): 候选物品索引，形状为 (batch_size,)
            history_indices (torch.LongTensor): 历史行为物品索引，形状为 (batch_size, max_hist_len)
            history_length (torch.LongTensor): 历史行为序列的实际长度，形状为 (batch_size,)
            click_indices (torch.LongTensor, optional): 点击物品索引，用于辅助损失
            noclick_indices (torch.LongTensor, optional): 未点击物品索引，用于辅助损失
            return_aux_loss (bool): 是否返回辅助损失
            
        返回:
            torch.Tensor: 预测的点击/转化概率
        """
        # 获取嵌入表示
        user_emb = self.user_embedding(user_indices)  # (batch_size, embedding_dim)
        item_emb = self.item_embedding(item_indices)  # (batch_size, embedding_dim)
        hist_emb = self.item_embedding(history_indices)  # (batch_size, max_hist_len, embedding_dim)
        
        # 步骤1: 兴趣抽取层
        # 使用GRU提取用户的历史行为兴趣表示
        gru_outputs, _ = self.interest_extractor(hist_emb, history_length)
        
        # 计算辅助损失（如果需要）
        aux_loss = None
        if return_aux_loss and click_indices is not None and noclick_indices is not None:
            click_emb = self.item_embedding(click_indices)
            noclick_emb = self.item_embedding(noclick_indices)
            click_prob, noclick_prob = self.auxiliary_net(gru_outputs, click_emb, noclick_emb)
            
            # 创建有效序列掩码
            mask = torch.arange(self.max_hist_len, device=history_length.device).expand(
                history_length.size(0), self.max_hist_len
            )
            mask = (mask < history_length.unsqueeze(1)).float()
            
            # 计算辅助损失
            click_loss = -torch.log(click_prob + 1e-8) * mask
            noclick_loss = -torch.log(1 - noclick_prob + 1e-8) * mask
            aux_loss = torch.sum(click_loss + noclick_loss) / torch.sum(mask)
        
        # 步骤2: 兴趣演化层
        # 计算候选物品与兴趣表示的注意力
        weighted_interest, attention_scores = self.attention_layer(item_emb, gru_outputs, history_length)
        
        # 通过兴趣演化层
        evolved_outputs, final_state = self.interest_evolution(gru_outputs, history_length)
        
        # 步骤3: 最终预测
        # 拼接特征
        concat_feature = torch.cat([user_emb, weighted_interest, item_emb], dim=1)
        
        # 通过MLP层
        for layer in self.mlp:
            concat_feature = layer(concat_feature)
            
        # 输出层
        output = self.output_layer(concat_feature)
        
        if return_aux_loss:
            return torch.sigmoid(output).squeeze(1), aux_loss
        else:
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
    
    def train_model(self, train_data, epochs=10, batch_size=256, learning_rate=0.001, aux_loss_weight=0.5):
        """
        训练模型
        
        参数:
            train_data (tuple): 训练数据，格式为 (user_indices, item_indices, 
                                history_indices, history_length, click_indices, 
                                noclick_indices, labels)
            epochs (int): 训练轮数
            batch_size (int): 批次大小
            learning_rate (float): 学习率
            aux_loss_weight (float): 辅助损失权重
            
        返回:
            list: 训练损失历史
        """
        user_indices, item_indices, history_indices, history_length, click_indices, noclick_indices, labels = train_data
        
        # 转换为PyTorch张量
        user_indices = torch.LongTensor(user_indices)
        item_indices = torch.LongTensor(item_indices)
        history_indices = torch.LongTensor(history_indices)
        history_length = torch.LongTensor(history_length)
        click_indices = torch.LongTensor(click_indices)
        noclick_indices = torch.LongTensor(noclick_indices)
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
                batch_click_indices = click_indices[batch_indices]
                batch_noclick_indices = noclick_indices[batch_indices]
                batch_labels = labels[batch_indices]
                
                # 前向传播
                outputs, aux_loss = self.forward(
                    batch_user_indices,
                    batch_item_indices,
                    batch_history_indices,
                    batch_history_length,
                    batch_click_indices,
                    batch_noclick_indices,
                    return_aux_loss=True
                )
                
                # 计算损失
                ctr_loss = criterion(outputs, batch_labels)
                loss = ctr_loss + aux_loss_weight * aux_loss
                
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