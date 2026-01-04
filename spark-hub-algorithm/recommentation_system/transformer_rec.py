import torch
import torch.nn as nn
import numpy as np
import math

class TransformerRecommender(nn.Module):
    """
    基于Transformer的推荐算法
    
    使用自注意力机制捕捉用户行为序列中的长期和短期兴趣
    实现了SASRec (Self-Attentive Sequential Recommendation) 的核心思想
    """
    def __init__(self, num_items, max_seq_len=50, hidden_dim=64, num_heads=4, 
                 num_layers=2, dropout=0.2):
        """
        初始化Transformer推荐模型
        
        参数:
            num_items (int): 物品数量
            max_seq_len (int): 最大序列长度
            hidden_dim (int): 隐藏层维度
            num_heads (int): 注意力头数量
            num_layers (int): Transformer层数
            dropout (float): Dropout比率
        """
        super().__init__()
        
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        
        # 物品嵌入层
        self.item_embeddings = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_dim)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, num_items)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def forward(self, input_seqs, attention_mask=None):
        """
        前向传播
        
        参数:
            input_seqs (torch.LongTensor): 输入序列，形状为 (batch_size, seq_len)
            attention_mask (torch.BoolTensor): 注意力掩码，形状为 (batch_size, seq_len)
            
        返回:
            torch.Tensor: 预测分数，形状为 (batch_size, num_items)
        """
        seq_length = input_seqs.size(1)
        
        # 获取物品嵌入
        item_emb = self.item_embeddings(input_seqs)
        
        # 添加位置编码
        positions = torch.arange(seq_length, device=input_seqs.device).expand(input_seqs.size(0), -1)
        pos_emb = self.position_embeddings(positions)
        
        # 组合物品嵌入和位置编码
        x = item_emb + pos_emb
        x = self.dropout(self.layer_norm(x))
        
        # 创建注意力掩码
        if attention_mask is None:
            attention_mask = input_seqs != 0
        
        # 转换掩码格式
        attention_mask = attention_mask.float().masked_fill(
            attention_mask == 0, float('-inf')).masked_fill(
            attention_mask == 1, float(0.0))
        
        # Transformer编码
        x = x.transpose(0, 1)  # 转换为 (seq_len, batch_size, hidden_dim)
        x = self.transformer_encoder(x, src_key_padding_mask=~attention_mask.bool())
        x = x.transpose(0, 1)  # 转换回 (batch_size, seq_len, hidden_dim)
        
        # 获取序列的最后一个非填充位置的表示
        last_positions = attention_mask.sum(1) - 1
        batch_size = x.size(0)
        last_hidden = x[torch.arange(batch_size), last_positions.long()]
        
        # 预测下一个物品
        logits = self.output_layer(last_hidden)
        
        return logits
    
    def predict(self, user_seq, k=5):
        """
        预测用户可能感兴趣的物品
        
        参数:
            user_seq (torch.LongTensor): 用户交互序列
            k (int): 推荐物品数量
            
        返回:
            list: 推荐的物品ID列表
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(user_seq)
            scores = torch.softmax(logits, dim=-1)
            _, indices = torch.topk(scores, k)
            return indices.cpu().numpy().tolist()
    
    def train_model(self, train_loader, optimizer, num_epochs=10, device='cuda'):
        """
        训练模型
        
        参数:
            train_loader (DataLoader): 训练数据加载器
            optimizer: 优化器
            num_epochs (int): 训练轮数
            device (str): 训练设备
            
        返回:
            list: 训练损失历史
        """
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        history = []
        
        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            
            for batch in train_loader:
                input_seqs = batch['input_seqs'].to(device)
                target_items = batch['target_items'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                optimizer.zero_grad()
                
                logits = self.forward(input_seqs, attention_mask)
                loss = criterion(logits, target_items)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            history.append(avg_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        return history 