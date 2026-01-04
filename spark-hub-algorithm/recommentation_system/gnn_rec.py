import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

class LightGCN(nn.Module):
    """
    LightGCN推荐算法
    
    使用简化的图卷积网络进行协同过滤
    论文：LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
    """
    def __init__(self, num_users, num_items, embedding_dim=64, n_layers=3):
        """
        初始化LightGCN模型
        
        参数:
            num_users (int): 用户数量
            num_items (int): 物品数量
            embedding_dim (int): 嵌入维度
            n_layers (int): 图卷积层数
        """
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        # 用户和物品的嵌入层
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
    
    def create_adj_matrix(self, user_items):
        """
        创建邻接矩阵
        
        参数:
            user_items (scipy.sparse.csr_matrix): 用户-物品交互矩阵
            
        返回:
            torch.Tensor: 归一化的邻接矩阵
        """
        # 创建用户-物品图的邻接矩阵
        adj_mat = sp.vstack([
            sp.hstack([sp.csr_matrix((user_items.shape[0], user_items.shape[0])), user_items]),
            sp.hstack([user_items.T, sp.csr_matrix((user_items.shape[1], user_items.shape[1]))])
        ])
        
        # 计算度矩阵
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        
        # 归一化邻接矩阵
        norm_adj = d_mat_inv.dot(adj_mat).dot(d_mat_inv)
        
        # 转换为PyTorch稀疏张量
        indices = torch.LongTensor(np.vstack((norm_adj.row, norm_adj.col)))
        values = torch.FloatTensor(norm_adj.data)
        shape = torch.Size(norm_adj.shape)
        
        return torch.sparse.FloatTensor(indices, values, shape)
    
    def forward(self, adj_matrix):
        """
        前向传播
        
        参数:
            adj_matrix (torch.Tensor): 归一化的邻接矩阵
            
        返回:
            tuple: (用户嵌入, 物品嵌入)
        """
        # 获取初始嵌入
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb])
        
        # 存储每一层的嵌入
        embs = [all_emb]
        
        # 图卷积传播
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(adj_matrix, all_emb)
            embs.append(all_emb)
        
        # 层间聚合
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        
        return users, items
    
    def calculate_loss(self, user_emb, item_emb, users, pos_items, neg_items):
        """
        计算BPR损失
        
        参数:
            user_emb (torch.Tensor): 用户嵌入
            item_emb (torch.Tensor): 物品嵌入
            users (torch.LongTensor): 用户索引
            pos_items (torch.LongTensor): 正样本物品索引
            neg_items (torch.LongTensor): 负样本物品索引
            
        返回:
            torch.Tensor: BPR损失
        """
        user_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]
        
        # BPR损失
        pos_scores = torch.sum(user_emb * pos_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=1)
        
        loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # L2正则化
        reg_loss = (1/2) * (user_emb.norm(2).pow(2) + 
                           pos_emb.norm(2).pow(2) + 
                           neg_emb.norm(2).pow(2)) / len(users)
        
        return loss + 1e-5 * reg_loss
    
    def predict(self, user_id, item_ids, adj_matrix):
        """
        预测用户对物品的评分
        
        参数:
            user_id (int): 用户ID
            item_ids (list): 物品ID列表
            adj_matrix (torch.Tensor): 归一化的邻接矩阵
            
        返回:
            numpy.ndarray: 预测分数
        """
        self.eval()
        with torch.no_grad():
            user_emb, item_emb = self.forward(adj_matrix)
            user_emb = user_emb[user_id]
            item_emb = item_emb[item_ids]
            
            scores = torch.matmul(user_emb, item_emb.t())
            return scores.cpu().numpy()
    
    def recommend(self, user_id, adj_matrix, k=5, exclude_interacted=True, interacted_items=None):
        """
        为用户推荐物品
        
        参数:
            user_id (int): 用户ID
            adj_matrix (torch.Tensor): 归一化的邻接矩阵
            k (int): 推荐物品数量
            exclude_interacted (bool): 是否排除已交互物品
            interacted_items (set): 用户已交互物品集合
            
        返回:
            list: 推荐的物品ID列表
        """
        self.eval()
        with torch.no_grad():
            user_emb, item_emb = self.forward(adj_matrix)
            user_emb = user_emb[user_id]
            
            scores = torch.matmul(user_emb, item_emb.t())
            
            if exclude_interacted and interacted_items is not None:
                scores[list(interacted_items)] = float('-inf')
            
            _, indices = torch.topk(scores, k)
            return indices.cpu().numpy().tolist()
    
    def train_model(self, train_loader, optimizer, adj_matrix, num_epochs=10, device='cuda'):
        """
        训练模型
        
        参数:
            train_loader (DataLoader): 训练数据加载器
            optimizer: 优化器
            adj_matrix (torch.Tensor): 归一化的邻接矩阵
            num_epochs (int): 训练轮数
            device (str): 训练设备
            
        返回:
            list: 训练损失历史
        """
        self.to(device)
        adj_matrix = adj_matrix.to(device)
        history = []
        
        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            
            for batch in train_loader:
                users = batch['users'].to(device)
                pos_items = batch['pos_items'].to(device)
                neg_items = batch['neg_items'].to(device)
                
                user_emb, item_emb = self.forward(adj_matrix)
                loss = self.calculate_loss(user_emb, item_emb, users, pos_items, neg_items)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            history.append(avg_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        return history 