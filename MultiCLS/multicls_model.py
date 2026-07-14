import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, seq_len):
        return self.pe[:, :seq_len, :]

class DiversityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, cls_embeddings):
        """
        計算 k 個 [CLS] 向量的正交性損失。
        :param cls_embeddings: [batch_size, k, hidden_dim]
        :return: scalar loss
        """
        batch_size, k, hidden_dim = cls_embeddings.shape
        
        # 將特徵在 hidden_dim 維度進行 L2 正規化
        norm_cls = F.normalize(cls_embeddings, p=2, dim=-1)
        
        # 計算批次中每組 k 個向量的餘弦相似度矩陣 (Cosine Similarity Matrix)
        # shape: [batch_size, k, k]
        sim_matrix = torch.bmm(norm_cls, norm_cls.transpose(1, 2))
        
        # 建立 Mask 遮蔽對角線 (對角線是向量與自己的相似度，必定為 1)
        mask = torch.eye(k, device=cls_embeddings.device).unsqueeze(0).bool()
        
        # 將對角線設為 0，只看不同的 [CLS] 之間的相似度
        sim_matrix.masked_fill_(mask, 0.0)
        
        # 我們希望非對角線的相似度越接近 0 越好。這裡取平方懲罰（或取絕對值）
        # 共有 batch_size * k * (k-1) 個非對角線元素
        off_diagonal_sims = sim_matrix ** 2
        div_loss = off_diagonal_sims.sum() / (batch_size * k * (k - 1))
        
        return div_loss

class ContinuousMultiCLS_Model(nn.Module):
    """
    專為 3D 連續數值序列 (如骨架座標) 設計的 Multi-CLS Transformer
    不使用 Tokenizer，而是直接透過 nn.Linear 映射輸入特徵，並加上可學習的 CLS tokens 與位置編碼。
    """
    def __init__(self, input_dim, num_classes=4, k=4, hidden_dim=256, 
                 num_layers=4, nhead=8, aggregation='concat', dropout=0.3,
                 proj_type='linear', pos_enc='learnable'):
        super().__init__()
        self.k = k
        self.aggregation = aggregation
        self.proj_type = proj_type
        self.pos_enc = pos_enc
        
        # 1. 數值特徵投射層
        if proj_type == 'linear':
            self.feature_proj = nn.Linear(input_dim, hidden_dim)
        elif proj_type == 'conv1d':
            self.feature_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        else:
            raise ValueError(f"Unknown proj_type: {proj_type}")
        
        # 2. 定義 k 個可學習的 [CLS] Tokens (取代 Special Tokens)
        self.cls_embeddings = nn.Parameter(torch.randn(1, k, hidden_dim))
        
        # 3. 定義位置編碼
        if pos_enc == 'learnable':
            self.pos_embedding = nn.Parameter(torch.randn(1, 1000, hidden_dim))
        elif pos_enc == 'sinusoidal':
            self.pos_embedding = PositionalEncoding(hidden_dim, max_len=1000)
        else:
            raise ValueError(f"Unknown pos_enc: {pos_enc}") 
        
        # 4. Transformer Encoder 骨幹
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. 分類頭
        classifier_input_dim = k * hidden_dim if aggregation == 'concat' else hidden_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.diversity_loss_fn = DiversityLoss()
        
    def forward(self, x, return_div_loss=False):
        # x shape: [batch_size, seq_len, input_dim] 
        batch_size, seq_len, _ = x.shape
        
        # 投影連續特徵: [batch_size, seq_len, hidden_dim]
        if self.proj_type == 'linear':
            seq_features = self.feature_proj(x)
        elif self.proj_type == 'conv1d':
            x_t = x.transpose(1, 2)
            seq_features = self.feature_proj(x_t).transpose(1, 2)
        
        # 將 k 個 [CLS] 擴展至目前的 batch size: [batch_size, k, hidden_dim]
        cls_tokens = self.cls_embeddings.expand(batch_size, -1, -1)
        
        # 在時間序列前方拼接 CLS tokens -> 新長度: seq_len + k
        # [batch_size, k + seq_len, hidden_dim]
        hidden_states = torch.cat([cls_tokens, seq_features], dim=1)
        
        # 加上位置編碼 (自動分配 0 到 k-1 給 CLS tokens)
        total_len = hidden_states.size(1)
        if self.pos_enc == 'learnable':
            hidden_states = hidden_states + self.pos_embedding[:, :total_len, :]
        elif self.pos_enc == 'sinusoidal':
            hidden_states = hidden_states + self.pos_embedding(total_len)
        
        # 通過 Transformer
        encoded = self.transformer_encoder(hidden_states)
        
        # 提取前 k 個 CLS
        cls_out = encoded[:, :self.k, :]
        
        # 計算多樣性損失
        div_loss = self.diversity_loss_fn(cls_out)
        
        # 聚合並分類
        features = cls_out.reshape(batch_size, -1) if self.aggregation == 'concat' else cls_out.mean(dim=1)
        logits = self.classifier(features)
        
        if return_div_loss:
            return logits, div_loss
        return logits

