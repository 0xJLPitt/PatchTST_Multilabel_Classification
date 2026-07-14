# Multi-CLS Transformer 架構解析與應用

## 什麼是 Multi-CLS Transformer？

在傳統的 Transformer Encoder（例如 BERT 或 ViT）中，通常只會在輸入序列的最前端加入單一個 `[CLS]` (Classification) 標記，並使用該標記最後的輸出向量來代表整段序列的全局特徵。
然而，一個複雜的時間序列（例如一段健身動作）可能包含多種層面或不同時間點的特徵，單一向量很難將所有重要資訊完美壓縮。

**Multi-CLS Transformer** 的核心概念是在序列前端插入多個不同的 `[CLS_1]`, `[CLS_2]`, ..., `[CLS_k]` 標記，並透過額外的「多樣性懲罰 (Diversity Loss)」強制這 $k$ 個標記學到**正交（Orthogonal）且不重複的特徵**。這相當於讓模型在一次前向傳播中，同時擁有 $k$ 個不同的視角或專家，來各自摘要這段動作的不同面向，最後再將它們組合（Concatenation 或 Mean Pooling）以提升分類的準確性與穩健度。

## 針對 3D Deadlift 動作辨識的架構拆解

因為 Deadlift 3D 骨架資料是連續數值（Continuous Time-Series）而非離散單字，所以我們不使用傳統 NLP 的 Tokenizer，而是直接修改 Transformer 的輸入層：

### 1. 連續數值投射層 (Feature Projection)
原來的 3D 特徵形狀為 `(Batch, Time, Channels)`。我們透過一個 `nn.Linear(Channels, Hidden_Dim)` 將連續數值特徵映射到 Transformer 內部統一的高維空間（例如 256 維）。這取代了 NLP 中查表（Embedding Lookup）的步驟。

### 2. $k$ 個可學習的分類標記 (Learnable [CLS] Tokens)
我們不使用 Tokenizer 特殊字元，而是直接宣告 `nn.Parameter` 初始化 $k$ 組隨機向量。在每個 batch 前向傳播時，將這 $k$ 個標記拼接（Concat）在映射好的時間序列最前方。
所以輸入到 Transformer 的序列總長度從 `Time` 變成了 `Time + k`。

### 3. 多樣性強制機制 (Diversity Constraint / Orthogonal Loss)
如果沒有這層約束，模型很容易讓這 $k$ 個標記學到一模一樣的資訊（發生特徵同質化），這樣就失去了使用多個 `[CLS]` 的意義。
**解決方法：** 我們取出 Transformer 最終層輸出的前 $k$ 個向量，並計算它們兩兩之間的**餘弦相似度矩陣 (Cosine Similarity Matrix)**。透過最小化這個矩陣的非對角線（Off-diagonal）元素，促使這些向量在特徵空間中彼此正交。

### 4. 特徵聚合與分類頭 (Aggregation & Classification)
最後，我們將這 $k$ 個 `[CLS]` 向量取出來，可以選擇：
*   **Concatenation（拼接）**：保留所有的向量細節，輸入到分類器。
*   **Mean Pooling（平均）**：綜合這 $k$ 個專家的意見，輸入到分類器。

## 為什麼 Multi-CLS Transformer 適合 Deadlift 資料？

1. **捕捉多重失敗原因**：硬舉的失敗往往是複合的（例如同時發生「龜背」與「槓鈴撞膝蓋」）。不同的 `[CLS]` token 有機會在訓練過程中自動分工，分別去專注偵測不同的異常模式。
2. **無需人為設計特徵切分**：與其手動把序列切片送入不同模型，Multi-CLS 利用自注意力機制（Self-Attention）讓不同的 `[CLS]` 自動去「看」序列中不同的時間點或特徵，這相當於內建了一個隱式的 Ensemble 系統。
3. **特徵空間利用率更高**：透過 Diversity Loss，強迫模型挖掘資料中所有潛在有用的資訊，而不會只依賴單一最明顯的特徵（例如只看速度最大值，而忽略了微小的角度偏移）。

## PyTorch 實作核心程式碼

以下為 ContinuousMultiCLS 的核心實作：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiversityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, cls_embeddings):
        # cls_embeddings: [batch_size, k, hidden_dim]
        batch_size, k, hidden_dim = cls_embeddings.shape
        
        # 正規化並計算餘弦相似度
        norm_cls = F.normalize(cls_embeddings, p=2, dim=-1)
        sim_matrix = torch.bmm(norm_cls, norm_cls.transpose(1, 2))
        
        # 遮蔽對角線
        mask = torch.eye(k, device=cls_embeddings.device).unsqueeze(0).bool()
        sim_matrix.masked_fill_(mask, 0.0)
        
        # 懲罰非對角線元素的平方
        off_diagonal_sims = sim_matrix ** 2
        div_loss = off_diagonal_sims.sum() / (batch_size * k * (k - 1))
        return div_loss

class ContinuousMultiCLS_Model(nn.Module):
    def __init__(self, input_dim, num_classes=4, k=4, hidden_dim=256, 
                 num_layers=4, nhead=8, aggregation='concat', dropout=0.3):
        super().__init__()
        self.k = k
        self.aggregation = aggregation
        
        # 1. 數值投影
        self.feature_proj = nn.Linear(input_dim, hidden_dim)
        
        # 2. k 個 CLS
        self.cls_embeddings = nn.Parameter(torch.randn(1, k, hidden_dim))
        
        # 3. 位置編碼
        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, hidden_dim)) 
        
        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. 分類器
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
        batch_size, seq_len, _ = x.shape
        
        seq_features = self.feature_proj(x) 
        cls_tokens = self.cls_embeddings.expand(batch_size, -1, -1)
        
        # 拼接並加上位置編碼
        hidden_states = torch.cat([cls_tokens, seq_features], dim=1)
        hidden_states = hidden_states + self.pos_embedding[:, :hidden_states.size(1), :]
        
        encoded = self.transformer_encoder(hidden_states)
        
        # 提取 k 個 CLS 並計算 Diversity Loss
        cls_out = encoded[:, :self.k, :]
        div_loss = self.diversity_loss_fn(cls_out)
        
        features = cls_out.reshape(batch_size, -1) if self.aggregation == 'concat' else cls_out.mean(dim=1)
        logits = self.classifier(features)
        
        if return_div_loss:
            return logits, div_loss
        return logits
```


python train.py --sport deadlift --type 3D --k 4 --hidden_dim 256 --num_layers 4 --diversity_weight 0.1 --num_workers 4

