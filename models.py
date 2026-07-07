import torchvision.models as models
import torch.nn as nn
import torch
    
class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, embed_dim, stride):
        super().__init__()
        # 在 Channel Independence 模式下，輸入維度永遠是 1
        self.proj = nn.Linear(patch_len * 1, embed_dim)
        self.stride = stride
        self.patch_len = patch_len

    def forward(self, x):
        # x shape: (B*C, T, 1)
        B_C, T, _ = x.shape
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride) 
        # x shape: (B*C, num_patches, 1, patch_len)
        x = x.reshape(B_C, -1, self.patch_len) # 展平 patch
        x = self.proj(x) # (B*C, num_patches, embed_dim)
        return x

class PatchTSTClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, input_len, patch_len=10, 
                 embed_dim=256, num_heads=4, num_layers=2, dropout=0.3, stride=1):
        super().__init__()
        
        # 修正點：這裡傳入 1，因為每個通道獨立處理
        self.patch_embed = PatchEmbedding(patch_len, embed_dim, stride)
        
        num_patches = (input_len - patch_len) // stride + 1
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        
        # 顯示指定 activation="gelu" (符合原論文設計)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 輕量化分類層：輸入維度改為 input_dim * embed_dim，參數量縮小約 100 倍以預防過擬合
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim * embed_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim * embed_dim, num_classes)
        )

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        
        # --- 論文核心：Channel Independence ---
        # 1. 重排維度: (B, T, C) -> (B, C, T) -> (B*C, T, 1)
        x = x.permute(0, 2, 1).reshape(B * C, T, 1)
        
        # 2. Patching & Embedding
        x = self.patch_embed(x)  # (B*C, num_patches, embed_dim)
        
        # 3. Transformer
        x = x + self.pos_embed
        x = self.transformer(x)
        
        # 4. 聚合資訊 (Readout)
        # 做法 A：時間軸平均池化 (Mean Pooling) - 在動作識別上通常更為穩健
        x = x.mean(dim=1)  # (B*C, embed_dim)
        
        # 做法 B：原論文做法，只取最後一個 Patch 的特徵
        # x = x[:, -1, :]  # (B*C, embed_dim)
        
        # 5. 還原維度並拉平為 [B, C * embed_dim]
        x = x.view(B, C, -1).reshape(B, -1)
        
        # 6. 分類層
        return self.classifier(x)
    