import torchvision.models as models
import torch.nn as nn
import torch
    
class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, embed_dim, stride, input_dim):
        super().__init__()
        # 早期融合：將同一個 patch 內的所有 Channel (input_dim) 拉平一起投影
        self.proj = nn.Linear(patch_len * input_dim, embed_dim)
        self.stride = stride
        self.patch_len = patch_len
        self.input_dim = input_dim

    def forward(self, x):
        # x shape: (B, T, C)
        B, T, C = x.shape
        # unfold 在時間軸(dim=1)上切 patch
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride) 
        # x shape: (B, num_patches, C, patch_len)
        
        # 轉換為 (B, num_patches, patch_len, C) 並在最後一維拉平
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(B, -1, self.patch_len * C) # (B, num_patches, patch_len * C)
        
        # 投影至 embed_dim
        x = self.proj(x) # (B, num_patches, embed_dim)
        return x

class PatchTSTClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, input_len, patch_len=16, 
                 embed_dim=256, num_heads=4, num_layers=2, dropout=0.3, stride=8):
        super().__init__()
        
        # 早期融合：傳入真實的 input_dim (例如 40)
        self.patch_embed = PatchEmbedding(patch_len, embed_dim, stride, input_dim)
        
        num_patches = (input_len - patch_len) // stride + 1
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        
        # 顯示指定 activation="gelu" (符合原論文設計)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 輕量化分類器：Transformer 內部已完美融合空間關係，分類器只需看 embed_dim (256維)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        
        # --- 打破 Channel Independence，改為 Early Fusion ---
        # 不需要把通道拆開，直接將原始 (B, T, C) 送入 PatchEmbedding
        
        # 1. Patching & Embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # 2. Transformer
        x = x + self.pos_embed
        x = self.transformer(x)
        
        # 3. 聚合資訊 (Readout)
        # 時間軸平均池化 (Mean Pooling)
        x = x.mean(dim=1)  # (B, embed_dim)
        
        # 不需要重新 reshape，因為 Batch Size 維度一直都是 B
        
        # 4. 分類層
        return self.classifier(x)
    