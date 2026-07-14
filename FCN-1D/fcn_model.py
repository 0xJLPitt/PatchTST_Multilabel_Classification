import torch
import torch.nn as nn

class FCN1DClassifier(nn.Module):
    """
    FCN-1D Classifier for Time-Series
    """
    def __init__(self, input_dim, num_classes, kernel_sizes=[7, 5, 3], dropout=0.3):
        super().__init__()
        
        layers = []
        in_channels = input_dim
        
        # 標準 FCN-1D 通道數配置: [128, 256, 128]
        out_channels_list = [128, 256, 128]
        
        # 根據 kernel_sizes 動態生成多層 Conv1d
        for i, k in enumerate(kernel_sizes):
            out_c = out_channels_list[i] if i < len(out_channels_list) else 128
            padding = k // 2 # 確保時間序列長度不變
            
            layers.append(nn.Conv1d(in_channels=in_channels, out_channels=out_c, kernel_size=k, padding=padding))
            layers.append(nn.BatchNorm1d(out_c))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
            in_channels = out_c
            
        self.conv_blocks = nn.Sequential(*layers)
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(in_channels, num_classes)
        
    def forward(self, x):
        # x shape: (B, T, C)
        # Conv1d 預期的 shape 為 (B, C, T)
        x = x.permute(0, 2, 1)
        
        # 卷積提取特徵
        x = self.conv_blocks(x)
        
        # 池化並分類
        x = self.global_pool(x)  # shape: (B, Channels, 1)
        x = x.squeeze(-1)        # shape: (B, Channels)
        
        return self.classifier(x)
