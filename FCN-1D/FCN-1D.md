# FCN-1D 架構解析與應用

## 什麼是 FCN-1D？

FCN-1D (Fully Convolutional Network for 1D) 是一種專門設計給「時間序列」分類的神經網路架構。它被稱為「Fully Convolutional（全卷積）」，是因為它**完全移除了傳統 CNN 網路最後面的「全連接層 (Fully Connected Layers/Linear Layers)」**，只在最後一步使用一個簡單的分類層來輸出結果。這個架構在多個學術基準測試上，都被證明是時間序列分類中最強且最穩定的 Baseline 之一。

## 針對 Deadlift 動作辨識的架構拆解

針對目前的 Deadlift 資料，輸入是一個形狀為 `(Batch, Time, Channels)` 的時間序列（其中 `Time = 110`，`Channels` 為 2D 的 40 維或 3D 的 70 維，包含多種角度、座標與它們的時間導數）。我們將模型拆解成三個核心部分：

### 1. 三層的卷積區塊 (Convolutional Blocks)
FCN 通常包含 3 層的 Conv1D Block，每一層都執行相同的操作，但在維度與視野上會逐步變化。

*   **Conv1d (一維卷積)**：
    資料有 110 個影格，每個影格有高維度的特徵（包含原始特徵、速度、加速度、Z-Score 等）。
    *   **第一層 (Kernel Size = 7)**：這個 Filter 會一次看連續的 7 個影格，並將這 7 個影格的所有特徵融合起來，學習到「局部的短期動作模式」（例如：這 0.2 秒內，臀核角度突然改變了）。我們設定它輸出 128 個 Channel。
    *   **第二層 (Kernel Size = 5)**：再把剛才學到的 128 種模式，用大小為 5 的視窗再進行一次組合。這能擴大感受野（Receptive Field），學習到「中期的動作模式」。
    *   **第三層 (Kernel Size = 3)**：最後再進行一次更精煉的特徵組合。
*   **BatchNorm1d (批次正規化)**：
    穩定並加速訓練，確保每個 Channel 輸出的數值不會暴增或過小。這對於時間序列資料尤其重要。
*   **ReLU (啟動函數)**：
    過濾掉無用的負值特徵，只保留被神經網路認為「有活化意義」的正向信號。

### 2. 全局平均池化層 (Global Average Pooling, GAP)
**這是 FCN 架構中最關鍵的靈魂設計。**

在經過前面 3 層卷積後，資料的形狀會變成 `(Batch Size, 128 Channels, 110 Time Steps)`。
傳統做法會用 Flatten 將資料攤平，產生上百萬個參數，極容易引發 Overfitting（死背訓練資料）。

**Global Average Pooling 的做法：**
把時間軸（110 個影格）直接「壓扁」，針對每一個 Channel，把 110 個時間點的數值**取平均值**。因此，原本 `(128, 110)` 的矩陣，瞬間變成了 `(128, 1)` 的向量！
*   每個 Channel 都可以視為一種「特定錯誤動作的偵測器」（例如專門偵測龜背，或槓剛鈴撞膝蓋）。
*   GAP 的意義在於詢問：「**在這 110 個影格中，這個特定錯誤特徵的『平均強烈程度』是多少？**」只要整個動作過程中，有某幾個影格觸發了強烈的錯誤特徵，平均值就會被拉高。它具備平移不變性（Translation Invariance），不在乎錯誤是發生在起頭或結尾。

### 3. 簡單分類層 (Classifier)
最後，我們把這 128 個代表整體動作強度的數值，送入一個簡單的 `nn.Linear(128, 4)`，直接輸出 4 種 Deadlift 錯誤類型的機率。參數極少，模型更不容易 Overfitting。

## 為什麼 FCN-1D 適合目前的 Deadlift 資料？

1.  **資料已經極度豐富**：目前資料已包含「一階差分 (速度)」、「二階差分 (加速度)」與「變化率」。比起 Transformer 需要自己學習相鄰影格的差異，Conv1d 可以直接掃描這些速度與加速度資料，提取「急降」、「驟停」等危險動態特徵。
2.  **抗雜訊能力強**：健身動作的關鍵點追蹤（Pose Tracking）常會有抖動雜訊。1D CNN 的卷積核本質上如同一個低通濾波器（Low-pass filter），能自動平滑化局部的異常抖動。
3.  **速度極快**：Transformer 算注意力機制的時間複雜度與序列長度的平方 ($O(T^2)$) 成正比，而 CNN 則是線性的 ($O(T)$)。這讓 FCN 在訓練與部署到邊緣裝置做即時推論時，速度皆極具優勢。

## PyTorch 實作程式碼

這可以直接加入 `models.py` 替換掉原本的 PatchTST 進行測試：

```python
import torch
import torch.nn as nn

class FCN1DClassifier(nn.Module):
    """
    強大的時間序列 Baseline 模型：Fully Convolutional Network (FCN)
    """
    def __init__(self, input_dim, num_classes, input_len=110, dropout=0.3):
        super().__init__()
        # 註：雖然不需要 input_len，但保留參數以相容原本的呼叫方式
        
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 使用全局平均池化將時間軸壓縮
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # 原本輸入 x shape: (B, T, C) -> 例如 (Batch, 110, 70)
        # 轉換為 Conv1d 需要的 shape: (Batch, Channels, Time)
        x = x.permute(0, 2, 1)  # shape 變為 (Batch, 70, 110)
        
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        # 池化
        x = self.global_pool(x)  # shape: (Batch, 128, 1)
        x = x.squeeze(-1)        # shape: (Batch, 128)
        
        return self.classifier(x)
```



python FCN-1D/train.py --sport deadlift --type 3d --num_workers 2
```

---

## Data Augmentation 實驗 (dtwwarp)

針對 FCN-1D 基礎架構，我們加入了在其他模型表現優異的擴增方法 `dtwwarp` 進行測試：

### 測試結果 (Tag: `fcn1d_dtwwarp_best`)
- **Macro F1**: 0.6676
- **Accuracy**: 0.3890
- **各類別 F1**:
  - Correct: 0.5430
  - Far from the shins: 0.7023
  - Hips rise first: 0.7188
  - Collide with the knees: 0.6000
  - Lower back rounding: 0.6491

### 結論
- **表現平穩但未突破天花板**：FCN-1D 加上 `dtwwarp` 後取得了 `0.6676` 的 Macro F1 與 `0.3890` 的不錯準確率。特別是在 `Correct` (0.54) 類別的判斷上表現優異。
- 總體而言，在所有架構的橫向比較中，`dtwwarp` 最能發揮作用的依然是 PatchTST 模型（逼近 0.7），在其他架構上（如 FCN-1D、iTransformer、MultiCLS）雖能訓練出具備競爭力的結果，但都會碰到約 0.67 ~ 0.68 的瓶頸。