# LSTM 架構解析與應用

## 什麼是 LSTM？

LSTM (Long Short-Term Memory) 是一種特殊的遞迴神經網路 (RNN)，專門用來處理「時間序列」或是具有先後順序的資料。有別於 FCN-1D 利用卷積視窗同時觀察多個影格，LSTM 是「一步一步」地讀取每個時間點的資料，並在內部維護一個「記憶狀態 (Cell State)」。這個機制讓 LSTM 能夠記住長期的動作模式（例如：起始動作與結尾動作的關聯），並且解決了傳統 RNN 常見的梯度消失問題。

## 針對 Deadlift 動作辨識的架構拆解

針對目前的 Deadlift 資料，輸入同樣是一個形狀為 `(Batch, Time, Channels)` 的時間序列（其中 `Time = 110`，`Channels` 為高維度的骨架特徵與差分資訊）。我們將 LSTM 模型拆解成兩個核心部分：

### 1. 雙向 LSTM 層 (Bidirectional LSTM)
在我們實作的 `LSTMClassifier` 中，使用了雙向的 LSTM (`bidirectional=True`)。

*   **順向與逆向讀取**：雙向 LSTM 不只從第 1 個影格讀到第 110 個影格，同時也從第 110 個影格倒著讀回第 1 個影格。這樣在判斷某個時間點的動作時，模型不僅擁有「過去的脈絡」，還能參考「未來的發展」。這對於判斷完整的連續動作（如硬舉的起槓到放下）非常有幫助。
*   **隱藏層與層數**：我們設定了預設 `hidden_size=128` 與 `num_layers=2`。這代表 LSTM 內部會用 128 維的向量來儲存記憶，並且有兩層 LSTM 疊加在一起，能夠提取更深層、更抽象的時序特徵。由於是雙向，最終輸出的特徵維度會是 `128 * 2 = 256`。

### 2. 特徵擷取與分類層 (Classifier)
**我們如何從 LSTM 的輸出中得到最終分類結果？**

LSTM 在讀取完整個 110 個影格的序列後，會在每一個時間點都吐出一個輸出狀態。資料經過 LSTM 後的形狀為 `(Batch Size, 110 Time Steps, 256 Hidden Features)`。

**擷取最後一個時間步 (Last Time Step)：**
有別於 FCN-1D 使用平均池化，LSTM 的最後一個時間步的輸出（即讀完所有影格後的最終狀態）理論上已經濃縮了前面所有時間點的記憶精華。
因此我們直接截取 `out[:, -1, :]`，將形狀縮減為 `(Batch Size, 256)`。

最後，將這 256 維的總結特徵，送入一個簡單的分類器 `nn.Sequential(nn.Dropout, nn.Linear(256, 4))`，輸出 4 種 Deadlift 錯誤類型的機率。

## 為什麼 LSTM 適合目前的 Deadlift 資料？

1.  **專精於長期依賴關係**：硬舉是一個有著明確階段性的動作（站位、起槓、鎖定、放下）。LSTM 的記憶單元非常適合捕捉「如果在起槓時發生了某個微小錯誤，是否會導致鎖定時的明顯失敗」這種跨越長時間的因果關係。
2.  **動態長度適應性**：雖然目前資料固定在 110 影格，但 LSTM 本質上能夠處理變動長度的序列。如果未來需要處理沒有經過長度正規化的原始影片片段，LSTM 會是非常彈性的選擇。
3.  **雙向脈絡分析**：透過雙向結構，模型在評估深蹲或硬舉的「底部」最容易出錯的位置時，可以同時參考下放時與起身時的軌跡，提供更全面的判斷。

## PyTorch 實作程式碼

這可以直接在 `LSTM/lstm_model.py` 中找到：

```python
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    """
    LSTM Classifier for Time-Series
    """
    def __init__(self, input_dim, num_classes, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        # bidirectional LSTM, so hidden_size * 2
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes)
        )
        
    def forward(self, x):
        # x shape: (B, T, C)
        out, _ = self.lstm(x)
        
        # Take the output from the last time step
        # out shape: (B, T, hidden_size * 2)
        last_out = out[:, -1, :]
        
        return self.classifier(last_out)
```

## 訓練指令

```bash
python LSTM/train.py --sport deadlift --type 3d --num_workers 4
```
