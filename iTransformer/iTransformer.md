# iTransformer (Inverted Transformer) 

## 簡介 (Introduction)
本文件說明如何使用 iTransformer 架構進行**時間序列二元/多標籤分類任務**。
不同於傳統的 Time Series Forecasting，此架構將輸出層設計為分類頭 (Classification Head)，並搭配 PyTorch 的 `BCEWithLogitsLoss` 進行訓練。

## 架構設計與實作要點 (Architecture & Implementation Details)

### 1. 資料維度反轉 (Data Inversion) ⚠️ 核心機制
- **原始輸入形狀：** `[batch_size, seq_len, num_variates]`
- **維度轉置：** 在進入模型的第一步執行 `x.transpose(1, 2)`，將形狀轉換為 `[batch_size, num_variates, seq_len]`。
- **Token 重新定義：** 轉置後，Token 數量不再是時間步（Time-steps），而是變數數量（`num_variates`）。這能讓注意力機制專注於捕捉「不同變數之間的關聯」。

### 2. 變數嵌入與正規化 (Variate Embedding & Normalization)
- **Layer Normalization：** 在 Embedding 之前，對每個變數的時間序列獨立進行正規化，以消除不同物理變數之間的量綱（Scale）差異。
- **Linear Projection：** 將每個變數的完整歷史序列（長度為 `seq_len`）投影到 `d_model` 維度。投影後形狀為 `[batch_size, num_variates, d_model]`。
- **無位置編碼 (No PE)：** 因為 Token 是「變數」而非「時間」，變數之間通常沒有絕對的順序關係。且時間序列的順序性已經被包含在整個 Linear Projection 層的權重中，因此**不需要**加入傳統的時間步位置編碼。

### 3. Transformer 編碼器 (Transformer Encoder Backbone)
- **跨變數注意力 (Cross-Variate Attention)：** 使用標準的 Transformer Encoder。此時的 Self-Attention 是在 `num_variates` 這個維度上計算的，其作用是學習「不同物理變數之間的連動關係」（Attention Map 大小為 `num_variates x num_variates`）。
- **前饋神經網路 (FFN)：** 負責在 `d_model` 維度上進一步提取該變數的時間特徵與非線性轉換。

### 4. 分類頭設計 (Classification Head)
- **Pooling 機制：** 將 Transformer Encoder 輸出的多個變數特徵 `[batch_size, num_variates, d_model]` 進行聚合。實作中支援 `flatten` (攤平為 `num_variates * d_model`) 或 `mean` pooling (在變數維度取平均，變成 `d_model`)。
- **輸出層：** 通過全連接層 `nn.Linear` 輸出預測結果。輸出的形狀為 `[batch_size, num_classes]`。
- **無 Sigmoid：** 網路最後一步**絕對不要**加上 `Sigmoid`，直接輸出原始的 Logits，以便交由 PyTorch 內建的 `BCEWithLogitsLoss` 處理。

### 5. 損失函數 (Loss Function)
- 訓練時使用 `nn.BCEWithLogitsLoss()`。這比在模型內加 Sigmoid 再算 BCE 具備更好的數值穩定性（Numerical Stability），能有效避免梯度消失或爆炸的問題。這也是實作二元分類最常犯的錯誤之一。

## 程式碼實作與驗證
完整的 PyTorch 實作與包含測試邏輯的腳本請參考同一資料夾下的 `itransformer_model.py`。
檔案中附有一段測試程式碼，將產生隨機 `[batch_size, seq_len, num_variates]` 輸入，並順利通過模型輸出 Logits 後，驗證 `BCEWithLogitsLoss` 的計算。



python train.py --sport deadlift --type 3D --d_model 128 --num_layers 3 --nhead 8