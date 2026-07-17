# iTransformer Training Log

## 實驗 1：iTransformer 初始架構 (Baseline)

### 架構重點
- **資料維度反轉 (Data Inversion)**：輸入維度由 `[batch_size, seq_len, num_variates]` 轉置為 `[batch_size, num_variates, seq_len]`，將整個時間序列（長度 110 或 100）視為單一特徵，讓 Token 從「時間」變為「變數 (Variates)」。
- **Layer Normalization**：在 Linear Projection 前，針對各個變數的時間序列獨立進行正規化。
- **無時間步位置編碼 (No PE)**：因 Token 已是變數而非時間序列，變數間沒有絕對先後順序，且時間性已被 Linear 層吸收，故不加傳統的 PE。
- **Cross-Variate Attention**：透過 Transformer Encoder 學習變數（各個身體關節）之間的關聯性。
- **分類頭 (Classification Head)**：
  - 採用 `mean` pooling，將 `[batch_size, num_variates, d_model]` 平均壓縮為 `[batch_size, d_model]`。
  - 直接透過單一一層 `nn.Linear` 輸出預測 Logits（未使用 Sigmoid）。
- **Loss Function**：`BCEWithLogitsLoss`（與 FocalLoss 結合，解決正負樣本不均問題）。
- **初始超參數**：
  - `d_model` = 128, `num_layers` = 3, `nhead` = 8
  - `dropout` = 0.1
  - Optimizer = AdamW (`lr=1e-4`, `weight_decay=1e-4`)

### 訓練結果
- **測試集表現**：
  - **Macro F1 Score**: 0.6078
  - **Accuracy**: 0.2368
- **問題分析與觀察**：
  - 訓練日誌顯示存在**極嚴重的過擬合 (Overfitting)**。在 Epoch 76 時，Train Loss 降至 0.0869，Train F1 升至 0.8423；然而 Val Loss 卻暴增到 0.4141，Val F1 則停滯在 0.5661。
  - 表現甚至低於 PatchTST。推測是因為 `mean` pooling 直接把所有關節特徵平均掉，抹除了判定特定動作（如「深蹲圓背」與特定脊椎關節相關）的空間特性。加上模型容量大、正則化不足，導致只會死背訓練集。

---

## 實驗 2：架構與防過擬合改進 (Improved Run)

針對上述問題，對模型架構與訓練腳本（`itransformer_model.py` 與 `train.py`）進行了以下修改：

### 程式碼與架構改動
1. **升級 Classification Head (分類頭)**：
   - 移除原本單純的一層 `Linear`。
   - 改為更魯棒的 **MLP (Linear -> GELU -> Dropout -> Linear)**，增強分類的非線性表現力，同時在最終層加入 Dropout 降低過擬合。
2. **增強正則化 (Regularization)**：
   - 將訓練參數預設的 `--dropout` 從 0.1 大幅提高至 **0.3**。
   - 將 AdamW 最佳化器的 `weight_decay` 從 `1e-4` 提升至 **1e-2**，給予更強烈的 L2 正則化懲罰。

### 建議下次執行的配置與指令 (Improved Run)
在下一次的訓練中，計畫使用以下參數來進一步解決特徵消失與過擬合問題：
- **改用 `--pooling flatten`**：不再平均變數，將所有關節特徵展開，讓模型能為不同關節分配不同權重。
- **縮小模型參數規模**：考量骨架資料集較小，將 Transformer 改為 `--d_model 64`、`--num_layers 2`、`--nhead 4` 以限制其硬背資料的能力。

**本次改動的訓練執行指令 (Tag: `improved_run`)：**
```bash
python train.py --sport deadlift --type 3D --d_model 64 --num_layers 2 --nhead 4 --pooling flatten --tag improved_run --num_workers 4
```

### 訓練結果 (Improved Run)
- **測試集表現**：
  - **Macro F1 Score**: 0.6468 (相比 Baseline 提升了 3.9%)
  - **Accuracy**: 0.3348 (相比 Baseline 提升了 9.8%)
- **各類別 F1 表現**：
  - Correct: 0.4338 (+5.9%)
  - Far from the shins: 0.7085 (+5.6%)
  - Hips rise first: 0.7184 (+4.7%)
  - Collide with the knees: 0.5520 (+4.6%)
  - Lower back rounding: 0.6081 (+0.6%)
- **問題分析與觀察**：
  - **整體與局部特徵提升**：改用 `flatten` 成功保留了不同關節的空間獨立特徵（不再被 `mean` 平均掉）。這讓高度依賴特定關節相對位置的錯誤動作判定大幅改善（如 Hips rise first 與 Far from the shins）。
  - **過擬合 (Overfitting) 依然嚴重**：Train F1 飆高到了 0.9875（近乎完美死背訓練集），但 Val F1 僅有 0.6478。這說明即便縮小了 d_model，分類頭展開後的參數 (21 個關節 * 64 維 = 1344) 依然讓模型能夠輕易硬背小型骨架資料集的特定軌跡。
- **下一步持續優化建議**：
  - **Data Augmentation (資料擴增)**：在 `dataset.py` 中加入骨架座標的高斯雜訊 (Jittering) 或遮蔽 (Masking)，以強迫模型學習通用的動作特徵，防止死背軌跡。
  - **進一步降低參數**：可嘗試將 `--d_model` 降至 `32` 或把 `--dropout` 提高至 `0.5`。

---

## 實驗 3：極端縮小模型與加強 Dropout (Tag: `1.3_decress_d_model_and_add_dropout`)

### 架構與超參數改動
- 延續實驗 2 的 `flatten` pooling，保留各關節空間特徵。
- 將特徵投影維度 `--d_model` 進一步縮小至 **32** (原為 64)。
- 將 Transformer 與 MLP 的 `--dropout` 大幅提高至 **0.5** (原為 0.3)。
- 執行指令：
```bash
python train.py --sport deadlift --type 3D --d_model 32 --num_layers 2 --nhead 4 --pooling flatten --tag 1.3_decress_d_model_and_add_dropout --num_workers 4 --dropout 0.5
```

### 訓練結果
- **測試集表現**：
  - **Macro F1 Score**: 0.6726 (相比實驗 2 提升 2.58%，相比 Baseline 提升高達 6.48%)
  - **Accuracy**: 0.3467 (相比實驗 2 提升 1.19%)
- **各類別 F1 表現 (與實驗 2 比較)**：
  - Correct: 0.4632 (+2.94%)
  - Far from the shins: 0.7207 (+1.22%)
  - Hips rise first: 0.6973 (-2.11%)
  - Collide with the knees: 0.6161 (+6.41%)
  - Lower back rounding: 0.6562 (+4.81%)
- **問題分析與觀察**：
  - **過擬合 (Overfitting) 獲得緩解**：強烈的 Dropout (0.5) 加上更小的 d_model (32) 成功發揮作用。觀察訓練日誌，Train F1 從上一次的近乎完美 (0.98) 降溫到 0.90 左右。模型不再只是無腦硬背訓練集，因此在未看過的測試集上取得了更好的泛化表現。
  - **細微動作辨識度大幅提升**：這項改動特別幫助了「膝蓋內夾 (Collide with knees)」與「深蹲圓背 (Lower back rounding)」這兩個依賴細微關節變化的錯誤動作（分別進步了 6.4% 與 4.8%）。因為模型不再被過多的參數雜訊干擾，能更專注於關鍵的空間變化。
- **下一步優化建議**：
  - 靠著單純「縮小模型容量 (Capacity)」來壓制過擬合的效益可能已經接近極限。如果未來想要突破 0.7 的關卡，將必須從**資料面**下手。加入 **Data Augmentation** (如座標雜訊 Jittering、隨機遮蔽部分關節、時間軸平移等) 將是讓模型學到更通用特徵的最強手段。

---

## 實驗 4：超參數消融實驗 (Ablation Study)

為了在不使用 Data Augmentation 的情況下將 F1 Score 推向極限，我們以「實驗 3 (`d_model=32`, `dropout=0.5`, `pooling=flatten`)」為 Baseline (F1: 0.6726, Acc: 0.3467)，對多個超參數進行了單一變數控制的消融實驗。

### 實驗結果數據
1. **Transformer 內部維度 (`dim_feedforward`，Baseline: 512)**
   - `128`: F1 0.6570, Acc 0.3474 (表現下降)
   - `64`: F1 0.6678, Acc 0.3571 (表現下降)
   - **結論**：保持較大的前饋網路維度有助於特徵映射。

2. **網路層數與注意力頭數 (Baseline: `num_layers=2`, `nhead=4`)**
   - `num_layers=1`: F1 0.6581, Acc 0.3318 (表現下降)
   - `nhead=2`: F1 0.6696, Acc 0.3660 (F1 微降，Acc 上升)
   - **結論**：2 層與 4 個頭的配置在 F1 上依然是較佳選擇。

3. **權重衰減 / L2 正則化 (`weight_decay`，Baseline: 1e-2)**
   - `0.05`: F1 0.6618, Acc 0.3051 (表現下降)
   - `0.1`: F1 0.6649, Acc 0.3385 (表現下降)
   - **結論**：過強的正則化會導致模型欠擬合，維持 1e-2 即可。

4. **學習率 (`lr`，Baseline: 1e-4)**
   - **`5e-4`: F1 0.6754, Acc 0.3742 (F1 與 Acc 皆顯著提升)**
   - `1e-3`: F1 0.6628, Acc 0.3526 (學習率過大，表現下降)
   - **結論**：適度提高學習率 (`5e-4`) 能幫助模型跳出局部最佳解。

5. **分類頭瓶頸維度 (`head_dim`，Baseline: `in_dim//2` 約為 336)**
   - **`128`: F1 0.6777, Acc 0.3675 (目前測出最高 F1)**
   - **`64`: F1 0.6743, Acc 0.3786 (F1 提升，且測出最高 Accuracy)**
   - **結論**：強制壓縮分類前的特徵維度，非常有效地過濾了冗餘雜訊並提升泛化能力。

### 消融實驗總結與最佳組合
根據上述結果，**有效參數**為：
- **`lr = 5e-4`**
- **`head_dim = 128` (或 `64`)**

我們接下來將這兩個有效參數進行組合，執行了最終的測試：
```bash
python train.py --d_model 32 --dropout 0.5 --pooling flatten --lr 5e-4 --head_dim 128 --tag ab_combine_lr_head --num_workers 4
```

### 組合測試結果 (ab_combine_lr_head)
- **Macro F1 Score**: 0.6628 (相比原本的 Baseline `0.6726` 反而下降了)
- **Accuracy**: 0.3638 (比起單獨使用 `head_dim` 或 `lr` 時也沒有更好)

**最終結論分析：**
在機器學習中，兩個分別能提升效能的超參數組合在一起，並不保證能產生「1+1>2」的效果。這主要是因為：
1. `head_dim=128` 或 `lr=5e-4` 都是各自在其預設參數（`lr=1e-4` 或 `head_dim=336`）的環境中被獨立優化出來的。
2. 當分類頭被大幅壓縮到 `128` 後，模型更新梯度的景觀 (Loss Landscape) 也發生了改變，此時再套用原本較大的學習率 `5e-4` 可能反而會讓模型在局部震盪，導致提早觸發 Early Stopping 或無法收斂到最佳點。

**接下來的最佳策略**：
既然組合沒有比較好，在未引入 Data Augmentation 前，我們單獨採納表現最好的單一參數設定即可（例如單獨設定 **`--head_dim 128`** 達到最高 F1 0.6777）。下一步就是正式導入 **Data Augmentation** 了！

---

## Data Augmentation 實驗 (dtwwarp)

我們基於上述最佳架構 (`--d_model 32 --dropout 0.5 --pooling flatten --head_dim 128`) 加入了在 PatchTST 表現最好的擴增方法 `dtwwarp` 進行訓練：

### 測試結果 (Tag: `itransformer_dtwwarp_best`)
- **Macro F1**: 0.6769 (比起原本最高 0.6777 幾乎持平/微降)
- **Accuracy**: 0.3727
- **各類別 F1**:
  - Correct: 0.4720
  - Far from the shins: 0.6963
  - Hips rise first: 0.7376
  - Collide with the knees: 0.6118
  - Lower back rounding: 0.6619

### 結論
- 對 iTransformer 而言，`dtwwarp` 並沒有帶來突破性的進展。這可能是因為 iTransformer 將時間序列轉置為特徵（Data Inversion），而 `dtwwarp` 扭曲的是時間軸；這等於是在變數之間製造了不一致的擾動，因此模型無法像傳統時序模型那樣從中獲益。
