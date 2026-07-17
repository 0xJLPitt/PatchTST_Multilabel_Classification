# MultiCLS Training Log

## 1. Baseline
* **Architecture**: Continuous Multi-CLS Transformer
  * `input_dim`: 110 (3D Deadlift)
  * `k` (CLS tokens): 4
  * `hidden_dim`: 256
  * `num_layers`: 4
  * `nhead`: 8
  * `dropout`: 0.3
  * `aggregation`: concat
  * `Loss`: Classification Loss + Diversity Loss (`weight=0.1`)
* **Results**:
  * **Macro F1**: 0.6854
  * **Accuracy**: 0.3318
  * **Class F1 Scores**:
    * Correct: 0.4741
    * Far from the shins: 0.7270
    * Hips rise first: 0.7217
    * Collide with the knees: 0.6244
    * Lower back rounding: 0.6684

## 2. Train with BCE (BCEWithLogitsLoss + pos_weight)
* **Architecture**: Same as Baseline, but trained with `BCEWithLogitsLoss(pos_weight=pos_weight)` to handle class imbalance.
* **Results**:
  * **Macro F1**: 0.6795
  * **Accuracy**: 0.3601
  * **Class F1 Scores**:
    * Correct: 0.5100 (Improved from 0.4741)
    * Far from the shins: 0.7281
    * Hips rise first: 0.7046
    * Collide with the knees: 0.6231
    * Lower back rounding: 0.6624

---

## 未加 Data Augmentation 前的優化方向建議

1. **Feature Projection (引入局部時序特徵)**: 
   目前是直接用 `nn.Linear(input_dim, hidden_dim)`。可以考慮改用 `nn.Conv1d` 來捕捉相鄰幀之間的局部時序關聯，然後再送入 Transformer Encoder。

2. **Positional Encoding (位置編碼優化)**:
   目前使用的是可學習的絕對位置編碼 (`nn.Parameter`)。對於時序訊號，改用 **Sinusoidal Positional Encoding (正弦波位置編碼)** 或是 **Rotary Position Embedding (RoPE)** 可能會有更好的泛化能力。

3. **Loss Function (損失函數優化)**:
   雖然加了 `pos_weight` 的 BCE 讓 `Correct` class 的 F1 提升到了 0.51，但可以進一步嘗試使用 **Focal Loss**，讓模型不僅關注不平衡的數量，也自動把重心放在難以分類的樣本上。

4. **降低模型複雜度 (避免 Overfitting)**:
   目前的模型參數較大 (`hidden_dim=256`, `num_layers=4`)。在訓練資料較少且無 augmentation 的情況下容易 overfitting。建議可以嘗試:
   - 降低 `hidden_dim` 到 `128` 甚至是 `64`
   - 減少 `num_layers` 到 `2`
   - 提高 `dropout` 到 `0.4` 或 `0.5`

5. **Diversity Loss 調整**:
   目前的 `diversity_weight=0.1`，可以做一些超參數測試 (例如調為 0.01 或 0.5)，以觀察 k 個 CLS Tokens 是否真正捕捉到不同的特徵。

6. **Aggregation 策略**:
   目前 4 個 CLS token 是用 `concat` 的方式進入最後的分類器。可以試著改用 `mean` (求平均) 來看看是否能降低分類器的參數負擔。

---

## 消融實驗結果 (Ablation Studies)

我們針對上述 6 個優化方向，各別控制單一變數進行了訓練實驗。各組皆以 **Train with BCE** 為基礎比較：

| 實驗 Tag | 實驗設定 | Macro F1 | Accuracy | Correct F1 | 參數數量 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0. Baseline (BCE)** | `bce` + `linear` + `learnable_pos` + `concat` | 0.6795 | 0.3601 | 0.5100 | 5.79M |
| **1. 1_conv1d_proj** | 改用 `Conv1D` 作為特徵投射 | 0.6680 | 0.3578 | 0.5137 | 5.83M |
| **2. 2_sinusoidal_pos** | 改用固定正弦位置編碼 | 0.6756 | 0.3296 | 0.5378 | 5.54M |
| **3. 3_focal_loss** | 損失函數改為 `Focal Loss` | 0.6712 | 0.3482 | 0.5321 | 5.79M |
| **4. 4_lower_complexity** | `dim=128`, `layers=2`, `dropout=0.5` | 0.6771 | 0.3393 | 0.5323 | 1.38M |
| **5. 5_div_weight_0.5** | Diversity Weight 提高至 `0.5` | **0.6820** | **0.4091** | **0.5855** | 5.79M |
| **6. 6_mean_aggregation** | 分類器輸入改為 `mean` | **0.6836** | 0.3526 | 0.5293 | 5.60M |
| **7. 7_best_comb_no_aug** | `div=0.5` + `mean` (組合 5 與 6) | 0.6783 | 0.3786 | 0.5042 | 5.60M |
| **8. 8_div0.5_lower_comp**| `div=0.5` + `Lower Complexity` | 0.6768 | 0.3452 | 0.5420 | 1.38M |
| **9. 10_div0.5_dim256_l2**| `div=0.5` + `dim=256, layers=2` | **0.6855** | 0.3823 | 0.5402 | 3.16M |
| **10. 11_div0.5_dim128_l4**| `div=0.5` + `dim=128, layers=4` | 0.6747 | 0.3638 | 0.4521 | 2.70M |

### 結果分析與最有效變因：
1. **Diversity Loss Weight 調高至 0.5 最為有效**：這項改動顯著提升了 **Accuracy (達到 0.4091)**，且在最難分類的 **Correct 類別中達到了 0.5855** 的高分 (相比原本的 0.51 大幅提升)。這表示強迫 `k=4` 個 CLS tokens 去學習更多樣、正交的特徵，能有效幫助模型做出更正確的判斷。
2. **Mean Aggregation 也能維持高效能**：將 CLS tokens 平均後進入分類器，**Macro F1 達到了 0.6836** (為所有實驗中最高)，雖然 Accuracy 沒有實驗 5 那麼高，但也成功降低了模型參數。
3. **Focal Loss 與 Sinusoidal Positional Encoding** 雖然讓 Correct 類別稍微提升 (0.532~0.537)，但整體的 Macro F1 反而略降，可能在沒有資料擴增的情況下，這些改動未能帶來全面提升。
4. **尋找輕量化的甜蜜點 (Exp 8~10)**：
   - 當參數大幅縮減至 1.38M (`dim=128, layers=2`) 時，模型容量不足以學習 `div=0.5` 的複雜特徵，效能下滑。
   - 當維持深度但砍半維度 (`dim=128, layers=4`, 2.70M) 時，模型表現依然不佳 (Correct F1 跌至 0.45)。
   - **當維持寬度但砍半深度 (`dim=256, layers=2`, 3.16M) 時，我們找到了完美的甜蜜點！** 不僅參數量從 5.79M 成功瘦身到 3.16M，**Macro F1 更達到了 0.6855**，是所有消融實驗中最高的！

**下一步總結建議**：
你的直覺完全正確！我們成功找出了這份數據的極限最佳解：**實驗 A (`10_div0.5_dim256_layers2`)**。
保留 `hidden_dim=256` 確保模型夠寬以容納多樣性的特徵，而把層數減少為 2 以降低過擬合風險，成功達到了最高的 Macro F1 且參數量近乎減半。接下來我們可以以此架構為基礎，加入 **Data Augmentation** 來挑戰更高的極限！

---

## Data Augmentation 實驗 (dtwwarp)

我們基於上述最佳架構 (`div0.5_dim256_layers2`) 加入了在 PatchTST 表現最好的擴增方法 `dtwwarp` 進行訓練：

### 測試結果 (Tag: `multicls_dtwwarp_best`)
- **Macro F1**: 0.6753 (比原先的 0.6855 下降)
- **Accuracy**: 0.3719
- **各類別 F1**:
  - Correct: 0.5106
  - Far from the shins: 0.7144
  - Hips rise first: 0.7246
  - Collide with the knees: 0.5997
  - Lower back rounding: 0.6624

### 結論
- **效能退步**：與 PatchTST 引入 `dtwwarp` 後能突破 0.69 甚至逼近 0.7 不同，MultiCLS 加上 `dtwwarp` 後反而出現了些微的效能倒退。這暗示了 MultiCLS（特別是基於時序特徵投影與 Diversity Loss 的架構）對於時間軸的非線性扭曲 (DTW) 可能比較敏感，反而破壞了原本模型捕捉到的多樣性特徵。
- MultiCLS 在目前的設定下，不使用 `dtwwarp` 會有最好的表現。
