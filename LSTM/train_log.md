# LSTM Training Log & Ablation Study

## Initial Training (Baseline)
- **Model**: LSTM
- **Hidden Size**: 128
- **Num Layers**: 2
- **Dropout**: 0.3
- **Weight Decay**: 1e-4
- **Observation**: 發生了嚴重的過擬合 (Overfitting) 現象。在約第 15~20 個 epoch 後，Training Loss 持續下降，但 Validation Loss 開始大幅往上飄移。Train F1-score 持續升高接近 0.9，但 Validation F1-score 停滯並開始下降。

## Ablation Study (消融實驗)
為了解決過擬合問題並找出符合目前資料集大小的最佳配置，我們安排了以下消融實驗：

1. **Higher Dropout (增加丟棄率)**
   - **改動**: `--dropout 0.5`
   - **目的**: 強迫模型不要過度依賴某些特定的神經元，減緩死背的情形。
   - **Tag**: `ablation_dropout_0.5`

2. **Higher Weight Decay (提高權重衰減/L2正規化)**
   - **改動**: `--weight_decay 1e-3`
   - **目的**: 限制模型權重不要變得太大，進而提高模型的泛化能力。
   - **Tag**: `ablation_wd_1e-3`

3. **Data Augmentation (資料增強)**
   - **改動**: `--aug_type jittering`
   - **目的**: 加入微小雜訊讓模型每次看到的資料都有點不一樣，減少死背訓練資料的機會。
   - **Tag**: `ablation_aug_jittering`

4. **Reduced Model Complexity (減少模型複雜度)**
   - **改動**: `--hidden_size 64 --num_layers 1`
   - **目的**: 由於目前的資料集規模較小，過於龐大的模型（128 hidden size / 2 layers）容易有過多參數去死背資料，因此將其縮小，測試是否能減輕過擬合。
   - **Tag**: `ablation_small_model`

---
*註記：所有實驗皆設定 `--num_workers 4` 加速資料讀取。*

## Experimental Results (實驗結果分析)

消融實驗已經跑完，以下是各組設定在測試集(Test Set)上的主要數據：

| Configuration | Parameters | Macro F1 | Accuracy | Observations |
|---------------|------------|----------|----------|--------------|
| 1. Dropout 0.5 | 601,092 | 0.6385 | 0.3177 | 未見明顯突破，整體表現持平。 |
| 2. Weight Decay 1e-3 | 601,092 | **0.6485** | 0.3081 | 取得最高的平均 F1 Score，證實提高權重衰減對抗過擬合有實質幫助。 |
| 3. Augmentation (jittering) | 601,092 | 0.6374 | **0.3363** | 取得最高的 Accuracy，且在「Correct」類別的 F1 (0.4598) 顯著高於其他配置 (~0.40)，這代表資料增強能幫助模型抓到關鍵動作特徵。 |
| 4. Small Model (64/1) | **70,148** | 0.6430 | 0.3148 | **模型參數減少了近 90%** 但表現依然持平甚至微幅提升。這證明小資料集不需要過大的模型，大模型多餘的參數的確都在「死背」訓練集的雜訊。 |
| **5. Combination (組合技)** | **70,148** | 0.6398 | 0.3192 | 結合 Small Model + WD 1e-3 + Jittering。**表現不如預期**，未能產生加乘效果。 |

### 「組合技」失效分析 (Analysis of Combination Failure)
我們將上述所有有效的正規化手段 (Small Model, Weight Decay 1e-3, Data Augmentation) 結合在一起進行訓練，但得到的 Macro F1 (0.6398) 反而不如單獨使用 Small Model (0.6430) 或單獨使用 Weight Decay (0.6485) 的表現。

失效的原因推測如下：
1. **正規化過度 (Over-Regularization)**：
   - `Small Model` 已經大幅移除了模型 90% 的參數，這本身就是非常強的限制。
   - 在模型容量 (Capacity) 已經極小的情況下，再疊加強烈的 `Weight Decay` 限制權重大小，並用 `Jittering` 替輸入資料加入雜訊，會讓模型面臨「學習能力不足 (Underfitting)」的窘境。
   - 模型沒有足夠的參數空間去消化經過 Jittering 增強後較為複雜的資料特徵，導致無法收斂到更好的局部最佳解。
2. **Jittering 帶來的雜訊干擾**：
   - 雖然 Jittering 在大模型中能稍微提升 Accuracy，但它引入的雜訊在輕量級模型 (Small Model) 身上，可能會蓋過真實的時間序列特徵，導致模型反而抓不到主要動作變化。

### 最終結論與後續建議
面對小資料集，**「把所有的正規化方法全部疊加」並不是好策略**，容易導致模型能力不足以學習特徵。

接下來的調整方向建議如下：
1. **退回最佳單一配置**：
   直接採用 `Weight Decay 1e-3` (搭配原本模型)，或是單純採用 `Small Model (64/1)`。這兩者都能在不傷害表現的前提下減緩 overfitting。
2. **微調組合**：
   如果你很想用 Small Model 以節省運算資源，那麼就**不要**疊加太強的 Weight Decay 或 Jittering。可以嘗試：`Small Model` 搭配預設的 `Weight Decay 1e-4` 即可，單純享受參數縮減帶來的防過擬合紅利。
