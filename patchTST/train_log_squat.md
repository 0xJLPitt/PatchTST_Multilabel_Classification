# PatchTST Squat 訓練日誌 (Train Log Squat)

本文件紀錄 PatchTST 模型在 Squat (深蹲) 2D 資料集上的訓練結果與實驗分析。

---

## 🏃 實驗 1：Baseline (無資料增強)
- **Tag**: `phase4.8_test_heads8_best_squat2`
- **超參數**: `--num_heads 8`, `--split_mode subject_exclusive` (8:1:1 split)
- **Augmentation**: None

### 測試集結果 (Fold 0)
- **Macro F1**: `0.6373`
- **Accuracy**: `0.4511`
- **各類別 F1 Score**:
  - Correct (完全正確): `0.4341`
  - Insufficient Depth (深度不足): `0.7173`
  - Excessive Knee Dominance (膝蓋過度主導): `0.5882`
  - Excessive Hip Dominance (髖部過度主導): `0.7232`
  - Posterior Pelvic Tilt (骨盆後傾/屁股眨眼): `0.2737`
  - Early Hip Rise (臀部上升過快): `0.8840`

---

## 🏃 實驗 2：加入 DTW Warping 資料增強
- **Tag**: `phase4.8_test_heads8_dtwwarp_squat`
- **超參數**: `--num_heads 8`, `--split_mode subject_exclusive` (8:1:1 split)
- **Augmentation**: `dtwwarp`

### 測試集結果 (Fold 0)
- **Macro F1**: `0.6252` (🔻 下降 0.0121)
- **Accuracy**: `0.4380` (🔻 下降 0.0131)
- **各類別 F1 Score**:
  - Correct (完全正確): `0.3451` (🔻 下降 0.0890)
  - Insufficient Depth (深度不足): `0.7054` (🔻 下降 0.0119)
  - Excessive Knee Dominance (膝蓋過度主導): `0.5933` (🔺 上升 0.0051)
  - Excessive Hip Dominance (髖部過度主導): `0.7468` (🔺 上升 0.0236)
  - Posterior Pelvic Tilt (骨盆後傾/屁股眨眼): `0.2723` (幾乎持平)
  - Early Hip Rise (臀部上升過快): `0.8081` (🔻 下降 0.0759)

---

## 🔬 實驗分析：為什麼加入 DTWWarp 後分數反而下降？

雖然 Dynamic Time Warping (DTW) 資料增強在 Deadlift (硬舉) 任務上取得了不錯的成效，但在 Squat (深蹲) 任務中，加入 DTWWarp 反而導致整體 Macro F1 與 Accuracy 雙雙下滑。我們可以從各個標籤的變化來深入分析原因：

1. **破壞了嚴格的「時序相對關係」(Timing & Synchronization)**
   - **Early Hip Rise (臀部上升過快)** 這個錯誤動作的核心定義在於「時間差」：在深蹲起身的階段，臀部上升的時間點早於肩膀/軀幹的上升。
   - DTWWarp 會對時間軸進行局部的拉伸與壓縮。當這項技術套用在深蹲動作時，**極有可能把原本正常的深蹲，在時間軸上扭曲成了「臀部先動、肩膀後動」的假象**，或者把原本的 Early Hip Rise 扭曲得不合物理邏輯。這導致模型學習到了充滿雜訊的時序特徵，使得 `Early_Hip_Rise` 的 F1 分數從 `0.8840` 暴跌至 `0.8081`。

2. **「Correct」(正確動作) 判斷標準變得模糊**
   - `Correct` 類別的 F1 分數受到重創（從 `0.4341` 跌至 `0.3451`）。深蹲的「正確節奏」是固定的（穩定的下蹲與流暢的起身）。時間軸的隨機扭曲（Warping）會把標準節奏的正確動作變成忽快忽慢的怪異動作，這讓模型更難去抓取「何謂標準動作」，導致 False Positive 或 False Negative 大幅增加。

3. **空間特徵 (Spatial Features) 無法從時間增強中獲益**
   - 像是 `Posterior_Pelvic_Tilt` (骨盆後傾) 以及 `Insufficient_Depth` (深度不足) 主要是依賴「空間座標的角度與相對位置」(例如：臀部低於膝蓋水平線、腰椎角度變化)。
   - DTWWarp 只有在時間維度做變化，並沒有創造出新的空間角度變異。因此，對於這類依賴空間極值的錯誤動作，時間軸扭曲無法提供有效的幫助，甚至可能因為極值所在的時間點被偏移，反而稍微干擾了 Attention 機制的對齊 (Alignment)。

4. **過擬合假象與物理限制 (2D 座標限制)**
   - 我們使用的是 2D 座標，在沒有 3D 深度資訊的情況下，扭曲時間會產生許多不符合人體力學的加速度與速度向量。PatchTST 依賴時間序列的 Patch 捕捉局部變化，如果一個 Patch 裡面的座標變化違反了重力或肌肉發力的物理原則，模型就會學到錯誤的捷徑 (Shortcut)。

### 💡 結論與未來方向
在深蹲這種**極度依賴特定關節在同一時間點的相對關係 (Synchronization)** 以及**固定發力節奏**的運動中，暴力的時間軸扭曲 (DTWWarp) 會破壞這份和諧。

針對 Squat 任務，未來建議改用以下幾種資料增強方式：
- **Jittering (加入座標雜訊)**：針對 2D 座標的抖動，幫助模型抵抗骨架偵測的不穩定性。
- **Scaling (空間縮放)**：模擬不同身高與身型的人。
- **Rotation (微小旋轉)**：模擬攝影機不同的架設角度。
- **Spawner / Sub-sampling**：均勻的降採樣或插值，保留全域的時間相對關係，而不是像 DTW 那樣局部扭曲。
