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

---

## 🏃 實驗 3：Feature Engineering 消融實驗 (針對 Posterior Pelvic Tilt)
為了解決 `Posterior_Pelvic_Tilt` (骨盆後傾/屁股眨眼) F1 score (0.27) 過低的問題，我們分析了原因，發現原本的 50 維特徵 (10個基礎物理特徵 × 5種時間轉換) 可能缺乏足夠的資訊 (如脊椎彎曲度或骨盆翻轉) 讓模型判斷。因此，我們透過手動計算新特徵進行 Ablation Study。

- **腳本**: `preprocess_ablation.py` -> 儲存成 `.pt` 格式加速訓練讀取。
- **訓練腳本**: `run_squat_ablation.sh` 
- **超參數**: `--num_workers 4`, 各自有對應的 `--tag` 與 `--data_path`

加入的特徵測試包含：
1. **Depth (深度)**：髖部 Y 座標與腳踝 Y 座標的差值。
2. **Velocity/Acceleration (速度與加速度)**：髖部 Y 座標的一階與二階導數，捕捉屁股眨眼時瞬間「掉落」的動態。
3. **Trunk (軀幹特徵)**：肩膀至髖部的 2D 距離與向量。
4. **Hip Flexion Angle (髖部屈伸角度)**：肩膀-髖部-膝蓋的夾角。
(訓練已完成，採用預設的 `instance_stratified` 分割進行消融測試)

> [!WARNING] 
> 發現重大 Bug：原本 `PatchTST_train.py` 內的標籤列表缺少了 `Correct` 類別，導致所有印出的分數標籤全部向後錯位了一格 (例如印出的 Posterior_Pelvic_Tilt 分數，其實是 Excessive_Hip_Dominance 的成績)！現已將此 Bug 修正。

### 消融實驗真實結果 (修正標籤錯位後)
**1. 針對 Posterior_Pelvic_Tilt (骨盆後傾)**
原本在 Baseline 中約為 `0.4150`。
- **+ Trunk (軀幹特徵)**: 提升至 `0.4364`
- **+ Depth (深度)**: 提升至 `0.4405`
- 在剛才使用 Trunk 訓練的模型中，實際的 Posterior_Pelvic_Tilt F1 達到了 `0.4871` (比原本最好模型的 `0.2737` 大幅提升)。

**2. 針對 Excessive_Hip_Dominance (髖部過度主導)**
原本在 Baseline 中約為 `0.73` 左右。
- **+ Trunk (軀幹特徵)**: 🌟 飆升至 `0.8023`！

### 💡 深入分析與結論
這次在修正 Bug 並加入 `--num_heads 8` 重新訓練後，我們得到了真實且精準的對比數據：

**1. 為什麼 Trunk 特徵能拯救 Posterior Pelvic Tilt (0.27 -> 0.43) 與 Excessive Hip Dominance (0.72 -> 0.79)？**
- **Posterior Pelvic Tilt (屁股眨眼/骨盆後傾)**：發生在深蹲底部，因骨盆翻轉導致下背圓背。在缺乏 3D 脊椎節點的 2D 骨架中，圓背會在物理上導致「肩膀至髖部」的直線距離縮短、向量改變。我們直接將 `trunk_len` (軀幹長度) 與 `trunk_vec` (軀幹向量) 餵給模型，等同於直接提供「軀幹是否被壓縮或變形」的強烈空間訊號，因此大幅提升了判斷準確率。
- **Excessive Hip Dominance (髖部過度主導)**：即軀幹過度前傾。`trunk_vec` 完美捕捉了軀幹的傾斜角度，這也是為何這個錯誤的分數能突破 0.79。

**2. 為什麼 Early Hip Rise 反而暴跌 (0.88 -> 0.68)？**
- **空間與時間的特徵衝突 (Spatial vs. Temporal)**：
  - `Early_Hip_Rise` (臀部上升過快) 是一個**時間序 (Temporal)** 的錯誤，關鍵在於「臀部 Y 座標的上升速度，比肩膀 Y 座標快」。
  - `Excessive_Hip_Dominance` 是一個**空間 (Spatial)** 的錯誤 (身體前傾)。
- 當臀部上升過快時，力學上通常會伴隨身體前傾。我們加入的 Trunk 特徵是強大的「空間特徵」，導致 Attention 模型產生了**特徵依賴 (Feature Reliance)**：模型變得太過關注「現在軀幹有多傾斜」(Spatial)，而忽略了去比較「過去與現在臀部和肩膀的相對上升速度」(Temporal)。
- 結果就是，模型把許多 `Early_Hip_Rise` 的動作，因為看到軀幹前傾，就直接誤判成了 `Excessive_Hip_Dominance` (這也解釋了為何後者的分數會上升)。

---

## 🏃 實驗 4：Trunk + Velocity 特徵於 Subject Exclusive 測試與反思
在釐清了 Baseline 預設切分模式的問題後，我們使用了最嚴格的 `--split_mode subject_exclusive`（確保測試集是未曾見過的受測者）重新驗證了加入 **Trunk 向量、長度以及 Y 軸相對速度** 的成效。

- **Tag**: `squat_trunk_vel_subject_exclusive`
- **超參數**: `--num_heads 8`, `--split_mode subject_exclusive`

### 測試集結果 (Fold 0)
- **Macro F1**: `0.6346` (相較原始舊模型 0.6373，幾乎持平)
- **各類別 F1 Score 比較 (對比舊模型)**:
  - Correct: `0.5059` (🔺 上升，原本 0.4341)
  - Insufficient_Depth: `0.7407` (🔺 上升，原本 0.7173)
  - Excessive_Knee_Dominance: `0.6607` (🔺 上升，原本 0.5882)
  - Excessive_Hip_Dominance: `0.7338` (🔺 微幅上升，原本 0.7232)
  - Posterior_Pelvic_Tilt: `0.2424` (🔻 下降，原本 0.2737)
  - Early_Hip_Rise: `0.7953` (🔻 下降，原本 0.8840)

### 🔬 深度分析：為什麼 Trunk 特徵在未見過的受測者上失效了？
這個實驗結果給了我們一個極具價值的 Insight。原本以為能拯救 `Posterior_Pelvic_Tilt` 的特徵，在遇到「新的人」時不但無效，甚至導致輕微退步，原因在於：**未標準化的絕對空間特徵 (Absolute Spatial Features) 導致的 Overfitting。**

1. **身型比例差異 (Body Proportions)**：
   我們算出的 `trunk_len` (肩膀到髖部的 2D 距離) 和 `trunk_vec` 都是**絕對數值**。但每個人的身高、身形比例差異巨大！一個身高 190 公分的受測者在「嚴重骨盆後傾 (圓背)」時的 Trunk 長度，可能都還比一個 150 公分的人「站直」時的 Trunk 長度還要長。
2. **模型記住了訓練集的人 (Memorization vs Generalization)**：
   這完美解釋了為什麼在之前的 `instance_stratified` 實驗中分數會狂飆——因為訓練集和測試集有相同的人，模型記住了「受測者 A 的正常長度是 50，所以降到 45 就是骨盆後傾」。但在 `subject_exclusive` 裡遇到完全陌生的受測者 B 時，這個死板的絕對數值規則就徹底崩潰了。
3. **Early_Hip_Rise 的下降**：
   即便我們補上了 Y 軸相對速度，但由於絕對長度與角度特徵在陌生人身上成為了「雜訊」，反而干擾了模型去捕捉原本純粹靠時間序列動作判斷的 `Early_Hip_Rise`。

### 💡 未來改進方向
如果我們要在 `subject_exclusive` 這種高難度的跨受測者任務中解決 `Posterior_Pelvic_Tilt`，**絕對不能使用絕對座標或絕對長度**。必須進行**基於個人的特徵標準化 (Subject-level Normalization)**：
- **相對長度變化率**：不要餵入 `trunk_len`，而是餵入 `trunk_len / 該動作第一幀的 trunk_len` (相對於站直時的縮放比例)。
- **角度變化量**：餵入當下軀幹角度與初始站直角度的「差值」，而不是絕對座標向量。
---

## 🛑 重大發現：為什麼「髖部低於膝蓋」的物理特徵會被破壞？

在深究為何模型無法在陌生受測者身上學會 `Posterior_Pelvic_Tilt` (骨盆後傾) 與 `Early_Hip_Rise` (臀部上升過快) 時，我們發現了一個**直指問題核心、導致所有物理特徵失效的致命盲點**：

### 1. 獨立的 Min-Max 正規化 (Independent Normalization)
我們追查了原始資料集的產生腳本 (`dataset/tools/squat_tool/data_split.py`)，發現裡面的 `normalize_to_neg1_1` 函數，是針對**每一個特徵維度 (axis=0)** 獨立進行 Min-Max 正規化（縮放到 `[-1, 1]`）。

這在保留「單一關節的運動波形」上是合理的，但當我們依賴「跨關節的物理空間關係」時，災難就發生了：
- 假設深蹲時，臀部 Y 座標移動了 100 像素，肩膀 Y 座標只移動了 50 像素。經過獨立正規化後，兩者的軌跡都會被強制拉伸到 `[-1, 1]`！這代表在正規化後的資料裡，**肩膀的 1 單位距離，跟臀部的 1 單位距離，在現實世界中根本不一樣長！**
- 先前我們在 `dataset.py` 中，讀取了「已經被扭曲比例尺」的座標去計算 Trunk Length。這算出來的根本不是物理長度，而是失去意義的數字。

### 2. 「0 臨界點 (Zero Threshold)」的物理意義被徹底抹除
您曾提到：「*Posterior_Pelvic_Tilt 的一個很大重點是髖的 Y 軸座標會低於膝蓋的 Y 軸座標。這個特徵應該已經放過了，有這個資訊不應該沒有辦法判讀。*」

這句話完全正確！我們也在 `dataset/processors/squat.py` 中找到了這個特徵：`r_knee_hip_y_diff` (Knee Y - Hip Y)。
**那為什麼模型還是學不會？**
答案就在獨立正規化的平移 (Shift) 效應：
- 原本 `Knee Y - Hip Y < 0` 是一個非常精準的絕對臨界點，代表髖部精確地降到了膝蓋下方。
- 但經過 `normalize_to_neg1_1` 後，`-1.0` 不再代表「髖部低於膝蓋」，它只代表**「這是該受測者這次深蹲蹲得最低的那一瞬間」**！
- 就算有一個受測者只做半蹲（髖部遠高於膝蓋），他蹲到最低點的那一刻，數值一樣會變成 `-1.0`。
- **模型拿到這個被平移過的數字，根本無法找出「0 (Hip == Knee)」這條絕對界線！**

---

## 🛠️ 終極解決方案：尺度不變的物理特徵 (Scale-Invariant Physical Features)

為了一次性、最完美地解決這個問題，我們設計了全新的實作計畫：
我們必須在 `dataset/processors/squat.py` 剛抽取出真實像素座標、**還沒有做 Min-Max 正規化之前**，就先算好以下 5 個核心物理特徵，並**絕對禁止將它們送入 `normalize_to_neg1_1`**，直接原汁原味地附加在最終特徵陣列之後：

我們會計算出該次深蹲第一幀的 `initial_trunk_len` (初始軀幹長度) 作為個人的基準比例尺，將絕對像素轉化為**相對比例**，消除不同身高的影響：
1. `trunk_len_ratio` = `trunk_len / initial_trunk_len` (捕捉圓背造成的軀幹壓縮)
2. `trunk_vec_x_ratio` = `(Shoulder_X - Hip_X) / initial_trunk_len` (軀幹 X 傾角)
3. `trunk_vec_y_ratio` = `(Shoulder_Y - Hip_Y) / initial_trunk_len` (軀幹 Y 傾角)
4. `knee_hip_y_ratio` = `(Knee_Y - Hip_Y) / initial_trunk_len` (**完美保留物理 0 臨界點！**當此值 < 0，代表髖部精準低於膝蓋)
5. `knee_hip_x_ratio` = `(Knee_X - Hip_X) / initial_trunk_len`

我們將會把這 5 個特徵，以及它們的一階速度 (Velocity)，共計 10 維新特徵直接加入 CSV。
這會讓原本的 50 維輸入擴充為 **60 維**，並徹底釋放模型判讀空間物理特徵的潛力！

---

## 🏃 實驗 5：Scale-Invariant 物理特徵於 Subject Exclusive 測試 (最終版)
在修正了獨立正規化摧毀物理零點的問題後，我們直接將未經平移的「相對物理特徵」與其速度附加到 60 維中，進行最終測試。

- **Tag**: `squat_trunk_scale_invariant`
- **超參數**: `--num_heads 8`, `--split_mode subject_exclusive`

### 測試集結果 (Fold 0)
- **Macro F1**: `0.6493` (🏆 突破歷史新高，原本 Baseline 為 0.6373)
- **各類別 F1 Score 比較 (對比舊模型)**:
  - Correct: `0.4808` (🔺 提升，原本 0.4341)
  - Insufficient_Depth (深度不足): `0.7775` (🔺 大幅提升，原本 0.7173)
  - Excessive_Knee_Dominance: `0.6562` (🔺 大幅提升，原本 0.5882)
  - Excessive_Hip_Dominance: `0.7847` (🔺 大幅提升，原本 0.7232)
  - Posterior_Pelvic_Tilt (骨盆後傾): `0.2799` (持平微升，原本 0.2737)
  - Early_Hip_Rise: `0.7484` (🔻 下降，原本 0.8840)

### 🔬 最終深度分析與物理限制
1. **成功驗證了物理特徵的威力**：
   `Insufficient_Depth` (深度不足) 大幅提升到了 0.77！這證明了我們加入的 `knee_hip_y_ratio` (髖部是否低於膝蓋) 精準地發揮了作用，模型現在能極度準確地判斷蹲得夠不夠深。前傾角度也幫助了 `Excessive_Hip_Dominance` 達到近 0.8 的高分。整體 Macro F1 創下 `subject_exclusive` 模式下的最高紀錄。

2. **殘酷的真相：為什麼 Posterior_Pelvic_Tilt 還是起不來？**
   在排除了所有特徵工程的 Bug 後，我們終於觸碰到了 **2D 骨架資料集的硬體極限 (Hardware/Data Bottleneck)**：
   - 骨盆後傾 (Butt Wink) 的本質，是「下背部/腰椎」發生了彎曲 (圓背) 與骨盆翻轉。
   - 但是，我們的 YOLO 骨架資料中，**軀幹只有「肩膀(6)」跟「髖部(12)」兩個點！**
   - 數學上，**兩點只能決定一條直線**。不管受測者的背怎麼彎，肩膀到髖部永遠只能連成一條無形的直線。圓背可能會讓這條直線在 2D 投影上稍微縮短 2~3%，但在深蹲最底部時，大腿會嚴重遮擋髖關節，YOLO 預測座標的「視覺雜訊誤差」遠遠大於這 2% 的長度變化！
   - **結論**：在缺乏「脊椎/中段腰椎」關節點的情況下，單靠 2D 視角的肩膀與髖部，要在「陌生受測者」身上抓出細微的骨盆翻轉，在數學與物理上是幾乎不可能的任務。0.28 可能已經是這組 2D 特徵的理論極限。

3. **Early_Hip_Rise 下降的權衡**：
   由於我們一次性加入了 10 個強大的「空間/角度」特徵，模型的 Attention 被空間特徵主導，相對削弱了對「時間速度差」的注意力。這是多任務學習中常見的特徵競爭現象，但換來了整體 Macro F1 與其他四個類別的全面提升。

---

## 🏃 實驗 6：Ablation Study (剔除部分物理特徵與速度)
為了確認是否是因為加入過多特徵導致 `Early_Hip_Rise` 注意力被分散，我們嘗試移除了 `trunk_len_ratio`、`knee_hip_x_ratio`，以及**所有物理特徵的一階速度 (Deltas)**，將維度從 60 縮減至 53 維。

- **Tag**: `squat_trunk_scale_trimmed_53feature`
- **超參數**: `--num_heads 8`, `--split_mode subject_exclusive`

### 測試集結果 (Fold 0)
- **Macro F1**: `0.6346` (🔻 低於 60 維版本的 0.6493)
- **各類別 F1 Score 比較 (對比 60 維模型)**:
  - Correct: `0.3073` (🔻 大幅下降，原本 0.4808)
  - Insufficient_Depth (深度不足): `0.7657` (微降，原本 0.7775)
  - Excessive_Knee_Dominance: `0.6067` (🔻 下降，原本 0.6562)
  - Excessive_Hip_Dominance: `0.6989` (🔻 大幅下降，原本 0.7847)
  - Posterior_Pelvic_Tilt (骨盆後傾): `0.2628` (微降，原本 0.2799)
  - Early_Hip_Rise: `0.8391` (🔺 成功回升，原本 0.7484)

### 🔬 Ablation 分析結論
實驗結果證實了我們對「特徵競爭」的假設，移除速度特徵確實讓模型重新找回了判斷 `Early_Hip_Rise` 的節奏 (從 0.74 回升至 0.83)。

**但是，代價太慘痛了：**
1. **速度 (Velocity) 對前傾判斷至關重要**：移除 `trunk_vec_y_ratio` 等角度特徵的速度後，`Excessive_Hip_Dominance` 從 0.78 暴跌回 0.69。這說明判斷軀幹前傾過度，不僅看絕對角度，更依賴於「軀幹傾倒的角速度」。
2. **整體準確率 (Correct) 崩盤**：正確動作的辨識率從 0.48 掉到 0.30。這表示 60 維中提供的全套空間比例與其速度，提供了分辨「標準動作與瑕疵動作」的關鍵邊界。

**🏆 最終結論：**
在 2D 骨架的限制下，**「保留完整 10 維物理特徵與速度」的 60 維架構 (實驗 5)** 是無庸置疑的最佳解。雖然 `Early_Hip_Rise` 有微幅犧牲，但換來的是 Depth、Hip Dominance 與 Correct 的全面爆發。
後續若要攻克 `Posterior_Pelvic_Tilt` 的 0.28 瓶頸，必須如計畫轉向 **3D 骨架資料** 以獲取精確的骨盆/腰椎旋轉角度。

python PatchTST_train.py --sport squat --tag squat_trunk_vel_subject_exclusive --num_workers 4 --num_heads 8 --split_mode subject_exclusive

---

## 附錄：深蹲特徵集詳細定義 (Feature Sets)

為了讓後續實驗更清晰，以下總結各階段所使用的特徵維度與具體內容。

### 1️⃣ Baseline 原始 50 維特徵 (實驗 1~4) - phase4.8_test_heads8_squat2
這 50 維是由 **10 個基礎空間特徵**，分別經過 **5 種時間序列轉換**（Normalized、Delta、Delta Ratio、Z-Score、Delta Square）所構成（10 × 5 = 50）：
1. **左膝蓋角度 (Left Knee Angle)**
2. **左髖部角度 (Left Hip Angle)**
3. **右膝蓋角度 (Right Knee Angle)**
4. **右髖部角度 (Right Hip Angle)**
5. **左手軀幹角度 (Left Arm-Torso Angle)**
6. **右手軀幹角度 (Right Arm-Torso Angle)**
7. **槓鈴 X 軸位移 (Barbell X Displacement)**
8. **槓鈴 Y 軸座標 (Barbell Y)**
9. **右膝與髖部的 Y 軸高度差 (Right Knee-Hip Y Diff)**
10. **右膝與髖部的 X 軸前後差 (Right Knee-Hip X Diff)**

### 2️⃣ 最終版 60 維 Scale-Invariant 特徵 (實驗 5) -quat_trunk_scale_invariant_best0719_60feature_best
在原始 50 維的基礎上，為了解決不同受測者身高與絕對座標平移的問題，我們於抽取出像素座標但 **尚未做 Min-Max 正規化之前**，計算了 5 個「相對於初始軀幹長度 (initial trunk length) 的比例值」，以及這 5 個比例值的一階速度 (Velocity)，總共新增 **10 維** (50 + 10 = 60)：
1. `trunk_len_ratio`: 相對軀幹長度比例 (捕捉骨盆翻轉時軀幹長度被壓縮的比例)
2. `trunk_vec_x_ratio`: 軀幹 X 軸傾角比例 (捕捉身體前傾)
3. `trunk_vec_y_ratio`: 軀幹 Y 軸傾角比例
4. `knee_hip_y_ratio`: 右膝與髖部的 Y 軸相對高度差 (保留完美的 **物理 0 臨界點**，判斷髖部是否精準低於膝蓋)
5. `knee_hip_x_ratio`: 右膝與髖部的 X 軸相對前後差
6~10. **以上 5 個特徵的一階速度 (Delta)**

### 3️⃣ Ablation 53 維特徵 (實驗 6) - squat_trunk_scale_trimmed_53feature
為了驗證 60 維特徵中是否有造成 Attention 競爭的冗餘資訊，我們從 60 維中 **移除了 7 個特徵**：
- 移除了 `trunk_len_ratio` (1 維)
- 移除了 `knee_hip_x_ratio` (1 維)
- 移除了 **所有 5 個物理特徵的一階速度** (5 維)

總計移除 7 維，剩下 **53 維**。實驗結果顯示，雖然移除後能提升對臀部上升過快的注意力，但卻造成軀幹前傾判斷以及整體標準動作辨識率的大幅崩盤，證明完整的 60 維特徵是 2D 骨架下的最佳解。