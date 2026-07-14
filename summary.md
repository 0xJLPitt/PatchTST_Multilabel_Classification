# 🏋️ 3D 硬舉動作辨識 (Deadlift Action Recognition) 
## PatchTST 模型極限優化完整報告 (Fine-Tuning Journey)

本報告記錄了基於 PatchTST 架構進行 3D 硬舉動作辨識的進階調優歷程。任務目標為多標籤分類 (Multi-label classification)，涵蓋五個類別：`Correct` (正確), `Far from the shins` (離腿太遠), `Hips rise first` (先抬臀), `Collide with the knees` (撞擊膝蓋), `Lower back rounding` (圓背)。

**優化基準點 (Commit `3ca80a378ee18692695c98d7fffb262985de6a06`)**：
本報告的內容整理從此 commit 之後開始。在此基準點上，模型的表現為 Macro F1 `0.646` 與嚴格準確率 (Accuracy) `0.288`。

### 🧬 特徵演進 (Feature Evolution)
為了讓模型能掌握空間物理資訊，我們在特徵工程上進行了大幅度的擴充：

**【基準點原始特徵 (8 個基礎幾何特徵)】**
在基準點時，模型已經預先加入了 8 個特徵，包含 6 個關節夾角與 2 個槓鈴絕對座標：
1. `Left knee angle` (左膝夾角)
2. `Left hip angle` (左髖夾角)
3. `Right knee angle` (右膝夾角)
4. `Right hip angle` (右髖夾角)
5. `Left arm-torso angle` (左臂與軀幹夾角)
6. `Right arm-torso angle` (右臂與軀幹夾角)
7. `Barbell X` (槓鈴水平座標)
8. `Barbell Y` (槓鈴垂直座標)

**【後續新增特徵 (5 個進階物理特徵)】**
在本次優化歷程 (Phase 4) 中，為了進一步解決特定類別的誤判，我們新增了 5 個特徵，使基礎特徵達到 **13 個** (這 13 個特徵再經過 5 種擴增轉換，形成最終的 65 維輸入)：
9. `Body length` (身體直線長度)：從原始 3D 數據中重新納入。
10. `Bar-Knee X Displacement` (槓鈴與膝蓋水平位移差)
11. `Bar-Knee Y Displacement` (槓鈴與膝蓋垂直位移差)
12. `Shoulder-Hip X Displacement` (肩髖水平位移差)
13. `Torso Angle to Ground` (軀幹絕對傾角)

接下來的歷程記錄了從基準點出發，經歷進階的架構重構、物理特徵工程、時間解析度調整以及嚴謹的消融實驗，最終成功將模型推向 **Macro F1 `0.687` 與 Accuracy `0.399`** 歷史巔峰的完整過程。

---

## 🚀 第一部：優化歷程 (Optimization Phases)

### Phase 1: 基礎抗噪與類別平衡 (Focal Loss & AdamW)
- **問題**：模型對 `Correct` 的判斷率極低，且在 `Far from the shins` 與 `Lower back rounding` 之間存在嚴重的特徵混淆。Validation F1 震盪劇烈。
- **改動**：
  - 引入 **Focal Loss** 取代 BCE，強制模型專注於難以分辨的樣本。
  - 優化器改為 **AdamW** (加入 Weight Decay 防過擬合)，Batch Size 擴大至 32。
  - 加入 3D 空間微小高斯雜訊 (Jittering=0.02) 增強泛化能力。
- **結果**：震盪趨緩，但 `Correct` 分數進一步下跌。這表明原版 PatchTST 的 Channel Independence (通道獨立) 機制破壞了關節間的空間幾何關係，導致模型無法理解「正確姿勢」。

### Phase 2 ~ 3: 破除通道獨立與早期融合 (Early Fusion Breakthrough)
- **問題**：原版 PatchTST 將 40 個特徵通道完全獨立，抹殺了人體骨架在同一幀內的空間關聯性。
- **改動 (早期融合 Early Fusion)**：
  - 打破 Channel Independence，讓每一塊 Patch 同時包含全身 40 個空間特徵。
  - 將分類器從龐大的 10240 維 MLP 大幅瘦身回 256 維 (參數量從千萬級降至 3 萬級)，根治過度擬合。
- **結果**：**Macro F1 提升至 0.663，Accuracy 突破 34%**。模型終於具備了空間幾何直覺。

### Phase 4.0 ~ 4.6: 物理特徵工程 (Feature Engineering)
- **問題**：模型難以自行從絕對座標推算出身體部位間的相對關係。
- **改動**：
  - 加入 **槓鈴與膝蓋的水平/垂直位移差 (Bar-Knee Displacement)**，精準打擊 `Collide with the knees` 誤判。
  - 加入 **圓背專屬特徵 (Anti-Rounding)**：包含身體直線長度 (Body Length) 與肩髖水平位移差。
  - 動態調整雜訊遮蔽 (Time/Channel/Point Masking) 逼迫模型學習泛用規律。
- **結果**：`Lower back rounding` (圓背) 的 F1 分數從谷底反彈，創下 `0.632` 的新高。

### Phase 4.7 ~ 4.8: 微觀時間解析度 (Micro-Temporal Resolution)
- **問題**：動作失誤 (如撞擊膝蓋或起槓瞬間圓背) 往往發生在零點幾秒內，預設的 Patch Size (16) 視野過大，容易錯失細節。
- **改動**：
  - 將 `patch_len` 縮小為 **8** (約 0.25 秒)，`stride` 縮小為 **4**。
  - 新增 **軀幹絕對傾角 (Torso Angle to Ground)**，完美捕捉上胸塌陷。
- **結果**：**Macro F1 來到 0.675，Accuracy 突破 38%**。模型對 `Correct` 的判斷力創下 0.537 的巔峰。

---

## 🔬 第二部：控制變因消融實驗 (Ablation Studies)

在達到 Phase 4.8 的瓶頸後，我們針對「注意力頭數」、「類別權重」、「訓練時間」與「動態角速度」進行了最後的交叉驗證：

| 測試項目 | 變更設定 | Macro F1 | Accuracy | 關鍵觀察 |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (4.8)** | 預設 (Input Dim 65, Heads=4) | 0.6754 | 0.3816 | 表現優異，但 `Hips rise first` 略顯疲軟。 |
| **Test 1 (👑 贏家)** | **`num_heads`=8** | **0.6875** | **0.3994** | 腦容量擴增，完美處理 65 維特徵，`Hips rise first` 飆升至 0.7279。 |
| **Test 2** | `focus_hips_rise`=1.2 | 0.6773 | 0.3497 | 手動干涉單一權重導致顧此失彼，Accuracy 崩盤。 |
| **Test 3** | `max_epochs`=200 | 0.6865 | 0.3801 | 延長訓練時間確實有助於複雜特徵的收斂。 |
| **Test 4** | 新增 **角速度特徵** (Dim 70) | 0.6583 | 0.3534 | 角速度確實讓 `Hips rise first` 提升，但資訊過載導致其他類別下跌。 |
| **Test 5 (終極版)** | Heads=8 + Epochs=200 + 角速度 | 0.6775 | 0.3690 | 結合所有優勢，但**依然無法擊敗單純只改 Heads 的 Test 1**。 |

### 🧠 深刻的 AI 物理學洞察
終極版 (Test 5) 雖然強大，但依然敗給了只有 65 維特徵的 Test 1。這個結果揭示了一個震撼的事實：
1. **Transformer 的時間追蹤能力**：當我們給予 8 個 Attention Heads 後，模型**自己就能在不同的時間幀之間，隱式地學到關節角速度 (Velocity) 的動態變化**。
2. **手動特徵變雜訊**：當我們「人為」把角速度差再算一次並塞進特徵（讓維度膨脹到 70 維）時，對強大的 8-Head Attention 來說反而是冗餘的雜訊（Overfitting），導致它在判斷 `Correct` 與 `Collide with the knees` 時分心。

---

## 🏆 結論：硬舉 3D AI 的地表最強形態

經過完整的榨汁與優化，我們正式確立了 PatchTST 模型在硬舉任務上的**黃金比例配置**：

1. **特徵維度**：**65 維 (13 個基礎幾何特徵 × 5 種變換)**。必須包含槓鈴位移、軀幹傾角、肩髖位移等空間物理特徵，但**不需要手動給予角速度**。
2. **時間解析度**：`patch_len=8`, `stride=4` (高頻微觀切片)。
3. **模型容量**：`embed_dim=256`, **`num_heads=8`** (極其關鍵), `num_layers=2`。
4. **架構核心**：必須採用 **Early Fusion (早期融合)**，並搭配極致輕量化的線性分類器 (256維)。
5. **資料擴增**：溫和的高斯雜訊 (0.02) 加上短暫的 Time Masking (1~3 幀)。

這套保存在 `models/deadlift/TST_Deadlift_3D/phase4.8_test_heads8` 裡面的模型，交出了 **Macro F1 0.6875** 與 **Accuracy 近 40%** 的傲人成績，正式宣告硬舉辨識任務的階段性破關！



目前3D deadlift資料集動作分布
總資料筆數：6,856 筆
完全正確的片段 (Correct)：1,316 筆
包含錯誤的片段 (包含至少一種錯誤)：5,540 筆
---------------------
正確 (Correct): 1,316 次
錯誤 1 (槓鈴遠離小腿 - Barbell moving away from the shins): 2,889 次
錯誤 2 (槓鈴離地前臀部先抬起 - Hips rising before the barbell leaves the ground): 1,656 次
錯誤 3 (槓鈴撞擊膝蓋 - Barbell colliding with the knees): 1,605 次
錯誤 4 (下背部圓起/龜背 - Lower back rounding): 3,021 次

四種資料處理方式:
一階差分 (Delta Feature)

實作方式：process_delta(filtered_interpolated)
意義：計算相鄰影格之間特徵的變化量。在物理意義上代表特徵變化的「速度」，有助於模型捕捉動作的瞬間變化趨勢。
二階差分 (Delta Square Feature)

實作方式：process_delta(delta_feature)
意義：對一階差分特徵再做一次差分。在物理意義上代表特徵變化的「加速度」，用來描述動作變化速度的急緩。
Z-Score 標準化 (Z-Score Feature)

實作方式：process_zscore(filtered_interpolated)
意義：對特徵進行 Z-score 標準化處理（減去平均值並除以標準差）。這可以消除不同特徵間尺度（Scale）的差異，並突顯特徵值相對於整體分佈的離群程度。
差分比例 / 變化率 (Delta Ratio Feature)

實作方式：process_delta_ratio(filtered_interpolated)
意義：計算特徵變化的比例或相對變化率。比起單純的差分數值，它更能反映特徵相對自身大小的變動幅度。



***
python PatchTST_train.py --sport deadlift --split_mode clip_random --tag 4.8with_clip_random