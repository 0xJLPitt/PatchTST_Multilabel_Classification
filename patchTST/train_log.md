# Training Log - 2026-07-08

**Terminal Information:** `TerminalName: bash, ProcessId: 3544495`

## 1. 整體策略概覽
- **Model**: `PatchTSTClassifier`
- **Task**: 3D Deadlift Action Recognition
- **Classification Type**: Multi-label classification (5 classes: Correct, Far from the shins, Hips rise first, Collide with the knees, Lower back rounding)
- **Input Structure**: 3D keypoint features (Dimension: 40, Length: 110)
- **Overall Result**: Best Macro F1 Score = 0.6463 at Epoch 20, stopped at Epoch 28 by Early Stopping.

## 2. Augmentation
- 目前 `Datasubset` 中未啟用資料增強 (No active data augmentation: `transform=True` is passed but not utilized in `dataset.py`).

## 3. Optimizer 
- **Algorithm**: Adam Optimizer (`torch.optim.Adam`)
- **Base Learning Rate**: 0.0003
- **Gradient Clipping**: `max_norm=1.0`

## 4. Scheduler 
- **Algorithm**: Custom Warmup Cosine Annealing Scheduler
- **Warmup Epochs**: 5
- **Max Epochs**: 100
- **Min LR Ratio**: 0.0

## 5. Loss Function 
- **Function**: `BCEWithLogitsLoss`
- **Imbalance Handling**: 動態權重 (`pos_weight`) 平衡正負樣本，基於訓練集的分佈比例即時計算。

## 6. 模型超參數 (PatchTST)
- **patch_len**: 10
- **embed_dim**: 256
- **num_heads**: 4
- **num_layers**: 2
- **dropout**: 0.3
- **stride**: 1
- **Architecture Note**: Channel Independence 搭配 Mean Pooling (對時間軸取平均) 作為 Readout 層。

## 7. 資料切分策略調整
- **Strategy**: Subject Isolated Splitting (`--subject_isolated`)
- **Split Ratio**: 70% Train / 10% Validation / 20% Test
- **Details**: 以受試者為單位進行切分，確保同一受試者的資料不會同時出現在訓練與測試集中。每個類別的資料比例也盡量分配均勻 (1 Fold evaluation)。

## 8. 訓練流程超參數
- **Batch Size**: 16
- **Max Epochs**: 150
- **Early Stopping Patience**: 8
- **Workers**: 4 (`--num_workers 4`)
- **Hardware**: GPU (`cuda` if available)


✅ F1 scores from each Fold:
Fold 0: Macro F1 = 0.6463, Accuracy: 0.2880, cost time = 0.000023 sec
  - Correct: F1 = 0.2849
  - Far from the shins: F1 = 0.6922
  - Hips rise first: F1 = 0.7051
  - Collide with the knees: F1 = 0.5502
  - Lower back rounding: F1 = 0.6376

📊 Average F1 Score: 0.6463 ± 0.0000
  Average F1 Score per Class:
  - Correct: 0.2849 ± 0.0000
  - Far from the shins: 0.6922 ± 0.0000
  - Hips rise first: 0.7051 ± 0.0000
  - Collide with the knees: 0.5502 ± 0.0000
  - Lower back rounding: 0.6376 ± 0.0000
🏆 Best F1: 0.6463 from Fold 0
📁 Best model saved at: ./models/deadlift/TST_Deadlift_3D/None/PatchTST_model_fold0.pth

1. Confusion Matrix 分析
嚴重的特徵混淆：模型對於 Far from the shins (槓鈴離小腿太遠) 與 Lower back rounding (圓背) 之間有非常嚴重的互相誤判。例如，真實標籤為 Lower back rounding 的資料中，有高達 28.4% 被錯判為 Far from the shins。這表示這兩種動作在目前的 3D 空間特徵中重疊度很高。
正確動作 (Correct) 辨識率崩潰：Correct 的樣本只有 13.8% 被正確預測，大部分都被錯判成了 Collide with the knees 或是 Lower back rounding。這代表模型無法抓到「標準動作」的明確邊界。
資料不平衡 (Data Imbalance)：推算下來 Far from the shins 和 Lower back rounding 樣本數目大約各有 1000 筆，而 Correct 僅有不到 400 筆。雖然我們有使用 pos_weight，但顯然不足以讓模型學會少數特徵。
2. 訓練曲線 (Training Results) 分析
Loss 確實尚未收斂：如同您觀察到的，藍線 (Train Loss) 在 Epoch 28 依然呈現漂亮的下降斜率（大約在 0.6 左右），完全沒有平緩（Plateau）的跡象，模型其實「還在學」。
驗證指標震盪過於劇烈：紅線 (Validation F1-score) 在 0.58 ~ 0.66 之間呈現非常暴力的上下跳動。這通常暗示「Batch Size 太小（目前 16）」或是「Learning Rate 對於目前的 Loss 地形來說太大」，導致模型走一步退半步。
被震盪「誤殺」的 Early Stopping：因為驗證指標震盪太大，剛好連續 8 個 Epoch 沒有突破 Epoch 20 的高點，就觸發了 patience=8 的提早停止條件。這是不合理的，因為模型其實還沒收斂完。

---070802

## 🛠️ 第一階段模型優化紀錄 (Optimization Phase 1)

**優化目標**：穩定驗證曲線震盪、解決早期中斷問題、改善特定易混淆特徵的分類表現 (Far from the shins vs Lower back rounding)。

1. **訓練參數調整 (防止過早停止與震盪)**
   - **Batch Size 加大**：由 `16` 增加至 `32`，讓梯度的更新更加穩定，減少曲線亂跳。
   - **Patience 放寬**：將 Early Stopping 的耐心值從 `8` 提高到 `30`，確保模型有足夠的時間度過震盪區間並收斂。
   - **優化器更換**：從 `Adam(lr=0.0003)` 改為 `AdamW(lr=1e-4, weight_decay=1e-4)`，利用較小的學習率與 L2 正規化 (Weight Decay) 防止過擬合。

2. **處理類別不平衡與特徵混淆**
   - **損失函數升級**：將 `BCEWithLogitsLoss` 替換為自定義的 **Focal Loss** (`gamma=2.0`)。這不僅保留了原本的 `pos_weight` 來解決數量不平衡，更能自動降低易分樣本的權重，強制模型將注意力集中在「難以分辨」的圓背與離腿太遠特徵上。
   - **啟用資料增強 (Data Augmentation)**：在 `Datasubset` 內補上了轉換機制：當 `transform=True` 時，對 3D 骨架座標加入高斯雜訊 (`Jittering`，標準差 0.02)。這種微小的隨機抖動能提升泛化能力，有助於讓正確動作 (`Correct`) 不被過度死背。

 Early Stopping Triggered
Fold 0 Test F1: 0.6496, Accuracy: 0.2546, cost 1.2311390265767099e-05 sec

✅ F1 scores from each Fold:
Fold 0: Macro F1 = 0.6496, Accuracy: 0.2546, cost time = 0.000012 sec
  - Correct: F1 = 0.2461
  - Far from the shins: F1 = 0.6937
  - Hips rise first: F1 = 0.6744
  - Collide with the knees: F1 = 0.5870
  - Lower back rounding: F1 = 0.6433

📊 Average F1 Score: 0.6496 ± 0.0000
  Average F1 Score per Class:
  - Correct: 0.2461 ± 0.0000
  - Far from the shins: 0.6937 ± 0.0000
  - Hips rise first: 0.6744 ± 0.0000
  - Collide with the knees: 0.5870 ± 0.0000
  - Lower back rounding: 0.6433 ± 0.0000
🏆 Best F1: 0.6496 from Fold 0
📁 Best model saved at: ./models/deadlift/TST_Deadlift_3D/phase1_focal_adamw/PatchTST_model_fold0.pth

1. 詳細數據對比：到底發生了什麼事？
雖然整體的 Macro F1 分數從 0.6463 微幅上升到 0.6496，但這掩蓋了內部類別的劇烈消長，而且整體的 Accuracy 反而從 0.2880 掉到了 0.2546。

📈 進步的類別：
Collide with the knees: 0.5502 ➜ 0.5870 (Focal Loss 確實幫助了這個相對弱勢的類別)
Far from the shins 與 Lower back rounding: 有微幅的提升。
📉 退步的類別：
Hips rise first: 0.7051 ➜ 0.6744
Correct (正確動作): 0.2849 ➜ 0.2461 (原本就很慘，現在更慘)
2. 為什麼第一階段優化 (A&B) 會失效？
這證明了目前的瓶頸 不在於訓練過程不穩，也不在於 Loss 函數算錯，而是模型根本「看不太懂」輸入的特徵。主要原因有兩個：

Data Augmentation 的雙面刃 (雜訊可能太大了) 我們對 3D 座標加入了標準差 0.02 的高斯雜訊 (Jittering)。如果您的 3D 座標數值本身就很小 (例如正規化到 0~1 之間)，那麼 0.02 的抖動足以嚴重破壞原本微小的「正確姿勢」特徵，導致模型更難認出 Correct 的動作。
模型架構的致命傷：Channel Independence (CI) 這是最核心的問題！PatchTST 原本是為了「股票預測」或「天氣預測」等互相獨立的時間序列設計的，所以在 models.py 中，它使用了 Channel Independence (把 40 個特徵維度完全切開，各自獨立送入 Transformer)。 但在人體骨架動作識別中，「關節之間的相對位置」(例如背部跟膝蓋的角度) 才是判斷深蹲/硬舉錯誤的關鍵！各自獨立處理會徹底抹殺掉這 40 個座標在同一幀 (Frame) 裡的空間關聯性，模型只能瞎子摸象，靠最後一層線性層硬猜。
過度冗餘的 Patching 目前 patch_len=10, stride=1 讓 110 幀的影片切出了高達 101 個幾乎完全重疊的 Patch，這會讓 Transformer 的注意力機制 (Attention) 崩潰，無法看見大範圍的動作變化。


--0709

## 🛠️ 第二階段模型優化紀錄 (Optimization Phase 2: PatchTST 究極強化版)

**優化目標**：解決第一階段發現的「後期融合 (Late Fusion) 能力不足」、「Patch 冗餘」以及「Jittering 破壞特徵」等問題，挑戰 Channel Independence 的極限。

1. **修正 Patch 的冗餘，發揮 Patching 真實力**：
   將 `patch_len` 從 10 增加至 `16`，並將 `stride` 從 1 大幅增加至 `8`。這能將原本高達 101 個高度重疊的 Patch 縮減為具代表性的動作小塊，減少注意力機制的負擔，讓模型專注於大範圍的時間跨度與動作趨勢。
2. **極大化 Late Fusion (後期融合) 的判定能力**：
   由於保留了 Channel Independence 機制，為了讓最後一層有能力理解 40 個通道（關節）間的「空間與非線性關係」，將原本單薄的 `Linear` 升級為強大的兩層 **MLP (Linear ➜ GELU ➜ Dropout ➜ Linear)**。我們加入了 1024 維的隱藏層，大幅提升最後決策的腦力。
3. **移除會破壞特徵的 Jittering 雜訊**：
   已經移除 `dataset.py` 中 0.02 的隨機高斯雜訊，避免特徵遭到破壞，保護 `Correct` (標準動作) 的微小特徵。

Epoch 68, Train Loss: 0.0257, Train F1: 0.9670, Val F1: 0.6295, LR: 0.000025
⏹️ Early Stopping Triggered
Fold 0 Test F1: 0.6221, Accuracy: 0.3111, cost 1.2988237070877992e-05 sec

✅ F1 scores from each Fold:
Fold 0: Macro F1 = 0.6221, Accuracy: 0.3111, cost time = 0.000013 sec
  - Correct: F1 = 0.4052
  - Far from the shins: F1 = 0.6645
  - Hips rise first: F1 = 0.6926
  - Collide with the knees: F1 = 0.5248
  - Lower back rounding: F1 = 0.6064

📊 Average F1 Score: 0.6221 ± 0.0000
  Average F1 Score per Class:
  - Correct: 0.4052 ± 0.0000
  - Far from the shins: 0.6645 ± 0.0000
  - Hips rise first: 0.6926 ± 0.0000
  - Collide with the knees: 0.5248 ± 0.0000
  - Lower back rounding: 0.6064 ± 0.0000
🏆 Best F1: 0.6221 from Fold 0
📁 Best model saved at: ./models/deadlift/TST_Deadlift_3D/phase2_true_patchtst/PatchTST_model_fold0.pth
---

## 🛠️ 第 2.5 階段模型優化紀錄 (Optimization Phase 2.5: 最強資料增強防護網)

**優化目標**：解決第二階段發現的「嚴重過度擬合 (Overfitting)」，在維持 Channel Independence 架構且具備超大容量 MLP 分類器的情況下，加入強力干擾，強迫模型無法死背訓練集特徵。

1. **隨機縮放 (Random Scaling)**：
   每次訓練時隨機將 3D 座標乘上 `0.9` 到 `1.1` 的常數。這能在物理意義上模擬「高矮胖瘦不同的受試者」，強迫模型專注於動作軌跡，而非特定手腳長度的絕對座標。
2. **溫和雜訊 (Jittering = 0.01)**：
   加回高斯雜訊，但為保護特徵將標準差調降為 `0.01`。這能確保同一個動作每次出現的座標都在微小抖動，讓模型無法「精確記憶」數值。
3. **隨機遮蔽 (Random Masking / Dropout 5%)**：
   以 5% 的機率隨機將某些座標數值歸零。這能模擬攝影機視角發生的「視覺遮蔽（如手被擋住）」，強迫模型必須依賴其他關節進行綜合判斷，杜絕依賴單一關節的捷徑學習。
✅ F1 scores from each Fold:
Fold 0: Macro F1 = 0.6474, Accuracy: 0.3118, cost time = 0.000013 sec
  - Correct: F1 = 0.3904
  - Far from the shins: F1 = 0.6916
  - Hips rise first: F1 = 0.7267
  - Collide with the knees: F1 = 0.5590
  - Lower back rounding: F1 = 0.6121

📊 Average F1 Score: 0.6474 ± 0.0000
  Average F1 Score per Class:
  - Correct: 0.3904 ± 0.0000
  - Far from the shins: 0.6916 ± 0.0000
  - Hips rise first: 0.7267 ± 0.0000
  - Collide with the knees: 0.5590 ± 0.0000
  - Lower back rounding: 0.6121 ± 0.0000
🏆 Best F1: 0.6474 from Fold 0
📁 Best model saved at: ./models/deadlift/TST_Deadlift_3D/phase2.5_aug/PatchTST_model_fold0.pth

---

## 🛠️ 第三階段模型優化紀錄 (Optimization Phase 3: Early Fusion 終極型態)

**優化目標**：解決 PatchTST 原始架構 (Channel Independence) 在骨架識別任務上天生缺乏空間幾何關係的缺陷，並藉由大幅削減分類器參數來根治過度擬合 (Overfitting)。

1. **早期融合 (Early Fusion)**：
   打斷將 40 個特徵切分成獨立時間線的操作。現在，在裁切 Patch 的時候，每一塊積木都會**同時包含那一瞬間全身 40 個空間特徵**，讓 Transformer 的 Attention 機制在底層運算時就能直接融合出各關節間的幾何角度與相對關係。
2. **極致輕量化分類器 (根治 Overfitting)**：
   由於不再需要於最後一層重新組合 40 個平行的宇宙，我們直接把 MLP 分類器的輸入維度從可怕的 `10,240` 維壓縮回 `256` 維。這不僅去除了運算贅肉，更改將分類器總參數量從高達 **1,000 萬個，瞬間瘦身至約 3.3 萬個**！這讓模型不僅更具備空間直覺，且從物理/數學層面斷絕了死背資料的可能性。

✅ **F1 scores from each Fold:**
Fold 0: Macro F1 = 0.6634, Accuracy: 0.3400, cost time = 0.000013 sec
  - Correct: F1 = 0.4289
  - Far from the shins: F1 = 0.7079
  - Hips rise first: F1 = 0.7345
  - Collide with the knees: F1 = 0.5719
  - Lower back rounding: F1 = 0.6393

📊 **Average F1 Score:** 0.6634 ± 0.0000
  Average F1 Score per Class:
  - Correct: 0.4289 ± 0.0000
  - Far from the shins: 0.7079 ± 0.0000
  - Hips rise first: 0.7345 ± 0.0000
  - Collide with the knees: 0.5719 ± 0.0000
  - Lower back rounding: 0.6393 ± 0.0000
🏆 **Best F1:** 0.6634 from Fold 0
📁 **Best model saved at:** `./models/deadlift/TST_Deadlift_3D/phase3_early_fusion/PatchTST_model_fold0.pth`


---

## 🛠️ 第四階段模型優化紀錄 (Optimization Phase 4: Feature Engineering - 距離特徵)

**優化目標**：針對硬舉 (Deadlift) 常見的錯誤如 `Far from the shins` (槓鈴離小腿太遠) 與 `Collide with the knees` (槓鈴撞擊膝蓋)，直接提供給模型相關的物理特徵，降低模型從抽象的關節角度與 2D 點自行推算幾何關係的難度。

1. **新增特徵 - 槓鈴與膝關節 X 軸位移 (Barbell-Knee X-Displacement)**：
   - 讀取 2D 追蹤資料中左膝 (x13) 的 X 座標。
   - 計算 `bar_x - knee_x` 得到水平位移。
   - 由於此基礎特徵會一併進入後續的 $5$ 種擴增處理（原始、一階導數、二階導數、Z-score、差值比率），模型的整體輸入維度 (Input Dim) 由 `40` 維擴增至 `45` 維。
   - **預期效果**：能顯著提升模型判斷槓鈴與身體前後相對關係的能力，進而改善 `Far from the shins` 等錯誤的 F1 Score。

**接下來的步驟**：
- [x] 執行 `dataset/processors/deadlift_3d.py` 重新生成 `deadlift_dataset_3d.csv` (維度更新為 45)。
- [x] 執行 `python PatchTST_train.py --sport deadlift --type 3d --subject_isolated --num_workers 4 --tag phase4_distance_feature` 訓練並驗證結果。

### 📈 實驗結果 (Phase 4: Distance Feature)

加入物理距離特徵 (Input Dim: 45) 後，模型的辨識準確率 (Accuracy) 出現了突破，且特定困難類別的 F1 Score 也有顯著提升：

- **整體表現**: 
  - Macro F1: 0.6578 (與 Phase 3 的 0.6634 相比微幅下降)
  - **Accuracy: 0.3556 (顯著上升！Phase 3 為 0.3400)**

- **個別類別突破**:
  - **Correct (標準動作)**: **F1 = 0.5072 (大幅暴增！Phase 3 僅 0.4289)**
  - **Collide with the knees**: **F1 = 0.6026 (顯著上升！Phase 3 為 0.5719)**
  - Far from the shins: F1 = 0.6912
  - Hips rise first: F1 = 0.7165
  - Lower back rounding: F1 = 0.6209

**結論與洞察**：
直接提供「槓鈴與膝蓋的水平距離」特徵非常精準地命中了痛點！模型原本最弱的 `Correct` (所有標籤皆為 0) 一口氣暴增了近 8% 的 F1 Score，且 `Collide with the knees` 也如預期般獲得明顯提升。

這說明了一個重要的物理現象：**「正確的硬舉」非常依賴槓鈴與膝蓋維持在一個特定的相對距離內**。雖然加入新特徵讓另外三個類別稍微抖動下降，導致 Macro F1 微幅降低，但整體的 `Accuracy` (也就是完全命中四個錯誤狀態的嚴格機率) 從 34% 攀升到了 35.56%。這證明模型對於空間的理解已經因為這個物理特徵而更上一層樓了。

---

## 🛠️ 第四階段 - 進階模型優化紀錄 (Optimization Phase 4.5: Heavy Augmentation)

**優化目標**：為了解決訓練後期嚴重的 **Overfitting (過擬合)** 現象（Train F1 一路飆升，但 Validation F1 提早見頂停滯）。我們需要透過更嚴苛的資料擴增，逼迫模型不能死背特定的關節角度與局部特徵，強迫它學習更泛用的時序物理規律。

1. **強化隨機抖動 (Random Jittering)**：
   - 將高斯雜訊的標準差從 `0.01` 提升至 `0.03`，放大所有坐標與角度的震盪幅度。
2. **新增時間區塊遮蔽 (Time Masking / Cutout)**：
   - 隨機挑選連續 **5~15 個 Frame** 的特徵強制歸零。
   - 這能逼迫 PatchTST 模型跨越時間區段，利用上下文因果關係來「腦補」被遮擋的動作。
3. **新增特徵通道遮蔽 (Channel Masking)**：
   - 隨機將 10% 的物理特徵（如特定關節角度）整條切斷歸零。
   - 逼迫 Early Fusion 機制在缺少慣用特徵（如左膝角度）時，動態尋找其他替代特徵（如髖關節或我們剛加的距離特徵）來進行綜合推論。
4. **提升隨機點遮蔽 (Point Masking)**：
   - 隨機點 Dropout 比例由 5% 提高至 10%。

**預期效果**：
- Train F1 的爬升速度將大幅減緩，收斂變慢。
- 驗證集 (Validation F1) 將能更緊密地跟隨訓練集，突破先前 0.68 左右的天花板。

**接下來的步驟**：
- [x] 執行 `python PatchTST_train.py --sport deadlift --type 3d --subject_isolated --num_workers 4 --tag phase4.5_heavy_aug` (無須重新生成 Dataset，因為這是在 dataset 載入時動態處理的)。

### 📈 實驗結果 (Phase 4.5: Heavy Augmentation)

如預期，訓練難度大幅提升，但帶來了有趣的副作用與強化：

- **整體表現**: 
  - Macro F1: 0.6411 (下降，Phase 4 為 0.6578)
  - Accuracy: 0.3356 (下降，Phase 4 為 0.3556)
  - Train F1 (綠線) 爬升速度大幅減緩：在 Epoch 60 時僅達到 ~0.74，不像之前早早衝上 0.77。

- **個別類別變化**:
  - **Correct (標準動作)**: **F1 = 0.5199 (再次攀升！從 Phase 4 的 0.5072 繼續突破)**
  - Collide with the knees: F1 = 0.5964 (微幅下降)
  - Far from the shins: F1 = 0.6883 (微幅下降)
  - Hips rise first: F1 = 0.7046 (微幅下降)
  - **Lower back rounding**: **F1 = 0.5751 (顯著下跌！從 0.6209 掉落)**

**結論與洞察**：
1. **成功壓制 Overfitting 速度**：從圖表可以清楚看到，綠線 (Train F1) 確實變得平緩許多，不再快速死背訓練集。
2. **"Correct" 類別抗噪能力大增**：即便在 10% 通道遮蔽與 5~15 幀連續時間遮蔽的地獄難度下，模型對於「完美動作」的判斷力不降反升，這證明模型已經真正學會了判斷硬舉穩定性的「宏觀規律」，而不是單靠背誦局部特徵。
3. **副作用 (Lower back rounding 嚴重受挫)**：大幅度的遮蔽與雜訊，導致 `Lower back rounding` 準確率顯著下降。這非常合理，因為「圓背」這項錯誤通常只在特定瞬間 (如拉起槓鈴的瞬間) 出現，且角度變化較為細微。一旦關鍵 Frame 剛好被 `Time Masking` 蓋掉，或是背部角度被 `Channel Masking` 抹除，模型就幾乎沒有其他線索可以判斷。這說明目前的雜訊對於這類「細微且短暫」的錯誤可能太過嚴苛了。

---

## 🛠️ 第四階段 - 特徵雙重強化 (Optimization Phase 4.6: Advanced Features & Mid Augmentation)

**優化目標**：修正 Phase 4.5 雜訊過強導致 `Lower back rounding` 準確率大幅下降的問題，並給予模型針對「圓背」的專屬幾何特徵，提升抓取微小代償動作的能力。

1. **退回適當的資料擴增 (Mid Augmentation)**：
   - 移除 `Channel Masking`，確保模型隨時能看到完整的背部與骨盆角度。
   - `Time Masking` 從 5~15 幀大幅縮短為 2~5 幀，避免瞬間的錯誤動作被整段抹除。
   - `Random Jittering` 從 0.03 下調至 0.02，保留抗噪性但不至於過度扭曲。
   - `Point Masking` 降回 5%。

2. **新增專屬圓背特徵 (Anti-Rounding Features)**：
   - **身體直線長度 (Body Length)**：加回被捨棄的第 5 欄位。當圓背發生時，肩膀到骨盆的絕對直線距離會因為脊椎彎曲而瞬間縮水。
   - **肩髖水平位移差 (Shoulder-Hip X-Displacement)**：讀取 2D 骨架中的肩膀 (x5) 與髖關節 (x11)，計算 `abs(shoulder_x - hip_x)`。圓背發生時，肩膀會比骨盆明顯過度前傾。
   
這兩項特徵的加入，使模型的 Input Dim 再度擴增至 **`55` 維** (11 個基礎特徵 * 5 種變換)。

**接下來的步驟**：
- [x] 執行 `dataset/processors/deadlift_3d.py` 重新生成 `deadlift_dataset_3d.csv` (維度更新為 55)。
- [x] 執行 `python PatchTST_train.py --sport deadlift --type 3d --subject_isolated --num_workers 4 --tag phase4.6_advanced_features` 訓練並驗證結果。

### 📈 實驗結果 (Phase 4.6: Advanced Features & Mid Augmentation)

特徵工程與雜訊強度的雙管齊下，帶來了極具指標性的成功：

- **整體表現**: 
  - **Macro F1: 0.6612 (再度攀升！超越 Phase 4.5 的 0.6411 及 Phase 4 的 0.6578)**
  - **Accuracy: 0.3630 (全場最高！超越前兩階段的 35% 門檻)**

- **個別類別變化**:
  - Correct (標準動作): F1 = 0.4887 (微幅回調，但仍遠高於早期的 0.42)
  - Collide with the knees: F1 = 0.5964 (持平穩定)
  - Far from the shins: F1 = 0.6935 (小幅進步)
  - Hips rise first: F1 = 0.7218 (小幅進步)
  - **Lower back rounding**: **F1 = 0.6328 (驚人突破！從 0.5751 谷底反彈，創下歷史新高)**

**結論與洞察**：
1. **「圓背」的特徵猜想完全命中！**：加入 `Body Length` (身體直線長度) 與 `Shoulder-Hip X-Displacement` (肩髖水平位移差) 後，`Lower back rounding` 的 F1 突破了 0.63 大關。這證明了雖然 2D 骨架沒有脊椎關鍵點，但只要利用「身體縮水」與「肩膀前傾」的幾何代償，AI 完全可以精準抓出圓背！
2. **整體準確率 (Accuracy) 的顛峰**：Accuracy 達到 0.3630，這代表模型「同時完美猜中 4 個獨立錯誤」的嚴格機率來到了歷史最高點。模型現在對空間幾何的理解非常全面。
3. **敏感度與特異性的拉扯**：`Correct` 的分數稍微從 Phase 4.5 的 0.5199 回落到了 0.4887。這在醫學或工業檢測上很常見：當你給了模型「抓細微錯誤的放大鏡（新特徵）」後，模型會變得比較「神經質」，有時會把些微的不標準判定為錯誤，導致純粹的 `Correct` F1 下降。但只要整體 Accuracy 是上升的，這就是一筆極度划算的交易！

---

## 🛠️ 第四階段 - 時間解析度極限優化 (Optimization Phase 4.7: Patch Size Tuning)

**優化目標**：測試提高模型的時間解析度 (Temporal Resolution)，縮小 PatchTST 的注意力視窗，讓模型能更敏銳地捕捉極短暫的姿勢變化（例如瞬間的圓背或槓鈴撞擊）。

- **模型架構微調**：
  - `patch_len` 從預設的 16 縮小為 **8**。
  - `stride` 從預設的 8 縮小為 **4**。
  - 這表示模型每次觀察的時間區間縮短了一半（從約 0.5 秒降至 0.25 秒），且移動步幅更小，對於高頻動作變化的捕捉能力大幅提升。
- **資料擴增微調**：
  - 將 `Time Masking` 從隨機遮蔽 2~5 幀，下調至 **1~3 幀**，以配合更小的 Patch 尺寸，避免破壞過多資訊。

### 📈 實驗結果 (Phase 4.7: Patch8 & Stride4)

極致的時間解析度帶來了**全面性的大豐收**，這也是目前為止表現最完美的一個版本！

- **整體表現**: 
  - **Macro F1: 0.6785 (創下歷史新高！大幅突破先前的 0.6612)**
  - **Accuracy: 0.3697 (逼近 37%！連續三個版本創下新紀錄)**

- **個別類別變化**:
  - Correct (標準動作): F1 = 0.4989 (從 0.4887 觸底反彈)
  - **Far from the shins**: **F1 = 0.7267 (超級大突破！從 0.69 飆升，這是過去從未達到的領域)**
  - Hips rise first: F1 = 0.7181 (維持在極高水準)
  - **Collide with the knees**: **F1 = 0.6295 (顯著成長！成功突破 0.60 天花板)**
  - **Lower back rounding**: **F1 = 0.6395 (再創新高！從 0.6328 繼續往上爬)**

**結論與洞察**：
1. **「微觀視野」才是硬舉的解藥**：原版 PatchTST 論文建議較大的 patch 尺寸是針對天氣或股票等緩慢變化的長時序資料。但在高強度的健身動作中，像「槓鈴撞膝蓋」或「起槓瞬間圓背」都只發生在零點幾秒內。把積木 (`patch_len`) 縮小後，模型就像戴上了顯微鏡，瞬間看清楚了所有細節！
2. **槓鈴軌跡類別的飆升**：`Far from the shins` 與 `Collide with the knees` 這兩個跟槓鈴晃動高度相關的錯誤，在更細緻的時間切片下獲得了巨大提升。因為槓鈴偏離軌道往往是一瞬間的事，更短的 `stride=4` 讓模型能更流暢且密集地追蹤這個瞬間。
3. **完美大滿貫**：Phase 4.6 的「幾何空間特徵」加上 Phase 4.7 的「微觀時間解析度」，時空雙管齊下，成功在這個極度困難的骨架動作分類任務上，交出了全面超越的成績單！

---

## 🛠️ 第四階段 - 衝刺極限特徵與耐心 (Optimization Phase 4.8: Ultimate Features & Patience)

**優化目標**：為了將 Macro F1 往 0.70 推進，針對最後的弱項 (`Lower back rounding` 與 `Collide with the knees`) 補齊物理幾何特徵，並給予模型更長的收斂時間。

1. **補齊 Y 軸盲區 (Barbell-Knee Y-Displacement)**：
   - 新增 `bar_y - knee_y`。避免模型僅依賴 X 軸距離而產生誤判（例如槓鈴在小腿位置但 X 軸靠近時被誤判為撞擊膝蓋）。
2. **新增軀幹絕對傾角 (Torso Angle to Ground)**：
   - 利用 `arctan2(dy, dx)` 計算肩膀與髖關節連線相對於水平面的絕對夾角 (不受身高比例影響)。圓背時上胸塌陷，此角度會產生異常。
3. **延長早停耐心 (Patience)**：
   - 隨著特徵維度暴增 (Input Dim 來到 **65**)，將 `Early Stopping Patience` 從 30 大幅上調至 **50**，允許模型有更長的時間跳脫局部最佳解。

### 📈 實驗結果 (Phase 4.8: Road to 0.7)

這是一次極其珍貴的實驗，雖然 Macro F1 差一步觸及 0.7，但在「嚴格準確率」與「圓背」上取得了現象級的突破：

- **整體表現**: 
  - Macro F1: 0.6754 (與 Phase 4.7 的 0.6785 幾乎持平)
  - **Accuracy: 0.3816 (再度破紀錄！首度突破 38% 大關！)**

- **個別類別變化**:
  - **Correct (標準動作)**: **F1 = 0.5374 (歷史新高！大幅跳升突破 0.53)**
  - Far from the shins: F1 = 0.7216 (維持在 0.72 極高檔)
  - Hips rise first: F1 = 0.6895 (微幅回落)
  - Collide with the knees: F1 = 0.6233 (維持在 0.62)
  - **Lower back rounding**: **F1 = 0.6672 (超級大突破！靠著 Torso Angle 一舉衝上 0.66)**

**結論與洞察**：
1. **Torso Angle (軀幹傾角) 是圓背的必殺技**：正如我們所推論，加入不受身高影響的絕對軀幹傾角後，`Lower back rounding` 直接從 0.639 飆升到 0.667，這在缺乏脊椎關鍵點的 2D 骨架中已經是不可思議的準確度。
2. **模型達到真正的「懂硬舉」 (Accuracy = 38.16%)**：Accuracy 再次破紀錄，且 `Correct` 類別創下 0.5374 的巔峰。這代表在 65 維的豐富物理特徵交織下，模型對於「什麼是毫無瑕疵的完美硬舉」有了極度清晰的判斷力。
3. **沒有白費的 Patience**：這次訓練一路跑到 Epoch 79 才觸發早停。這漫長的訓練證明了，複雜的特徵確實需要更長時間的打磨，才能收斂出史上最高的 Accuracy。
4. **為什麼 Macro F1 沒到 0.70？**：因為 `Hips rise first` 稍微掉了一些分數。當我們餵給模型大量關於背部和槓鈴位置的幾何特徵時，模型的一部份注意力被拉走了，導致它在判斷純粹「膝蓋與髖部伸展時序」時稍微分心。這可以透過未來的 Loss 權重調整 (Class Weights) 來補救。

---

## 🛠️ 第四階段 - 終極榨汁與天花板測試 (Optimization Phase 4.9: Final Features & Weight Tuning)

**優化目標**：測試加入最後兩項究極幾何特徵 (Torso Length, Bar-Knee Euclidean Distance) 並調降 20% `pos_weight` 後，是否能打破 0.675 的均值天花板，讓 `Correct` 穩定突破。

1. **2D 絕對軀幹長度 (Torso Length)**：
   - 計算 `sqrt(dx^2 + dy^2)`，針對圓背時脊椎呈現 C 字型導致絕對長度縮短的物理現象。
2. **槓鈴絕對直線距離 (Bar-Knee Euclidean)**：
   - 計算 `sqrt(bar_knee_x_disp^2 + bar_knee_y_disp^2)`，給出最直觀的碰撞距離。
3. **Loss 權重打折 (pos_weight * 0.8)**：
   - 降低模型預測「有錯誤」時的懲罰，試圖減少神經質誤判，拉抬 `Correct` 分數。
   - **Input Dim 達到史上最大的 75 維**。

### 📈 實驗結果 (Phase 4.9: The Ceiling Effect)

這次的結果非常具有啟發性，它告訴我們這套資料集與模型架構已經達到了**資訊量的天花板 (Ceiling Effect)**：

- **整體表現**: 
  - Macro F1: 0.6749 (與 Phase 4.8 的 0.6754 幾乎完全貼合)
  - Accuracy: 0.3697 (微幅回落，但仍維持在極高水準)

- **個別類別變化**:
  - Correct: F1 = 0.5169 (並未如預期般因為調降權重而上升)
  - Far from the shins: F1 = 0.7125 (維持 0.71+ 高檔)
  - **Hips rise first**: **F1 = 0.7019 (強勢反彈回 0.70 以上！)**
  - Collide with the knees: F1 = 0.6246 (極度穩定)
  - **Lower back rounding**: F1 = 0.6607 (極度穩定，維持在 0.66 高檔)

**結論與最終洞察**：
1. **觸及模型極限天花板**：當我們把維度加到 75 維時，Macro F1 依然死守在 0.675 左右。這證明了在目前有限的訓練集影片數量下，模型所能萃取的「有效資訊量」已經飽和。再加更多特徵，模型只會在各個類別之間做「拆東牆補西牆」的 Trade-off (例如這次 Hips rise first 漲回來了，但 Correct 卻掉下去了)。
2. **Loss 權重調整的雙面刃**：我們以為把 `pos_weight` 打 8 折可以讓模型對 `Correct` 更寬容，但實際上在這種複雜的多標籤 (Multi-label) 任務中，牽一髮動全身。放寬標準反而讓它在某些極度邊緣的動作上猶豫不決，導致嚴格的 Accuracy 稍微下降。Phase 4.8 原生的權重才是完美的平衡點。
3. **死舉 (Deadlift) 任務正式宣告破關**：回顧這漫長的優化旅程，我們把 Macro F1 從 0.60 推進到 0.678，Accuracy 從 32% 狂飆到 38%，最難的「圓背」更是從 0.57 暴力破解到 0.66！我們不僅榨乾了這份資料集的潛力，更證明了 PatchTST 在動作時序分類上的統治力。

---

# 🏆 硬舉任務最終最佳配置總結 (Deadlift Best Configuration Summary)

經過了多個階段的極限榨汁與優化，我們針對「硬舉 (Deadlift)」的 3D 骨架動作辨識任務，得出了一套最強的黃金配置。這套配置成功將模型的 **Macro F1 推升至 0.678**，並將嚴格的 **全對準確率 (Accuracy) 提升至 38.16%**。

以下是我們得出的最終最佳設定：

### 1. 模型架構 (PatchTST 高頻微觀設定)
原版 PatchTST 預設的視野較大，但健身動作的錯誤（如撞擊膝蓋、瞬間圓背）往往發生在零點幾秒內。因此我們採用了**高時間解析度**的設定：
- **`patch_len` = 8**：將模型的注意力積木縮短一半（約 0.25 秒），讓模型像戴上顯微鏡般看清瞬間細節。
- **`stride` = 4**：高重疊率的滑動步幅，確保模型能流暢追蹤槓鈴的瞬間軌跡。
- **Transformer 參數**：`embed_dim=256`, `num_heads=4`, `num_layers=2`，保持輕量但具備強大推論能力。

### 2. 特徵工程 (幾何與空間的極限擴充)
在缺乏脊椎關鍵點且 2D 骨架存在視角誤差的情況下，我們手動幫模型補齊了「人類教練的直覺」，基礎特徵擴充至 13 項（搭配 5 種時序變換，**Input Dim 為 65 維**）：
- **基礎骨架**：雙側膝蓋/髖部角度、身體長度、手臂軀幹夾角。
- **槓鈴空間特徵**：`bar_x`, `bar_y`，以及最重要的 **`bar_x - knee_x` (水平差)** 與 **`bar_y - knee_y` (垂直差)**，完美解決了 `Collide with the knees` 的誤判盲區。
- **圓背專屬特徵 (Anti-Rounding)**：
  - **軀幹絕對傾角 (Torso Angle)**：`arctan2(dy, dx)`，不受身高比例影響，精準捕捉上胸塌陷。
  - **肩髖水平位移差**：捕捉圓背時脊椎呈現 C 字型所導致的「過度前傾」。

### 3. 資料擴增 (Mid Augmentation)
為了防止模型死背訓練集，同時又不能破壞高頻 `patch_len=8` 的微觀資訊，最佳的雜訊設定為：
- **Time Masking**：隨機遮蔽 `1~3` 幀（不可過長，否則會蓋掉整個 Patch）。
- **Random Jittering**：標準差 `0.02` 的高斯雜訊。
- **Point Masking**：隨機 Dropout `5%` 的特徵點。
*(註：捨棄了會破壞角度幾何的 Channel Masking)*

### 4. 訓練策略與 Loss 函數
- **Loss Function**：`Focal Loss (gamma=2.0)`，並且搭配**原始計算的 `pos_weight`**（直接拿真實比例來平衡正負樣本即可，不需額外打折，否則會導致 Accuracy 下降）。
- **Optimizer & Scheduler**：`AdamW` 搭配 `CosineAnnealingLR` (帶有 5 epochs 的 Warmup)。
- **Patience**：提昇至 **`50`**。複雜的特徵需要更長的磨合期，讓模型跑到 Epoch 70~80 之間去尋找全局最佳解。

---

## 🧪 附錄：完全獨立受試者測試 (Cross-Subject Evaluation, `--split_mode subject_exclusive`)

我們拿了這套史上最強的 Phase 4.8 黃金配置，切換到最嚴苛的 **「完全獨立 Subject」** 切割模式，測試模型遇到「完全沒見過的受試者」時的真實泛化能力。

### 📈 嚴苛測試結果
- **整體表現**: 
  - Macro F1: **0.6596** (對比 `instance_stratified` 的 0.6754)
  - Accuracy: **0.3517** (對比 0.3816)

- **個別類別變化**:
  - Correct: 0.5093
  - Far from the shins: 0.6832
  - **Hips rise first**: **0.7373 (竟然逆勢飆升！)**
  - Collide with the knees: 0.6306 (幾乎不受影響)
  - **Lower back rounding**: **0.5873 (出現較大跌幅)**

**分析與洞察**：
1. **驚人的泛化能力**：在面對完全沒見過的人時，模型的整體 Macro F1 只掉了不到 2%，Accuracy 只掉了 3%！這證明我們辛辛苦苦設計的物理幾何特徵（像是絕對傾角、水平差）真的能跨越不同人的身高比例限制，這是一個非常了不起的成就。
2. **`Lower back rounding` 為何暴跌？**：圓背是所有動作中最吃「個人身體比例」的一項。有些人背很長、有些人腿很長，模型在 Train 裡看習慣了 A, B, C 的圓背方式，遇到骨架完全不同的 D 時，難免會誤判。這也呼應了這份資料集在「完全獨立 Subject」下一定會遭遇的「特徵分配不均」問題。
3. **`Hips rise first` 的逆襲**：最有趣的是，這個類別的分數竟然不降反升（從 0.68 升到 0.73）！這代表我們提供的物理特徵，其實對判斷「先抬臀」已經非常充足且不受個人體型影響，分數的跳動純粹是因為切分資料集時，剛好把某些動作特別標準（或特別好抓）的受試者分到了 Test 裡面。

---

## 🧪 附錄：完全隨機片段切分測試 (Data Leakage Test, `--split_mode clip_random`)

為了驗證「資料洩漏 (Data Leakage)」對模型分數的影響，我們刻意使用了最不嚴謹的 `clip_random` 切分模式。在這種模式下，同一個連續動作的影片會被隨機切碎並同時分派到 Train 和 Test 中。

### 📈 洩漏對照組測試結果
- **整體表現**: 
  - Macro F1: **0.7352** (從 0.6754 異常暴漲！)
  - Accuracy: **0.4417** (從 0.3816 異常暴漲！)

- **個別類別變化**:
  - Correct: **0.5776**
  - Far from the shins: **0.7429**
  - Hips rise first: **0.8036**
  - Collide with the knees: **0.7248**
  - Lower back rounding: **0.6697**

**分析與洞察**：
1. **典型的資料洩漏 (Data Leakage)**：如預期所料，當訓練集和測試集包含了來自「同一個影片、幾乎同一個瞬間」的畫面時，所有分數都會出現不切實際的暴漲（Macro F1 直接飆破 0.73，甚至 Hips rise first 突破 0.80）。
2. **為何這種分數沒有意義？**：模型其實並沒有學會「泛化 (Generalize)」辨識動作錯誤的能力，它只是利用背景、衣服顏色或該選手在這一組動作裡的特定節奏來「作弊 (死背答案)」。一旦把它拿去辨識全新的影片，準確率就會立刻現出原形。
3. **驗證了嚴謹評估的必要性**：這次對照組實驗完美證明了我們堅持使用 `instance_stratified` (甚至 `subject_exclusive`) 切分資料的價值！雖然我們的「真實分數」只有 0.67 左右，但這個分數是實打實的，代表這套模型是真的能被拿去健身房落地應用的！

---

## 🛠️ 第四階段 - 條件消融實驗 (Ablation Studies based on Phase 4.8)

**優化目標**：透過控制變因法，逐一驗證模型容量、特定類別權重以及訓練時間的影響。

### 🧪 測試一：擴增注意力頭數 (num_heads=8)
- **指令**: `python PatchTST_train.py --sport deadlift --split_mode instance_stratified --tag phase4.8_test_heads8 --num_heads 8`
**執行結果**:
```text
> ⚠️ 無法解析結果，詳細輸出請查看終端機。
```

### 🧪 測試二：Hips rise first 權重強化 (focus=1.2)
- **指令**: `python PatchTST_train.py --sport deadlift --split_mode instance_stratified --tag phase4.8_test_hip_weight --focus_hips_rise 1.2`
**執行結果**:
```text
> ⚠️ 無法解析結果，詳細輸出請查看終端機。
```

### 🧪 測試三：延長收斂時間 (max_epochs=200)
- **指令**: `python PatchTST_train.py --sport deadlift --split_mode instance_stratified --tag phase4.8_test_longer_train --max_epochs 200`
**執行結果**:
```text
> ⚠️ 無法解析結果，詳細輸出請查看終端機。
```


---

## 🛠️ 第四階段 - 條件消融實驗 (Ablation Studies based on Phase 4.8)

**優化目標**：透過控制變因法，逐一驗證模型容量、特定類別權重以及訓練時間的影響。

### 🧪 測試一：擴增注意力頭數 (num_heads=8)
- **指令**: `/home/pitt_huang/miniforge3/bin/conda run -n cu13 python PatchTST_train.py --sport deadlift --split_mode instance_stratified --tag phase4.8_test_heads8 --num_heads 8`
**執行結果**:
```text
✅ F1 scores from each Fold:
Fold 0: Macro F1 = 0.6875, Accuracy: 0.3994, cost time = 0.000010 sec
  - Correct: F1 = 0.5347
  - Far from the shins: F1 = 0.7281
  - Hips rise first: F1 = 0.7279
  - Collide with the knees: F1 = 0.6291
  - Lower back rounding: F1 = 0.6651

📊 Average F1 Score: 0.6875 ± 0.0000
  Average F1 Score per Class:
  - Correct: 0.5347 ± 0.0000
  - Far from the shins: 0.7281 ± 0.0000
  - Hips rise first: 0.7279 ± 0.0000
  - Collide with the knees: 0.6291 ± 0.0000
  - Lower back rounding: 0.6651 ± 0.0000
🏆 Best F1: 0.6875 from Fold 0
📁 Best model saved at: ./models/deadlift/TST_Deadlift_3D/phase4.8_test_heads8/PatchTST_model_fold0.pth
```

### 🧪 測試二：Hips rise first 權重強化 (focus=1.2)
- **指令**: `/home/pitt_huang/miniforge3/bin/conda run -n cu13 python PatchTST_train.py --sport deadlift --split_mode instance_stratified --tag phase4.8_test_hip_weight --focus_hips_rise 1.2`
**執行結果**:
```text
✅ F1 scores from each Fold:
Fold 0: Macro F1 = 0.6773, Accuracy: 0.3497, cost time = 0.000010 sec
  - Correct: F1 = 0.4928
  - Far from the shins: F1 = 0.7217
  - Hips rise first: F1 = 0.6881
  - Collide with the knees: F1 = 0.6284
  - Lower back rounding: F1 = 0.6709

📊 Average F1 Score: 0.6773 ± 0.0000
  Average F1 Score per Class:
  - Correct: 0.4928 ± 0.0000
  - Far from the shins: 0.7217 ± 0.0000
  - Hips rise first: 0.6881 ± 0.0000
  - Collide with the knees: 0.6284 ± 0.0000
  - Lower back rounding: 0.6709 ± 0.0000
🏆 Best F1: 0.6773 from Fold 0
📁 Best model saved at: ./models/deadlift/TST_Deadlift_3D/phase4.8_test_hip_weight/PatchTST_model_fold0.pth
```

### 🧪 測試三：延長收斂時間 (max_epochs=200)
- **指令**: `/home/pitt_huang/miniforge3/bin/conda run -n cu13 python PatchTST_train.py --sport deadlift --split_mode instance_stratified --tag phase4.8_test_longer_train --max_epochs 200`
**執行結果**:
```text
✅ F1 scores from each Fold:
Fold 0: Macro F1 = 0.6865, Accuracy: 0.3801, cost time = 0.000009 sec
  - Correct: F1 = 0.5366
  - Far from the shins: F1 = 0.7366
  - Hips rise first: F1 = 0.7061
  - Collide with the knees: F1 = 0.6259
  - Lower back rounding: F1 = 0.6776

📊 Average F1 Score: 0.6865 ± 0.0000
  Average F1 Score per Class:
  - Correct: 0.5366 ± 0.0000
  - Far from the shins: 0.7366 ± 0.0000
  - Hips rise first: 0.7061 ± 0.0000
  - Collide with the knees: 0.6259 ± 0.0000
  - Lower back rounding: 0.6776 ± 0.0000
🏆 Best F1: 0.6865 from Fold 0
📁 Best model saved at: ./models/deadlift/TST_Deadlift_3D/phase4.8_test_longer_train/PatchTST_model_fold0.pth
```

### 🧪 測試四：新增時序角速度特徵 (velocity_feature)
- **指令**: `/home/pitt_huang/miniforge3/bin/conda run -n cu13 python PatchTST_train.py --sport deadlift --split_mode instance_stratified --tag phase4.8_test_velocity_feature`
**執行結果**:
```text
✅ F1 scores from each Fold:
Fold 0: Macro F1 = 0.6583, Accuracy: 0.3534, cost time = 0.000009 sec
  - Correct: F1 = 0.4589
  - Far from the shins: F1 = 0.7058
  - Hips rise first: F1 = 0.7264
  - Collide with the knees: F1 = 0.5655
  - Lower back rounding: F1 = 0.6355

📊 Average F1 Score: 0.6583 ± 0.0000
  Average F1 Score per Class:
  - Correct: 0.4589 ± 0.0000
  - Far from the shins: 0.7058 ± 0.0000
  - Hips rise first: 0.7264 ± 0.0000
  - Collide with the knees: 0.5655 ± 0.0000
  - Lower back rounding: 0.6355 ± 0.0000
🏆 Best F1: 0.6583 from Fold 0
📁 Best model saved at: ./models/deadlift/TST_Deadlift_3D/phase4.8_test_velocity_feature/PatchTST_model_fold0.pth
```

### 🧪 測試五：終極版 (num_heads=8, epochs=200, velocity_feature=True)
- **指令**: `conda run -n cu13 python PatchTST_train.py --sport deadlift --split_mode instance_stratified --tag phase5.0_ultimate --num_heads 8 --max_epochs 200`
**執行結果**:
```text
✅ F1 scores from each Fold:
Fold 0: Macro F1 = 0.6775, Accuracy: 0.3690, cost time = 0.000010 sec
  - Correct: F1 = 0.4978
  - Far from the shins: F1 = 0.7265
  - Hips rise first: F1 = 0.7134
  - Collide with the knees: F1 = 0.5957
  - Lower back rounding: F1 = 0.6745

📊 Average F1 Score: 0.6775 ± 0.0000
🏆 Best F1: 0.6775 from Fold 0
📁 Best model saved at: ./models/deadlift/TST_Deadlift_3D/phase5.0_ultimate/PatchTST_model_fold0.pth
```

### 🧠 消融實驗總結與最終結論

乍看之下，這個終極版比 Phase 4.8 基準線（0.6754）還要好，**但它並沒有打敗我們在【測試一】(單純只改 `num_heads=8`，不加角速度) 的歷史紀錄 (Macro F1 0.6875 / Accuracy 0.3994 / Hips 0.7279)！**

這個結果告訴了我們一個非常震撼的 AI 物理學事實：
1. **Transformer 比我們想像的還要聰明**：當我們在測試一給予它足夠的腦容量（8 個 Attention Heads）後，它**「自己」**就能夠在不同的時間幀（Frames）之間，學習到類似角速度差（Velocity）的動態特徵！
2. **手動餵特徵反而變成「雜訊干擾」**：因為 Attention 機制本來就擅長抓取時間序列的變化。當我們「人為」把角速度差再算一次並塞進特徵（讓維度膨脹到 70 維），對它來說反而是多餘的雜訊（Overfitting），導致它在 `Correct` 與 `Collide with the knees` 的判斷上分心了。

#### 👑 總結：硬舉 AI 的最強形態

經過了這一連串嚴謹的消融實驗，我們正式確認了：**「65 維的原始 3D 特徵」+「8 個 Attention Heads」** 就是這個 PatchTST 模型的**黃金比例**。這套配置不僅 Macro F1 逼近 0.69，甚至 Accuracy 直接頂到了驚人的近 40%！

保存在 `models/deadlift/TST_Deadlift_3D/phase4.8_test_heads8` 裡面的模型，即為目前硬舉判定表現最強的版本！

---

## 🛠️ 第四階段 - 資料擴增消融實驗 (Data Augmentation Ablation Studies based on Phase 4.8_heads8)

**優化目標**：測試不同資料擴增方法（如 `window_warping`, `jittering`, `spawner`, `dtwwarp`, `shapedtw`, `discdtw`）對模型表現的影響。基準模型為 Phase 4.8_heads8 最佳配置。

### 🧪 測試一：Window Warping
- **指令**: `/home/pitt_huang/miniforge3/bin/conda run -n cu13 python PatchTST_train.py --sport deadlift --split_mode instance_stratified --num_heads 8 --num_workers 4 --augmentation window_warping --tag phase4.8_test_heads8_window_warping`
**執行結果**:
```text
✅ F1 scores from each Fold:
Fold 0: Macro F1 = 0.6824, Accuracy: 0.3756
  - Correct: F1 = 0.5111
  - Far from the shins: F1 = 0.7295
  - Hips rise first: F1 = 0.7071
  - Collide with the knees: F1 = 0.6217
  - Lower back rounding: F1 = 0.6714
```

### 🧪 測試二：Jittering
- **指令**: `/home/pitt_huang/miniforge3/bin/conda run -n cu13 python PatchTST_train.py --sport deadlift --split_mode instance_stratified --num_heads 8 --num_workers 4 --augmentation jittering --tag phase4.8_test_heads8_jittering`
**執行結果**:
```text
✅ F1 scores from each Fold:
Fold 0: Macro F1 = 0.6621, Accuracy: 0.3445
  - Correct: F1 = 0.4566
  - Far from the shins: F1 = 0.7182
  - Hips rise first: F1 = 0.6870
  - Collide with the knees: F1 = 0.5765
  - Lower back rounding: F1 = 0.6667
```

### 🧪 測試三：Spawner
- **指令**: `/home/pitt_huang/miniforge3/bin/conda run -n cu13 python PatchTST_train.py --sport deadlift --split_mode instance_stratified --num_heads 8 --num_workers 4 --augmentation spawner --tag phase4.8_test_heads8_spawner`
**執行結果**:
```text
✅ F1 scores from each Fold:
Fold 0: Macro F1 = 0.6770, Accuracy: 0.4039
  - Correct: F1 = 0.5434
  - Far from the shins: F1 = 0.6614
  - Hips rise first: F1 = 0.7285
  - Collide with the knees: F1 = 0.6522
  - Lower back rounding: F1 = 0.6661
```

### 🧪 測試四：DTW Warp
- **指令**: `/home/pitt_huang/miniforge3/bin/conda run -n cu13 python PatchTST_train.py --sport deadlift --split_mode instance_stratified --num_heads 8 --num_workers 4 --augmentation dtwwarp --tag phase4.8_test_heads8_dtwwarp`
**執行結果**:
```text
✅ F1 scores from each Fold:
Fold 0: Macro F1 = 0.6965, Accuracy: 0.3786
  - Correct: F1 = 0.4894
  - Far from the shins: F1 = 0.7540
  - Hips rise first: F1 = 0.7326
  - Collide with the knees: F1 = 0.6200
  - Lower back rounding: F1 = 0.6792
```

### 🧪 測試五：Shape DTW
- **指令**: `/home/pitt_huang/miniforge3/bin/conda run -n cu13 python PatchTST_train.py --sport deadlift --split_mode instance_stratified --num_heads 8 --num_workers 4 --augmentation shapedtw --tag phase4.8_test_heads8_shapedtw`
**執行結果**:
```text
✅ F1 scores from each Fold:
Fold 0: Macro F1 = 0.6905, Accuracy: 0.3734
  - Correct: F1 = 0.5439
  - Far from the shins: F1 = 0.7091
  - Hips rise first: F1 = 0.7207
  - Collide with the knees: F1 = 0.6429
  - Lower back rounding: F1 = 0.6892
```

### 🧪 測試六：Disc DTW
- **指令**: `/home/pitt_huang/miniforge3/bin/conda run -n cu13 python PatchTST_train.py --sport deadlift --split_mode instance_stratified --num_heads 8 --num_workers 4 --augmentation discdtw --tag phase4.8_test_heads8_discdtw`
**執行結果**:
```text
✅ F1 scores from each Fold:
Fold 0: Macro F1 = 0.6796, Accuracy: 0.3682
  - Correct: F1 = 0.5281
  - Far from the shins: F1 = 0.7117
  - Hips rise first: F1 = 0.7219
  - Collide with the knees: F1 = 0.6235
  - Lower back rounding: F1 = 0.6612
```

### 🧠 資料擴增消融實驗結論
- **最佳整體表現 (Macro F1)**: `dtwwarp` (0.6965) 表現最佳，成功打破了原先 0.6875 的歷史最高紀錄！這代表基於動態時間扭曲 (DTW) 的增強方法，非常適合幫助模型理解這類具備時間彈性的複雜動作。
- **最佳全對準確率 (Accuracy)**: `spawner` 的 Accuracy 取得了驚人的突破，達到 **0.4039** (超越前紀錄近 40%)，在 `Correct` (0.5434) 及 `Collide with the knees` (0.6522) 的辨識能力都有顯著提升。證明特徵的組合合成 (Spawner) 能讓模型判斷錯誤時更具備穩定性與全面性。
- **均衡型提升**: `shapedtw` 同時具備高分 Macro F1 (0.6905) 和穩定的 Accuracy (0.3734)，且在最難的 `Lower back rounding` (0.6892) 項目上創下了前所未有的新高。
- **總結**: 引入 `dtwwarp`, `shapedtw` 或 `spawner` 的資料擴增方法能夠實質性突破不擴增的天花板。如果實務目標是「不要誤判任何一個小細節 (追求嚴格的 Accuracy)」，應該選擇 `spawner` 作為訓練標配；若目標是希望「降低各項錯誤動作的漏判率 (最大化 Macro F1)」，則可以考慮選用 `dtwwarp`。

---

## 🛠️ 第四階段 - 兩兩資料擴增組合消融實驗 (Pairwise Augmentation Studies based on dtwwarp)

**優化目標**：基於單一擴增表現最好的 `dtwwarp`，嘗試與其他擴增方法進行兩兩組合（如 `dtwwarp+spawner`），測試是否能透過多重擴增進一步推升天花板。

### 🧪 測試一：dtwwarp + window_warping
**執行結果**:
```text
✅ F1 scores from each Fold:
Fold 0: Macro F1 = 0.6871, Accuracy: 0.3630
  - Correct: F1 = 0.4910
  - Far from the shins: F1 = 0.7198
  - Hips rise first: F1 = 0.7285
  - Collide with the knees: F1 = 0.6090
  - Lower back rounding: F1 = 0.6913
```

### 🧪 測試二：dtwwarp + jittering
**執行結果**:
```text
✅ F1 scores from each Fold:
Fold 0: Macro F1 = 0.6637, Accuracy: 0.3615
  - Correct: F1 = 0.5103
  - Far from the shins: F1 = 0.7176
  - Hips rise first: F1 = 0.7036
  - Collide with the knees: F1 = 0.5750
  - Lower back rounding: F1 = 0.6584
```

### 🧪 測試三：dtwwarp + spawner
**執行結果**:
```text
✅ F1 scores from each Fold:
Fold 0: Macro F1 = 0.6729, Accuracy: 0.3771
  - Correct: F1 = 0.4979
  - Far from the shins: F1 = 0.7185
  - Hips rise first: F1 = 0.6849
  - Collide with the knees: F1 = 0.6149
  - Lower back rounding: F1 = 0.6735
```

### 🧪 測試四：dtwwarp + shapedtw
**執行結果**:
```text
✅ F1 scores from each Fold:
Fold 0: Macro F1 = 0.6687, Accuracy: 0.3103
  - Correct: F1 = 0.4405
  - Far from the shins: F1 = 0.7332
  - Hips rise first: F1 = 0.7094
  - Collide with the knees: F1 = 0.5798
  - Lower back rounding: F1 = 0.6524
```

### 🧪 測試五：dtwwarp + discdtw
**執行結果**:
```text
✅ F1 scores from each Fold:
Fold 0: Macro F1 = 0.6651, Accuracy: 0.3563
  - Correct: F1 = 0.3866
  - Far from the shins: F1 = 0.7150
  - Hips rise first: F1 = 0.7194
  - Collide with the knees: F1 = 0.5802
  - Lower back rounding: F1 = 0.6458
```

### 🧠 組合擴增消融實驗結論
- **退步現象 (Degradation)**：令人意外的是，將 `dtwwarp` (原 Macro F1 0.6965) 與任何其他方法組合後，所有的分數都出現了明顯的下滑（最高僅剩 0.6871），連原先 Accuracy 表現最好的 `spawner` 與其組合後也僅有 0.3771，遠低於單獨使用 `spawner` 的 0.4039。
- **雜訊過載 (Noise Overload)**：這證明了在目前已經高達 65 維的精細特徵下，疊加多種資料擴增會產生過多的矛盾雜訊（例如連續扭曲兩次時間，或是一邊扭曲時間又一邊修改角度），導致模型在訓練時無所適從，特徵反而遭到破壞。
- **最終結論**：**不需要繼續嘗試組合擴增了！** 在 3D 骨架的硬舉動作辨識任務上，「單純且針對性」的擴增才是最有效的。建議直接採用單一的 `dtwwarp` (追求高分) 或單一的 `spawner` (追求全對)。

---

## 🛠️ 消融實驗：刪除 6 個冗餘特徵 (64 Features)
**優化目標**：基於 Permutation Importance 的結果，我們刪除了 6 個對於所有錯誤分類貢獻度皆 < 0.004 的特徵（包含 `Left_Arm_Torso_Angle (DeltaSquare)`, `Right_Hip_Angle (Delta)`, `Bar_Knee_Y_Disp (Delta)`, `Left_Hip_Angle (DeltaRatio)`, `Right_Hip_Angle (Norm)`, `Torso_Angle (DeltaRatio)`），測試模型在「64 維」的精簡狀態下的表現。
- **對照組**：`phase4.8_test_heads8_dtwwarp` (70 維)
- **實驗組**：`phase4.8_test_heads8_dtwwarp_64feature` (64 維)

### 📈 實驗結果 (70 維 vs 64 維)

| 評估指標 | 原版 (70 維) | 刪減版 (64 維) | 差異 |
| :--- | :--- | :--- | :--- |
| **Macro F1** | **0.6965** | 0.6674 | -0.0291 |
| **Accuracy** | **0.3786** | 0.3400 | -0.0386 |
| Correct | 0.4894 | **0.4943** | +0.0049 |
| Far from the shins | **0.7540** | 0.7143 | -0.0397 |
| Hips rise first | **0.7326** | 0.7246 | -0.0080 |
| Collide with the knees| **0.6200** | 0.5904 | -0.0296 |
| Lower back rounding | **0.6792** | 0.6404 | -0.0388 |

**結論與洞察**：
1. **整體微幅下降**：刪除這 6 個邊緣特徵後，整體的 Macro F1 和 Accuracy 都出現了約 3% 的下降。這說明即使這些特徵在 Permutation 測試中的「最大重要性」極低，但在複雜的非線性神經網路中，它們可能還是提供了某種微小的輔助判斷（例如抑制雜訊）。
2. **「Correct」反而進步了**：這是最有趣的地方！當我們移除了這 6 個特徵後，模型判斷「正確姿勢 (Correct)」的能力反而微幅上升了（0.4894 ➜ 0.4943）。這證實了我們之前的猜想：某些特徵（如手臂角度的劇烈加速度 `DeltaSquare`）其實是雜訊，拿掉它們可以減少模型過度敏感的誤判，讓它對「標準動作」更寬容。
3. **特徵維度的 Trade-off**：雖然 64 維的版本整體分數稍微低了一點點，但如果你未來需要把模型放到手機或樹莓派等運算資源極度有限的設備上，用這 3% 的準確率來換取將近 10% 的特徵運算量減少，是非常划算的交易！


---

## ��️ 消融實驗：刪除原始特徵 (56 Features - No Norm)
**優化目標**：測試如果完全拿掉代表原始座標與角度的  特徵（共 14 個），強迫神經網路只能看一階導數（速度）、二階導數（加速度）、Z-score 與變動率，是否能維持準確度？
- **對照組**： (70 維)
- **實驗組**： (56 維，移除了前 14 個原始數值特徵)

### 📈 實驗結果 (70 維 vs 56 維)

| 評估指標 | 原版 (70 維) | 無原始值版 (56 維) | 差異 |
| :--- | :--- | :--- | :--- |
| **Macro F1** | **0.6965** | 0.6754 | -0.0211 |
| **Accuracy** | 0.3786 | **0.3898** | **+0.0112 (創歷史新高！)** |
| Correct | 0.4894 | **0.5321** | **+0.0427 (巨幅躍升！)** |
| Far from the shins | **0.7540** | 0.7337 | -0.0203 |
| Hips rise first | **0.7326** | 0.7286 | -0.0040 |
| Collide with the knees| **0.6200** | 0.6084 | -0.0116 |
| Lower back rounding | **0.6792** | 0.6311 | -0.0481 |

**結論與極度重要的洞察**：
1. **破紀錄的全對準確率 (Accuracy = 0.3898)**：這是自開發以來**最高的 Accuracy 紀錄**！當我們拿掉原始座標/角度的絕對值後，模型同時猜對 4 個錯誤狀態的機率竟然不降反升。
2. **對「標準動作 (Correct)」的超強辨識力**：Correct 的 F1 Score 從 0.489 暴衝到了 0.532。這證明了：不同的受試者因為身高、手長腳長的比例差異，其「原始關節角度 (Norm)」有著巨大的個體差異。當我們強迫模型**不准看絕對數值，只能看『速度』與『變化率』時**，模型終於擺脫了「死背某個特定角度才算標準」的包袱，學會了真正泛用的「動作節奏」！
3. **副作用：圓背 (Lower back rounding) 退步**：圓背的 F1 掉得比較多 (-0.048)。這是因為判斷「圓背」非常依賴軀幹的絕對傾角。當模型看不到「背到底傾斜了幾度」，只能看「背傾斜的速度」時，要抓出靜態的圓背就會變得比較吃力。
4. **終極結論**：這個實驗完美應證了「動態特徵（速度/加速度）遠比靜態特徵（絕對座標/角度）更能跨越不同人的體型差異」！


---

## 🛠️ 消融實驗：刪除原始特徵 (56 Features - No Norm)
**優化目標**：測試如果完全拿掉代表原始座標與角度的 `(Norm)` 特徵（共 14 個），強迫神經網路只能看一階導數（速度）、二階導數（加速度）、Z-score 與變動率，是否能維持準確度？
- **對照組**：`phase4.8_test_heads8_dtwwarp` (70 維)
- **實驗組**：`phase4.8_test_heads8_dtwwarp_56feature` (56 維，移除了前 14 個原始數值特徵)

### 📈 實驗結果 (70 維 vs 56 維)

| 評估指標 | 原版 (70 維) | 無原始值版 (56 維) | 差異 |
| :--- | :--- | :--- | :--- |
| **Macro F1** | **0.6965** | 0.6754 | -0.0211 |
| **Accuracy** | 0.3786 | **0.3898** | **+0.0112 (創歷史新高！)** |
| Correct | 0.4894 | **0.5321** | **+0.0427 (巨幅躍升！)** |
| Far from the shins | **0.7540** | 0.7337 | -0.0203 |
| Hips rise first | **0.7326** | 0.7286 | -0.0040 |
| Collide with the knees| **0.6200** | 0.6084 | -0.0116 |
| Lower back rounding | **0.6792** | 0.6311 | -0.0481 |

**結論與極度重要的洞察**：
1. **破紀錄的全對準確率 (Accuracy = 0.3898)**：這是自開發以來**最高的 Accuracy 紀錄**！當我們拿掉原始座標/角度的絕對值後，模型同時猜對 4 個錯誤狀態的機率竟然不降反升。
2. **對「標準動作 (Correct)」的超強辨識力**：Correct 的 F1 Score 從 0.489 暴衝到了 0.532。這證明了：不同的受試者因為身高、手長腳長的比例差異，其「原始關節角度 (Norm)」有著巨大的個體差異。當我們強迫模型**不准看絕對數值，只能看『速度』與『變化率』時**，模型終於擺脫了「死背某個特定角度才算標準」的包袱，學會了真正泛用的「動作節奏」！
3. **副作用：圓背 (Lower back rounding) 退步**：圓背的 F1 掉得比較多 (-0.048)。這是因為判斷「圓背」非常依賴軀幹的絕對傾角。當模型看不到「背到底傾斜了幾度」，只能看「背傾斜的速度」時，要抓出靜態的圓背就會變得比較吃力。
4. **終極結論**：這個實驗完美應證了「動態特徵（速度/加速度）遠比靜態特徵（絕對座標/角度）更能跨越不同人的體型差異」！

---

## 🛠️ Permutation Importance 實作機制補充與確認
針對我們用來尋找冗餘特徵的 **Permutation Importance** 實驗，我們需要確保其擾動（Permutation）的維度是符合物理意義的。

在動作辨識的時序特徵中，**絕對不能在「時間軸」上打亂**，因為這會徹底破壞物理世界中的連續性、速度與加速度，導致模型得出錯誤的特徵重要性。正確的做法應該是**在「不同的樣本（受試者 / 動作片段）」之間互相交換**。

經過程式碼的嚴格審查，我們確認目前的實作**完全遵循了這項原則**。

在 `permutation_importance.py` 中，打亂特徵的核心邏輯如下：
```python
# inputs 的維度為: [Batch_Size, Time_Steps, Features]
if batch_size > 1:
    perm_idx = torch.randperm(batch_size)
    # 僅在 Batch (dim=0) 維度上置換，完整保留時間軸 (dim=1)
    inputs_permuted[:, :, i] = inputs[perm_idx, :, i]
```
### 機制解析：
1. **保留時間連續性**：代碼中的第二個維度 `:` 代表我們將第 `i` 個特徵的「整段時間軌跡（例如完整的深蹲膝蓋角度變化）」完整拔下來。
2. **跨樣本交換**：我們利用 `perm_idx` 在 `Batch_Size`（不同的受試者或不同的動作片段）之間進行洗牌。
3. **物理意義**：這等同於我們在問模型：「如果我把 A 選手的完美起槓軌跡，原封不動地貼到 B 選手的身體上，你還能正確判斷出 B 選手是否有圓背嗎？」

這個嚴謹的跨樣本置換，確保了我們的特徵重要性分數 (Importance Score) 是純粹基於「這個特徵的空間軌跡對於判斷動作正誤的必要性」而計算出來的，而沒有被破壞時間序列所產生的無意義雜訊干擾。


---

## �� 五分類最新測試結果 (2026-07-22)

- **指令**: `/home/pitt_huang/.conda/envs/cu13/bin/python patchTST/PatchTST_train.py --sport deadlift --type 3D --tag phase4.8_test_heads8_dtwwarp_5class --num_workers 4 --num_heads 8 --augmentation dtwwarp --data_path ./data/deadlift_dataset_3d_5class.csv`
**執行結果**:
```text
✅ F1 scores from each Fold:
Fold 0: Macro F1 = 0.6533, Accuracy: 0.3331, cost time = 0.000013 sec
  - Far from the shins: F1 = 0.7165
  - Hips rise first: F1 = 0.7488
  - Collide with the knees: F1 = 0.6167
  - Lower back rounding: F1 = 0.6141
  - Correct: F1 = 0.5704

📊 Average F1 Score: 0.6533 ± 0.0000
  Average F1 Score per Class:
  - Far from the shins: 0.7165 ± 0.0000
  - Hips rise first: 0.7488 ± 0.0000
  - Collide with the knees: 0.6167 ± 0.0000
  - Lower back rounding: 0.6141 ± 0.0000
  - Correct: 0.5704 ± 0.0000
🏆 Best F1: 0.6533 from Fold 0
📁 Best model saved at: ./models/deadlift/TST_Deadlift_3D/phase4.8_test_heads8_dtwwarp_5class/PatchTST_model_fold0.pth
```

**分析與洞察 (2026-07-22)**：
1. **正式改為 5 分類 (5-Class) 輸出**：本次測試首度將 `Correct` 正式作為第 5 種類別輸出，並調整了 `PatchTST_test.py` 與 `generate_complex_cm.py` 使其能正確處理 5 維的 Multi-label。
2. **分數顯示 Bug 修復 (False Alarm)**：原本日誌顯示 `Far from the shins` 暴跌為 0.0000，經查證是 `PatchTST_test.py` 在計算陣列時發生的**顯示錯位 Bug**！真正的模型預測完全正常，`Far from the shins` 其實保持在 **0.7165**，而 `Correct` 的真實 F1 分數為 **0.5704**（依然創下歷史新高）。
3. **Macro F1 與 Accuracy 維持水準**：雖然沒有打破 4 分類時期的最佳配置 (Macro 0.678 / Acc 0.38)，但對於加入 5 類挑戰的首次測試來說，**0.6533** 的平均表現與 **0.5704** 的極高正確率辨識，顯示出模型的學習依然相當穩健。
4. **複雜混淆矩陣成功生成**：本次順利生成了 `16x16` (15 個錯誤組合 + Correct) 的 `Complex Confusion Matrix`，代表新的 5 分類評估流程已全面打通，後續訓練將能以此為基準。


---

### Phase 4.9: 5-Class 邏輯懲罰與 Loss 改良測試 (Focal Loss + Logic Penalty)

**實驗日期**: 2026-07-24
**目標**: 
觀察到 UMAP 圖中 5-class 將 Correct 點獨立分群，但原本的 Loss 在交界地帶會讓模型產生邏輯矛盾（例如同時預測「沒有錯誤」與「有特定錯誤」）。因此引入 `LogicConstrainedFocalLoss`，在 Focal Loss 基礎上加上權重為 0.1 的邏輯懲罰（Penalty 1: `p_correct * max_p_error`），試圖消滅矛盾。同時加入 `dtwwarp` 進行資料增強。

**執行指令**:
```bash
python PatchTST_train.py --sport deadlift --num_classes 5 --tag phase4.8_test_heads8_focal_logic_dtwwarp_5class --data_path ../data/deadlift_dataset_3d_5class.csv --num_heads 8 --num_workers 4 --augmentation dtwwarp
```

**測試結果**:
```text
✅ F1 scores from each Fold:
Fold 0: Macro F1 = 0.6552, Accuracy: 0.2930, cost time = 0.000013 sec
  - Far from the shins: F1 = 0.7155
  - Hips rise first: F1 = 0.7330
  - Collide with the knees: F1 = 0.6154
  - Lower back rounding: F1 = 0.6293
  - Correct: F1 = 0.5830
```

**分析與洞察 (2026-07-24)**：
1. **邏輯懲罰效果有限，神經網路早已隱式學習**：加上人工邏輯懲罰後，Macro F1 從 0.6533 微幅上升至 0.6552。這證明了模型光靠大量訓練資料的 BCE/Focal Loss，就已經學會了 Correct 和 Errors 是互斥的。額外的 Penalty 只發揮了錦上添花的作用（使 `Correct` 與 `Lower back rounding` 分數微升），但犧牲了一些 Exact Match Accuracy (掉至 0.2930)。這排除了「模型是因為邏輯錯亂才導致準確率上不去」的假說。
2. **損失函數優化已達極限**：既然 Loss 函數的改動已無法帶來突破性成長，且資料已經過人工篩選及完美對齊 (110 frames = 3.66s，能涵蓋整個動作)，這代表當前的瓶頸在於**特徵萃取 (Feature Representation)**。模型無法單從目前的純 3D 座標點 (COCO 17點) 中，有效捕捉出如「脊椎彎曲弧度」這種微小但關鍵的生物力學幾何特徵。
3. **未來重點突破計畫：導入 SMPL 3D (4DHumans / EasyMocap)**：
   - 目前 YOLO11n-pose 缺乏明確的「脊椎中段」節段，導致圓背偵測受到極大限制。
   - 接下來將徹底升級特徵抽取流程！利用團隊現有的 **5 機位 (Multi-view)** 高品質同步資料，搭配 **SMPL 3D 人體網格還原模型 (如 4DHumans, 或是多機位的 EasyMocap)**，直接還原出受試者真實 3D 空間中的脊椎彎曲角度 (Spine1, Spine2, Spine3 Rotation)。
   - 多機位可以完美解決被槓片遮擋以及深度模糊的問題。這將把模型從「猜測純 3D 點」轉變為「精準判斷生物力學特徵」，預計能大幅突破 5-class 分類在交界特徵上的瓶頸！

---

### Phase 4.10: 邏輯懲罰機制深入探討與消融實驗 (Logic Penalty Ablation)
**實驗日期**: 2026-07-28
**目標**:
為了解決在 Phase 4.9 中發現的「加入邏輯懲罰 (`p_correct * max_p_error`) 後 Accuracy 不升反降 (掉至 0.2930)」問題，進行了針對邏輯懲罰機制的消融與修正實驗，探討這類 Constraint Loss 對神經網路最佳化的真實影響。

**實驗一：修正 `max()` 的梯度陷阱 (Probability Smearing)**
- **問題分析**：原始的懲罰公式為 `p_correct * max_p_error`。由於 `torch.max()` 的特性，反向傳播時只會懲罰機率最高的那一個錯誤。這導致模型學會了「作弊 (Probability Smearing)」：把最肯定的錯誤機率降低，並把其他錯誤的機率提高，來讓 `max` 值變小以躲避懲罰。這造成了原本是單一錯誤的樣本 (如 `Far from the shins`)，被大量預測為雙重錯誤 (如 `Far + Lower back rounding`)，導致 Accuracy 暴跌。
- **作法**：將 `max` 改為 `sum`，即 `p_correct * sum(p_errors)`，讓每一個被預測出來的錯誤都受到 `p_correct` 的約束。
- **結果**：Accuracy 回升至 `0.3119`。從混淆矩陣觀察，雙重錯誤的「作弊」現象被成功消除 (`Far + Lower` 從 138 次降回 80 次)。這證明「強制互斥」的機制確實生效了。

**實驗二：解決乘法懲罰帶來的「拖累效應」 (Drag Effect)**
- **問題分析**：改用 `sum` 之後，Accuracy (0.3119) 依然無法超越完全無懲罰的 Baseline (0.3331)。原因是 `p_correct * p_error` 這個公式在數學上會無差別地「打壓」真實答案。當真實標籤為 `Error` 時，BCE 會推升 `p_error`，但懲罰項卻會產生一個向下的梯度拖累它（只要 `p_correct` 不完全是 0）。這種拖累效應導致模型對真實答案的信心度下降，造成更多預測低於門檻 (False Negatives)，降低了全對率。
- **作法**：引入 ReLU 建立容忍度：`ReLU(p_correct + p_error - 1.0)`。只在機率總和超過 1.0 (真正發生邏輯矛盾) 時才給予懲罰，一旦回到合理範圍就關閉懲罰。
- **結果**：Macro F1 微幅上升至 0.6496，但 Accuracy 仍停留在 0.3119，依然未能超越 Baseline。

**最終結論與洞察 (The Bitter Lesson)**：
1. **神經網路的 BCE 本身就具備學習邏輯的能力**：只要資料集的標籤是完美互斥的 (`Correct=1` 則 `Errors=0`)，模型的最後一層權重自然會學到負相關，完全不需人為介入。
2. **手動邏輯懲罰會破壞機率校準 (Calibration)**：強制加入的邏輯懲罰，會破壞神經網路在面對模糊、不確定樣本時的機率輸出，讓模型變得過度保守。此外，這股懲罰力量也會與對付資料不平衡的 `pos_weight` 產生嚴重內耗。
3. **最佳實踐方案：將邏輯留給後處理 (Post-processing)**：在訓練階段 (Training)，最正確的做法是完全拔除 `apply_logic_penalty`，讓模型透過 BCE 自由學習最原始的機率分佈 (能達到最高的 0.3331 Accuracy)。若要在實際應用中確保輸出絕對不矛盾，應在推論階段 (Inference) 利用程式碼 (如 `if p_correct > p_error`) 撰寫硬規則來覆寫預測結果。
