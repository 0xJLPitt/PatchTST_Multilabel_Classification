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

---

## 🛠️ 第 2.5 階段模型優化紀錄 (Optimization Phase 2.5: 最強資料增強防護網)

**優化目標**：解決第二階段發現的「嚴重過度擬合 (Overfitting)」，在維持 Channel Independence 架構且具備超大容量 MLP 分類器的情況下，加入強力干擾，強迫模型無法死背訓練集特徵。

1. **隨機縮放 (Random Scaling)**：
   每次訓練時隨機將 3D 座標乘上 `0.9` 到 `1.1` 的常數。這能在物理意義上模擬「高矮胖瘦不同的受試者」，強迫模型專注於動作軌跡，而非特定手腳長度的絕對座標。
2. **溫和雜訊 (Jittering = 0.01)**：
   加回高斯雜訊，但為保護特徵將標準差調降為 `0.01`。這能確保同一個動作每次出現的座標都在微小抖動，讓模型無法「精確記憶」數值。
3. **隨機遮蔽 (Random Masking / Dropout 5%)**：
   以 5% 的機率隨機將某些座標數值歸零。這能模擬攝影機視角發生的「視覺遮蔽（如手被擋住）」，強迫模型必須依賴其他關節進行綜合判斷，杜絕依賴單一關節的捷徑學習。


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
