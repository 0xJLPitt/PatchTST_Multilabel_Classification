# Permutation Importance 特徵重要性分析腳本

我們已經建立了特徵排列重要性（Permutation Importance）的分析腳本，位於：`patchTST/permutation_importance.py`。
這份腳本專為你的 **Multi-label 模型**設計，以下是幾個關鍵的設計亮點與使用說明：

## 實作細節說明

1. **改用 Per-class F1 評估指標**
   - 由於模型是 Multi-label（多標籤分類），腳本內部統一採用 `f1_score(..., average=None)`。
   - 這樣一來，腳本會同時算出所有錯誤動作（例如 Deadlift 的 4 種錯誤）各自的 Baseline F1 與 Permuted F1，而不再只給出單一一個平均分數。

2. **迴圈架構**
   - 外層迴圈會讀取你透過 `--model_paths` 傳入的權重檔（支援單一模型，也支援傳入多個對照組模型）。
   - 內層迴圈則自動根據 Dataset 決定的特徵維度（`dim`，例如 60 或 70），逐一對各個特徵（Feature Index）進行干擾測試。

3. **時間序列洗牌邏輯（核心關鍵）與自動還原**
   - 在內層迴圈中，我們使用 `inputs_permuted[:, :, i] = inputs[torch.randperm(batch_size), :, i]` 針對 **Batch 維度**上的特定特徵進行打亂。
   - 這樣做能**完美保留該特徵在時間序列上的順序與動態變化**，只是把樣本 A 的該特徵完整抽換成樣本 B 的該特徵，藉此打破特徵與錯誤動作間的關聯性。
   - 由於每次測試特定特徵時，都會從乾淨的 `inputs.clone()` 開始操作，因此測試完畢後會**自動還原**，不影響後續特徵的評估。

4. **為每個錯誤獨立記錄與排序**
   - 特徵重要性定義為 `Baseline F1 - Permuted F1`。
   - 在最終輸出的 CSV 中，每一個錯誤類別都會有自己專屬的 Importance 欄位（例如 `Away_from_shins`, `Hips_rise_first` 等），讓你一目瞭然每個特徵對哪個錯誤特別關鍵。
   - 最終會計算所有類別重要性的 `Mean_Importance` 並進行降冪排序，自動匯出為 CSV 檔。

## 如何執行腳本？

你可以直接傳入這個 Multi-label 模型的 `.pth` 檔。假設你的運動類型是 `deadlift`，可以這樣下指令：

```bash
cd /home/pitt_huang/workspace/fitness_action_recognition/patchTST

python permutation_importance.py \
    --sport deadlift \
    --model_paths /home/pitt_huang/workspace/fitness_action_recognition/patchTST/models/deadlift/TST_Deadlift_3D/phase4.8_test_heads8_dtwwarp/PatchTST_model_fold0.pth \
    --num_heads 8 \
    --output_csv deadlift_feature_importance.csv
```

> **注意**：指令中加入了 `--num_heads 8` 來配合你這版使用 8 heads 訓練的模型。如果未來有其他對照組模型，也可以在 `--model_paths` 後面接續加上第二個、第三個路徑，腳本都能自動處理並合併為一份 CSV。



python permutation_importance.py \
    --sport squat \
    --model_paths /home/pitt_huang/workspace/fitness_action_recognition/patchTST/models/squat/TST_Squat/phase4.8_test_heads8_squat2/PatchTST_model_fold0.pth \
    --num_heads 8 \
    --output_csv squat_feature_importance.csv