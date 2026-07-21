# Fitness Action Recognition Rules

## Squat Model Training & Evaluation
- When training or evaluating Squat models (e.g., using `PatchTST_train_squat.py`), the model handles 5 multi-label classes.
- A basic 5x5 confusion matrix is insufficient for multi-label errors. You MUST always generate the "Multi-error Compound Confusion Matrix" (usually 15x15 or 17x17 depending on model predictions).
- This is done by running `generate_complex_cm_squat.py` and passing the model save directory as an argument.
- The training script `PatchTST_train_squat.py` has been updated to automatically run this script at the end of training. Always ensure this automated step completes successfully.
- For Squat, the correct data split ratio is generally **8:1:1** (Train:Val:Test).

