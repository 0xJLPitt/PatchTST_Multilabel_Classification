#!/bin/bash

echo "Starting Ablation Studies for LSTM Overfitting..."

PYTHON_CMD="/home/pitt_huang/.conda/envs/cu13/bin/python"

# 1. Higher Dropout
echo "Running Ablation: Higher Dropout (0.5)"
$PYTHON_CMD train.py --tag ablation_dropout_0.5 --dropout 0.5 --num_workers 4

# 2. Higher Weight Decay
echo "Running Ablation: Higher Weight Decay (1e-3)"
$PYTHON_CMD train.py --tag ablation_wd_1e-3 --weight_decay 1e-3 --num_workers 4

# 3. Data Augmentation (jittering)
echo "Running Ablation: Data Augmentation (jittering)"
$PYTHON_CMD train.py --tag ablation_aug_jittering --aug_type jittering --num_workers 4

# 4. Reduced Model Complexity
echo "Running Ablation: Smaller Model (Hidden 64, Layers 1)"
$PYTHON_CMD train.py --tag ablation_small_model --hidden_size 64 --num_layers 1 --num_workers 4

echo "Ablation Studies Completed."
