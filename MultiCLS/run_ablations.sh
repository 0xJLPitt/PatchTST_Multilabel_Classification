#!/bin/bash
set -e

cd /home/pitt_huang/workspace/fitness_action_recognition/MultiCLS

PYTHON="/home/pitt_huang/.conda/envs/cu13/bin/python"

echo "Starting Experiment 1: Feature Projection (Conv1D)"
$PYTHON train.py --tag 1_conv1d_proj --proj_type conv1d --num_workers 4

echo "Starting Experiment 2: Positional Encoding (Sinusoidal)"
$PYTHON train.py --tag 2_sinusoidal_pos --pos_enc sinusoidal --num_workers 4

echo "Starting Experiment 3: Loss Function (Focal Loss)"
$PYTHON train.py --tag 3_focal_loss --loss_type focal --num_workers 4

echo "Starting Experiment 4: Lower Complexity"
$PYTHON train.py --tag 4_lower_complexity --hidden_dim 128 --num_layers 2 --dropout 0.5 --num_workers 4

echo "Starting Experiment 5: Diversity Loss Weight (0.5)"
$PYTHON train.py --tag 5_div_weight_0.5 --diversity_weight 0.5 --num_workers 4

echo "Starting Experiment 6: Aggregation (Mean)"
$PYTHON train.py --tag 6_mean_aggregation --aggregation mean --num_workers 4

echo "All 6 experiments completed successfully!"
