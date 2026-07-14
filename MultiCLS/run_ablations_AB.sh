#!/bin/bash
set -e

cd /home/pitt_huang/workspace/fitness_action_recognition/MultiCLS
PYTHON="/home/pitt_huang/.conda/envs/cu13/bin/python"

echo "Starting Exp A: dim=256, layers=2"
$PYTHON train.py --tag 10_div0.5_dim256_layers2 --diversity_weight 0.5 --hidden_dim 256 --num_layers 2 --dropout 0.3 --num_workers 4

echo "Starting Exp B: dim=128, layers=4"
$PYTHON train.py --tag 11_div0.5_dim128_layers4 --diversity_weight 0.5 --hidden_dim 128 --num_layers 4 --dropout 0.3 --num_workers 4

echo "All experiments completed!"
