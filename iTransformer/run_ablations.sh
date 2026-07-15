#!/bin/bash
cd /home/pitt_huang/workspace/fitness_action_recognition/iTransformer

LOG_FILE="ablation_results.log"
echo "--- iTransformer Ablation Results ---" > $LOG_FILE

PYTHON="/home/pitt_huang/.conda/envs/cu13/bin/python"

run_exp() {
    CMD=$1
    TAG=$2
    echo "Running: $TAG" | tee -a $LOG_FILE
    echo "CMD: $CMD" | tee -a $LOG_FILE
    
    # Run the command, capture the final F1 result
    OUTPUT=$($CMD)
    echo "$OUTPUT" | grep -E "Fold 0 Test F1" | tee -a $LOG_FILE
    echo "-----------------------------------" | tee -a $LOG_FILE
}

# 1. dim_feedforward
run_exp "$PYTHON train.py --d_model 32 --dropout 0.5 --pooling flatten --dim_feedforward 128 --tag ab_ff_128 --num_workers 4" "ab_ff_128"
run_exp "$PYTHON train.py --d_model 32 --dropout 0.5 --pooling flatten --dim_feedforward 64 --tag ab_ff_64 --num_workers 4" "ab_ff_64"

# 2. num_layers / nhead
run_exp "$PYTHON train.py --d_model 32 --dropout 0.5 --pooling flatten --num_layers 1 --tag ab_layer_1 --num_workers 4" "ab_layer_1"
run_exp "$PYTHON train.py --d_model 32 --dropout 0.5 --pooling flatten --nhead 2 --tag ab_head_2 --num_workers 4" "ab_head_2"

# 3. weight_decay
run_exp "$PYTHON train.py --d_model 32 --dropout 0.5 --pooling flatten --weight_decay 0.05 --tag ab_wd_0.05 --num_workers 4" "ab_wd_0.05"
run_exp "$PYTHON train.py --d_model 32 --dropout 0.5 --pooling flatten --weight_decay 0.1 --tag ab_wd_0.1 --num_workers 4" "ab_wd_0.1"

# 4. learning rate
run_exp "$PYTHON train.py --d_model 32 --dropout 0.5 --pooling flatten --lr 5e-4 --tag ab_lr_5e4 --num_workers 4" "ab_lr_5e4"
run_exp "$PYTHON train.py --d_model 32 --dropout 0.5 --pooling flatten --lr 1e-3 --tag ab_lr_1e3 --num_workers 4" "ab_lr_1e3"

# 5. classification head bottleneck
run_exp "$PYTHON train.py --d_model 32 --dropout 0.5 --pooling flatten --head_dim 128 --tag ab_head_dim_128 --num_workers 4" "ab_head_dim_128"
run_exp "$PYTHON train.py --d_model 32 --dropout 0.5 --pooling flatten --head_dim 64 --tag ab_head_dim_64 --num_workers 4" "ab_head_dim_64"

echo "All ablation experiments completed." >> $LOG_FILE
