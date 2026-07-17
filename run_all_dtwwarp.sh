#!/bin/bash
export CONDA_ENV="cu13"
CONDA_RUN="/home/pitt_huang/miniforge3/bin/conda run -n $CONDA_ENV python train.py"

echo "=== Running MultiCLS with dtwwarp ==="
cd /home/pitt_huang/workspace/fitness_action_recognition/MultiCLS
$CONDA_RUN --num_workers 4 --augmentation dtwwarp --tag multicls_dtwwarp_best --diversity_weight 0.5 --hidden_dim 256 --num_layers 2 > dtwwarp_run.log 2>&1

echo "=== Running iTransformer with dtwwarp ==="
cd /home/pitt_huang/workspace/fitness_action_recognition/iTransformer
$CONDA_RUN --d_model 32 --dropout 0.5 --pooling flatten --head_dim 128 --num_workers 4 --augmentation dtwwarp --tag itransformer_dtwwarp_best > dtwwarp_run.log 2>&1

echo "=== Running FCN-1D with dtwwarp ==="
cd /home/pitt_huang/workspace/fitness_action_recognition/FCN-1D
$CONDA_RUN --num_workers 4 --augmentation dtwwarp --tag fcn1d_dtwwarp_best > dtwwarp_run.log 2>&1

echo "All runs completed."
