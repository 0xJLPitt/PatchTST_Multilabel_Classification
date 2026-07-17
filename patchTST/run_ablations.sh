#!/bin/bash
# Ablation studies for data augmentations

export CONDA_ENV="cu13"
CONDA_RUN="/home/pitt_huang/miniforge3/bin/conda run -n $CONDA_ENV python PatchTST_train.py"

BASE_ARGS="--sport deadlift --split_mode instance_stratified --num_heads 8 --num_workers 4"

# 1. window_warping
$CONDA_RUN $BASE_ARGS --augmentation window_warping --tag phase4.8_test_heads8_window_warping

# 2. jittering
$CONDA_RUN $BASE_ARGS --augmentation jittering --tag phase4.8_test_heads8_jittering

# 3. spawner
$CONDA_RUN $BASE_ARGS --augmentation spawner --tag phase4.8_test_heads8_spawner

# 4. dtwwarp
$CONDA_RUN $BASE_ARGS --augmentation dtwwarp --tag phase4.8_test_heads8_dtwwarp

# 5. shapedtw
$CONDA_RUN $BASE_ARGS --augmentation shapedtw --tag phase4.8_test_heads8_shapedtw

# 6. discdtw
$CONDA_RUN $BASE_ARGS --augmentation discdtw --tag phase4.8_test_heads8_discdtw

echo "All 6 augmentation ablation studies completed successfully."
