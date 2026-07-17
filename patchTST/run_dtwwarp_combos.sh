#!/bin/bash
export CONDA_ENV="cu13"
CONDA_RUN="/home/pitt_huang/miniforge3/bin/conda run -n $CONDA_ENV python PatchTST_train.py"
BASE_ARGS="--sport deadlift --split_mode instance_stratified --num_heads 8 --num_workers 4"

echo "Starting dtwwarp+window_warping"
$CONDA_RUN $BASE_ARGS --augmentation dtwwarp+window_warping --tag dtwwarp_window_warping

echo "Starting dtwwarp+jittering"
$CONDA_RUN $BASE_ARGS --augmentation dtwwarp+jittering --tag dtwwarp_jittering

echo "Starting dtwwarp+spawner"
$CONDA_RUN $BASE_ARGS --augmentation dtwwarp+spawner --tag dtwwarp_spawner

echo "Starting dtwwarp+shapedtw"
$CONDA_RUN $BASE_ARGS --augmentation dtwwarp+shapedtw --tag dtwwarp_shapedtw

echo "Starting dtwwarp+discdtw"
$CONDA_RUN $BASE_ARGS --augmentation dtwwarp+discdtw --tag dtwwarp_discdtw

echo "All dtwwarp combinations completed."
