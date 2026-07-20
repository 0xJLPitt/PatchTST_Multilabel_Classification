#!/bin/bash
# Ablation study for Squat Posterior Pelvic Tilt features

# Ensure we are in the correct directory
cd /home/pitt_huang/workspace/fitness_action_recognition

# 1. Baseline (original 50 features)
echo "Running Baseline..."
conda run -n cu13 python patchTST/PatchTST_train.py --sport squat --tag ablation_baseline --num_workers 4 --data_path data/squat_ablation_baseline.pt

# 2. Depth Feature
echo "Running Depth Ablation..."
conda run -n cu13 python patchTST/PatchTST_train.py --sport squat --tag ablation_depth --num_workers 4 --data_path data/squat_ablation_depth.pt

# 3. Velocity/Acceleration Feature
echo "Running Velocity Ablation..."
conda run -n cu13 python patchTST/PatchTST_train.py --sport squat --tag ablation_velocity --num_workers 4 --data_path data/squat_ablation_velocity.pt

# 4. Trunk Angle/Length Feature
echo "Running Trunk Ablation..."
conda run -n cu13 python patchTST/PatchTST_train.py --sport squat --tag ablation_trunk --num_workers 4 --data_path data/squat_ablation_trunk.pt

# 5. Hip Flexion Angle Feature
echo "Running Hip Flexion Ablation..."
conda run -n cu13 python patchTST/PatchTST_train.py --sport squat --tag ablation_hipflex --num_workers 4 --data_path data/squat_ablation_hipflex.pt

# 6. All Features Combined
echo "Running All Features Ablation..."
conda run -n cu13 python patchTST/PatchTST_train.py --sport squat --tag ablation_all --num_workers 4 --data_path data/squat_ablation_all.pt

echo "Ablation Study Completed."
