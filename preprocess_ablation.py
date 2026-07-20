import pandas as pd
import ast
import torch
import numpy as np
import os

def calculate_angle(p1, p2, p3):
    # p1, p2, p3 are tensors of shape (N, 2)
    ba = p1 - p2
    bc = p3 - p2
    
    dot_prod = (ba * bc).sum(dim=-1)
    norm_ba = torch.norm(ba, dim=-1)
    norm_bc = torch.norm(bc, dim=-1)
    
    # Avoid division by zero
    denominator = norm_ba * norm_bc
    denominator = torch.clamp(denominator, min=1e-6)
    
    cosine_angle = dot_prod / denominator
    cosine_angle = torch.clamp(cosine_angle, -1.0, 1.0)
    
    return torch.acos(cosine_angle) * (180.0 / np.pi)

def process_dataset():
    csv_file = 'data/squat_dataset_2d.csv'
    print("Loading dataset...")
    df = pd.read_csv(csv_file)
    
    all_features = []
    all_labels = []
    subjects = []
    instances = []
    
    print("Parsing CSV...")
    for _, row in df.iterrows():
        if 'features' in row and 'label' in row:
            features = ast.literal_eval(str(row['features']))
            labels = ast.literal_eval(str(row['label']))
            subject = str(row['subject'])
            task = str(row['task']) if 'task' in row else '1'
            instance = f"{subject}_{task}"
            
            all_features.append(features)
            all_labels.append(labels)
            subjects.append(subject)
            instances.append(instance)

    # Convert to tensors
    features_t = torch.tensor(all_features, dtype=torch.float32) # (B, 110, 50)
    labels_t = torch.tensor(all_labels, dtype=torch.float32)     # (B, 5)
    
    B, T, C = features_t.shape
    
    print(f"Original features shape: {features_t.shape}")
    
    # Assuming standard indices based on data_produce.py
    # 6: Shoulder, 12: Hip, 14: Knee, 16: Ankle
    idx_shoulder = 6
    idx_hip = 12
    idx_knee = 14
    idx_ankle = 16
    
    def get_pt(idx):
        return features_t[:, :, 2*idx:2*idx+2] # (B, 110, 2)
        
    p_shoulder = get_pt(idx_shoulder)
    p_hip = get_pt(idx_hip)
    p_knee = get_pt(idx_knee)
    p_ankle = get_pt(idx_ankle)
    
    # 1. Depth: Hip Y - Ankle Y (or just distance)
    # Using Y-coordinate distance for depth (normalized)
    depth_feat = (p_hip[:, :, 1] - p_ankle[:, :, 1]).unsqueeze(-1) # (B, 110, 1)
    
    # 2. Velocity & Accel of Hip
    # Delta Y over time
    hip_y = p_hip[:, :, 1] # (B, 110)
    vel_hip = torch.zeros_like(hip_y)
    vel_hip[:, 1:] = hip_y[:, 1:] - hip_y[:, :-1]
    
    accel_hip = torch.zeros_like(hip_y)
    accel_hip[:, 2:] = vel_hip[:, 2:] - vel_hip[:, 1:-1]
    
    vel_accel_feat = torch.stack([vel_hip, accel_hip], dim=-1) # (B, 110, 2)
    
    # 3. Trunk features
    # Trunk vector: Shoulder - Hip
    trunk_vec = p_shoulder - p_hip # (B, 110, 2)
    trunk_len = torch.norm(trunk_vec, dim=-1, keepdim=True) # (B, 110, 1)
    trunk_feat = torch.cat([trunk_vec, trunk_len], dim=-1) # (B, 110, 3)
    
    # 4. Hip Flexion Angle
    # Angle between Shoulder-Hip and Knee-Hip
    hip_angle = calculate_angle(p_shoulder, p_hip, p_knee).unsqueeze(-1) # (B, 110, 1)
    
    # Combine datasets
    datasets = {
        'baseline': features_t,
        'depth': torch.cat([features_t, depth_feat], dim=-1),
        'velocity': torch.cat([features_t, vel_accel_feat], dim=-1),
        'trunk': torch.cat([features_t, trunk_feat], dim=-1),
        'hipflex': torch.cat([features_t, hip_angle], dim=-1),
        'all': torch.cat([features_t, depth_feat, vel_accel_feat, trunk_feat, hip_angle], dim=-1)
    }
    
    os.makedirs('data', exist_ok=True)
    
    for name, data_feat in datasets.items():
        save_path = f'data/squat_ablation_{name}.pt'
        save_dict = {
            'features': data_feat,
            'labels': labels_t,
            'subjects': subjects,
            'instances': instances
        }
        torch.save(save_dict, save_path)
        print(f"Saved {name} dataset to {save_path} with feature shape {data_feat.shape}")

if __name__ == "__main__":
    process_dataset()
