import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler

class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, embed_dim, stride):
        super().__init__()
        # 在 Channel Independence 模式下，輸入維度永遠是 1
        self.proj = nn.Linear(patch_len * 1, embed_dim)
        self.stride = stride
        self.patch_len = patch_len

    def forward(self, x):
        # x shape: (B*C, T, 1)
        B_C, T, _ = x.shape
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride) 
        # x shape: (B*C, num_patches, 1, patch_len)
        x = x.reshape(B_C, -1, self.patch_len) # 展平 patch
        x = self.proj(x) # (B*C, num_patches, embed_dim)
        return x

class PatchTSTClassifier(nn.Module):
    def __init__(self, input_dim=52, num_classes=4, input_len=100, patch_len=10, 
                embed_dim=256, num_heads=4, num_layers=4, dropout=0.3, stride=1):
        super().__init__()
        
        # 修正點：這裡傳入 1，因為每個通道獨立處理
        self.patch_embed = PatchEmbedding(patch_len, embed_dim, stride)
        
        num_patches = (input_len - patch_len) // stride + 1
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim * num_patches * embed_dim), # 這裡要改成 51200
            nn.Linear(input_dim * num_patches * embed_dim, num_classes)
        )

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        
        # --- 論文核心：Channel Independence ---
        # 1. 重排維度: (B, T, C) -> (B, C, T) -> (B*C, T, 1)
        x = x.permute(0, 2, 1).reshape(B * C, T, 1)
        
        # 2. Patching & Embedding
        x = self.patch_embed(x)  # (B*C, num_patches, embed_dim)
        
        # 3. Transformer
        x = x + self.pos_embed
        x = self.transformer(x)
        
        # 4. 聚合資訊 (Readout)
        # 先做時間維度平均 (Global Average Pooling over patches)
        # x = x.mean(dim=1)  # (B*C, embed_dim)
        
        # 再做通道間的聚合: (B*C, embed_dim) -> (B, C, embed_dim) -> (B, embed_dim)
        # x = x.view(B, C, -1).mean(dim=1) 
        x = x.view(B, -1)
        # 5. 分類層
        return self.classifier(x)

def get_angle(a, b, c):
    """Calculates angle ABC at vertex B."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab, cb = a - b, c - b
    norm_ab, norm_cb = np.linalg.norm(ab), np.linalg.norm(cb)
    if norm_ab == 0 or norm_cb == 0: return np.nan
    cosv = np.clip(np.dot(ab, cb) / (norm_ab * norm_cb), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cosv))
    return angle_deg if angle_deg <= 180 else 360 - angle_deg

def distance_point_to_line(p, p1, p2):
    """Perpendicular distance from point p to line passing through p1 and p2."""
    p, p1, p2 = np.array(p), np.array(p1), np.array(p2)
    line_vec = p2 - p1
    if np.all(line_vec == 0): return np.linalg.norm(p - p1)
    line_len = np.linalg.norm(line_vec)
    return np.linalg.norm(np.cross(line_vec, p1 - p)) / line_len

def angle_line_to_line(p1, p2, p3, p4):
    """Angle between extended line (p1, p2) and line (p3, p4)."""
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p4) - np.array(p3)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return np.nan
    cosv = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cosv))
    return angle_deg if angle_deg <= 90 else 180 - angle_deg

# --- Normalization Techniques (from step8) ---

def normalize_to_neg1_1(data):
    min_val = np.min(data)
    max_val = np.max(data)
    scale = max_val - min_val
    if scale == 0:
        scale = 1.0
    return 2 * (data - min_val) / scale - 1.0

def z_score_normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    if std < 1e-8:
        z = np.zeros_like(data)
    else:
        z = (data - mean) / std
    return normalize_to_neg1_1(z)

def variation_normalize(data):
    out = np.zeros(len(data))
    out[1:] = data[:-1] - data[1:]
    return normalize_to_neg1_1(out)

def variation_acceleration_normalize(data):
    out = np.zeros(len(data))
    for i in range(2, len(data)):
        out[i] = (data[i] - data[i-1]) - (data[i-1] - data[i-2])
    return normalize_to_neg1_1(out)

def variation_ratio_normalize(data, eps=1e-3):
    out = np.zeros(len(data))
    for i in range(1, len(data)):
        prev = data[i - 1]
        # Treat near-zero denominators as unstable, not only exact zeros.
        # Distance-like features can be nudged close to zero by augmentation,
        # which would otherwise turn a small absolute change into an enormous
        # ratio feature and destabilize training.
        out[i] = (prev - data[i]) / prev if abs(prev) >= eps else 0
    return normalize_to_neg1_1(out)

def remove_outliers_and_interpolate(data):
    """Simple 3-sigma outlier removal and 1D interpolation."""
    if len(data) < 3: return data
    # Use nanmean and nanstd to handle existing nan values in the input data
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(data)
        std = np.nanstd(data)
    
    # If the whole column is nan, nanmean returns nan
    if np.isnan(mean):
        # Fill with a default value (e.g. 0.0) to prevent propagates of NaNs
        return np.zeros_like(data)
        
    if std == 0 or np.isnan(std):
        std = 1e-8
        
    lower, upper = mean - 3 * std, mean + 3 * std
    clean = np.copy(data)
    
    # Ignore nan warning during comparison
    with np.errstate(invalid='ignore'):
        clean[(data < lower) | (data > upper)] = np.nan
        
    valid = ~np.isnan(clean)
    if np.sum(valid) > 1:
        idx = np.arange(len(clean))
        return np.interp(idx, idx[valid], clean[valid])
    elif np.sum(valid) == 1:
        # If only one valid element, return an array filled with that element
        return np.full_like(data, clean[valid][0])
    return np.full_like(data, mean)

import json

# --- Main Preprocessing & Prediction Pipeline ---

def extract_raw_features(video_path, bar_dict, rear_ske_dict, top_ske_dict, angle_dicts=None):
    """
    Extracts time-series features from raw coordinates.
    Returns a DataFrame with calculated metrics for each frame.
    """
    features = []
    # Intersection of all frames
    frames = sorted(set(bar_dict.keys()) & set(rear_ske_dict.keys()) & set(top_ske_dict.keys()))
    
    # Joint mappings based on expected inputs
    # Rear: 0:L_SHO, 1:R_SHO, 2:L_ELB, 3:R_ELB, 4:L_WRI, 5:R_WRI
    # Top: 0:L_SHO, 1:R_SHO, 2:L_HIP, 3:R_HIP, 4:L_ELB, 5:R_ELB, 6:L_WRI, 7:R_WRI
    
    for f in frames:
        try:
            bar_d = bar_dict[f]
            rear_d = rear_ske_dict[f]
            top_d = top_ske_dict[f]
            if len(rear_d) < 12 or len(top_d) < 16: continue

            # 1. Bar Features
            bar_x, bar_y = bar_d[0], bar_d[1]
            bar_ratio = bar_y / bar_x if bar_x != 0 else 0

            # 2. Rear Features
            rl_sho, rr_sho = rear_d[0:2], rear_d[2:4]
            rl_elb, rr_elb = rear_d[4:6], rear_d[6:8]
            rl_wri, rr_wri = rear_d[8:10], rear_d[10:12]

            if angle_dicts:
                l_elb_angle = angle_dicts.get("left_elbow", {}).get(f, np.nan)
                r_elb_angle = angle_dicts.get("right_elbow", {}).get(f, np.nan)
                l_sho_angle = angle_dicts.get("left_shoulder", {}).get(f, np.nan)
                r_sho_angle = angle_dicts.get("right_shoulder", {}).get(f, np.nan)
            else:
                l_elb_angle = get_angle(rl_wri, rl_elb, rl_sho)
                r_elb_angle = get_angle(rr_wri, rr_elb, rr_sho)
                # Shoulder angle: Angle between (L_SHO->R_SHO) and (L_SHO->L_ELB)
                l_sho_angle = angle_line_to_line(rl_sho, rr_sho, rl_sho, rl_elb)
                r_sho_angle = angle_line_to_line(rr_sho, rl_sho, rr_sho, rr_elb)
            
            l_sho_y, r_sho_y = rl_sho[1], rr_sho[1]

            # 3. Top Features
            tl_sho, tr_sho = top_d[0:2], top_d[2:4]
            tl_hip, tr_hip = top_d[4:6], top_d[6:8]
            tl_elb, tr_elb = top_d[8:10], top_d[10:12]
            tl_wri, tr_wri = top_d[12:14], top_d[14:16]

            if angle_dicts:
                l_torso_arm = angle_dicts.get("left_torso-arm", {}).get(f, np.nan)
                r_torso_arm = angle_dicts.get("right_torso-arm", {}).get(f, np.nan)
            else:
                l_torso_arm = get_angle(tl_hip, tl_sho, tl_elb)
                r_torso_arm = get_angle(tr_hip, tr_sho, tr_elb)
            
            # Wrist distance to extended shoulder line
            l_dist = distance_point_to_line(tl_wri, tl_sho, tr_sho)
            r_dist = distance_point_to_line(tr_wri, tl_sho, tr_sho)

            row = [
                f, bar_x, bar_y, bar_ratio, 
                l_elb_angle, r_elb_angle, 
                l_sho_angle, r_sho_angle, 
                l_sho_y, r_sho_y,
                l_torso_arm, r_torso_arm, 
                l_dist, r_dist
            ]
            features.append(row)
        except Exception as e:
            print(f"Skipping frame {f} due to error: {e}")
            
    cols = [
        "frame", "bar_x", "bar_y", "bar_ratio",
        "left_elbow", "right_elbow",
        "left_shoulder", "right_shoulder",
        "left_shoulder_y", "right_shoulder_y",
        "left_torso-arm", "right_torso-arm",
        "left_dist", "right_dist"
    ]
    df = pd.DataFrame(features, columns=cols)
        
    return df
