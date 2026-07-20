import os
import sys
import json
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dataset.tools.squat_tool.utils import calculate_angle1, interpolate_features
from dataset.tools.squat_tool.data_split import (
    process_delta, process_delta_ratio, process_zscore, normalize_to_neg1_1
)
from dataset.tools.squat_tool.hampel import hampel_filter, interpolate_hampel_dict

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Ensure signal length > padlen (3 * max(len(a), len(b)))
    padlen = 3 * max(len(a), len(b))
    if data.shape[0] <= padlen:
        return data  # Too short to filter
    y = filtfilt(b, a, data, axis=0)
    return y

def clean_skeleton_data(input_path, expected_joints=17):
    """ Read YOLO skeleton and clean with Hampel + interpolation """
    data = {}
    if not os.path.exists(input_path):
        return {}

    with open(input_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            try:
                frame_idx = int(parts[0])
                joint_idx = int(parts[1])
                x, y = float(parts[2]), float(parts[3])
                
                if frame_idx not in data:
                    data[frame_idx] = {}
                data[frame_idx][joint_idx] = (x, y)
            except ValueError:
                continue

    all_frames = sorted(data.keys())
    if not all_frames:
        return {}
        
    all_points = []
    valid_frames = []
    
    for f_idx in all_frames:
        joints = data[f_idx]
        p_list = []
        for j_idx in range(expected_joints):
            if j_idx in joints:
                p_list.append(joints[j_idx])
            else:
                p_list.append((np.nan, np.nan))
        all_points.append(p_list)
        valid_frames.append(f_idx)

    all_points = np.array(all_points) # (num_frames, expected_joints, 2)
    mask = np.zeros_like(all_points[:, :, 0], dtype=bool)

    # Apply Hampel filter independently for each joint and each axis
    for joint_idx in range(expected_joints):
        x_series = all_points[:, joint_idx, 0]
        y_series = all_points[:, joint_idx, 1]
        x_outliers = hampel_filter(x_series)
        y_outliers = hampel_filter(y_series)
        mask[:, joint_idx] = x_outliers | y_outliers

    cleaned_points = all_points.copy().astype(float)
    cleaned_points[mask] = np.nan

    # Convert to dict for interpolation
    results = {}
    for i, frame in enumerate(valid_frames):
        coords_flat = []
        for j in range(expected_joints):
            coords_flat.extend([cleaned_points[i, j, 0], cleaned_points[i, j, 1]])
        results[frame] = coords_flat
        
    # Interpolate
    results = interpolate_hampel_dict(results)
    
    # Repack to {frame: {joint: (x, y)}}
    final_data = {}
    for f_idx, flat_coords in results.items():
        final_data[f_idx] = {}
        for j in range(expected_joints):
            final_data[f_idx][j] = (flat_coords[2*j], flat_coords[2*j+1])
            
    return final_data

def clean_barbell_data(input_path):
    """ Read YOLO coordinates and clean Barbell X, Y with Hampel + interpolation """
    data = {}
    if not os.path.exists(input_path):
        return {}

    frames = []
    values = []
    with open(input_path, "r") as file:
        for line in file:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            try:
                frame_count = int(parts[0])
                # Only use center x and center y
                x, y = float(parts[1]), float(parts[2])
                frames.append(frame_count)
                values.append([x, y])
            except ValueError:
                continue

    if not values:
        return {}

    values = np.array(values)
    x_outliers = hampel_filter(values[:, 0])
    y_outliers = hampel_filter(values[:, 1])

    outlier_frames = x_outliers | y_outliers
    values_filtered = values.copy().astype(float)
    values_filtered[outlier_frames] = np.nan

    results = {frame: values_filtered[i].tolist() for i, frame in enumerate(frames)}
    
    # Interpolate
    results = interpolate_hampel_dict(results)
    return results

def process_dataset(annotations_file, dataset_dir, output_dir):
    with open(annotations_file, 'r', encoding='utf-8') as f:
        annot_data = json.load(f)["data"]
        
    error_order = [
        "下蹲深度不足",
        "下蹲時膝蓋過度主導",
        "下蹲時髖部過度主導",
        "骨盆後傾",
        "臀部上升過快"
    ]
    
    # We will aggregate by subject
    # subject -> list of feature dicts
    subject_data = defaultdict(list)
    
    for task in annot_data:
        task_name = task.get("task_name")
        subject = task.get("subject")
        annotations = task.get("annotations", [])
        
        if not annotations:
            continue
            
        result = annotations[0].get("result", {})
        global_errors = result.get("errors", [])
        clips = result.get("clips", [])
        
        # Determine multi-hot label
        label_vec = [0, 0, 0, 0, 0]
        if "正常" not in global_errors:
            for i, err in enumerate(error_order):
                if err in global_errors:
                    label_vec[i] = 1
                    
        # Load skeleton and barbell for this recording
        recording_dir = os.path.join(dataset_dir, task_name)
        skeleton_file = os.path.join(recording_dir, "yolo_skeleton.txt")
        barbell_file = os.path.join(recording_dir, "yolo_coordinates.txt")
        
        if not os.path.exists(skeleton_file) or not os.path.exists(barbell_file):
            print(f"Skipping {recording_dir} (Missing files)")
            continue
            
        print(f"Processing {recording_dir}...")
        skeleton_data = clean_skeleton_data(skeleton_file)
        barbell_data = clean_barbell_data(barbell_file)
        
        for clip in clips:
            start_frame = clip["start_frame"]
            end_frame = clip["end_frame"]
            
            clip_features = []
            physical_raw_features = []
            valid_start_x = None
            initial_trunk_len = None
            
            for frame in range(start_frame, end_frame + 1):
                if frame not in skeleton_data or frame not in barbell_data:
                    continue
                    
                skel = skeleton_data[frame]
                bar = barbell_data[frame]
                
                # 1. Left knee angle (Hip(11) - Knee(13) - Ankle(15))
                l_knee_angle = calculate_angle1(skel[11][0], skel[11][1], skel[13][0], skel[13][1], skel[15][0], skel[15][1])
                # 2. Left hip angle (Shoulder(5) - Hip(11) - Knee(13))
                l_hip_angle = calculate_angle1(skel[5][0], skel[5][1], skel[11][0], skel[11][1], skel[13][0], skel[13][1])
                # 3. Right knee angle (Hip(12) - Knee(14) - Ankle(16))
                r_knee_angle = calculate_angle1(skel[12][0], skel[12][1], skel[14][0], skel[14][1], skel[16][0], skel[16][1])
                # 4. Right hip angle (Shoulder(6) - Hip(12) - Knee(14))
                r_hip_angle = calculate_angle1(skel[6][0], skel[6][1], skel[12][0], skel[12][1], skel[14][0], skel[14][1])
                # 5. Left arm-torso angle (Elbow(7) - Shoulder(5) - Hip(11))
                l_arm_torso_angle = calculate_angle1(skel[7][0], skel[7][1], skel[5][0], skel[5][1], skel[11][0], skel[11][1])
                # 6. Right arm-torso angle (Elbow(8) - Shoulder(6) - Hip(12))
                r_arm_torso_angle = calculate_angle1(skel[8][0], skel[8][1], skel[6][0], skel[6][1], skel[12][0], skel[12][1])
                
                # 7. Barbell X (displacement)
                bar_x = bar[0]
                if valid_start_x is None:
                    valid_start_x = bar_x
                bar_x_disp = bar_x - valid_start_x
                
                # 8. Barbell Y
                bar_y = bar[1]
                
                # 9. Knee-Hip Y Diff (Right side: Knee(14) - Hip(12))
                r_knee_hip_y_diff = skel[14][1] - skel[12][1]
                
                # 10. Knee-Hip X Diff (Right side: Knee(14) - Hip(12))
                r_knee_hip_x_diff = skel[14][0] - skel[12][0]
                
                feat = [
                    l_knee_angle, l_hip_angle, r_knee_angle, r_hip_angle,
                    l_arm_torso_angle, r_arm_torso_angle, 
                    bar_x_disp, bar_y, r_knee_hip_y_diff, r_knee_hip_x_diff
                ]
                clip_features.append(feat)
                
                # --- Scale-Invariant Physical Features ---
                r_shoulder = skel[6]
                r_hip = skel[12]
                r_knee = skel[14]
                
                trunk_vec_x = r_shoulder[0] - r_hip[0]
                trunk_vec_y = r_shoulder[1] - r_hip[1]
                trunk_len = np.sqrt(trunk_vec_x**2 + trunk_vec_y**2)
                
                if initial_trunk_len is None:
                    initial_trunk_len = trunk_len if trunk_len > 0 else 1.0
                
                trunk_len_ratio = trunk_len / initial_trunk_len
                trunk_vec_x_ratio = trunk_vec_x / initial_trunk_len
                trunk_vec_y_ratio = trunk_vec_y / initial_trunk_len
                knee_hip_y_ratio = (r_knee[1] - r_hip[1]) / initial_trunk_len
                knee_hip_x_ratio = (r_knee[0] - r_hip[0]) / initial_trunk_len
                # Keep all 5 essential features
                physical_feat = [
                    trunk_len_ratio, trunk_vec_x_ratio, trunk_vec_y_ratio,
                    knee_hip_y_ratio, knee_hip_x_ratio
                ]
                physical_raw_features.append(physical_feat)
                
            if len(clip_features) < 15:
                print(f"Clip {start_frame}-{end_frame} too short, skipping.")
                continue
                
            clip_features = np.array(clip_features)
            physical_raw_features = np.array(physical_raw_features)
            
            # Filtering
            clip_features = butter_lowpass_filter(clip_features, cutoff=1, fs=30, order=4)
            physical_raw_features = butter_lowpass_filter(physical_raw_features, cutoff=1, fs=30, order=4)
            
            # Interpolation (110 frames)
            interpolated = interpolate_features(clip_features, 110)
            physical_interpolated = interpolate_features(physical_raw_features, 110)
            
            # Package for data_split functions which expect {id: data}
            filtered_interpolated = {"0": interpolated}
            
            delta_feature = process_delta(filtered_interpolated)
            delta_square_feature = process_delta(delta_feature)
            zscore_feature = process_zscore(filtered_interpolated)
            delta_ratio_feature = process_delta_ratio(filtered_interpolated)
            
            fn = normalize_to_neg1_1(filtered_interpolated["0"])
            fdn = normalize_to_neg1_1(delta_feature["0"])
            fd2n = normalize_to_neg1_1(delta_ratio_feature["0"])
            fzn = normalize_to_neg1_1(zscore_feature["0"])
            fdsn = normalize_to_neg1_1(delta_square_feature["0"])
            
            # Calculate physical delta
            physical_delta = np.vstack([np.zeros(physical_interpolated.shape[1]), np.diff(physical_interpolated, axis=0)])
            
            # Combine 5 transformations -> [110, 50] array + physical -> [110, 60]
            all_feat = np.concatenate([fn, fdn, fd2n, fzn, fdsn, physical_interpolated, physical_delta], axis=-1)
            
            subject_data[subject].append({
                "subject": subject,
                "task": task_name,
                "clip_idx": clip.get("clip_index"),
                "features": str(all_feat.tolist()),
                "label": str(label_vec)
            })
            
    all_data = []
    for s_data in subject_data.values():
        all_data.extend(s_data)
        
    # Here output_dir actually contains the file path (e.g. ./data/squat_dataset_2d.csv)
    out_dir_path = os.path.dirname(output_dir)
    if out_dir_path:
        os.makedirs(out_dir_path, exist_ok=True)
        
    pd.DataFrame(all_data).to_csv(output_dir, index=False)
    
    print(f"Data saved to {output_dir}")
    print(f"Total clips: {len(all_data)}")

if __name__ == "__main__":
    annot_file = "/cats/shared/squat_dataset_0527/annotations.json"
    data_dir = "/cats/shared/squat_dataset_0527"
    out_csv = "./data/squat_dataset_2d.csv"
    process_dataset(annot_file, data_dir, out_csv)
