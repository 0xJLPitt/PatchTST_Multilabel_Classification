import os
import sys
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dataset.tools.interpolate import run_interpolation
from dataset.tools.Benchpress_tool.hampel import run_hampel_bar, run_hampel_yolo_ske_left_front
from dataset.tools.Deadlift_tool.data_produce import run_data_produce
from dataset.tools.Deadlift_tool.data_split import run_data_split

def pre_process(video_path: str):
    # Run the standard pipeline
    import time
    memo = {}
    
    def run_step(name, func, args, kwargs={}):
        t0 = time.time()
        res = func(*args, **kwargs)
        memo[name] = res
        print(f"[DeadliftProcessor] {name} time : {time.time() - t0:.2f}s")
        return res

    run_step("Interpolation", run_interpolation, [video_path])
    run_step("Hampel Bar", run_hampel_bar, [video_path], {"sport": 'deadlift'})
    run_step("Hampel Skeleton", run_hampel_yolo_ske_left_front, [video_path])
    run_step("Angle Data", run_data_produce, [video_path])
    res = run_step("Data Split", run_data_split, [video_path])
    return memo, res

# deadliftdim=8
# bar x, y
# knee, hip, torso-arm

# benchpress dim=12
# bar x, y
# shoulder(y, angle), torso-arm, elbow, distance(wrist to shoulder line)

def apply_augmentation(df):
    """
    Placeholder for data augmentation (e.g., jittering, scaling, time-warping).
    """
    # Example: return df + np.random.normal(0, 0.01, df.shape)
    return df

def generate_csv(dataset_dir, output_csv):
    import os
    import pandas as pd
    import numpy as np
    import json
    
    # Load multi-error mapping
    multi_error_path = os.path.join(dataset_dir, "multierror.json")
    from collections import defaultdict
    pass_list = defaultdict(set)
    clip_errors_map = {} # (subject, set, clip_idx) -> set of errors
    
    if os.path.exists(multi_error_path):
        with open(multi_error_path, 'r') as f:
            me_data = json.load(f)
            for subject, mistake_groups in me_data.items():
                for group in mistake_groups:
                    all_errors = {e["error"] for e in group}
                    for i, error_info in enumerate(group):
                        err_name = error_info["error"]
                        set_name = error_info["set"]
                        clips = error_info["clips"]
                        if i == 0:
                            for clip in clips:
                                clip_errors_map[(subject, set_name, int(clip))] = all_errors
                        else:
                            key = f"{subject}_{set_name}_{err_name}"
                            pass_list[key].update(clips)

    data = []
    
    error_order = [
        "Barbell_moving_away_from_the_shins",
        "Hips_rising_before_the_barbell_leaves_the_ground",
        "Barbell_colliding_with_the_knees",
        "Lower_back_rounding"
    ]
    
    # Process DeadliftDataset
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory {dataset_dir} not found.")
        return
        
    for label_dir in os.listdir(dataset_dir):
        full_label_dir = os.path.join(dataset_dir, label_dir)
        if not os.path.isdir(full_label_dir):
            continue
            
        for subject_dir in os.listdir(full_label_dir):
            subject_path = os.path.join(full_label_dir, subject_dir)
            if not os.path.isdir(subject_path):
                continue
                
            for set_dir in os.listdir(subject_path):
                set_path = os.path.join(subject_path, set_dir)
                if not os.path.isdir(set_path):
                    continue
                
                angle_3d_dir = os.path.join(set_path, "Angle", "3D")
                bar_dir = os.path.join(set_path, "Coordinate", "bar")
                if os.path.exists(angle_3d_dir) and os.path.isdir(angle_3d_dir):
                    for file in os.listdir(angle_3d_dir):
                        if file.endswith(".csv"):
                            clip_idx = file.replace("angle_", "").replace(".csv", "")
                            
                            # 1. 根據與 deadlift.py 相同的方式，依賴 multierror.json 過濾重複動作
                            key = f"{subject_dir}_{set_dir}_{label_dir}"
                            if key in pass_list and int(clip_idx) in pass_list[key]:
                                continue
                            
                            file_path = os.path.join(angle_3d_dir, file)
                            bar_file = os.path.join(bar_dir, f"bar_{clip_idx}.csv")
                            
                            print(f"Processing 3D Angle file: {file_path}")
                            try:
                                # 2. Construct the Multi-label
                                label_vec = [0, 0, 0, 0]
                                active_errors = set()
                                
                                # 若是 Correct 資料夾，則標籤全部為 0；否則才讀取錯誤標籤
                                if label_dir != 'Correct':
                                    if label_dir in error_order:
                                        active_errors.add(label_dir)
                                    
                                    me_key = (subject_dir, set_dir, int(clip_idx))
                                    if me_key in clip_errors_map:
                                        active_errors.update(clip_errors_map[me_key])
                                
                                for i, err in enumerate(error_order):
                                    if err in active_errors:
                                        label_vec[i] = 1
                                        
                                # 2. Start extracting and merging features
                                df_3d = pd.read_csv(file_path, header=None)
                                
                                # Drop nothing (Keep col 5 body length). Keep joints: 1: left knee angle, 2: left hip angle, 3: right knee angle, 4: right hip angle, 5: body length, 6: left arm-torso angle, 7: right arm-torso angle
                                # Note: index 0 is frame, so we skip it.
                                df_3d_filtered = df_3d.iloc[:, [1, 2, 3, 4, 5, 6, 7]]
                                
                                # Add bar_x and bar_y from bar file
                                if os.path.exists(bar_file):
                                    df_bar = pd.read_csv(bar_file, header=None)
                                    features_bar_arr = df_bar.iloc[:, [1, 2]].values
                                else:
                                    features_bar_arr = np.zeros((len(df_3d_filtered), 2))
                                    
                                # Add knee_x, knee_y from 2D_L
                                coord_2dl_file = os.path.join(set_path, "Coordinate", "2D_L", f"clip_{clip_idx}_2d.csv")
                                if os.path.exists(coord_2dl_file):
                                    df_coord = pd.read_csv(coord_2dl_file)
                                    # left knee x is 'x13', y is 'y13'
                                    knee_x_arr = df_coord['x13'].values.reshape(-1, 1)
                                    knee_y_arr = df_coord['y13'].values.reshape(-1, 1)
                                    # left shoulder x is 'x5', y is 'y5', left hip x is 'x11', y is 'y11'
                                    shoulder_x = df_coord['x5'].values.reshape(-1, 1)
                                    shoulder_y = df_coord['y5'].values.reshape(-1, 1)
                                    hip_x = df_coord['x11'].values.reshape(-1, 1)
                                    hip_y = df_coord['y11'].values.reshape(-1, 1)
                                    
                                    shoulder_hip_disp = np.abs(shoulder_x - hip_x)
                                    
                                    # Torso Angle to Ground (90=vertical, 0=horizontal)
                                    dx = np.abs(shoulder_x - hip_x) + 1e-6
                                    dy = np.abs(shoulder_y - hip_y)
                                    torso_angle = np.degrees(np.arctan2(dy, dx))
                                else:
                                    knee_x_arr = np.zeros((len(df_3d_filtered), 1))
                                    knee_y_arr = np.zeros((len(df_3d_filtered), 1))
                                    shoulder_hip_disp = np.zeros((len(df_3d_filtered), 1))
                                    torso_angle = np.zeros((len(df_3d_filtered), 1))
                                
                                # Merge frame by frame
                                # Make sure they have the same length
                                min_len = min(len(df_3d_filtered), len(features_bar_arr), len(knee_x_arr))
                                
                                # Calculate displacement
                                bar_knee_disp = features_bar_arr[:min_len, 0:1] - knee_x_arr[:min_len]
                                bar_knee_y_disp = features_bar_arr[:min_len, 1:2] - knee_y_arr[:min_len]
                                
                                # Calculate angular velocity difference (hip vs knee)
                                left_knee_angle = df_3d.iloc[:, 1].values.reshape(-1, 1)
                                left_hip_angle = df_3d.iloc[:, 2].values.reshape(-1, 1)
                                left_knee_vel = np.gradient(left_knee_angle, axis=0)
                                left_hip_vel = np.gradient(left_hip_angle, axis=0)
                                hip_knee_vel_diff = left_hip_vel - left_knee_vel

                                merged_features = np.concatenate([
                                    df_3d_filtered.values[:min_len], 
                                    features_bar_arr[:min_len], 
                                    bar_knee_disp,
                                    bar_knee_y_disp,
                                    shoulder_hip_disp[:min_len],
                                    torso_angle[:min_len],
                                    hip_knee_vel_diff[:min_len]
                                ], axis=1)
                                
                                from dataset.tools.Deadlift_tool.utils import interpolate_features
                                from dataset.tools.Deadlift_tool.data_split import process_delta, process_delta_ratio, process_zscore, normalize_to_neg1_1
                                
                                # Data Augmentation (Placeholder)
                                merged_features = apply_augmentation(merged_features)
                                
                                from scipy.signal import butter, filtfilt
                                fs = 30
                                cutoff = 1
                                order = 4
                                nyq = 0.5 * fs
                                normal_cutoff = cutoff / nyq
                                b, a = butter(order, normal_cutoff, btype='low')
                                if min_len > 15:
                                    merged_features = filtfilt(b, a, merged_features, axis=0)

                                filtered_interpolated = {"0": interpolate_features(merged_features, 110)}
                                delta_feature = process_delta(filtered_interpolated)
                                delta_square_feature = process_delta(delta_feature)
                                zscore_feature = process_zscore(filtered_interpolated)
                                delta_ratio_feature = process_delta_ratio(filtered_interpolated)
                                
                                fn = normalize_to_neg1_1(filtered_interpolated["0"]).tolist()
                                fdn = normalize_to_neg1_1(delta_feature["0"]).tolist()
                                fd2n = normalize_to_neg1_1(delta_ratio_feature["0"]).tolist()
                                fzn = normalize_to_neg1_1(zscore_feature["0"]).tolist()
                                fdsn = normalize_to_neg1_1(delta_square_feature["0"]).tolist()
                                
                                # Multi-label logic moved to start of block
                                    
                                    
                                # Combine all 5 normalizations into [110, 40] array
                                all_feat = np.concatenate([
                                    np.array(fn), 
                                    np.array(fdn), 
                                    np.array(fd2n), 
                                    np.array(fzn), 
                                    np.array(fdsn)
                                ], axis=-1)
                                
                                import re
                                set_val = int(re.search(r'\d+', os.path.basename(set_dir)).group())
                                clip_val = int(re.search(r'\d+', os.path.basename(file)).group())
                                
                                sub_match = re.search(r'subject?\d+', os.path.basename(subject_dir))
                                sub_name = sub_match.group() if sub_match else os.path.basename(subject_dir)
                                
                                data.append({
                                    "subject": sub_name,
                                    "set": set_val,
                                    "clip": clip_val,
                                    "features": str(all_feat.tolist()),
                                    "label": str(label_vec)
                                })
                            except Exception as e:
                                print(f"Error processing {file_path}: {e}")
                                
                else:
                    print(f"Skipping {set_path} (No 3D Angle Data found)")
                    
    df = pd.DataFrame(data)
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")

if __name__ == "__main__":
    generate_csv(r"/cats/dataset/DeadliftDataset_0408", "./data/deadlift_dataset_3d.csv")
