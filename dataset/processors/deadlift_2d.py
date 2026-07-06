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

def apply_augmentation(df):
    """
    Placeholder for data augmentation (e.g., jittering, scaling, time-warping).
    """
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
                
                # Check for 2D Angle and Coordinate subdirectories
                rl_angle_dir = os.path.join(set_path, "Angle", "2D_RL")
                fl_angle_dir = os.path.join(set_path, "Angle", "2D_FL")
                bar_dir = os.path.join(set_path, "Coordinate", "bar")
                l_coord_dir = os.path.join(set_path, "Coordinate", "2D_L")
                
                if (os.path.isdir(rl_angle_dir) and os.path.isdir(fl_angle_dir) and 
                    os.path.isdir(bar_dir) and os.path.isdir(l_coord_dir)):
                    
                    for file in os.listdir(rl_angle_dir):
                        if file.endswith(".csv"):
                            clip_idx = file.replace("angle_", "").replace(".csv", "")
                            
                            # 1. 根據與 deadlift_3d.py 相同的方式，依賴 multierror.json 過濾重複動作
                            key = f"{subject_dir}_{set_dir}_{label_dir}"
                            if key in pass_list and int(clip_idx) in pass_list[key]:
                                continue
                            
                            rl_file = os.path.join(rl_angle_dir, f"angle_{clip_idx}.csv")
                            fl_file = os.path.join(fl_angle_dir, f"angle_{clip_idx}.csv")
                            bar_file = os.path.join(bar_dir, f"bar_{clip_idx}.csv")
                            l_coord_file = os.path.join(l_coord_dir, f"clip_{clip_idx}_2d.csv")
                            
                            # Check if all four necessary feature files are present
                            if (os.path.exists(rl_file) and os.path.exists(fl_file) and 
                                os.path.exists(bar_file) and os.path.exists(l_coord_file)):
                                
                                print(f"Processing 2D files: {rl_file}")
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
                                            
                                    # Read files
                                    df_rl = pd.read_csv(rl_file, header=None)
                                    df_fl = pd.read_csv(fl_file, header=None)
                                    df_bar = pd.read_csv(bar_file, header=None)
                                    df_l_coord = pd.read_csv(l_coord_file) # Contains header: frame,x0,y0...
                                    
                                    # Align lengths
                                    min_len = min(len(df_rl), len(df_fl), len(df_bar), len(df_l_coord))
                                    if min_len == 0:
                                        print(f"Empty data for clip {clip_idx} in {set_path}")
                                        continue
                                        
                                    # Calculate custom features from Coordinate/2D_L
                                    # x5, y5 = L Shoulder
                                    # x11, y11 = L Hip
                                    x5 = df_l_coord['x5'].values[:min_len]
                                    y5 = df_l_coord['y5'].values[:min_len]
                                    x11 = df_l_coord['x11'].values[:min_len]
                                    y11 = df_l_coord['y11'].values[:min_len]
                                    
                                    # Body length: Euclidean distance between shoulder (x5, y5) and hip (x11, y11)
                                    body_lengths = np.sqrt((x5 - x11)**2 + (y5 - y11)**2)
                                    
                                    # Barbell X and Y
                                    bar_x = df_bar.iloc[:min_len, 1].values
                                    bar_y = df_bar.iloc[:min_len, 2].values
                                    
                                    # Deviation between barbell endpoint X and shoulder point X
                                    deviation = bar_x - x5
                                    
                                    # Stack the 8 requested 2D features:
                                    # 1. left knee angle (2D_RL)
                                    # 2. left hip angle (2D_RL)
                                    # 3. right knee angle (2D_FL)
                                    # 4. right hip angle (2D_FL)
                                    # 5. barbell endpoint X (2D_L)
                                    # 6. barbell endpoint Y (2D_L)
                                    # 7. body length (2D_L, self-calculated)
                                    # 8. deviation between barbell endpoint X and shoulder point X (2D_L, self-calculated)
                                    merged_features = np.stack([
                                        df_rl.iloc[:min_len, 2].values,
                                        df_rl.iloc[:min_len, 1].values,
                                        df_fl.iloc[:min_len, 2].values,
                                        df_fl.iloc[:min_len, 1].values,
                                        bar_x,
                                        bar_y,
                                        body_lengths,
                                        deviation
                                    ], axis=1)
                                    
                                    from dataset.tools.Deadlift_tool.utils import interpolate_features
                                    from dataset.tools.Deadlift_tool.data_split import process_delta, process_delta_ratio, process_zscore, normalize_to_neg1_1
                                    
                                    # Data Augmentation (Placeholder)
                                    merged_features = apply_augmentation(merged_features)

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
                                    print(f"Error processing clip {clip_idx} in {set_path}: {e}")
                                
                else:
                    print(f"Skipping {set_path} (Missing required 2D directories)")
                    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")

if __name__ == "__main__":
    generate_csv("DeadliftDataset_0408", "./data/deadlift_dataset_2d.csv")
