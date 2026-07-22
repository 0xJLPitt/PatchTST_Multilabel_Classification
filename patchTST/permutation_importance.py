import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
import argparse
import sys
import os
import random
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import Dataset_Deadlift, Dataset_Benchpress, Dataset_Squat, Dataset_Squat_PT, Datasubset
from models import PatchTSTClassifier
from torch.utils.data import DataLoader

def get_dataloaders(args):
    if args.sport == 'deadlift':
        data_path = args.data_path if args.data_path else os.path.join(os.path.dirname(__file__), '..', 'data', f'deadlift_dataset_{args.type.lower()}.csv')
        full_dataset = Dataset_Deadlift(data_path)
        num_classes = 4
        input_len = 110
    elif args.sport == 'benchpress':
        data_path = args.data_path if args.data_path else os.path.join(os.path.dirname(__file__), '..', 'data', 'benchpress_dataset.csv')
        full_dataset = Dataset_Benchpress(data_path)
        num_classes = 4
        input_len = 100
    elif args.sport == 'squat':
        data_path = args.data_path if args.data_path else os.path.join(os.path.dirname(__file__), '..', 'data', 'squat_dataset_2d.csv')
        if data_path.endswith('.pt'):
            full_dataset = Dataset_Squat_PT(data_path)
        else:
            full_dataset = Dataset_Squat(data_path)
        num_classes = 5
        input_len = 110
    else:
        raise ValueError("Invalid sport")

    dataset_folds = []
    if args.split_mode == 'subject_exclusive':
        subject_indices = defaultdict(list)
        for idx in range(len(full_dataset)):
            sub = full_dataset.subjects[idx]
            subject_indices[sub].append(idx)
            
        unique_subs = sorted(list(subject_indices.keys()))
        random.seed(42)
        random.shuffle(unique_subs)
        
        n_sub = len(unique_subs)
        tr_end = max(1, int(0.7 * n_sub))
        vl_end = max(tr_end + 1, int(0.8 * n_sub))
        
        train_subs = unique_subs[:tr_end]
        val_subs = unique_subs[tr_end:vl_end]
        test_subs = unique_subs[vl_end:]
        
        train_indices = []
        valid_indices = []
        test_indices = []
        for sub in train_subs: train_indices.extend(subject_indices[sub])
        for sub in val_subs: valid_indices.extend(subject_indices[sub])
        for sub in test_subs: test_indices.extend(subject_indices[sub])
            
        dataset_folds = [(train_indices, valid_indices, test_indices)]
        
    elif args.split_mode == 'instance_stratified':
        train_indices = []
        valid_indices = []
        test_indices = []
        
        instance_subject = {}
        instance_primary_label = {}
        instance_indices = defaultdict(list)
        
        for idx in range(len(full_dataset)):
            sub = full_dataset.subjects[idx]
            inst = full_dataset.instances[idx]
            label = tuple(full_dataset.labels[idx].int().tolist())
            
            instance_indices[inst].append(idx)
            if inst not in instance_primary_label:
                instance_primary_label[inst] = label
                instance_subject[inst] = sub
                
        subject_instances = defaultdict(list)
        for inst, label in instance_primary_label.items():
            sub = instance_subject[inst]
            subject_instances[sub].append(inst)
            
        train_insts = set()
        val_insts = set()
        test_insts = set()
        
        class_free_insts = defaultdict(list)
        class_total_count = defaultdict(int)
        class_already_train = defaultdict(int)
        
        for inst, label in instance_primary_label.items():
            class_total_count[label] += 1
            
        for sub in sorted(subject_instances.keys()):
            insts = list(subject_instances[sub])
            sub_rng = random.Random(sub)
            sub_rng.shuffle(insts)
            
            first_inst = insts[0]
            train_insts.add(first_inst)
            first_label = instance_primary_label[first_inst]
            class_already_train[first_label] += 1
            
            for free_inst in insts[1:]:
                free_label = instance_primary_label[free_inst]
                class_free_insts[free_label].append(free_inst)
                
        for label, free_list in class_free_insts.items():
            label_rng = random.Random(str(label))
            label_rng.shuffle(free_list)
            
            total_class = class_total_count[label]
            target_train = round(0.7 * total_class)
            target_val = round(0.1 * total_class)
            target_test = total_class - target_train - target_val
            
            already_tr = class_already_train[label]
            need_tr = max(0, target_train - already_tr)
            
            free_for_val_test = len(free_list) - need_tr
            if free_for_val_test < 0:
                for inst in free_list:
                    train_insts.add(inst)
            else:
                for inst in free_list[:need_tr]:
                    train_insts.add(inst)
                
                rem_list = free_list[need_tr:]
                denom = target_val + target_test
                val_ratio = target_val / denom if denom > 0 else 0.60
                
                n_val = round(val_ratio * len(rem_list))
                
                for inst in rem_list[:n_val]:
                    val_insts.add(inst)
                for inst in rem_list[n_val:]:
                    test_insts.add(inst)
                    
        for inst in train_insts:
            train_indices.extend(instance_indices[inst])
        for inst in val_insts:
            valid_indices.extend(instance_indices[inst])
        for inst in test_insts:
            test_indices.extend(instance_indices[inst])
                
        dataset_folds = [(train_indices, valid_indices, test_indices)]
        
    elif args.split_mode == 'clip_random':
        seeds = [42]
        for se in seeds:
            random.seed(se)
            all_indices = list(range(len(full_dataset)))
            random.shuffle(all_indices)
            
            n_total = len(all_indices)
            tr_end = int(0.75 * n_total)
            vl_end = int(0.90 * n_total)
            
            train_idx = all_indices[:tr_end]
            val_idx = all_indices[tr_end:vl_end]
            test_idx = all_indices[vl_end:]
            dataset_folds.append((train_idx, val_idx, test_idx))

    # We use fold 0's validation set
    _, v_idx, _ = dataset_folds[0]
    valid_dataset = Datasubset(full_dataset, v_idx, transform=False)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return full_dataset.dim, num_classes, input_len, valid_loader

def main():
    parser = argparse.ArgumentParser(description="Permutation Importance for PatchTST")
    parser.add_argument('--model_paths', type=str, nargs='+', required=True, help="Paths to the model weights (.pth files)")
    parser.add_argument('--sport', type=str, choices=['benchpress', 'deadlift', 'squat'], required=True)
    parser.add_argument('--type', type=str, choices=['2d', '3d', '2D', '3D'], default='3D')
    parser.add_argument('--split_mode', type=str, choices=['subject_exclusive', 'instance_stratified', 'clip_random'], default='instance_stratified')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--num_heads', type=int, default=8) # updated default to 8 since you use heads=8 sometimes
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_csv', type=str, default='permutation_importance_results.csv', help="Output CSV file path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_dim, num_classes, input_len, valid_loader = get_dataloaders(args)
    print(f"Data loaded. Input Dimension: {input_dim}, Validation samples: {len(valid_loader.dataset)}")
    
    # Class names mapping
    if args.sport == 'deadlift':
        class_names = ["Correct", "Away_from_shins", "Hips_rise_first", "Collide_with_knees", "Lower_back_rounding"]
    elif args.sport == 'squat':
        class_names = ["Correct", "Insufficient_Depth", "Excessive_Knee_Dom", "Excessive_Hip_Dom", "Posterior_Pelvic_Tilt", "Early_Hip_Rise"]
    else:
        class_names = ["Correct"] + [f"Class_{i}" for i in range(num_classes)]

    # Store importances. Format: importances_dict['Feature_Index'] = [0, 1, 2...]
    importances_dict = {"Feature_Index": list(range(input_dim))}
    
    for model_path in args.model_paths:
        model_name = os.path.basename(model_path)
        print(f"\nEvaluating model: {model_name}")
        
        # Initialize and load model
        model = PatchTSTClassifier(input_dim, num_classes, input_len, num_heads=args.num_heads)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # 1. Calculate Baseline F1 on clean validation set
        baseline_y_true = []
        baseline_y_pred = []
        
        with torch.no_grad():
            for inputs, labels, _ in valid_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).int()
                
                baseline_y_true.extend(labels.numpy())
                baseline_y_pred.extend(preds.cpu().numpy())
                
        # Calculate per-class F1 for error labels
        y_true_np = np.array(baseline_y_true)
        y_pred_np = np.array(baseline_y_pred)
        baseline_f1_err = f1_score(y_true_np, y_pred_np, average=None)
        
        # Calculate F1 for 'Correct' (all error labels are 0)
        y_true_corr = (y_true_np.sum(axis=1) == 0).astype(int)
        y_pred_corr = (y_pred_np.sum(axis=1) == 0).astype(int)
        baseline_f1_corr = f1_score(y_true_corr, y_pred_corr, zero_division=0)
        
        # Combine Correct + Errors
        baseline_f1_per_class = np.insert(baseline_f1_err, 0, baseline_f1_corr)
        
        for i, cname in enumerate(class_names):
            print(f"[{model_name}] Baseline F1-score ({cname}): {baseline_f1_per_class[i]:.4f}")
        
        # Initialize list of lists for per-class importances
        model_importances = {cname: [] for cname in class_names}
        
        # 2. Permutation Importance loop
        for i in tqdm(range(input_dim), desc=f"Permuting features ({model_name})"):
            permuted_y_true = []
            permuted_y_pred = []
            
            for inputs, labels, _ in valid_loader:
                batch_size = inputs.size(0)
                inputs_permuted = inputs.clone()
                
                # Permute feature i across the batch dimension
                if batch_size > 1:
                    perm_idx = torch.randperm(batch_size)
                    inputs_permuted[:, :, i] = inputs[perm_idx, :, i]
                    
                inputs_permuted = inputs_permuted.to(device)
                
                with torch.no_grad():
                    outputs = model(inputs_permuted)
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).int()
                    
                permuted_y_true.extend(labels.numpy())
                permuted_y_pred.extend(preds.cpu().numpy())
                
            # Calculate F1 for this permuted feature
            p_y_true_np = np.array(permuted_y_true)
            p_y_pred_np = np.array(permuted_y_pred)
            
            p_f1_err = f1_score(p_y_true_np, p_y_pred_np, average=None)
            
            p_y_true_corr = (p_y_true_np.sum(axis=1) == 0).astype(int)
            p_y_pred_corr = (p_y_pred_np.sum(axis=1) == 0).astype(int)
            p_f1_corr = f1_score(p_y_true_corr, p_y_pred_corr, zero_division=0)
            
            permuted_f1_per_class = np.insert(p_f1_err, 0, p_f1_corr)
            
            # Importance = Baseline F1 - Permuted F1
            importance_per_class = baseline_f1_per_class - permuted_f1_per_class
            
            for j, cname in enumerate(class_names):
                model_importances[cname].append(importance_per_class[j])
            
        # Add to main dict (prefix with model name if there are multiple models to avoid collision)
        prefix = f"{model_name}_" if len(args.model_paths) > 1 else ""
        for cname in class_names:
            importances_dict[f"{prefix}{cname}"] = model_importances[cname]
        
    # 3. Create DataFrame and export
    df = pd.DataFrame(importances_dict)
    
    # Calculate Mean Importance across all error classes
    model_cols = [c for c in df.columns if c != "Feature_Index"]
    df['Mean_Importance'] = df[model_cols].mean(axis=1)
    
    # Generate Feature Names based on Deadlift 3D 14 base features * 5 derivations
    if args.sport == 'deadlift' and input_dim == 70:
        base_features = [
            "Left_Knee_Angle", "Left_Hip_Angle", "Right_Knee_Angle", "Right_Hip_Angle",
            "Body_Length", "Left_Arm_Torso_Angle", "Right_Arm_Torso_Angle",
            "Bar_X", "Bar_Y", "Bar_Knee_X_Disp", "Bar_Knee_Y_Disp",
            "Shoulder_Hip_X_Disp", "Torso_Angle", "Hip_Knee_Vel_Diff"
        ]
        derivations = ["Norm", "Delta", "DeltaRatio", "ZScore", "DeltaSquare"]
        
        feature_names = []
        for idx in df['Feature_Index']:
            base_idx = idx % 14
            deriv_idx = idx // 14
            feature_names.append(f"{base_features[base_idx]} ({derivations[deriv_idx]})")
            
        df.insert(1, 'Feature_Name', feature_names)
    elif args.sport == 'squat' and input_dim == 60:
        # Assuming squat has 12 base features * 5 derivations based on code
        base_features = [f"BaseFeature_{i}" for i in range(12)]
        derivations = ["Norm", "Delta", "DeltaRatio", "ZScore", "DeltaSquare"]
        feature_names = [f"{base_features[idx % 12]} ({derivations[idx // 12]})" for idx in df['Feature_Index']]
        df.insert(1, 'Feature_Name', feature_names)
    else:
        df.insert(1, 'Feature_Name', [f"Feature_{idx}" for idx in df['Feature_Index']])
    
    # Sort by Mean Importance descending
    df = df.sort_values(by='Mean_Importance', ascending=False)
    
    df.to_csv(args.output_csv, index=False)
    print(f"\nPermutation Importance analysis complete. Results saved to {args.output_csv}")
    print("Top 10 Features (by Mean Importance across all errors):")
    print(df.head(10))

if __name__ == "__main__":
    main()
