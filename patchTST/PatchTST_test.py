import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import time
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay, f1_score
import json
import matplotlib.pyplot as plt
import numpy as np
from tools import *
import argparse
from torch.utils.data import DataLoader, random_split
from models import PatchTSTClassifier
from sklearn.metrics import accuracy_score


def get_test_indices_instance_stratified(full_dataset):
    import random
    from collections import defaultdict
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
        
    train_insts, val_insts, test_insts = set(), set(), set()
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
            for inst in free_list: train_insts.add(inst)
        else:
            for inst in free_list[:need_tr]: train_insts.add(inst)
            rem_list = free_list[need_tr:]
            denom = target_val + target_test
            val_ratio = target_val / denom if denom > 0 else 0.60
            n_val = round(val_ratio * len(rem_list))
            for inst in rem_list[:n_val]: val_insts.add(inst)
            for inst in rem_list[n_val:]: test_insts.add(inst)
                
    for inst in test_insts:
        test_indices.extend(instance_indices[inst])
        
    return test_indices

def test_model_with_path_tracking(model, test_loader, criterion, txt_dir, save_path, num_classes, sport='deadlift'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.to(device)
    model.eval()
    
    # **存放測試過程的數據**
    total_loss, total_time = 0.0, 0.0  
    y_true, y_pred = [], []
    all_indices = []
    
    with torch.no_grad():
        for inputs, labels, indices in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if inputs.ndim != 3:
                raise ValueError(f"Expected 3D input (B, T, F), got shape {inputs.shape}")

            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            total_time += (end_time - start_time)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs = torch.sigmoid(outputs)  # [B, num_classes]
            preds = (probs > 0.5).int()     # [B, num_classes]
            
            y_true.extend(labels.cpu().numpy().tolist())    # labels shape: (batch, num_classes)
            y_pred.extend(preds.tolist())                   # preds shape: (batch, num_classes)
            all_indices.extend(indices.cpu().numpy().tolist())
    avg_loss = total_loss / len(test_loader)
    avg_time_per_sample = total_time / len(y_true)
    f1 = f1_score(y_true, y_pred, average='macro')

    # 繪製混淆矩陣
    if sport == 'deadlift':
        if num_classes == 5:
            classes = ['Far from the shins', 'Hips rise first', 'Collide with the knees', 'Lower back rounding', 'Correct']
        else:
            classes = ['Correct', 'Far from the shins', 'Hips rise first', 'Collide with the knees', 'Lower back rounding']
    elif sport == 'squat':
        if num_classes == 6:
            classes = ['Insufficient_Depth', 'Excessive_Knee_Dominance', 'Excessive_Hip_Dominance', 'Posterior_Pelvic_Tilt', 'Early_Hip_Rise', 'Correct']
        else:
            classes = ['Correct', 'Insufficient_Depth', 'Excessive_Knee_Dominance', 'Excessive_Hip_Dominance', 'Posterior_Pelvic_Tilt', 'Early_Hip_Rise']
    else:
        classes = ['Correct', 'tilting to the left', 'tilting to the right', 'scapular protraction', 'elbows flaring']
    binary_classes = classes if num_classes == len(classes) else classes[1:]
    
    cm = multilabel_confusion_matrix(y_true, y_pred, sample_weight=None, labels=None, samplewise=False)
    n_classes = cm.shape[0]
    fig, axes = plt.subplots(nrows=(n_classes+1)//2, ncols=2, figsize=(12, 10))
    axes = axes.flatten()

    for i in range(n_classes):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm[i], display_labels=['False', 'True'])
        disp.plot(include_values=True, cmap="Blues", ax=axes[i], 
                xticks_rotation="horizontal", values_format="d")
        axes[i].set_title(f'Class: {binary_classes[i]}')

    plt.tight_layout()
    plt.savefig(f"{txt_dir}/confusion_matrix.png")
    plt.close()
    
    cm = multilabel_confusion_matrix_mix(y_true, y_pred, num_classes)
    plot_custom_confusion_matrix(cm, classes, f"{txt_dir}/confusion_matrix_mix.png")
    accuracy = accuracy_score(y_true, y_pred)    

    # 計算每個類別的 F1 score (包含 Correct)
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    f1_errors = f1_score(y_true_np, y_pred_np, average=None, zero_division=0)
    
    if num_classes == len(classes): # 5 classes (explicit Correct)
        class_f1 = list(f1_errors)
    else: # 4 classes (implicit Correct)
        y_true_correct = (np.sum(y_true_np, axis=1) == 0).astype(int)
        y_pred_correct = (np.sum(y_pred_np, axis=1) == 0).astype(int)
        f1_correct = f1_score(y_true_correct, y_pred_correct, average='binary', zero_division=0)
        class_f1 = [f1_correct] + list(f1_errors)

    # 儲存異常資料夾清單 (通用所有模型)
    try:
        if hasattr(test_loader.dataset, 'dataset'):
            instances = test_loader.dataset.dataset.instances
        else:
            instances = test_loader.dataset.instances
            
        from collections import defaultdict
        misclassifications = defaultdict(set)
        
        offset = 1
        dummy_class = 0
        
        for i in range(len(y_true_np)):
            yt = y_true_np[i]
            yp = y_pred_np[i]
            idx = all_indices[i]
            inst = instances[idx]
            
            true_classes = np.where(yt == 1)[0]
            pred_classes = np.where(yp == 1)[0]
            
            if len(true_classes) == 0:
                true_classes = [dummy_class]
            else:
                true_classes = [c + offset for c in true_classes]
                
            if len(pred_classes) == 0:
                pred_classes = [dummy_class]
            else:
                pred_classes = [c + offset for c in pred_classes]
                
            for t in true_classes:
                for p in pred_classes:
                    if t != p:  # Only record misclassifications
                        misclassifications[(t, p)].add(inst)
                        
        with open(os.path.join(txt_dir, "misclassified_instances.md"), "w", encoding="utf-8") as f:
            f.write("# Misclassified Instances (Folders)\n\n")
            
            for (t, p), inst_set in sorted(misclassifications.items()):
                if len(inst_set) > 0:
                    t_name = classes[t]
                    p_name = classes[p]
                    f.write(f"## True: {t_name}, Pred: {p_name} (Total: {len(inst_set)})\n")
                    for inst in sorted(list(inst_set)):
                        f.write(f"- `{inst}`\n")
                    f.write("\n")
                    
    except Exception as e:
        print(f"Warning: Could not save misclassified instances mapping: {e}")

    return avg_loss, f1, avg_time_per_sample, accuracy, class_f1

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--sport', type=str, choices=['benchpress', 'deadlift'])
    parser.add_argument('--num_classes', type=int, choices=[4, 5], default=4, help='Number of classes for deadlift (4 or 5)')
    parser.add_argument('--type', type=str, choices=['2d', '3d', '2D', '3D'], default='3D', help='Feature type (2D or 3D) for deadlift')
    parser.add_argument('--tag', type=str, required=True, help='Tag for save_dir to locate the model')
    args = parser.parse_args()
    
    from dataset import *
    if args.sport == 'deadlift':
        feat_type = args.type.upper()
        if args.num_classes == 5:
            data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'deadlift_dataset_3d_5class.csv')
        else:
            data_path = os.path.join(os.path.dirname(__file__), '..', 'data', f'deadlift_dataset_{args.type.lower()}.csv')
            
        output_dir = f'./models/deadlift/TST_Deadlift_{feat_type}/{args.tag}'
        save_dir = f'./models/deadlift/TST_Deadlift_{feat_type}/{args.tag}'
        test_dataset = Dataset_Deadlift(data_path)
        num_classes = args.num_classes
        input_len = 110
    elif args.sport == 'squat':
        feat_type = args.type.upper()
        if args.num_classes == 6:
            data_path = os.path.join(os.path.dirname(__file__), '..', 'data', f'squat_dataset_{args.type.lower()}_6class.csv')
        else:
            data_path = os.path.join(os.path.dirname(__file__), '..', 'data', f'squat_dataset_{args.type.lower()}.csv')
            
        output_dir = f'./models/squat/TST_Squat_{feat_type}/{args.tag}'
        save_dir = f'./models/squat/TST_Squat_{feat_type}/{args.tag}'
        test_dataset = Dataset_Squat(data_path)
        num_classes = args.num_classes
        input_len = 110
    elif args.sport == 'benchpress':
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'benchpress_dataset.csv')
        test_dataset  = Dataset_Benchpress(data_path)
        output_dir = './models/benchpress/TST_Benchpress/Exp1/no_wrist_press'
        save_dir = './models/benchpress/TST_Benchpress/9/no_wrist_press'
        num_classes = 4
        input_len = 100
    input_dim = test_dataset.dim
    print('Input dimension', input_dim)
    
    best_f1 = -1
    best_seed = None
    best_model_path = ""

    all_f1_scores = []
    cost_times = []
    accuracies = []
    all_class_f1_scores = []
    seeds = [42, 2023, 7, 88, 100, 999]
    
    for se in seeds:
        set_seed(se)

        # 分割資料
        gen = torch.Generator().manual_seed(se)  # 為每個seed創建獨立生成器
        
        test_indices = get_test_indices_instance_stratified(test_dataset)
        from torch.utils.data import Subset
        test_data = Subset(test_dataset, test_indices)
        
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

        # 訓練與測試
        model = PatchTSTClassifier(input_dim, num_classes, input_len).to(device)
        criterion = torch.nn.BCEWithLogitsLoss()

        save_path = os.path.join(save_dir, f"PatchTST_model_seed{se}.pth")
        txt_dir = os.path.join(output_dir, f"seed{se}")
        os.makedirs(txt_dir, exist_ok=True)
        
        avg_loss, f1, avg_time_per_sample, accuracy, class_f1 = test_model_with_path_tracking(
            model, test_loader, criterion, txt_dir, save_path, num_classes, sport=args.sport
        )
        print(f"Seed {se} Test F1: {f1:.4f}, Accuracy: {accuracy:.4f}, cost {avg_time_per_sample} sec")
        all_f1_scores.append(f1)
        cost_times.append(avg_time_per_sample)
        accuracies.append(accuracy)
        all_class_f1_scores.append(class_f1)

        if f1 > best_f1:
            best_f1 = f1
            best_seed = se
            best_model_path = save_path

    if args.sport == 'deadlift':
        classes = ['Correct', 'Far from the shins', 'Hips rise first', 'Collide with the knees', 'Lower back rounding']
    else:
        classes = ['Correct', 'tilting to the left', 'tilting to the right', 'scapular protraction', 'elbows flaring']

    write_result(model, seeds, all_f1_scores, accuracies, cost_times, output_dir, best_f1, best_seed, best_model_path, class_names=classes, class_f1_scores=all_class_f1_scores)