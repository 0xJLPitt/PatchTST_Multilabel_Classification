from random import choice
import torch
import os, json
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import PatchTSTClassifier
from sklearn.metrics import f1_score
from tools import *
import argparse
from PatchTST_test import test_model_with_path_tracking
import math

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, save_path, fig_path, num_epochs=150, patience=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_f1 = 0.0  # 用來儲存最佳 F1-score
    patience_counter = 0

    # **存放訓練過程的數據**
    train_losses = []
    train_f1_scores = []
    val_f1_scores = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        y_true, y_pred = [], []

        for inputs, labels, indices in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)  # [B, num_classes]
            preds = (probs > 0.5).int()     # [B, num_classes]
            y_true.extend(labels.cpu().numpy().tolist())    # labels shape: (batch, num_classes)
            y_pred.extend(preds.tolist())                   # preds shape: (batch, num_classes)

        avg_loss = total_loss / len(train_loader)
        train_f1 = f1_score(y_true, y_pred, average='macro')
        val_f1 = compute_f1_score(model, valid_loader)

        scheduler.step()
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # **紀錄數據**
        train_losses.append(avg_loss)
        train_f1_scores.append(train_f1)
        val_f1_scores.append(val_f1)

        # 根據 F1-score 儲存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print("✅ Model Saved (Best F1-score)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️ Early Stopping Triggered")
                break

    # **繪製 Loss 和 F1-score**
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, label="Train Loss", color='blue', marker='o')
    plt.plot(epochs, train_f1_scores, label="Train F1-score", color='green', marker='s')
    plt.plot(epochs, val_f1_scores, label="Validation F1-score", color='red', marker='d')

    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title("Training Loss & F1-score per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")  # 儲存高解析度圖片

def get_warmup_cosine_scheduler(
    optimizer,
    warmup_epochs: int,
    max_epochs: int,
    min_lr_ratio: float = 0.0,
):
    """
    warmup_epochs:   線性 warmup epoch 數（從 0 -> base_lr）
    max_epochs:      總訓練 epoch 數
    min_lr_ratio:    最小 lr / base_lr，比方 0.0 就是衰到 0
    """
    assert warmup_epochs < max_epochs

    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            # 線性 warmup: 0 -> 1
            return float(current_epoch + 1) / float(warmup_epochs)
        # cosine decay: 1 -> min_lr_ratio
        progress = (current_epoch - warmup_epochs) / float(max_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--sport', type=str, choices=['benchpress', 'deadlift'])
    parser.add_argument('--type', type=str, choices=['2d', '3d', '2D', '3D'], default='3D', help='Feature type (2D or 3D) for deadlift')
    parser.add_argument('--subject_isolated', action='store_true', help='Whether to split the dataset by subject')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of subset workers for DataLoader')
    parser.add_argument('--tag', type=str, help='Tag for save_dir, default is your data argumentation') # spawner, ...
    args = parser.parse_args()
    seeds = [42, 2023, 7, 88, 100, 999]
    
    from dataset import *
    
    if args.sport == 'deadlift':
        feat_type = args.type.upper()
        data_path = os.path.join(os.getcwd(), 'data', f'deadlift_dataset_{args.type.lower()}.csv')
        full_dataset = Dataset_Deadlift(data_path)
        save_dir = f'./models/deadlift/TST_Deadlift_{feat_type}/{args.tag}'
        num_classes = 4
        input_len = 110
        
    elif args.sport == 'benchpress':
        data_path = os.path.join(os.getcwd(), 'data', 'benchpress_dataset.csv')
        full_dataset = Dataset_Benchpress(data_path)
        save_dir = f'./models/benchpress/TST_Benchpress/{args.tag}'
        num_classes = 4
        input_len = 100
    
    
    dataset_folds = []
    if args.subject_isolated:
        train_indices = []
        valid_indices = []
        test_indices = []
        
        from collections import defaultdict
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
                
        # 按照受試者對實例進行分組
        subject_instances = defaultdict(list)
        for inst, label in instance_primary_label.items():
            sub = instance_subject[inst]
            subject_instances[sub].append(inst)
            
        train_insts = set()
        val_insts = set()
        test_insts = set()
        
        # 統計各類別的實例總數與已分配給訓練集的數量
        class_free_insts = defaultdict(list)
        class_total_count = defaultdict(int)
        class_already_train = defaultdict(int)
        
        for inst, label in instance_primary_label.items():
            class_total_count[label] += 1
            
        # 優先條件：每個受試者必須至少有 1 組在 trainingset
        for sub in sorted(subject_instances.keys()):
            insts = list(subject_instances[sub])
            # 用受試者名稱作為 Seed 進行 Shuffle，確保切分結果可重現
            sub_rng = random.Random(sub)
            sub_rng.shuffle(insts)
            
            first_inst = insts[0]
            train_insts.add(first_inst)
            first_label = instance_primary_label[first_inst]
            class_already_train[first_label] += 1
            
            # 剩餘的實例作為自由分配組別
            for free_inst in insts[1:]:
                free_label = instance_primary_label[free_inst]
                class_free_insts[free_label].append(free_inst)
                
        # 針對每個類別，分配其剩餘的自由組別以達成全域的 75%:15%:10% 比例
        for label, free_list in class_free_insts.items():
            label_rng = random.Random(str(label))
            label_rng.shuffle(free_list)
            
            total_class = class_total_count[label]
            target_train = round(0.75 * total_class)
            target_val = round(0.15 * total_class)
            target_test = total_class - target_train - target_val
            
            already_tr = class_already_train[label]
            need_tr = max(0, target_train - already_tr)
            
            free_for_val_test = len(free_list) - need_tr
            if free_for_val_test < 0:
                # 自由組數不足以填滿目標訓練集，將剩餘自由組全部分給訓練集
                for inst in free_list:
                    train_insts.add(inst)
            else:
                # 分配 need_tr 個自由組給訓練集
                for inst in free_list[:need_tr]:
                    train_insts.add(inst)
                
                # 分配剩下的自由組給驗證集和測試集
                rem_list = free_list[need_tr:]
                denom = target_val + target_test
                val_ratio = target_val / denom if denom > 0 else 0.60
                
                n_val = round(val_ratio * len(rem_list))
                
                for inst in rem_list[:n_val]:
                    val_insts.add(inst)
                for inst in rem_list[n_val:]:
                    test_insts.add(inst)
                    
        # 將實例映射回原始資料集的索引
        for inst in train_insts:
            train_indices.extend(instance_indices[inst])
        for inst in val_insts:
            valid_indices.extend(instance_indices[inst])
        for inst in test_insts:
            test_indices.extend(instance_indices[inst])
                
        dataset_folds = [(train_indices, valid_indices, test_indices)]
        num_folds = 1
    else:
        num_folds = len(seeds)
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
            
    best_f1 = -1
    best_seed = None
    best_model_path = ""

    all_f1_scores = []
    cost_times = []
    accuracies = []
    all_class_f1_scores = []

    for i, (t_idx, v_idx, test_indices) in enumerate(dataset_folds):
        train_dataset = Datasubset(full_dataset, t_idx, transform=True)
        valid_dataset = Datasubset(full_dataset, v_idx, transform=False)
        test_dataset = Datasubset(full_dataset, test_indices, transform=False)

        input_dim = full_dataset.dim
        print(f'Fold {i} | Input Dim: {input_dim} | Train: {len(train_dataset)}, Val: {len(valid_dataset)}, Test: {len(test_dataset)}')

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        # 訓練與測試
        # 計算此 Fold 的類別權重 (pos_weight) 以平衡正負樣本失衡
        train_labels = full_dataset.labels[t_idx]
        pos_counts = train_labels.sum(dim=0)
        neg_counts = len(t_idx) - pos_counts
        pos_counts[pos_counts == 0] = 1.0
        pos_weight = (neg_counts / pos_counts).to(device)

        model = PatchTSTClassifier(input_dim, num_classes, input_len).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0003)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs=5, max_epochs=100, min_lr_ratio=0.0)

        save_path = os.path.join(save_dir, f"PatchTST_model_fold{i}.pth")
        txt_dir = os.path.join(save_dir, f"PatchTST_model_fold{i}_results")
        fig_path = os.path.join(txt_dir, f"train_results_fold{i}.png")
        os.makedirs(txt_dir, exist_ok=True)

        train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, save_path, fig_path)

        avg_loss, f1, avg_time_per_sample, accuracy, class_f1 = test_model_with_path_tracking(
            model, test_loader, criterion, txt_dir, save_path, num_classes, sport=args.sport
        )
        print(f"Fold {i} Test F1: {f1:.4f}, Accuracy: {accuracy:.4f}, cost {avg_time_per_sample} sec")
        all_f1_scores.append(f1)
        cost_times.append(avg_time_per_sample)
        accuracies.append(accuracy)
        all_class_f1_scores.append(class_f1)

        if f1 > best_f1:
            best_f1 = f1
            best_seed = i
            best_model_path = save_path

    if args.sport == 'deadlift':
        classes = ['Correct', 'Far from the shins', 'Hips rise first', 'Collide with the knees', 'Lower back rounding']
    else:
        classes = ['Correct', 'tilting to the left', 'tilting to the right', 'scapular protraction', 'elbows flaring']

    write_result(model, num_folds, all_f1_scores, accuracies, cost_times, save_dir, best_f1, best_seed, best_model_path, class_names=classes, class_f1_scores=all_class_f1_scores)