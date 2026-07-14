import sys
import os
# 加入父目錄以取得 dataset 與 tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import random
import json
import math
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# 引入原本的 Dataset 與工具
from dataset import Dataset_Deadlift, Datasubset, Dataset_Benchpress
from tools import compute_f1_score, write_result
from PatchTST_test import test_model_with_path_tracking
from itransformer_model import iTransformer_Classification # 從當前目錄引入 iTransformer 模型

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        # inputs 已經是 logits (無 Sigmoid)，剛好適合使用 binary_cross_entropy_with_logits
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=self.pos_weight)
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * bce_loss
        return focal_loss.mean()

def get_warmup_cosine_scheduler(optimizer, warmup_epochs: int, max_epochs: int, min_lr_ratio: float = 0.0):
    assert warmup_epochs < max_epochs
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(warmup_epochs)
        progress = (current_epoch - warmup_epochs) / float(max_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, save_path, fig_path, num_epochs=150, patience=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_f1 = 0.0
    patience_counter = 0

    train_losses, train_f1_scores, val_losses, val_f1_scores = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        y_true, y_pred = [], []

        for inputs, labels, indices in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # 模型 forward，輸出為 logits
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.tolist())

        avg_loss = total_loss / len(train_loader)
        train_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Validation phase
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for val_inputs, val_labels, _ in valid_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                v_loss = criterion(val_outputs, val_labels)
                val_loss_total += v_loss.item()
        avg_val_loss = val_loss_total / len(valid_loader) if len(valid_loader) > 0 else 0.0
        val_f1 = compute_f1_score(model, valid_loader)

        scheduler.step()
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        train_losses.append(avg_loss)
        val_losses.append(avg_val_loss)
        train_f1_scores.append(train_f1)
        val_f1_scores.append(val_f1)

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    epochs = range(1, len(train_losses) + 1)
    
    # Loss Plot
    ax1.plot(epochs, train_losses, label="Train Loss", color='blue', marker='o', markersize=3)
    ax1.plot(epochs, val_losses, label="Validation Loss", color='orange', marker='s', markersize=3)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss per Epoch")
    ax1.legend()
    ax1.grid(True)
    
    # F1-score Plot
    ax2.plot(epochs, train_f1_scores, label="Train F1-score", color='green', marker='o', markersize=3)
    ax2.plot(epochs, val_f1_scores, label="Validation F1-score", color='red', marker='d', markersize=3)
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Training & Validation F1-score per Epoch")
    ax2.legend()
    ax2.grid(True)
    
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--sport', type=str, choices=['benchpress', 'deadlift'], default='deadlift')
    parser.add_argument('--type', type=str, choices=['2d', '3d', '2D', '3D'], default='3D', help='Feature type for deadlift')
    parser.add_argument('--split_mode', type=str, choices=['instance_stratified'], default='instance_stratified', help='Data split mode')
    parser.add_argument('--tag', type=str, default='baseline', help='Tag for save_dir')
    parser.add_argument('--max_epochs', type=int, default=150, help='Max training epochs')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    
    # iTransformer specific arguments
    parser.add_argument('--d_model', type=int, default=128, help='Hidden size for iTransformer')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of encoder layers')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dim_feedforward', type=int, default=512, help='Feedforward hidden size')
    parser.add_argument('--pooling', type=str, choices=['mean', 'flatten'], default='mean', help='Pooling strategy')
    
    args = parser.parse_args()

    print(f"--- iTransformer Training Settings ---")
    print(f"Sport: {args.sport}, Type: {args.type}")
    print(f"d_model: {args.d_model}, Num Layers: {args.num_layers}, nhead: {args.nhead}, Dropout: {args.dropout}")
    
    if args.sport == 'deadlift':
        feat_type = args.type.upper()
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', f'deadlift_dataset_{args.type.lower()}.csv')
        full_dataset = Dataset_Deadlift(data_path)
        save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints', f'iTransformer_Deadlift_{feat_type}', args.tag)
        num_classes = 4
        input_len = 110 # seq_len
    else:
        # Benchpress
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'benchpress_dataset.csv')
        full_dataset = Dataset_Benchpress(data_path)
        save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints', f'iTransformer_Benchpress', args.tag)
        num_classes = 4
        input_len = 100 # seq_len
        
    os.makedirs(save_dir, exist_ok=True)
        
    # Data Splitting logic (instance_stratified)
    train_indices, valid_indices, test_indices = [], [], []
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
            
    subject_instances = defaultdict(list)
    for inst, label in instance_primary_label.items():
        sub = instance_subject[inst]
        subject_instances[sub].append(inst)
        
    train_insts, val_insts, test_insts = set(), set(), set()
    class_free_insts, class_total_count, class_already_train = defaultdict(list), defaultdict(int), defaultdict(int)
    
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
                
    for inst in train_insts: train_indices.extend(instance_indices[inst])
    for inst in val_insts: valid_indices.extend(instance_indices[inst])
    for inst in test_insts: test_indices.extend(instance_indices[inst])
            
    dataset_folds = [(train_indices, valid_indices, test_indices)]
    
    best_f1, best_seed, best_model_path = -1, None, ""
    all_f1_scores, cost_times, accuracies, all_class_f1_scores = [], [], [], []

    for i, (t_idx, v_idx, test_indices) in enumerate(dataset_folds):
        train_dataset = Datasubset(full_dataset, t_idx, transform=True)
        valid_dataset = Datasubset(full_dataset, v_idx, transform=False)
        test_dataset = Datasubset(full_dataset, test_indices, transform=False)

        input_dim = full_dataset.dim # i.e. num_variates
        print(f'Fold {i} | Seq Len: {input_len} | Input Dim(Variates): {input_dim} | Train: {len(train_dataset)}, Val: {len(valid_dataset)}, Test: {len(test_dataset)}')

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=args.num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=args.num_workers)

        train_labels = full_dataset.labels[t_idx]
        pos_counts = train_labels.sum(dim=0)
        neg_counts = len(t_idx) - pos_counts
        pos_counts[pos_counts == 0] = 1.0
        pos_weight = (neg_counts / pos_counts).to(device)
        
        # Initialize iTransformer
        model = iTransformer_Classification(
            seq_len=input_len, 
            num_variates=input_dim, 
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            num_classes=num_classes,
            dropout=args.dropout,
            pooling=args.pooling
        ).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        criterion = FocalLoss(gamma=2.0, pos_weight=pos_weight)
        scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs=5, max_epochs=args.max_epochs, min_lr_ratio=0.0)

        save_path = os.path.join(save_dir, f"iTransformer_model_fold{i}.pth")
        txt_dir = os.path.join(save_dir, f"iTransformer_model_fold{i}_results")
        fig_path = os.path.join(txt_dir, f"train_results_fold{i}.png")
        os.makedirs(txt_dir, exist_ok=True)

        train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, save_path, fig_path, num_epochs=args.max_epochs, patience=50)

        # 這裡會使用到 tools 裡面的 PatchTST_test 來進行推論測試
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

    write_result(model, 1, all_f1_scores, accuracies, cost_times, save_dir, best_f1, best_seed, best_model_path, class_names=classes, class_f1_scores=all_class_f1_scores)
