import os
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


def test_model_with_path_tracking(model, test_loader, criterion, txt_dir, save_path, num_classes, sport='deadlift'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.to(device)
    model.eval()
    
    # **存放測試過程的數據**
    total_loss, total_time = 0.0, 0.0  
    y_true, y_pred = [], []
    
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

    avg_loss = total_loss / len(test_loader)
    avg_time_per_sample = total_time / len(y_true)
    f1 = f1_score(y_true, y_pred, average='macro')

    # 繪製混淆矩陣
    if sport == 'deadlift':
        classes = ['Correct', 'Far from the shins', 'Hips rise first', 'Collide with the knees', 'Lower back rounding']
    else:
        classes = ['Correct', 'tilting to the left', 'tilting to the right', 'scapular protraction', 'elbows flaring']
    binary_classes = classes[1:]
    
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
    y_true_correct = (np.sum(y_true_np, axis=1) == 0).astype(int)
    y_pred_correct = (np.sum(y_pred_np, axis=1) == 0).astype(int)
    f1_correct = f1_score(y_true_correct, y_pred_correct, average='binary', zero_division=0)
    
    f1_errors = f1_score(y_true_np, y_pred_np, average=None, zero_division=0)
    class_f1 = [f1_correct] + list(f1_errors)

    return avg_loss, f1, avg_time_per_sample, accuracy, class_f1

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--sport', type=str, choices=['benchpress', 'deadlift'])
    args = parser.parse_args()
    
    from dataset import *
    if args.sport == 'deadlift':
        data_path = os.path.join(os.getcwd(), 'data', 'deadlift_dataset.csv')
        test_dataset = Dataset_Deadlift(data_path)
        output_dir = './models/deadlift/TST_Deadlift/12'
        save_dir = './models/deadlift/TST_Deadlift/12'
        num_classes = 4
        input_len = 110
    elif args.sport == 'benchpress':
        data_path = os.path.join(os.getcwd(), 'data', 'benchpress_dataset.csv')
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
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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