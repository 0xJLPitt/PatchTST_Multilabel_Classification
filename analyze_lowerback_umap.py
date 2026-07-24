"""
Example usage:
conda run -n cu13 python analyze_lowerback_umap.py --csv data/deadlift_dataset_3d_5class.csv --model patchTST/models/deadlift/TST_Deadlift_3D/phase4.8_test_heads8_dtwwarp_5class/PatchTST_model_fold0.pth
"""

import os
import sys
import ast
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Ensure local umap repository is correctly imported by pointing to its root
repo_umap_path = os.path.join(os.path.dirname(__file__), 'umap')
if os.path.exists(repo_umap_path):
    sys.path.insert(0, repo_umap_path)

import umap

# Add patchTST path so we can import the model
sys.path.append(os.path.join(os.path.dirname(__file__), 'patchTST'))
from patchTST.models import PatchTSTClassifier

def get_deadlift_classes(num_classes):
    if num_classes == 5:
        return ['Far from the shins', 'Hips rise first', 'Collide with the knees', 'Lower back rounding', 'Correct']
    elif num_classes == 4:
        return ['Far from the shins', 'Hips rise first', 'Collide with the knees', 'Lower back rounding']
    else:
        return ['Correct', 'Far from the shins', 'Hips rise first', 'Collide with the knees', 'Lower back rounding']

def load_data(csv_file):
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    features = []
    labels = []
    for _, row in df.iterrows():
        if 'features' in row and 'label' in row:
            feat = ast.literal_eval(str(row['features']))
            lbl = ast.literal_eval(str(row['label']))
            features.append(feat)
            labels.append(lbl)
    
    features = np.array(features)
    labels = np.array(labels)
    print(f"Data shape: {features.shape}, Labels shape: {labels.shape}")
    return torch.tensor(features).float(), torch.tensor(labels).float()

def extract_embeddings_and_predictions(features, model_path, input_dim, num_classes, input_len):
    print(f"Loading model from {model_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = PatchTSTClassifier(input_dim, num_classes, input_len, num_heads=8).to(device)
    state_dict = torch.load(model_path, map_location=device)
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    
    model.eval()
    
    embeddings = []
    predictions = []
    
    def hook(module, input, output):
        embeddings.append(input[0].detach().cpu().numpy())
    
    handle = model.classifier[0].register_forward_hook(hook)
    
    print("Extracting embeddings and predictions...")
    batch_size = 64
    features = features.to(device)
    
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch = features[i:i+batch_size]
            logits = model(batch)
            preds = (logits > 0).float().cpu().numpy()
            predictions.append(preds)
            
    handle.remove()
    
    embeddings = np.concatenate(embeddings, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    print(f"Embeddings shape: {embeddings.shape}, Predictions shape: {predictions.shape}")
    return embeddings, predictions

def plot_confusion_umap(embedding_2d, labels, predictions, class_names, target_class, save_path, title):
    if target_class not in class_names:
        raise ValueError(f"Target class '{target_class}' not found in classes {class_names}")
    
    target_idx = class_names.index(target_class)
    
    y_true = labels[:, target_idx]
    y_pred = predictions[:, target_idx]
    
    # Classify into TP, FP, FN, TN
    tp_mask = (y_true == 1) & (y_pred == 1)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    tn_mask = (y_true == 0) & (y_pred == 0)
    
    print(f"--- Confusion Matrix for '{target_class}' ---")
    print(f"True Positives (TP): {tp_mask.sum()}")
    print(f"False Positives (FP): {fp_mask.sum()}")
    print(f"False Negatives (FN): {fn_mask.sum()}")
    print(f"True Negatives (TN): {tn_mask.sum()}")
    
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    # Plot TN first (background)
    if tn_mask.sum() > 0:
        ax.scatter(embedding_2d[tn_mask, 0], embedding_2d[tn_mask, 1], 
                   color='lightgray', label=f'TN (Normal)', alpha=0.5, s=20)
        
    # Plot FN (Blue)
    if fn_mask.sum() > 0:
        ax.scatter(embedding_2d[fn_mask, 0], embedding_2d[fn_mask, 1], 
                   color='blue', label=f'FN (Missed/沒抓到)', alpha=0.9, s=40)
        
    # Plot FP (Green)
    if fp_mask.sum() > 0:
        ax.scatter(embedding_2d[fp_mask, 0], embedding_2d[fp_mask, 1], 
                   color='green', label=f'FP (Wrong/誤判)', alpha=0.9, s=40)
        
    # Plot TP (Orange)
    if tp_mask.sum() > 0:
        ax.scatter(embedding_2d[tp_mask, 0], embedding_2d[tp_mask, 1], 
                   color='orange', label=f'TP (Correct/正確抓到)', alpha=0.9, s=40)

    plt.title(title, fontsize=16)
    plt.legend(loc='best', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved UMAP plot to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='data/deadlift_dataset_3d_5class.csv')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--target_class', type=str, default='Lower back rounding')
    parser.add_argument('--use_raw', action='store_true', help='Use raw features for UMAP instead of model embeddings')
    args = parser.parse_args()
    
    out_dir = os.path.join("umap_results", os.path.basename(args.csv).split('.')[0])
    os.makedirs(out_dir, exist_ok=True)
    
    features, labels = load_data(args.csv)
    num_samples, input_len, input_dim = features.shape
    num_classes = labels.shape[1]
    class_names = get_deadlift_classes(num_classes)
    
    embeddings, predictions = extract_embeddings_and_predictions(
        features, args.model, input_dim, num_classes, input_len
    )
    
    if args.use_raw:
        print("Using Raw Data for UMAP...")
        data_to_umap = features.reshape(features.shape[0], -1).numpy()
        out_name = f"umap_{args.target_class.replace(' ', '_')}_confusion_raw.png"
        title = f"UMAP Confusion (Raw Data) - {args.target_class}"
    else:
        print("Using Model Embeddings for UMAP...")
        data_to_umap = embeddings
        out_name = f"umap_{args.target_class.replace(' ', '_')}_confusion.png"
        title = f"UMAP Confusion (Model Embeddings) - {args.target_class}"
    
    print("Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    embedding_2d = reducer.fit_transform(data_to_umap)
    
    save_path = os.path.join(out_dir, out_name)
    
    plot_confusion_umap(embedding_2d, labels.numpy(), predictions, class_names, args.target_class, save_path, title)

if __name__ == "__main__":
    main()
