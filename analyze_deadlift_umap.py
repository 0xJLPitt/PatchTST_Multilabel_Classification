"""
Example usage:
1. For 5-class deadlift dataset:
conda run -n cu13 python analyze_deadlift_umap.py --csv data/deadlift_dataset_3d_5class.csv --model patchTST/models/deadlift/TST_Deadlift_3D/phase4.8_test_heads8_dtwwarp_5class/PatchTST_model_fold0.pth

2. For 4-class deadlift dataset:
conda run -n cu13 python analyze_deadlift_umap.py --csv data/deadlift_dataset_3d.csv --model patchTST/models/deadlift/TST_Deadlift_3D/phase4.8_test_heads8_dtwwarp/PatchTST_model_fold0.pth
"""

import os
import sys
import ast
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
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
        # According to PatchTST_train.py
        return ['Far from the shins', 'Hips rise first', 'Collide with the knees', 'Lower back rounding', 'Correct']
    elif num_classes == 4:
        return ['Far from the shins', 'Hips rise first', 'Collide with the knees', 'Lower back rounding']
    else:
        return ['Correct', 'Far from the shins', 'Hips rise first', 'Collide with the knees', 'Lower back rounding']

def get_class_colors():
    # Assign distinct colors to the 5 classes
    return ['red', 'green', 'blue', 'orange', 'black']

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

def extract_embeddings(features, model_path, input_dim, num_classes, input_len):
    print(f"Loading model from {model_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Instantiate the model with same hyperparams as training
    # For heads8 model, num_heads=8.
    model = PatchTSTClassifier(input_dim, num_classes, input_len, num_heads=8).to(device)
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    
    model.eval()
    
    # We will use a forward hook to extract the embedding before the classifier
    embeddings = []
    
    def hook(module, input, output):
        # input to classifier is a tuple, we want the first element (B, embed_dim)
        embeddings.append(input[0].detach().cpu().numpy())
    
    # Register hook on the first layer of the classifier (LayerNorm)
    handle = model.classifier[0].register_forward_hook(hook)
    
    print("Extracting embeddings...")
    batch_size = 64
    features = features.to(device)
    
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch = features[i:i+batch_size]
            model(batch)
            
    handle.remove()
    
    # embeddings is a list of batches, concatenate them
    embeddings = np.concatenate(embeddings, axis=0)
    print(f"Embeddings extracted, shape: {embeddings.shape}")
    return embeddings

def plot_umap(embedding_2d, labels, class_names, colors, save_path, title):
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    legend_elements = []
    for i, (c_name, color) in enumerate(zip(class_names, colors)):
        legend_elements.append(mpatches.Patch(color=color, label=c_name))
        
    if len(class_names) == 4:
        legend_elements.append(mpatches.Patch(color='black', label='Correct (All 0s)'))
        
    for i in range(len(embedding_2d)):
        x, y = embedding_2d[i]
        lbl = labels[i]
        
        active_classes = np.where(lbl == 1)[0]
        
        if len(active_classes) == 0:
            ax.plot(x, y, marker='o', color='black', linestyle='None', markersize=8)
        elif len(active_classes) == 1:
            # Single error or single "Correct" label
            c_idx = active_classes[0]
            ax.plot(x, y, marker='o', color=colors[c_idx], linestyle='None', markersize=8, alpha=0.8, markeredgecolor='none')
        elif len(active_classes) == 2:
            # Two errors -> left/right half coloring
            c1_idx = active_classes[0]
            c2_idx = active_classes[1]
            c1_color = colors[c1_idx]
            c2_color = colors[c2_idx]
            
            ax.plot(x, y, marker='o', markerfacecolor=c1_color, markerfacecoloralt=c2_color,
                    fillstyle='left', linestyle='None', markersize=9, markeredgecolor='none', alpha=0.9)
        else:
            # 3 or more errors
            ax.plot(x, y, marker='*', color='purple', linestyle='None', markersize=12, alpha=0.9)

    plt.title(title, fontsize=16)
    plt.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved UMAP plot to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Run UMAP on deadlift datasets")
    parser.add_argument('--csv', type=str, default='data/deadlift_dataset_3d_5class.csv', help='Path to dataset CSV')
    parser.add_argument('--model', type=str, default=None, help='Path to PatchTST model weights')
    args = parser.parse_args()
    
    # Ensure output dir exists
    out_dir = os.path.join("umap_results", os.path.basename(args.csv).split('.')[0])
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load Data
    features, labels = load_data(args.csv)
    
    num_samples, input_len, input_dim = features.shape
    num_classes = labels.shape[1]
    
    class_names = get_deadlift_classes(num_classes)
    colors = get_class_colors()
    
    # 2. Extract Features
    if args.model and os.path.exists(args.model):
        print("Using Embeddings from model...")
        data_to_umap = extract_embeddings(features, args.model, input_dim, num_classes, input_len)
        title_prefix = "UMAP (Model Embeddings)"
        out_name = "umap_embeddings.png"
    else:
        print("Using Raw Data...")
        data_to_umap = features.view(num_samples, -1).numpy()
        title_prefix = "UMAP (Raw Data Flattened)"
        out_name = "umap_raw.png"
        
    # 3. Run UMAP
    print("Running UMAP dimensionality reduction (this might take a few moments)...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    embedding_2d = reducer.fit_transform(data_to_umap)
    
    # 4. Plot
    save_path = os.path.join(out_dir, out_name)
    title = f"{title_prefix} - {os.path.basename(args.csv)}"
    plot_umap(embedding_2d, labels.numpy(), class_names, colors, save_path, title)
    
    print("Analysis complete.")

if __name__ == "__main__":
    main()
