import os
import argparse
import ast
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import sys
repo_umap_path = os.path.join(os.path.dirname(__file__), 'umap')
if os.path.exists(repo_umap_path):
    sys.path.insert(0, repo_umap_path)
import umap

# Add patchTST path to use models
sys.path.append(os.path.join(os.path.dirname(__file__), 'patchTST'))
from models import PatchTSTClassifier

def get_test_indices_instance_stratified(full_dataset):
    import random
    from collections import defaultdict
    test_indices = []
    
    instance_subject = {}
    instance_primary_label = {}
    instance_indices = defaultdict(list)
    
    for idx in range(len(full_dataset.labels)):
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

class MockDataset:
    def __init__(self, labels, subjects, instances):
        self.labels = labels
        self.subjects = subjects
        self.instances = instances

def load_data(csv_file):
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    features = []
    labels = []
    subjects = []
    instances = []
    
    for _, row in df.iterrows():
        f = ast.literal_eval(str(row['features']))
        l = ast.literal_eval(str(row['label']))
        features.append(f)
        labels.append(l)
        subjects.append(str(row['subject']))
        instances.append(str(row['clip']) if 'clip' in row else (str(row['instance']) if 'instance' in row else str(row['subject'])))
        
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    print(f"Total data shape: {features.shape}, Labels shape: {labels.shape}")
    
    mock_dataset = MockDataset(labels, subjects, instances)
    test_indices = get_test_indices_instance_stratified(mock_dataset)
    test_indices = sorted(list(set(test_indices)))
    print(f"Extracted {len(test_indices)} Test Set samples!")
    
    return features[test_indices], labels[test_indices]

def get_deadlift_classes(num_classes):
    if num_classes == 5:
        return ['Far from the shins', 'Hips rise first', 'Collide with the knees', 'Lower back rounding', 'Correct']
    else:
        return ['Far from the shins', 'Hips rise first', 'Collide with the knees', 'Lower back rounding']

def get_class_colors():
    return ['red', 'green', 'blue', 'orange', 'black']

def extract_embeddings(features, model_path, input_dim, num_classes, input_len):
    print(f"Loading model from {model_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PatchTSTClassifier(input_dim=input_dim, num_classes=num_classes, input_len=input_len)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    embeddings = []
    def hook(module, input, output):
        embeddings.append(input[0].detach().cpu().numpy())
    
    handle = model.classifier[0].register_forward_hook(hook)
    
    print("Extracting embeddings for Test Set...")
    batch_size = 64
    num_samples = features.shape[0]
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_x = features[i:i+batch_size].to(device)
            _ = model(batch_x)
            
    handle.remove()
    embeddings = np.concatenate(embeddings, axis=0)
    print(f"Test Set Embeddings shape: {embeddings.shape}")
    return embeddings

def plot_umap(embedding_2d, labels, class_names, num_classes, save_path):
    print("Plotting Test Set UMAP...")
    colors = get_class_colors()
    
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    for i in range(len(embedding_2d)):
        active_classes = [j for j, label in enumerate(labels[i]) if label == 1]
        
        x, y = embedding_2d[i, 0], embedding_2d[i, 1]
        
        if len(active_classes) == 1:
            c = colors[active_classes[0]]
            ax.scatter(x, y, color=c, alpha=0.7, s=20, edgecolors='w', linewidth=0.5)
        elif len(active_classes) == 2:
            c1, c2 = colors[active_classes[0]], colors[active_classes[1]]
            ax.scatter(x, y, color=c1, alpha=0.7, s=20, marker='<')
            ax.scatter(x, y, color=c2, alpha=0.7, s=20, marker='>')
        elif len(active_classes) > 2:
            ax.scatter(x, y, color='purple', alpha=0.9, s=30, marker='*', edgecolors='w', linewidth=0.5)
        else:
            if num_classes == 4:
                ax.scatter(x, y, color='black', alpha=0.7, s=20, edgecolors='w', linewidth=0.5)
            else:
                ax.scatter(x, y, color='lightgray', alpha=0.7, s=20, edgecolors='w', linewidth=0.5)

    legend_elements = []
    for i, class_name in enumerate(class_names):
        if class_name != 'Correct' or num_classes == 5:
            legend_elements.append(Patch(facecolor=colors[i], label=class_name))
            
    if num_classes == 4:
        legend_elements.append(Patch(facecolor='black', label='Correct (All 0s)'))
        
    legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='purple', markersize=10, label='>= 3 errors'))

    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title="Classes")
    plt.title(f'UMAP of PatchTST Test Set Embeddings ({num_classes} Classes)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Test Set UMAP plot to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze Deadlift Test Set UMAP')
    parser.add_argument('--csv', type=str, default='data/deadlift_dataset_3d_5class.csv')
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    
    out_dir = os.path.join("umap_results", os.path.basename(args.csv).split('.')[0])
    os.makedirs(out_dir, exist_ok=True)
    
    features, labels = load_data(args.csv)
    
    input_dim = features.shape[2]
    num_classes = labels.shape[1]
    input_len = features.shape[1]
    
    class_names = get_deadlift_classes(num_classes)
    
    embeddings = extract_embeddings(
        features, args.model, input_dim, num_classes, input_len
    )
    
    print("Running UMAP dimensionality reduction on Test Set...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    out_name = f"umap_testset_{num_classes}class.png"
    save_path = os.path.join(out_dir, out_name)
    
    plot_umap(embedding_2d, labels.numpy(), class_names, num_classes, save_path)

if __name__ == '__main__':
    main()
