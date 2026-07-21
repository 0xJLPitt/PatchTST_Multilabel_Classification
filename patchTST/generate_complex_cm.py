import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from models import PatchTSTClassifier
from dataset import Dataset_Deadlift
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

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

def vector_to_label(vec):
    classes = ['Far from the shins', 'Hips rise first', 'Collide with the knees', 'Lower back rounding']
    active = [classes[i] for i, val in enumerate(vec) if val == 1]
    if not active:
        return 'Correct'
    return ' + '.join(active)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'deadlift_dataset_3d.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'deadlift_dataset_3d.csv')
        
    full_dataset = Dataset_Deadlift(data_path)
    test_indices = get_test_indices_instance_stratified(full_dataset)
    test_dataset = Subset(full_dataset, test_indices)
    
    loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    import sys
    
    input_dim = full_dataset.dim
    num_classes = 4
    input_len = 110
    
    # Allow passing base_dir as an argument
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = '/home/pitt_huang/workspace/fitness_action_recognition/patchTST/models/deadlift/TST_Deadlift_3D/phase4.8_test_heads8_dtwwarp'
        
    model = PatchTSTClassifier(input_dim, num_classes, input_len, num_heads=8).to(device)
    save_path = os.path.join(base_dir, 'PatchTST_model_fold0.pth')
    
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    
    y_true_str = []
    y_pred_str = []
    
    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()
            
            labels_np = labels.cpu().numpy()
            preds_np = preds.cpu().numpy()
            
            for i in range(len(labels_np)):
                y_true_str.append(vector_to_label(labels_np[i]))
                y_pred_str.append(vector_to_label(preds_np[i]))

    unique_true_labels = set(y_true_str)
    unique_pred_labels = set(y_pred_str)
    all_unique_labels = sorted(list(unique_true_labels.union(unique_pred_labels)))
    
    if 'Correct' in all_unique_labels:
        all_unique_labels.remove('Correct')
        all_unique_labels = ['Correct'] + all_unique_labels
        
    cm = confusion_matrix(y_true_str, y_pred_str, labels=all_unique_labels)
    
    out_dir = os.path.join(base_dir, 'PatchTST_model_fold0_results_new_metrix')
    os.makedirs(out_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(20, 18))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_unique_labels)
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha('right')
        tick.set_rotation_mode('anchor')
        
    plt.title('Multi-label Compound Confusion Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix_complex.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    df_cm = pd.DataFrame(cm, index=all_unique_labels, columns=all_unique_labels)
    df_cm.to_csv(os.path.join(out_dir, 'confusion_matrix_complex.csv'))
    
    print(f"Successfully generated 15x15 (or size {len(all_unique_labels)}x{len(all_unique_labels)}) compound confusion matrix.")
    print(f"Results saved to: {out_dir}")

if __name__ == "__main__":
    main()
