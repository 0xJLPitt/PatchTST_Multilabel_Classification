import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
from torch.utils.data import DataLoader
from models import PatchTSTClassifier
from dataset import Dataset_Deadlift, Datasubset
import argparse

def find_misclassified():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'deadlift_dataset_3d.csv')
    full_dataset = Dataset_Deadlift(data_path)
    
    input_dim = full_dataset.dim
    num_classes = 4
    input_len = 110
    
    model = PatchTSTClassifier(input_dim, num_classes, input_len, num_heads=8).to(device)
    save_path = './models/deadlift/TST_Deadlift_3D/phase4.8_test_heads8_dtwwarp/PatchTST_model_fold0.pth'
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    
    loader = DataLoader(full_dataset, batch_size=32, shuffle=False)
    
    instances = full_dataset.instances
    
    case1 = set() # True: Correct, Pred: Far from the shins
    case2 = set() # True: Correct, Pred: Collide with the knees
    case3 = set() # True: Lower back rounding, Pred: Far from the shins
    case4 = set() # True: Collide with the knees, Pred: Lower back rounding
    
    with torch.no_grad():
        for inputs, labels, indices in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()
            
            labels = labels.cpu().numpy()
            preds = preds.cpu().numpy()
            indices = indices.cpu().numpy()
            
            for i in range(len(labels)):
                true_lbl = labels[i]
                pred_lbl = preds[i]
                idx = indices[i]
                inst = instances[idx]
                
                is_correct_true = np.sum(true_lbl) == 0
                is_correct_pred = np.sum(pred_lbl) == 0
                
                # Case 1: True: Correct, Pred: Far from the shins
                if is_correct_true and pred_lbl[0] == 1:
                    case1.add(inst)
                
                # Case 2: True: Correct, Pred: Collide with the knees
                if is_correct_true and pred_lbl[2] == 1:
                    case2.add(inst)
                    
                # Case 3: True: Lower back rounding, Pred: Far from the shins
                if true_lbl[3] == 1 and pred_lbl[0] == 1:
                    case3.add(inst)
                    
                # Case 4: True: Collide with the knees, Pred: Lower back rounding
                if true_lbl[2] == 1 and pred_lbl[3] == 1:
                    case4.add(inst)
                    
    with open('/home/pitt_huang/.gemini/antigravity-ide/brain/7aac2756-802a-4861-b574-4e9f40655f93/misclassified_folders.md', 'w') as f:
        f.write("# Misclassified Instances (Folders)\n\n")
        
        f.write(f"## 1. True: Correct, Pred: Far from the shins (Total: {len(case1)})\n")
        for inst in sorted(list(case1)):
            f.write(f"- `{inst}`\n")
            
        f.write(f"\n## 2. True: Correct, Pred: Collide with the knees (Total: {len(case2)})\n")
        for inst in sorted(list(case2)):
            f.write(f"- `{inst}`\n")
            
        f.write(f"\n## 3. True: Lower back rounding, Pred: Far from the shins (Total: {len(case3)})\n")
        for inst in sorted(list(case3)):
            f.write(f"- `{inst}`\n")
            
        f.write(f"\n## 4. True: Collide with the knees, Pred: Lower back rounding (Total: {len(case4)})\n")
        for inst in sorted(list(case4)):
            f.write(f"- `{inst}`\n")

if __name__ == "__main__":
    find_misclassified()
