from torch.utils.data import Dataset
import torch
import random

class Dataset_Benchpress(Dataset):
    def __init__(self, csv_file):
        import pandas as pd
        import ast
        self.features = []
        self.labels = []
        self.subjects = []
        self.instances = []
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            if 'features' in row and 'label' in row:
                features = ast.literal_eval(str(row['features']))
                labels = ast.literal_eval(str(row['label']))
                subject = str(row['subject'])
                instance = str(row['instance']) if 'instance' in row else subject
                self.features.append(torch.tensor(features).float())
                self.labels.append(torch.tensor(labels).float())
                self.subjects.append(subject)
                self.instances.append(instance)
        
        self.features = torch.stack(self.features) if self.features else torch.tensor([])
        self.labels = torch.stack(self.labels) if self.labels else torch.tensor([])
        self.dim = self.features.shape[-1] if len(self.features) > 0 else 0

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return x, y, idx

class Dataset_Deadlift(Dataset):
    def __init__(self, csv_file):
        import pandas as pd
        import ast
        self.features = []
        self.labels = []
        self.subjects = []
        self.instances = []
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            if 'features' in row and 'label' in row:
                features = ast.literal_eval(str(row['features']))
                labels = ast.literal_eval(str(row['label']))
                subject = str(row['subject'])
                set_val = str(row['set']) if 'set' in row else '1'
                instance = f"{subject}_set{set_val}"
                self.features.append(torch.tensor(features).float())
                self.labels.append(torch.tensor(labels).float())
                self.subjects.append(subject)
                self.instances.append(instance)
        
        self.features = torch.stack(self.features) if self.features else torch.tensor([])
        self.labels = torch.stack(self.labels) if self.labels else torch.tensor([])
        self.dim = self.features.shape[-1] if len(self.features) > 0 else 0

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return x, y, idx


class Datasubset(Dataset):
    def __init__(self, dataset, indices, transform=False):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y, true_idx = self.dataset[self.indices[idx]]
        if self.transform:
            # 1. Random Scaling (0.9 to 1.1)
            scale = 0.9 + 0.2 * torch.rand(1).item()
            x = x * scale
            
            # 2. Random Jittering (稍微調弱：std=0.03 -> 0.02)
            noise = torch.randn_like(x) * 0.02
            x = x + noise
            
            # 3. Time Masking (配合更小的 patch_len=8，將遮蔽時間縮小至隨機 1~3 個 frame)
            seq_len, dim = x.shape
            mask_len = torch.randint(1, 4, (1,)).item()
            if seq_len > mask_len:
                start = torch.randint(0, seq_len - mask_len, (1,)).item()
                x[start:start+mask_len, :] = 0.0

            # 4. Point Masking (隨機點 dropout 降回 5%)
            mask = (torch.rand_like(x) > 0.05).float()
            x = x * mask
        return x, y, true_idx
