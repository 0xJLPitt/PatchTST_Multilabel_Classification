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
    def __init__(self, dataset, indices, transform=False, aug_type=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.aug_type = aug_type
        
        self.augmented_x_list = []
        if self.aug_type:
            aug_types = self.aug_type.split('+')
            print(f"Applying augmentations: {aug_types} (this might take a while)...")
            # extract subset features
            all_x = []
            all_y = []
            for i in self.indices:
                x, y, _ = self.dataset[i]
                all_x.append(x)
                all_y.append(y)
            X = torch.stack(all_x).numpy()
            Y = torch.stack(all_y).numpy()
            
            import sys, os
            ts_aug_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ts_aug'))
            if ts_aug_path not in sys.path:
                sys.path.append(ts_aug_path)
            from utils import augmentation
            
            for aug in aug_types:
                if aug == "window_warping":
                    X_aug = augmentation.window_warp(X)
                elif aug == "jittering":
                    X_aug = augmentation.jitter(X)
                elif aug == "spawner":
                    X_aug = augmentation.spawner(X, Y)
                elif aug == "dtwwarp":
                    X_aug = augmentation.random_guided_warp(X, Y)
                elif aug == "shapedtw":
                    X_aug = augmentation.random_guided_warp_shape(X, Y)
                elif aug == "discdtw":
                    X_aug = augmentation.discriminative_guided_warp(X, Y)
                else:
                    X_aug = X
                self.augmented_x_list.append(torch.from_numpy(X_aug).float())
            print(f"Augmentations {aug_types} applied successfully.")

    def __len__(self):
        if self.aug_type:
            return len(self.indices) * (1 + len(self.aug_type.split('+')))
        return len(self.indices)

    def __getitem__(self, idx):
        n = len(self.indices)
        if self.aug_type:
            if idx >= n:
                aug_idx = (idx - n) // n
                true_idx_offset = (idx - n) % n
                true_idx = self.indices[true_idx_offset]
                _, y, _ = self.dataset[true_idx]
                x = self.augmented_x_list[aug_idx][true_idx_offset]
                return x, y, true_idx
            
        true_idx = self.indices[idx]
        x, y, _ = self.dataset[true_idx]
        return x, y, true_idx
