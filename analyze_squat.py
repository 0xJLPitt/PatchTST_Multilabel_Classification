import pandas as pd
import ast
import torch
import numpy as np

csv_file = 'data/squat_dataset_2d.csv'
print("Loading dataset...")
df = pd.read_csv(csv_file)
print(f"Total rows: {len(df)}")

all_labels = []
all_features = []
subjects = []

for _, row in df.iterrows():
    if 'features' in row and 'label' in row:
        features = ast.literal_eval(str(row['features']))
        labels = ast.literal_eval(str(row['label']))
        subject = str(row['subject'])
        
        all_features.append(features)
        all_labels.append(labels)
        subjects.append(subject)

all_labels = np.array(all_labels)
# labels shape: (num_samples, 5)
classes = ['Insufficient_Depth', 'Excessive_Knee_Dominance', 'Excessive_Hip_Dominance', 'Posterior_Pelvic_Tilt', 'Early_Hip_Rise']

print("\n--- Label Distribution ---")
for i, cls in enumerate(classes):
    pos_count = np.sum(all_labels[:, i] == 1)
    neg_count = np.sum(all_labels[:, i] == 0)
    print(f"{cls}: Positives={pos_count}, Negatives={neg_count}, PosRatio={pos_count/len(all_labels):.2%}")

print(f"\nFeature shape (single sample): {np.array(all_features[0]).shape}")
