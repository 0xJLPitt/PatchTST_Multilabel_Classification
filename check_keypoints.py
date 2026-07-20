import pandas as pd
import ast
import numpy as np

csv_file = 'data/squat_dataset_2d.csv'
df = pd.read_csv(csv_file, nrows=10) # load a few rows

for _, row in df.iterrows():
    if 'features' in row:
        features = ast.literal_eval(str(row['features']))
        features = np.array(features)
        # features shape: (110, 50)
        mean_y = np.mean(features, axis=0)[1::2] # Odd indices are Y
        print("Mean Y coordinates for the 25 keypoints:")
        for i, y in enumerate(mean_y):
            print(f"Keypoint {i}: {y:.2f}")
        break
