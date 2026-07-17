import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../ts_aug')))
from utils import augmentation
print(dir(augmentation))
