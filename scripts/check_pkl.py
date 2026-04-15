import joblib
import numpy as np
from pprint import pprint

file_path = '/home/ubuntu/Desktop/BAK/BFM/models/data/lafan_29dof_10s-clipped.pkl'

try:
    data = joblib.load(file_path)
    print("--- File Loaded Successfully with Joblib ---")
    
    # Quick inspection
    if isinstance(data, dict):
        print(f"Keys found: {list(data.keys())[:5]}...") 
except Exception as e:
    print(f"Joblib also failed: {e}")

pprint(data["walk3_subject2_clip8"].keys())