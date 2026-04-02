import pandas as pd
import sys

file_path = "Выгрузка по выданным субсидиям 2025 год (обезлич) (1).xlsx"

try:
    # Read the first few rows to understand the structure
    df = pd.read_excel(file_path, nrows=5)
    print("Columns in the dataset:")
    print(df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.to_string())
    
    # Get basic info
    df_full_info = pd.read_excel(file_path, nrows=1) # Just to get headers if needed
    print("\nDataset summary (first 5 rows):")
    print(df.info())
except Exception as e:
    print(f"Error reading file: {e}")
