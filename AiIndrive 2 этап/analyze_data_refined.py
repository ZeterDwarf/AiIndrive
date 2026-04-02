import pandas as pd

file_path = "Выгрузка по выданным субсидиям 2025 год (обезлич) (1).xlsx"

try:
    # Row 3 (index 3) seems to contain the headers
    df = pd.read_excel(file_path, skiprows=3)
    
    # Let's see the actual columns now
    print("Cleaned Columns:")
    print(df.columns.tolist())
    
    print("\nFirst 10 rows of cleaned data:")
    print(df.head(10).to_string())
    
    # Check for missing values and basic stats
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    print("\nDataset Shape:")
    print(df.shape)
    
    # Unique values in some key columns
    if 'Статус заявки' in df.columns:
        print("\nStatuses Distribution:")
        print(df['Статус заявки'].value_counts())
    
    if 'Область' in df.columns:
        print("\nRegion Distribution:")
        print(df['Область'].value_counts())

except Exception as e:
    print(f"Error reading file: {e}")
