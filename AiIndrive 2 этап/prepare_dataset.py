import pandas as pd
import numpy as np
import os

def prepare_data():
    file_path = "Выгрузка по выданным субсидиям 2025 год (обезлич) (1).xlsx"
    print(f"Reading {file_path} using fast engine...")
    
    # Try reading without header parsing first to see what's happening
    try:
        # Use openpyxl specifically, sometimes it handles large files better in non-streaming mode
        # or we can try reading only the first 5000 rows for the prototype if the file is too large
        df = pd.read_excel(file_path, skiprows=3, engine='openpyxl')
        
        # Identify the header row (contains '№ п/п', 'Дата поступления', etc.)
        # The first row of the resulting dataframe should be our header
        df.columns = df.iloc[0]
        df = df.drop(0).reset_index(drop=True)
        
        # Filter out empty rows or non-data rows
        # The actual data column names might be tricky, let's map them by index if names fail
        # Typical columns based on previous 'analyze' call:
        # 0: № п/п, 1: Дата поступления, 4: Область, 5: Акимат, 6: Номер заявки, 
        # 7: Направление водства, 8: Наименование субсидирования, 9: Статус заявки, 
        # 10: Норматив, 11: Причитающая сумма, 12: Район хозяйства
        
        print(f"Columns found: {df.columns.tolist()[:5]}...")
        
        # Rename columns to standard ones for easier processing
        col_map = {
            df.columns[0]: 'ID',
            df.columns[1]: 'Date',
            df.columns[4]: 'Region',
            df.columns[5]: 'Akimat',
            df.columns[6]: 'App_Number',
            df.columns[7]: 'Category',
            df.columns[8]: 'Sub_Name',
            df.columns[9]: 'Status',
            df.columns[10]: 'Quota',
            df.columns[11]: 'Amount',
            df.columns[12]: 'District'
        }
        df = df.rename(columns=col_map)
        
        # Filter valid rows
        df = df[df['ID'].notnull()]
        print(f"Valid data rows: {len(df)}")

        # 1. Clean Numeric Data
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df['Quota'] = pd.to_numeric(df['Quota'], errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # 2. Synthetic Feature Engineering (Merit-based)
        np.random.seed(42)
        size = len(df)
        
        # Productivity Growth (%)
        df['Productivity_Growth'] = np.random.normal(7, 12, size) 
        
        # Tax/Social Return Index (Taxes paid per Million Tenge of subsidy)
        df['Tax_Return_Index'] = np.random.uniform(0.05, 0.9, size)
        
        # Technological Modernization Score (0-100)
        df['Tech_Score'] = np.random.randint(30, 100, size)
        
        # Past Violation History (0 = Clean, 1 = Violation)
        df['Past_Violations'] = np.random.choice([0, 1], size=size, p=[0.94, 0.06])
        
        # Regional Priority Multiplier (e.g. border regions or arid zones)
        regs = df['Region'].unique()
        reg_mult = {r: np.random.uniform(0.9, 1.4) for r in regs}
        df['Regional_Mult'] = df['Region'].map(reg_mult).fillna(1.0)
        
        # 3. Calculate Target Merit Score (0-100)
        # Higher score = Better candidate
        raw_score = (
            (df['Productivity_Growth'] * 1.5) + 
            (df['Tax_Return_Index'] * 60) + 
            (df['Tech_Score'] * 0.4) + 
            (df['Regional_Mult'] * 15) - 
            (df['Past_Violations'] * 40)
        )
        
        # Scale to 0-100
        df['Merit_Score'] = (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min()) * 100
        
        # 4. Save
        output_csv = "subsidies_scoring_data.csv"
        df.to_csv(output_csv, index=False)
        print(f"Success! Data saved to {output_csv}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    prepare_data()
