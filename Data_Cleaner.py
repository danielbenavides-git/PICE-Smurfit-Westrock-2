# ========================
# Clean CSV for Supabase
# ========================

import pandas as pd

def clean_csv_for_supabase(database_path):
    """
    Cleans a CSV file to ensure compatibility with Supabase by converting float columns
    that contain only integer values to integer type.
    
    Parameters:
    -----------
    database_path (str): Path to the CSV file to be cleaned.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    
    df = pd.read_csv(database_path)
    
    for col in df.columns:
        if df[col].dtype == 'float64':
            if (df[col].dropna() % 1 == 0).all():
                df[col] = df[col].astype('Int64')

    df.to_csv(database_path, index=False)

    print(f"CSV file at {database_path} has been cleaned for Supabase compatibility.")

    return df

df1 = clean_csv_for_supabase('PICE BD 2025.csv')
df2 = clean_csv_for_supabase('PICE BD 2025-ML.csv')
df3 = clean_csv_for_supabase('PICE - iForest_anomalies_test.csv')