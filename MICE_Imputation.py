# ======================================================================
# MICE (Multiple Imputation by Chained Equations) for Mixed Data Types
# ======================================================================

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from prettytable import PrettyTable

def mice_imputation(db, variables_to_impute):
    if 'id' in db.columns:
        transaction_id = db['id'].copy()
    else:
        print("WARNING: 'id' column not found. Adding it now.")
        transaction_id = pd.Series(range(len(db)), name='id')
        db['id'] = transaction_id

    # Identify data types
    numeric_vars = []
    categorical_vars = []

    for var in variables_to_impute:
        if var in db.columns:
            if pd.api.types.is_numeric_dtype(db[var]):
                numeric_vars.append(var)
            else:
                categorical_vars.append(var)

    # Create a copy of the data
    db_imputed = db[variables_to_impute].copy()

    # Store missing counts before imputation
    missing_before = db_imputed.isnull().sum()

    # Encode categorical variables
    encoders = {}
    encoded_cols = {}

    for var in categorical_vars:
        encoded_col_name = f"{var}_encoded"
        le = LabelEncoder()
        
        non_null_mask = db_imputed[var].notna()
        if non_null_mask.sum() > 0:
            le.fit(db_imputed.loc[non_null_mask, var].astype(str))
            db_imputed[encoded_col_name] = np.nan
            db_imputed.loc[non_null_mask, encoded_col_name] = le.transform(
                db_imputed.loc[non_null_mask, var].astype(str)
            )
            encoders[var] = le
            encoded_cols[var] = encoded_col_name

    # Prepare data for MICE imputation
    cols_to_impute = numeric_vars + list(encoded_cols.values())
    imputation_data = db_imputed[cols_to_impute].copy()

    # Apply MICE Imputation
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1),
        max_iter=10,
        random_state=42,
        verbose=0
    )

    imputed_array = imputer.fit_transform(imputation_data)
    imputed_df = pd.DataFrame(imputed_array, columns=cols_to_impute, index=db_imputed.index)

    # Decode categorical variables back to original values
    for var, encoded_col in encoded_cols.items():
        imputed_encoded = imputed_df[encoded_col].round().astype(int)
        le = encoders[var]
        max_label = len(le.classes_) - 1
        imputed_encoded = imputed_encoded.clip(0, max_label)
        db_imputed[var] = le.inverse_transform(imputed_encoded)

    # Update numeric variables with imputed values
    for var in numeric_vars:
        db_imputed[var] = imputed_df[var]

    # Keep only original variables
    db_imputed = db_imputed[variables_to_impute]

    # Store missing counts after imputation
    missing_after = db_imputed.isnull().sum()

    # Update the original dataframe
    for var in variables_to_impute:
        if var in db_imputed.columns:
            db[var] = db_imputed[var]

    # ======================================================================
    # FINAL SUMMARY
    # ======================================================================

    # Data types summary
    print("\nDATA TYPES:")
    print(f"Numeric variables: {len(numeric_vars)}")
    print(f"Categorical variables: {len(categorical_vars)}")

    # Create comparison table

    table = PrettyTable()
    table.field_names = ["Variable", "Type", "Missing Before", "Missing After", "Values Imputed"]
    table.align["Variable"] = "l"
    table.align["Type"] = "c"
    table.align["Missing Before"] = "r"
    table.align["Missing After"] = "r"
    table.align["Values Imputed"] = "r"

    for var in variables_to_impute:
        if var in db.columns:
            var_type = "Numeric" if var in numeric_vars else "Categorical"
            before = missing_before[var] if var in missing_before.index else 0
            after = missing_after[var] if var in missing_after.index else 0
            imputed = before - after
            
            if before > 0:  # Only show variables that had missing values
                table.add_row([
                    var,
                    var_type,
                    f"{before:,}",
                    f"{after:,}",
                    f"{imputed:,}"
                ])

    print(table)

    # Overall statistics
    total_before = missing_before.sum()
    total_after = missing_after.sum()
    total_imputed = total_before - total_after

    print(f"Total missing values before: {total_before:,}")
    print(f"Total missing values after:  {total_after:,}")
    print(f"Total values imputed:        {total_imputed:,}")

    if total_after == 0:
        print("\nStatus: All missing values successfully imputed")
    else:
        print(f"\nStatus: {total_after:,} missing values remain")

    if 'id' not in db.columns:
        db['id'] = transaction_id
    
    db.to_csv("PICE BD 2025-Imputed.csv", index=False)

    return db