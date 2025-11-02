# =============================================
# Missing Data Analysis for Imputation Strategy
# =============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
from prettytable import PrettyTable
import warnings

# Suppress precision loss warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, 
                       message='.*Precision loss occurred in moment calculation.*')

# ============================================
# 1. MISSINGNESS PATTERN ANALYSIS
# ============================================

def analyze_missingness_patterns(df, variables):
    """
    Analyze patterns of missingness across variables
    """
    # Create missingness indicator matrix
    missing_matrix = df[variables].isnull().astype(int)
    
    # Correlation between missingness indicators
    missing_corr = missing_matrix.corr()
    
    # Find high correlations (excluding diagonal)
    high_corr_pairs = []
    for i in range(len(missing_corr.columns)):
        for j in range(i+1, len(missing_corr.columns)):
            corr_value = missing_corr.iloc[i, j]
            if abs(corr_value) > 0.3:  # Threshold for moderate correlation
                high_corr_pairs.append((
                    missing_corr.columns[i], 
                    missing_corr.columns[j], 
                    corr_value
                ))
    
    return missing_matrix, missing_corr, high_corr_pairs

# ============================================
# 2. TEST FOR MCAR (Little's MCAR Test Alternative)
# ============================================

def test_mcar_using_ttest(df, var_with_missing, numeric_vars):
    """
    Test if missingness in a variable is related to values in other variables
    Uses t-tests to compare means of numeric variables between missing/not-missing groups
    """
    missing_indicator = df[var_with_missing].isnull()
    significant_diffs = []
    
    for num_var in numeric_vars:
        if num_var == var_with_missing:
            continue
            
        # Split data into missing and non-missing groups
        group_missing = df.loc[missing_indicator, num_var].dropna()
        group_present = df.loc[~missing_indicator, num_var].dropna()
        
        if len(group_missing) > 0 and len(group_present) > 0:
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(group_missing, group_present, equal_var=False)
            
            if p_value < 0.05:
                significant_diffs.append({
                    'variable': num_var,
                    'p_value': p_value
                })
    
    if significant_diffs:
        return "MAR or MNAR"
    else:
        return "MCAR"

# ============================================
# 3. TEST FOR MAR vs MNAR
# ============================================

def test_mar_vs_mnar_categorical(df, var_with_missing, categorical_vars):
    """
    Test if missingness is related to categorical variables (MAR)
    """
    missing_indicator = df[var_with_missing].isnull()
    significant_assoc = []
    
    for cat_var in categorical_vars:
        if cat_var == var_with_missing:
            continue
        
        # Create contingency table
        contingency_table = pd.crosstab(missing_indicator, df[cat_var])
        
        if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            # Calculate Cramér's V for effect size
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
            
            if p_value < 0.05 and cramers_v > 0.1:
                significant_assoc.append({
                    'variable': cat_var,
                    'p_value': p_value,
                    'cramers_v': cramers_v
                })
    
    if significant_assoc:
        return "MAR"
    else:
        return "Inconclusive"

# ============================================
# 4. VISUALIZATION OF MISSINGNESS
# ============================================

def visualize_missingness(df, variables):
    """
    Create visualizations to understand missingness patterns
    """
    missing_matrix = df[variables].isnull().astype(int)
    
    plt.figure(figsize=(10, 6))
    
    # Correlation heatmap of missingness
    missing_corr = missing_matrix.corr()
    sns.heatmap(missing_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Correlation Between Missingness Indicators', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()

# ============================================
# 5. COMPREHENSIVE ANALYSIS FUNCTION
# ============================================

def comprehensive_missingness_analysis(df, variables_to_impute):
    """
    Run complete missingness analysis and provide recommendations
    """

    # Identify numeric and categorical variables
    numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_vars = [col for col in df.columns if col not in numeric_vars]
    
    # Filter to only include variables present in df
    variables_present = [v for v in variables_to_impute if v in df.columns]
    
    # 1. Pattern analysis
    missing_matrix, missing_corr, high_corr_pairs = analyze_missingness_patterns(df, variables_present)
    
    # 2. Test each variable for MCAR
    results = {}
    for var in variables_present:
        if df[var].isnull().sum() > 0:
            mcar_result = test_mcar_using_ttest(df, var, numeric_vars)
            mar_result = test_mar_vs_mnar_categorical(df, var, categorical_vars)
            results[var] = {'MCAR_test': mcar_result, 'MAR_test': mar_result}
    
    # 3. Display correlation matrix
    print("=" * 80)
    print("CORRELATION BETWEEN MISSINGNESS INDICATORS")
    print("=" * 80)
    
    # 4. Visualizations
    visualize_missingness(df, variables_present)
    
    # 5. Summary and recommendations
    print("\n" + "=" * 80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 80)
    
    # Create PrettyTable
    table = PrettyTable()
    table.field_names = ["Variable", "Missing %", "Type", "Recommended Method"]
    table.align["Variable"] = "l"
    table.align["Missing %"] = "r"
    table.align["Type"] = "c"
    table.align["Recommended Method"] = "l"
    table.max_width["Recommended Method"] = 45
    
    # Collect data for sorting
    table_data = []
    for var, result in results.items():
        # Determine missingness type
        if result['MCAR_test'] == 'MCAR':
            missingness_type = 'MCAR'
            recommendation = 'Simple imputation (mean/median/mode)'
        elif result['MAR_test'] == 'MAR':
            missingness_type = 'MAR'
            recommendation = 'MICE or KNN Imputation'
        elif result['MCAR_test'] == 'MAR or MNAR' and result['MAR_test'] == 'MAR':
            missingness_type = 'MAR'
            recommendation = 'MICE or KNN Imputation'
        elif result['MCAR_test'] == 'MAR or MNAR':
            missingness_type = 'MNAR'
            recommendation = 'Random Forest or Deep Learning'
        else:
            missingness_type = 'Inconclusive'
            recommendation = 'KNN as default'
        
        # Calculate missing percentage
        missing_pct = df[var].isnull().mean() * 100
        
        table_data.append({
            'var': var,
            'pct': missing_pct,
            'type': missingness_type,
            'rec': recommendation
        })
    
    # Sort by missing percentage (highest to lowest)
    table_data.sort(key=lambda x: x['pct'], reverse=True)
    
    # Add rows to table
    for row in table_data:
        table.add_row([
            row['var'],
            f"{row['pct']:.2f}%",
            row['type'],
            row['rec']
        ])
    
    print("\n")
    print(table)
    
    print("\n" + "=" * 80)
    print("GENERAL RECOMMENDATIONS")
    print("=" * 80)
    print("""
1. MCAR (Missing Completely At Random):
   - Simple imputation: mean, median, mode
   - Listwise deletion (if small proportion)
   
2. MAR (Missing At Random):
   - Multiple Imputation by Chained Equations (MICE)
   - K-Nearest Neighbors (KNN) imputation
   - Regression-based imputation
   
3. MNAR (Missing Not At Random):
   - Random Forest imputation
   - Deep learning-based imputation
   - Domain expert consultation
   - Consider keeping missingness as a feature
    """)
    
    return results