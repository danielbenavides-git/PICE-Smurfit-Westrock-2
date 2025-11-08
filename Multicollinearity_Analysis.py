# ============================================
# Multicollinearity_Analysis.py
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from statsmodels.stats.outliers_influence import variance_inflation_factor
from prettytable import PrettyTable
import warnings
warnings.filterwarnings('ignore')

def comprehensive_multicollinearity_analysis(
    data, 
    vif_threshold=10, 
    corr_threshold=0.85,
    show_plots=True,
    verbose=True
):
    """
    Perform comprehensive multicollinearity analysis and feature selection.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe with all numeric features (categorical variables should be encoded first)
    vif_threshold : float, default=10
        VIF threshold for identifying severe multicollinearity
    corr_threshold : float, default=0.85
        Correlation threshold for identifying highly correlated features
    show_plots : bool, default=True
        Whether to display visualizations (correlation heatmap, dendrogram)
    verbose : bool, default=True
        Whether to print detailed analysis results
    
    Returns:
    --------
    df_selected : pd.DataFrame
        Dataframe with selected features (redundant features removed)
    features_to_drop : list
        List of features recommended for removal
    analysis_results : dict
        Dictionary containing:
            - 'high_corr_pairs': List of highly correlated feature pairs
            - 'corr_matrix': Full correlation matrix
            - 'vif_data': DataFrame with VIF values for all features
            - 'cluster_info': DataFrame with cluster assignments
            - 'drop_reasons': Dictionary with reasons for dropping each feature
    """
    
    # --------------------------------------------
    # 1. Correlation Analysis
    # --------------------------------------------
    
    def analyze_correlations(df, threshold):
        """
        Calculate pairwise correlations between all features and identify highly correlated 
        pairs in order to find features that move together (correlation > threshold), 
        indicating redundancy.
        
        Returns:
        - high_corr_pairs: List of tuples (feature1, feature2, correlation_value)
        - corr_matrix: Full correlation matrix for all features
        """

        corr_matrix = df.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        high_corr_pairs = [
            (column, row, upper_triangle.loc[row, column])
            for column in upper_triangle.columns
            for row in upper_triangle.index
            if upper_triangle.loc[row, column] > threshold
        ]
        high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
        
        if show_plots:
            plt.figure(figsize=(10, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdYlBu_r', 
                        center=0, square=True, linewidths=0.5, 
                        cbar_kws={"shrink": 0.8, "label": "Absolute Correlation"})
            plt.title('Feature Correlation Matrix', fontsize=16, weight='bold', pad=20)
            plt.tight_layout()
            plt.show()
        
        if verbose and high_corr_pairs:
            print(f"Found {len(high_corr_pairs)} highly correlated pairs (|r| > {threshold}):")
            table = PrettyTable()
            table.field_names = ["Feature 1", "Feature 2", "Correlation"]
            table.align["Feature 1"] = "l"
            table.align["Feature 2"] = "l"
            table.align["Correlation"] = "r"
            
            display_count = min(20, len(high_corr_pairs))
            for feat1, feat2, corr_val in high_corr_pairs[:display_count]:
                table.add_row([feat1, feat2, f"{corr_val:.3f}"])
            
            print(table)
            
            if len(high_corr_pairs) > 20:
                print(f"... and {len(high_corr_pairs) - 20} more pairs\n")
        
        return high_corr_pairs, corr_matrix
    
    # --------------------------------------------
    # 2. VIF Analysis
    # --------------------------------------------
    
    def calculate_vif(df, threshold):
        """
        Calculate Variance Inflation Factor for each feature to detect multicollinearity.
        
        Purpose: VIF measures how much a feature can be predicted by all other features.
        - VIF = 1: No correlation with other features
        - VIF = 5-10: Moderate multicollinearity
        - VIF > 10: Severe multicollinearity (feature is redundant)
        
        Returns:
        - vif_data: DataFrame with columns [Feature, VIF, Status]
        """

        vif_data = pd.DataFrame()
        vif_data["Feature"] = df.columns
        
        vif_values = []
        for i in range(df.shape[1]):
            try:
                vif = variance_inflation_factor(df.values, i)
                vif_values.append(vif if not np.isinf(vif) else 999)
            except:
                vif_values.append(999)
        
        vif_data["VIF"] = vif_values
        vif_data = vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)
        
        def categorize_vif(vif):
            if vif > threshold:
                return 'Severe'
            elif vif > 5:
                return 'Moderate'
            else:
                return 'Low'
        
        vif_data['Status'] = vif_data['VIF'].apply(categorize_vif)
        
        if verbose:
            print(f"\nTop 20 Features by VIF:")
            print(vif_data[['Feature', 'VIF', 'Status']].head(20).to_string(index=False))
            
            severe_count = (vif_data['VIF'] > threshold).sum()
            moderate_count = ((vif_data['VIF'] > 5) & (vif_data['VIF'] <= threshold)).sum()
            low_count = (vif_data['VIF'] <= 5).sum()
            
            print(f"\nVIF Summary: {severe_count} Severe | {moderate_count} Moderate | {low_count} Low\n")
        
        return vif_data
    
    # --------------------------------------------
    # 3. Hierarchical Clustering
    # --------------------------------------------
    
    def cluster_features(df, threshold):
        """
        Group features into clusters based on correlation using hierarchical clustering.
        
        Purpose: Visualize which features form redundant groups (e.g., all amount-related 
        features might cluster together).
        
        Method: Uses Spearman correlation (rank-based, robust to outliers) and creates
        a dendrogram showing feature relationships.
        
        Returns:
        - cluster_df: DataFrame with columns [Feature, Cluster]
        - linkage_matrix: Hierarchical clustering linkage for dendrogram
        """

        from scipy.stats import rankdata
        
        # Rank each column
        ranked_data = np.column_stack([rankdata(df[col]) for col in df.columns])
        
        # Calculate correlation on ranked data (Spearman correlation)
        corr = np.corrcoef(ranked_data.T)
        
        # Ensure perfect symmetry
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1.0)
        
        # Handle any NaN values (in case of constant columns)
        corr = np.nan_to_num(corr, nan=0.0)
        
        # Convert correlation to distance
        dissimilarity = 1 - np.abs(corr)
        
        # Ensure dissimilarity is also symmetric and non-negative
        dissimilarity = np.clip(dissimilarity, 0, 2)
        dissimilarity = (dissimilarity + dissimilarity.T) / 2
        
        # Convert to condensed distance matrix
        condensed_dist = hierarchy.distance.squareform(dissimilarity, checks=False)
        
        # Hierarchical clustering
        linkage_matrix = hierarchy.linkage(condensed_dist, method='complete')
        
        if show_plots:
            plt.figure(figsize=(18, 8))
            dendro = hierarchy.dendrogram(
                linkage_matrix,
                labels=df.columns,
                leaf_rotation=90,
                leaf_font_size=9
            )
            plt.axhline(y=1-threshold, color='red', linestyle='--', 
                        linewidth=2, label=f'Threshold (corr = {threshold})')
            plt.title('Hierarchical Clustering of Features by Correlation', 
                    fontsize=16, weight='bold', pad=20)
            plt.xlabel('Features', fontsize=12)
            plt.ylabel('Distance (1 - |correlation|)', fontsize=12)
            plt.legend(fontsize=11)
            plt.tight_layout()
            plt.show()
        
        clusters = hierarchy.fcluster(linkage_matrix, 1-threshold, criterion='distance')
        cluster_df = pd.DataFrame({
            'Feature': df.columns,
            'Cluster': clusters
        }).sort_values('Cluster')
        
        if verbose:
            multi_feature_clusters = []
            for cluster_id in sorted(cluster_df['Cluster'].unique()):
                features_in_cluster = cluster_df[cluster_df['Cluster'] == cluster_id]['Feature'].tolist()
                if len(features_in_cluster) > 1:
                    multi_feature_clusters.append((cluster_id, features_in_cluster))
            
            if multi_feature_clusters:
                print(f"Clusters with multiple features ({len(multi_feature_clusters)} total):")
                for cluster_id, features in multi_feature_clusters:
                    print(f"\nCluster {cluster_id} ({len(features)} features):")
                    for feat in features:
                        print(f"  - {feat}")
                print()
        
        return cluster_df, linkage_matrix
    
    # --------------------------------------------
    # 4. Feature Selection
    # --------------------------------------------
    
    def select_features(corr_pairs, vif_df, cluster_df, vif_thresh, corr_thresh):
        """
        Determine which features to drop based on correlation, VIF, and clustering analysis.
        
        Strategy:
        1. From correlated pairs: Drop the feature with higher VIF (more redundant overall)
        2. From clusters: Keep only the feature with lowest VIF per cluster
        
        Purpose: Automatically select which redundant features to remove while keeping
        the most informative representative from each redundant group.
        
        Returns:
        - features_to_drop: List of feature names to remove
        - drop_reasons: Dictionary mapping each dropped feature to why it was dropped
        """
        
        features_to_drop = set()
        drop_reasons = {}
        
        # Strategy 1: From correlated pairs, drop feature with higher VIF
        for feat1, feat2, corr_val in corr_pairs:
            if feat1 in features_to_drop or feat2 in features_to_drop:
                continue
            
            vif1 = vif_df[vif_df['Feature'] == feat1]['VIF'].values[0]
            vif2 = vif_df[vif_df['Feature'] == feat2]['VIF'].values[0]
            
            if vif1 > vif2:
                features_to_drop.add(feat1)
                drop_reasons[feat1] = f"High correlation with {feat2} (r={corr_val:.3f}), VIF={vif1:.1f} > {vif2:.1f}"
            else:
                features_to_drop.add(feat2)
                drop_reasons[feat2] = f"High correlation with {feat1} (r={corr_val:.3f}), VIF={vif2:.1f} > {vif1:.1f}"
        
        # Strategy 2: From clusters, keep feature with lowest VIF
        for cluster_id in cluster_df['Cluster'].unique():
            cluster_features = cluster_df[cluster_df['Cluster'] == cluster_id]['Feature'].tolist()
            
            if len(cluster_features) > 1:
                cluster_vifs = vif_df[vif_df['Feature'].isin(cluster_features)].copy()
                cluster_vifs = cluster_vifs.sort_values('VIF')
                keep_feature = cluster_vifs.iloc[0]['Feature']
                
                for feat in cluster_features:
                    if feat != keep_feature and feat not in features_to_drop:
                        features_to_drop.add(feat)
                        drop_reasons[feat] = f"Clustered with {keep_feature} (kept for lower VIF)"
        
        if verbose:
            reduction_pct = len(features_to_drop)/len(vif_df)*100 if len(vif_df) > 0 else 0
            print(f"Feature Selection: {len(vif_df)} -> {len(vif_df) - len(features_to_drop)} features ({reduction_pct:.1f}% reduction)")
            
            if features_to_drop:
                print(f"\nFeatures to drop ({len(features_to_drop)}):")
                table = PrettyTable()
                table.field_names = ["Feature", "Reason"]
                table.align["Feature"] = "l"
                table.align["Reason"] = "l"
                table.max_width["Reason"] = 70
                
                for feat in sorted(features_to_drop):
                    reason = drop_reasons.get(feat, "Redundant feature in cluster")
                    table.add_row([feat, reason])
                
                print(table)
        
        return list(features_to_drop), drop_reasons
    
    # --------------------------------------------
    # Execute Analysis Pipeline
    # --------------------------------------------
    
    high_corr_pairs, corr_matrix = analyze_correlations(data, corr_threshold)
    vif_data = calculate_vif(data, vif_threshold)
    cluster_info, linkage_matrix = cluster_features(data, corr_threshold)
    features_to_drop, drop_reasons = select_features(
        high_corr_pairs, vif_data, cluster_info, vif_threshold, corr_threshold
    )
    
    # Apply feature selection
    df_selected = data.drop(columns=features_to_drop, errors='ignore')
    
    if verbose:
        selected_features = df_selected.columns.tolist()
        print(f"\nRetained features ({len(selected_features)}):")
        for i, feat in enumerate(selected_features, 1):
            print(f"  {i:2d}. {feat}")
    
    # Package results
    analysis_results = {
        'high_corr_pairs': high_corr_pairs,
        'corr_matrix': corr_matrix,
        'vif_data': vif_data,
        'cluster_info': cluster_info,
        'linkage_matrix': linkage_matrix,
        'drop_reasons': drop_reasons
    }
    
    return df_selected, features_to_drop, analysis_results