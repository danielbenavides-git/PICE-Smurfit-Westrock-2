# ================================================
# Multicollinearity Analysis and Feature Selection
# ================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, rankdata
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
    verbose=True,
    protected_features=None
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
    protected_features : list, default=None
        List of feature names that should never be dropped (e.g., critical business features
        like 'Monto', 'Cantidad'). These features will be kept even if they are highly 
        correlated with other features.
    
    Returns:
    --------
    df_selected : pd.DataFrame
        Dataframe with selected features (redundant features removed, protected features kept)
    features_to_drop : list
        List of features recommended for removal
    analysis_results : dict
        Dictionary containing:
            - 'high_corr_pairs': List of highly correlated feature pairs
            - 'corr_matrix': Full correlation matrix
            - 'vif_data': DataFrame with VIF values for all features
            - 'cluster_info': DataFrame with cluster assignments
            - 'drop_reasons': Dictionary with reasons for dropping each feature
            - 'protected_features': List of features that were protected from removal
    """
    
    # Initialize protected features
    if protected_features is None:
        protected_features = []
    else:
        # Ensure protected features exist in data
        protected_features = [f for f in protected_features if f in data.columns]
        if verbose and protected_features:
            print(f"The following {len(protected_features)} features will be kept regardless of multicollinearity:")
            for i, feat in enumerate(protected_features, 1):
                print(f"  {i}. {feat}")
            print()
    
    # Handle 'id' column
    if 'id' in data.columns:
        transaction_id = data['id'].copy()
        data_for_analysis = data.drop(columns=['id'])
    else:
        transaction_id = None
        data_for_analysis = data.copy()

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
            plt.figure(figsize=(10, 6))
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
                        protected_marker = " [PROTECTED]" if feat in protected_features else ""
                        print(f"  - {feat}{protected_marker}")
                print()
        
        return cluster_df, linkage_matrix
    
    # --------------------------------------------
    # 4. Feature Selection with Protection
    # --------------------------------------------
    
    def select_features(corr_pairs, vif_df, cluster_df, vif_thresh, corr_thresh, protected):
        """
        Determine which features to drop based on correlation, VIF, and clustering analysis.
        Protected features are never dropped.
        
        Strategy:
        1. From correlated pairs: Drop the feature with higher VIF (unless protected)
        2. From clusters: Keep protected features or feature with lowest VIF per cluster
        
        Purpose: Automatically select which redundant features to remove while keeping
        the most informative representative from each redundant group and all protected features.
        
        Returns:
        - features_to_drop: List of feature names to remove
        - drop_reasons: Dictionary mapping each dropped feature to why it was dropped
        """
        
        features_to_drop = set()
        drop_reasons = {}
        protected_kept_count = 0
        
        # Strategy 1: From correlated pairs, drop feature with higher VIF (respect protection)
        for feat1, feat2, corr_val in corr_pairs:
            if feat1 in features_to_drop or feat2 in features_to_drop:
                continue
            
            # Check if either feature is protected
            feat1_protected = feat1 in protected
            feat2_protected = feat2 in protected
            
            if feat1_protected and feat2_protected:
                # Both protected, keep both, skip
                if verbose:
                    print(f"  Note: {feat1} and {feat2} are both protected despite high correlation (r={corr_val:.3f})")
                continue
            elif feat1_protected:
                # Protect feat1, drop feat2
                features_to_drop.add(feat2)
                drop_reasons[feat2] = f"High correlation with {feat1} (r={corr_val:.3f}) - {feat1} is protected"
                protected_kept_count += 1
                continue
            elif feat2_protected:
                # Protect feat2, drop feat1
                features_to_drop.add(feat1)
                drop_reasons[feat1] = f"High correlation with {feat2} (r={corr_val:.3f}) - {feat2} is protected"
                protected_kept_count += 1
                continue
            
            # Neither protected, use normal VIF logic
            vif1 = vif_df[vif_df['Feature'] == feat1]['VIF'].values[0]
            vif2 = vif_df[vif_df['Feature'] == feat2]['VIF'].values[0]
            
            if vif1 > vif2:
                features_to_drop.add(feat1)
                drop_reasons[feat1] = f"High correlation with {feat2} (r={corr_val:.3f}), VIF={vif1:.1f} > {vif2:.1f}"
            else:
                features_to_drop.add(feat2)
                drop_reasons[feat2] = f"High correlation with {feat1} (r={corr_val:.3f}), VIF={vif2:.1f} > {vif1:.1f}"
        
        # Strategy 2: From clusters, keep protected features or feature with lowest VIF
        for cluster_id in cluster_df['Cluster'].unique():
            cluster_features = cluster_df[cluster_df['Cluster'] == cluster_id]['Feature'].tolist()
            
            if len(cluster_features) > 1:
                # Check if any features in cluster are protected
                protected_in_cluster = [f for f in cluster_features if f in protected]
                
                if protected_in_cluster:
                    # Keep all protected features, drop others
                    for feat in cluster_features:
                        if feat not in protected_in_cluster and feat not in features_to_drop:
                            features_to_drop.add(feat)
                            drop_reasons[feat] = f"Clustered with protected feature(s): {', '.join(protected_in_cluster)}"
                else:
                    # No protected features, keep feature with lowest VIF
                    cluster_vifs = vif_df[vif_df['Feature'].isin(cluster_features)].copy()
                    cluster_vifs = cluster_vifs.sort_values('VIF')
                    keep_feature = cluster_vifs.iloc[0]['Feature']
                    
                    for feat in cluster_features:
                        if feat != keep_feature and feat not in features_to_drop:
                            features_to_drop.add(feat)
                            drop_reasons[feat] = f"Clustered with {keep_feature} (kept for lower VIF)"
        
        # FINAL CHECK: Ensure no protected features are in drop list
        protected_saved = []
        for protected_feat in protected:
            if protected_feat in features_to_drop:
                features_to_drop.remove(protected_feat)
                protected_saved.append(protected_feat)
        
        if verbose:
            reduction_pct = len(features_to_drop)/len(vif_df)*100 if len(vif_df) > 0 else 0
            print(f"Feature Selection: {len(vif_df)} -> {len(vif_df) - len(features_to_drop)} features ({reduction_pct:.1f}% reduction)")
            
            if protected_saved:
                print(f"\nProtected features that were saved from removal:")
                for feat in protected_saved:
                    print(f"  - {feat}")
            
            if protected_kept_count > 0:
                print(f"\n{protected_kept_count} correlated features dropped to preserve protected features")
            
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
    
    high_corr_pairs, corr_matrix = analyze_correlations(data_for_analysis, corr_threshold)
    vif_data = calculate_vif(data_for_analysis, vif_threshold)
    cluster_info, linkage_matrix = cluster_features(data_for_analysis, corr_threshold)
    features_to_drop, drop_reasons = select_features(
        high_corr_pairs, vif_data, cluster_info, vif_threshold, corr_threshold, protected_features
    )
    
    # Apply feature selection
    df_selected = data_for_analysis.drop(columns=features_to_drop, errors='ignore')

    # Add back 'id' column if it existed
    if transaction_id is not None:
        df_selected.insert(0, 'id', transaction_id)

    if verbose:
        selected_features = [f for f in df_selected.columns if f != 'id']
        protected_in_final = [f for f in protected_features if f in selected_features]
        
        print(f"\nRetained features ({len(selected_features)}):")
        for i, feat in enumerate(selected_features, 1):
            protected_marker = " [PROTECTED]" if feat in protected_features else ""
            print(f"  {i:2d}. {feat}{protected_marker}")
        
        if protected_in_final:
            print(f"\n✓ All {len(protected_in_final)} protected features successfully retained")
    
    # Package results
    analysis_results = {
        'high_corr_pairs': high_corr_pairs,
        'corr_matrix': corr_matrix,
        'vif_data': vif_data,
        'cluster_info': cluster_info,
        'linkage_matrix': linkage_matrix,
        'drop_reasons': drop_reasons,
        'protected_features': protected_features
    }
    
    return df_selected, features_to_drop, analysis_results