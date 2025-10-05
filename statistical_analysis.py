import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from preprocessing import create_engineered_features

def rename_features(df):
    """
    Renames the DataFrame columns to match the abbreviated names as in the paper.
    """
    rename_map = {
        'Cement (component 1)(kg in a m^3 mixture)': 'C',
        'Blast Furnace Slag (component 2)(kg in a m^3 mixture)': 'BFS',
        'Fly Ash (component 3)(kg in a m^3 mixture)': 'FAsh',
        'Water  (component 4)(kg in a m^3 mixture)': 'W',
        'Superplasticizer (component 5)(kg in a m^3 mixture)': 'SP',
        'Coarse Aggregate  (component 6)(kg in a m^3 mixture)': 'CA',
        'Fine Aggregate (component 7)(kg in a m^3 mixture)': 'FA',
        'Age (day)': 'Age',
        'Concrete compressive strength(MPa, megapascals) ': 'Strength',
        'water_cement_ratio': 'w/c',
        'total_cementitious': 'Total_Cem',
        'cementitious_water_ratio': 'Cem/w',
        'total_aggregate': 'Total_Agg',
        'fine_coarse_ratio': 'FA/CA',
        'percent_Cement': '%C',
        'percent_Blast Furnace Slag': '%BFS',
        'percent_Fly Ash': '%FAsh',
        'percent_Water': '%W',
        'percent_Superplasticizer': '%SP',
        'percent_Coarse Aggregate': '%CA',
        'percent_Fine Aggregate': '%FA',
        'log_age': 'log_Age',
        'age_category': 'Age_cat',
        'age_cement_interaction': 'Age*C',
        'cement_squared': 'C^2',
        'water_squared': 'W^2',
        'paste_volume_ratio': 'PVR'
    }
    df = df.rename(columns=rename_map)
    return df

def calculate_shap(df, target, test_size=0.2, random_state=42):
    """
    Trains a LightGBM model and computes SHAP values for the test set.
    
    Returns:
        model: Trained LightGBM model.
        X_test: Features for SHAP analysis (test split).
        shap_values: SHAP values for X_test.
    """
    # Separate features and target
    X = df.drop(columns=[target])
    y = df[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Train LightGBM model
    model = LGBMRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    
    # Create SHAP explainer and compute SHAP values for the test set
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    
    return model, X_test, shap_values

if __name__ == "__main__":
    data_path = "data/Concrete_Data.xls"  
    original_df = pd.read_excel(data_path)
    
    # Define the target column
    target = 'Concrete compressive strength(MPa, megapascals) '
    
    # 1) Original Data
    original_df_renamed = rename_features(original_df)
    model_orig, X_orig, shap_values_orig = calculate_shap(original_df_renamed, 'Strength')
    
    # 2) Engineered Data
    engineered_df = create_engineered_features(original_df)
    engineered_df = rename_features(engineered_df)
    model_eng, X_eng, shap_values_eng = calculate_shap(engineered_df, 'Strength')
    
    # ---- Plot 1: Bar chart for ORIGINAL features ----
    plt.figure(figsize=(8, 6))
    shap.plots.bar(shap_values_orig, max_display=None)  # or set max_display=N
    plt.title("LightGBM Mean SHAP Value (Original Features)")
    plt.tight_layout()
    plt.show()
    
    # ---- Plot 2: Beeswarm summary for ORIGINAL features ----
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values_orig.values, X_orig, plot_type="dot", show=False)
    plt.title("LightGBM SHAP Feature Importance (Original Features)")
    plt.tight_layout()
    plt.show()
    
    # ---- Plot 3: Bar chart for ENGINEERED features ----
    plt.figure(figsize=(8, 6))
    shap.plots.bar(shap_values_eng, max_display=None)  # or set max_display=N
    plt.title("LightGBM Mean SHAP Value (Engineered Features)")
    plt.tight_layout()
    plt.show()
    
    # ---- Plot 4: Beeswarm summary for ENGINEERED features ----
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values_eng.values, X_eng, plot_type="dot", show=False)
    plt.title("LightGBM SHAP Feature Importance (Engineered Features)")
    plt.tight_layout()
    plt.show()
