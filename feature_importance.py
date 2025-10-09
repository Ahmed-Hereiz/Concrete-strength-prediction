import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

def rename_features(df):
    """
    Renames the DataFrame columns to match the abbreviated names as in the paper.
    """
    rename_map = {
        "Cement (component 1)(kg in a m^3 mixture)": "cement",
        "Blast Furnace Slag (component 2)(kg in a m^3 mixture)": "blast-furnace slag",
        "Fly Ash (component 3)(kg in a m^3 mixture)": "fly-ash",
        "Water  (component 4)(kg in a m^3 mixture)": "water",
        "Superplasticizer (component 5)(kg in a m^3 mixture)": "superplasticizer",
        "Coarse Aggregate  (component 6)(kg in a m^3 mixture)": "coarse aggregate",
        "Fine Aggregate (component 7)(kg in a m^3 mixture)": "fine aggregate",
        "Age (day)": "age",
        "Concrete compressive strength(MPa, megapascals) ": "concrete CS",
        "water_cement_ratio": "Water-to-Cement",
        "total_cementitious": "Total Cementitious Materials",
        "cementitious_water_ratio": "Cementitious-to-Water",
        "total_aggregate": "Total Aggregate",
        "fine_coarse_ratio": "Fine-to-Coarse Aggregate",
        "paste_volume_ratio": "Paste Volume",
        "percent_Cement": "Cement Percentage Composition",
        "percent_Blast Furnace Slag": "blast-furnace slag Percentage Composition",
        "percent_Fly Ash": "fly-ash Percentage Composition",
        "percent_Water": "water Percentage Composition",
        "percent_Superplasticizer": "superplasticizer Percentage Composition",
        "percent_Coarse Aggregate": "coarse aggregate Percentage Composition",
        "percent_Fine Aggregate": "fine aggregate Percentage Composition",
        "log_age": "Log-Transformed Age",
        "age_category": "Age Categorization",
        "age_cement_interaction": "Age-Cement Interaction",
        "cement_squared": "Polynomial Term (P_C)",
        "water_squared": "Polynomial Term (P_W)"
    }
    df = df.rename(columns=rename_map)
    return df

def calculate_feature_importance(df, target_col, test_size=0.2, random_state=42):
    # Split the data into features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Initialize and train the LGBM model
    model = LGBMRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    
    # Extract feature importances
    importance = model.feature_importances_
    features = X.columns
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    return model, importance_df, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Load your data (adjust the path and file type as needed)
    data = pd.read_excel("data/Concrete_Data.xls")
    
    # Import your feature engineering function
    from preprocessing import create_engineered_features
    df = create_engineered_features(data)
    
    # Rename features according to the paper's abbreviations
    df = rename_features(df)
    
    # Define the target variable name (after renaming)
    target = 'concrete CS'
    
    # Calculate feature importance using the function
    model, importance_df, X_train, X_test, y_train, y_test = calculate_feature_importance(df, target)
    
    # Display the feature importance DataFrame with updated feature names
    print("Feature Importances:")
    print(importance_df)
    
    # Plot the feature importances using the updated names
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title("Feature Importance (LGBM)")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
