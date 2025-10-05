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

def calculate_feature_importance(df, target_col, test_size=0.2, random_state=42):
    """
    Trains an LGBMRegressor and returns a DataFrame of feature importances with renamed features.
    
    Parameters:
        df (pd.DataFrame): DataFrame with engineered features.
        target_col (str): The name of the target column (after renaming).
        test_size (float): Proportion of data to be used as test set.
        random_state (int): Seed for reproducibility.
        
    Returns:
        model: The trained LGBMRegressor.
        importance_df (pd.DataFrame): DataFrame containing features and their importance scores.
        X_train, X_test, y_train, y_test: Data splits.
    """
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
    target = 'Strength'
    
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
