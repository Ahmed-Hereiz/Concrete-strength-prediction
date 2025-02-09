import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

def calculate_feature_importance(df, target_col, test_size=0.2, random_state=42):
    """
    Trains an LGBMRegressor and returns a DataFrame of feature importances.
    
    Parameters:
        df (pd.DataFrame): DataFrame with engineered features.
        target_col (str): The name of the target column.
        test_size (float): Proportion of data to be used as test set.
        random_state (int): Seed for reproducibility.
        
    Returns:
        model: The trained LGBMRegressor.
        importance_df (pd.DataFrame): DataFrame containing features and their importance scores.
        X_train, X_test, y_train, y_test: Data splits (optional, for further analysis).
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
    
    # Extract feature importance (using built-in importance from LGBM)
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
    
    # Define the target variable name (ensure it matches exactly)
    target = 'Concrete compressive strength(MPa, megapascals) '
    
    # Calculate feature importance using the function
    model, importance_df, X_train, X_test, y_train, y_test = calculate_feature_importance(df, target)
    
    # Display the feature importance DataFrame
    print("Feature Importances:")
    print(importance_df)
    
    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title("Feature Importance (LGBM)")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
