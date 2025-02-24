from preprocessing import create_engineered_features, preprocess_full_data
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
# from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap 
import warnings

warnings.filterwarnings('ignore')

def train_models(X_train, X_test, y_train, y_test):
    # Define models
    models = {
        'Random Forest': RandomForestRegressor(random_state=42),
        #'XGBoost': XGBRegressor(random_state=42),
        'Extra Trees': ExtraTreesRegressor(random_state=42),
        'LightGBM': LGBMRegressor(random_state=42),
        # 'CatBoost': CatBoostRegressor(random_state=42, verbose=False)
    }
    metrics = {}  
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        joblib.dump(model, f'saved/models/{name.replace(" ", "_").lower()}.joblib')
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        metrics[name] = {'RMSE': rmse, 'R2': r2}
        print(f"{name}: RMSE = {rmse}, R2 = {r2}")
        
        print(f"Generating SHAP explanations for {name}...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False, plot_size=(10, 6))
        plt.title(f'{name} SHAP Feature Importance')
        plt.savefig(f'saved/assets/shap_method/{name.replace(" ", "_").lower()}_shap_summary.png', bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, plot_size=(10, 6))
        plt.title(f'{name} Mean SHAP Value')
        plt.savefig(f'saved/assets/shap_method/{name.replace(" ", "_").lower()}_shap_bar.png', bbox_inches='tight')
        plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(metrics.keys(), [m['RMSE'] for m in metrics.values()], color='blue', alpha=0.7)
    plt.title('Model RMSE')
    plt.ylabel('RMSE Score')
    plt.xticks(rotation=45)
    
    for i, v in enumerate([m['RMSE'] for m in metrics.values()]):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.savefig('saved/assets/model_rmse.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(metrics.keys(), [m['R2'] for m in metrics.values()], color='orange', alpha=0.7)
    plt.title('Model R2')
    plt.ylabel('R2 Score')
    plt.xticks(rotation=45)
    
    for i, v in enumerate([m['R2'] for m in metrics.values()]):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.savefig('saved/assets/model_r2.png')
    plt.close()
    
    return metrics


if __name__ == "__main__":
    print("Loading data...")
    data = pd.read_excel("data/Concrete_Data.xls")

    # Feature engineering
    print("Creating engineered features...")
    df = create_engineered_features(data)
    new_features = set(df.columns) - set(data.columns)
    original_target = 'Concrete compressive strength(MPa, megapascals) '
    original_features = df.columns.tolist()
    original_features.remove(original_target)

    # Preprocess data
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_full_data(df=df)

    # Train models and generate SHAP explanations
    print("Training models and generating SHAP explanations...")
    metrics = train_models(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    print("Training and explanation process completed!")
    