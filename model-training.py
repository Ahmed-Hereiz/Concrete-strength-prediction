from preprocessing import create_engineered_features, preprocess_full_data
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd


def train_models(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': RandomForestRegressor(),
        'XGBoost': XGBRegressor(),
        'Extra Trees': ExtraTreesRegressor(),
        'LightGBM': LGBMRegressor()
    }
    
    metrics = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, f'saved/models/{name.replace(" ", "_").lower()}.joblib')
        
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        metrics[name] = {'RMSE': rmse, 'R2': r2}
        print(f"{name}: RMSE = {rmse}, R2 = {r2}")

    voting_reg = VotingRegressor(estimators=[(name.lower(), model) for name, model in models.items()])
    voting_reg.fit(X_train, y_train)
    joblib.dump(voting_reg, 'saved/models/voting_regressor.joblib')
    
    y_pred = voting_reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    metrics['Voting Regressor'] = {'RMSE': rmse, 'R2': r2}
    print(f"Voting Regressor: RMSE = {rmse}, R2 = {r2}")

    # Plot RMSE
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
    data = pd.read_excel("data/Concrete_Data.xls")

    df = create_engineered_features(data)
    new_features = set(df.columns) - set(data.columns)
    original_target = 'Concrete compressive strength(MPa, megapascals) '
    original_features = df.columns.tolist()
    original_features.remove(original_target)

    X_train, X_test, y_train, y_test = preprocess_full_data(df=df)
    metrics = train_models(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)
    