import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os


def create_engineered_features(df):
    df_new = df.copy()
    
    df_new['water_cement_ratio'] = df['Water  (component 4)(kg in a m^3 mixture)'] / df['Cement (component 1)(kg in a m^3 mixture)']
    
    df_new['total_cementitious'] = (df['Cement (component 1)(kg in a m^3 mixture)'] + 
                                   df['Blast Furnace Slag (component 2)(kg in a m^3 mixture)'] + 
                                   df['Fly Ash (component 3)(kg in a m^3 mixture)'])
    
    df_new['cementitious_water_ratio'] = df_new['total_cementitious'] / df['Water  (component 4)(kg in a m^3 mixture)']
    
    df_new['total_aggregate'] = (df['Coarse Aggregate  (component 6)(kg in a m^3 mixture)'] + 
                                df['Fine Aggregate (component 7)(kg in a m^3 mixture)'])
    df_new['fine_coarse_ratio'] = (df['Fine Aggregate (component 7)(kg in a m^3 mixture)'] / 
                                  df['Coarse Aggregate  (component 6)(kg in a m^3 mixture)'])
    
    total_weight = df.iloc[:, 0:7].sum(axis=1)
    for col in df.iloc[:, 0:7].columns:
        df_new[f'percent_{col.split("(")[0].strip()}'] = df[col] / total_weight * 100
    
    df_new['log_age'] = np.log1p(df['Age (day)'])
    
    df_new['age_category'] = pd.cut(df['Age (day)'], 
                                   bins=[0, 7, 27, float('inf')],
                                   labels=['early', 'medium', 'mature'])
    
    df_new['age_cement_interaction'] = df['Age (day)'] * df['Cement (component 1)(kg in a m^3 mixture)']
    
    df_new['cement_squared'] = df['Cement (component 1)(kg in a m^3 mixture)'] ** 2
    df_new['water_squared'] = df['Water  (component 4)(kg in a m^3 mixture)'] ** 2
    
    df_new['paste_volume_ratio'] = (df['Cement (component 1)(kg in a m^3 mixture)'] + 
                                   df['Water  (component 4)(kg in a m^3 mixture)'] + 
                                   df['Superplasticizer (component 5)(kg in a m^3 mixture)']) / total_weight
    
    return df_new


def preprocess_full_data(df, scale_data=True):
    x = df.drop(columns=['Concrete compressive strength(MPa, megapascals) '])
    y = df['Concrete compressive strength(MPa, megapascals) ']    

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)
    numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns

    scaler = StandardScaler()
    encoder = OrdinalEncoder()

    if scale_data:
        X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
        X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])
        # Save the scaler only if scaling is used
        os.makedirs('saved/preprocessing', exist_ok=True)
        joblib.dump(scaler, 'saved/preprocessing/scaler.joblib')
    else:
        # If not scaling, just copy the data (no transformation)
        X_train[numerical_columns] = X_train[numerical_columns].copy()
        X_test[numerical_columns] = X_test[numerical_columns].copy()

    if len(categorical_columns) > 0:
        X_train[categorical_columns] = encoder.fit_transform(X_train[categorical_columns])
        X_test[categorical_columns] = encoder.transform(X_test[categorical_columns])
        # Save the encoder
        os.makedirs('saved/preprocessing', exist_ok=True)
        joblib.dump(encoder, 'saved/preprocessing/encoder.joblib')

    return X_train, X_test, y_train, y_test
