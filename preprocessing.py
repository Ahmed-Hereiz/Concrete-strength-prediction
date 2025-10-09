import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Mapping from original to new names
_RENAME_MAP = {
    "Cement (component 1)(kg in a m^3 mixture)": "cement",
    "Water  (component 4)(kg in a m^3 mixture)": "water",
    "Coarse Aggregate  (component 6)(kg in a m^3 mixture)": "coarse aggregate",
    "Fine Aggregate (component 7)(kg in a m^3 mixture)": "fine aggregate",
    "Superplasticizer (component 5)(kg in a m^3 mixture)": "superplasticizer",
    "Blast Furnace Slag (component 2)(kg in a m^3 mixture)": "blast-furnace slag",
    "Fly Ash (component 3)(kg in a m^3 mixture)": "fly-ash",
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

def _is_new_naming(df):
    # Check for a few key new names to determine if new naming is used
    required_new_names = [
        "cement", "water", "coarse aggregate", "fine aggregate", "superplasticizer",
        "blast-furnace slag", "fly-ash", "age", "concrete CS"
    ]
    return all(name in df.columns for name in required_new_names)

def create_engineered_features(df):
    df_new = df.copy()
    # Decide which naming to use
    if _is_new_naming(df):
        # Use new names
        cement = "cement"
        water = "water"
        bfs = "blast-furnace slag"
        flyash = "fly-ash"
        coarse_agg = "coarse aggregate"
        fine_agg = "fine aggregate"
        superplasticizer = "superplasticizer"
        age = "age"
        strength = "concrete CS"
    else:
        # Use original names
        cement = "Cement (component 1)(kg in a m^3 mixture)"
        water = "Water  (component 4)(kg in a m^3 mixture)"
        bfs = "Blast Furnace Slag (component 2)(kg in a m^3 mixture)"
        flyash = "Fly Ash (component 3)(kg in a m^3 mixture)"
        coarse_agg = "Coarse Aggregate  (component 6)(kg in a m^3 mixture)"
        fine_agg = "Fine Aggregate (component 7)(kg in a m^3 mixture)"
        superplasticizer = "Superplasticizer (component 5)(kg in a m^3 mixture)"
        age = "Age (day)"
        strength = "Concrete compressive strength(MPa, megapascals) "

    # Water-to-Cement
    df_new['Water-to-Cement'] = df[water] / df[cement]

    # Total Cementitious Materials
    df_new['Total Cementitious Materials'] = df[cement] + df[bfs] + df[flyash]

    # Cementitious-to-Water
    # Use the correct column for total cementitious (just created above)
    df_new['Cementitious-to-Water'] = df_new['Total Cementitious Materials'] / df[water]

    # Total Aggregate
    df_new['Total Aggregate'] = df[coarse_agg] + df[fine_agg]

    # Fine-to-Coarse Aggregate
    df_new['Fine-to-Coarse Aggregate'] = df[fine_agg] / df[coarse_agg]

    # Percentage Composition for each component (first 7 columns)
    # Try to get the correct columns for the 7 components
    if _is_new_naming(df):
        comp_cols = [
            "cement", "blast-furnace slag", "fly-ash", "water",
            "superplasticizer", "coarse aggregate", "fine aggregate"
        ]
    else:
        comp_cols = [
            "Cement (component 1)(kg in a m^3 mixture)",
            "Blast Furnace Slag (component 2)(kg in a m^3 mixture)",
            "Fly Ash (component 3)(kg in a m^3 mixture)",
            "Water  (component 4)(kg in a m^3 mixture)",
            "Superplasticizer (component 5)(kg in a m^3 mixture)",
            "Coarse Aggregate  (component 6)(kg in a m^3 mixture)",
            "Fine Aggregate (component 7)(kg in a m^3 mixture)"
        ]
    total_weight = df[comp_cols].sum(axis=1)
    for col in comp_cols:
        # Use the new naming for the output column
        if _is_new_naming(df):
            out_col = f"{col.capitalize()} Percentage Composition"
        else:
            out_col = f"{col.split('(')[0].strip()} Percentage Composition"
        df_new[out_col] = df[col] / total_weight * 100

    # Log-Transformed Age
    df_new['Log-Transformed Age'] = np.log1p(df[age])

    # Age Categorization
    df_new['Age Categorization'] = pd.cut(df[age], bins=[0, 7, 27, float('inf')], labels=['early', 'medium', 'mature'])

    # Age-Cement Interaction
    df_new['Age-Cement Interaction'] = df[age] * df[cement]

    # Cement squared
    df_new['Polynomial Term (P_C)'] = df[cement] ** 2

    # Water squared
    df_new['Polynomial Term (P_W)'] = df[water] ** 2

    # Paste Volume
    df_new['Paste Volume'] = (df[cement] + df[water] + df[superplasticizer]) / total_weight

    return df_new

def preprocess_full_data(df, scale_data=True):
    # Determine target column name
    if _is_new_naming(df):
        target_col = "concrete CS"
    else:
        target_col = "Concrete compressive strength(MPa, megapascals) "
    x = df.drop(columns=[target_col])
    y = df[target_col]

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
