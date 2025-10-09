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
        "cement_squared": "cement squared",
        "water_squared": "water squared"
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
    # Set global matplotlib params for high-quality output
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "lines.linewidth": 2,
        "axes.linewidth": 1.5,
        "figure.constrained_layout.use": False
    })

    data_path = "data/Concrete_Data.xls"
    original_df = pd.read_excel(data_path)

    # Define the target column
    target = 'Concrete compressive strength(MPa, megapascals) '

    # 1) Original Data
    original_df_renamed = rename_features(original_df)
    model_orig, X_orig, shap_values_orig = calculate_shap(original_df_renamed, 'concrete CS')

    # 2) Engineered Data
    engineered_df = create_engineered_features(original_df)
    engineered_df = rename_features(engineered_df)
    model_eng, X_eng, shap_values_eng = calculate_shap(engineered_df, 'concrete CS')

    # ---- Plot 1: Bar chart for ORIGINAL features ----
    fig1, ax1 = plt.subplots(figsize=(14, 10), dpi=200)
    shap.plots.bar(shap_values_orig, max_display=None, show=False, ax=ax1)
    ax1.set_title("LightGBM Mean SHAP Value (Original Features)", fontsize=20)
    fig1.tight_layout()
    plt.show()

    # ---- Plot 2: Beeswarm summary for ORIGINAL features ----
    shap.summary_plot(
        shap_values_orig.values,
        X_orig,
        plot_type="dot",
        show=False,
        color_bar=True,
        alpha=0.8,
        max_display=20,
        title="LightGBM SHAP Feature Importance (Original Features)"
    )
    plt.tight_layout()
    plt.show()

    # ---- Plot 3: Bar chart for ENGINEERED features ----
    fig3, ax3 = plt.subplots(figsize=(14, 10), dpi=200)
    shap.plots.bar(shap_values_eng, max_display=None, show=False, ax=ax3)
    ax3.set_title("LightGBM Mean SHAP Value (Engineered Features)", fontsize=20)
    fig3.tight_layout()
    plt.show()

    # ---- Plot 4: Beeswarm summary for ENGINEERED features ----
    shap.summary_plot(
        shap_values_eng.values,
        X_eng,
        plot_type="dot",
        show=False,
        color_bar=True,
        alpha=0.8,
        max_display=20,
        title="LightGBM SHAP Feature Importance (Engineered Features)"
    )
    plt.tight_layout()
    plt.show()
