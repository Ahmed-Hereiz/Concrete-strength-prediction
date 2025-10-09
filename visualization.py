import pandas as pd
from make_visuals import (
    visualize_feature_distribution,
    visualize_feature_pairplot,
    visualize_scatterplot_bivariate,
    visualize_correlation_heatmap
)
from preprocessing import create_engineered_features

def load_data_without_feature_engineering(file_path):
    df = pd.read_excel(file_path)
    df.rename(columns={
        "Cement (component 1)(kg in a m^3 mixture)": "cement",
        "Blast Furnace Slag (component 2)(kg in a m^3 mixture)": "blast-furnace slag",
        "Fly Ash (component 3)(kg in a m^3 mixture)": "fly-ash",
        "Water  (component 4)(kg in a m^3 mixture)": "water",
        "Superplasticizer (component 5)(kg in a m^3 mixture)": "superplasticizer",
        "Coarse Aggregate  (component 6)(kg in a m^3 mixture)": "coarse aggregate",
        "Fine Aggregate (component 7)(kg in a m^3 mixture)": "fine aggregate",
        "Age (day)": "age",
        "Concrete compressive strength(MPa, megapascals) ": "concrete CS"
    }, inplace=True)
    return df

def load_data_with_feature_engineering(file_path):
    df = create_engineered_features(pd.read_excel(file_path))
    df.rename(columns={
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
    }, inplace=True)
    return df


if __name__ == "__main__":
    df_original = load_data_without_feature_engineering("data/Concrete_Data.xls")
    visualize_feature_distribution(df_original,name="original_features")
    visualize_feature_pairplot(df_original)
    visualize_scatterplot_bivariate(df_original,x="cement",unit_x="kg/m³",unit_y="MPa") 
    visualize_scatterplot_bivariate(df_original, x="age", trendline="lowess",unit_x="(days)",unit_y="MPa")   

    df_engineered = load_data_with_feature_engineering("data/Concrete_Data.xls")
    visualize_feature_distribution(df_engineered,name="engineered_features")
    visualize_correlation_heatmap(df_engineered)