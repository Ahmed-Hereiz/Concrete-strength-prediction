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
        "Cement (component 1)(kg in a m^3 mixture)": "Cement",
        "Blast Furnace Slag (component 2)(kg in a m^3 mixture)": "Slag",
        "Fly Ash (component 3)(kg in a m^3 mixture)": "FlyAsh",
        "Water  (component 4)(kg in a m^3 mixture)": "Water",
        "Superplasticizer (component 5)(kg in a m^3 mixture)": "Plasticizer",
        "Coarse Aggregate  (component 6)(kg in a m^3 mixture)": "CoarseAgg",
        "Fine Aggregate (component 7)(kg in a m^3 mixture)": "FineAgg",
        "Age (day)": "Age_Days",
        "Concrete compressive strength(MPa, megapascals) ": "Strength"
    }, inplace=True)
    return df

def load_data_with_feature_engineering(file_path):
    df = create_engineered_features(pd.read_excel(file_path))
    df.rename(columns={
        "Cement (component 1)(kg in a m^3 mixture)": "Cement",
        "Blast Furnace Slag (component 2)(kg in a m^3 mixture)": "Slag",
        "Fly Ash (component 3)(kg in a m^3 mixture)": "FlyAsh",
        "Water  (component 4)(kg in a m^3 mixture)": "Water",
        "Superplasticizer (component 5)(kg in a m^3 mixture)": "Plasticizer",
        "Coarse Aggregate  (component 6)(kg in a m^3 mixture)": "CoarseAgg",
        "Fine Aggregate (component 7)(kg in a m^3 mixture)": "FineAgg",
        "Age (day)": "Age_Days",
        "Concrete compressive strength(MPa, megapascals) ": "Strength",
        "water_cement_ratio": "Water_Cement_Ratio",
        "total_cementitious": "Total_Cementitious",
        "cementitious_water_ratio": "Cementitious_Water_Ratio",
        "total_aggregate": "Total_Aggregate",
        "fine_coarse_ratio": "Fine_Coarse_Ratio",
        "percent_Cement": "Percent_Cement",
        "percent_Blast Furnace Slag": "Percent_Slag",
        "percent_Fly Ash": "Percent_FlyAsh",
        "percent_Water": "Percent_Water",
        "percent_Superplasticizer": "Percent_Superplasticizer",
        "percent_Coarse Aggregate": "Percent_CoarseAgg",
        "percent_Fine Aggregate": "Percent_FineAgg",
        "log_age": "Log_Age",
        "age_category": "Age_Category",
        "age_cement_interaction": "Age_Cement_Interaction",
        "cement_squared": "Cement_Squared",
        "water_squared": "Water_Squared",
        "paste_volume_ratio": "Paste_Volume_Ratio"
    }, inplace=True)
    return df


if __name__ == "__main__":
    df_original = load_data_without_feature_engineering("data/Concrete_Data.xls")
    visualize_feature_distribution(df_original)
    visualize_feature_pairplot(df_original)
    visualize_scatterplot_bivariate(df_original,x="Cement") 
    visualize_scatterplot_bivariate(df_original, x="Age_Days", trendline="lowess")   

    df_engineered = load_data_with_feature_engineering("data/Concrete_Data.xls")
    visualize_feature_distribution(df_engineered)
    visualize_correlation_heatmap(df_engineered)