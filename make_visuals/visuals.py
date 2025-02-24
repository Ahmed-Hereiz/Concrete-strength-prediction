import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def visualize_feature_distributaion(df, show_plot=True):

    features = df.columns

    n_features = len(features)
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))
    
    # Calculate dynamic width and height based on number of features
    base_size = 300  
    width = n_cols * base_size
    height = n_rows * base_size
    width = max(800, width)
    height = max(800, height)
    
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=features)
    for i, feature in enumerate(features):
        row = i // 3 + 1
        col = i % 3 + 1
        fig.add_trace(go.Histogram(x=df[feature], name=feature, marker=dict(color="steelblue")), row=row, col=col)

    fig.update_layout(
        title="Feature Distributions",
        template="plotly_white",
        height=height,
        width=width,
        showlegend=False
    )

    if show_plot:
        fig.show()

    fig.write_html("saved/visualization/feature_distribution_histograms.html")


def visualize_feature_pairplot(df, show_plot=True):
    fig = px.scatter_matrix(
        df,
        dimensions=df.columns,
        title="Pair Plot of Features",
        color_discrete_sequence=["#2E7FBA"]  
    )

    fig.update_layout(
        width=1200, height=1200,  
        template="plotly_white"
    )

    if show_plot:
        fig.show()

    fig.write_html("saved/visualization/feature_pairplot.html")

def visualize_scatterplot_bivariate(df, x, y="Strength", show_plot=True, trendline=None):
    fig = px.scatter(df, x=x,y=y,
                    template="plotly_white", opacity=0.7, trendline=trendline)

    fig.update_layout(title=f"{x} vs {y}", height=500, width=900)
    
    if show_plot:
        fig.show()

    fig.write_html(f"saved/visualization/{x}_vs_{y}.html")


def visualize_correlation_heatmap(df, show_plot=True):
    
    corr_matrix = df.drop(columns=["Age_Category"]).corr().round(2)

    # Create Heatmap with values displayed inside each square
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="Blues",
            zmin=-1, zmax=1,
            colorbar=dict(title="Correlation"),
            text=corr_matrix.values,
            texttemplate="%{text}",
            hoverinfo="z"
        )
    )

    fig.update_layout(
        title="Correlation Heatmap with Values (Excluding Age_Category)",
        width=1000, height=800,
        template="plotly_white"
    )

    if show_plot:
        fig.show()

    fig.write_html("saved/visualization/correlation_heatmap.html")