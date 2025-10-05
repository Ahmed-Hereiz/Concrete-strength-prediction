import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def visualize_feature_distributaion(df, show_plot=True):

    features = df.columns

    n_features = len(features)
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))
    
    # Enhanced resolution: increase base size and bin count
    base_size = 600  # was 300
    width = n_cols * base_size
    height = n_rows * base_size
    width = max(1600, width)
    height = max(1600, height)
    
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=features)
    for i, feature in enumerate(features):
        row = i // 3 + 1
        col = i % 3 + 1
        # Use more bins for higher resolution
        fig.add_trace(
            go.Histogram(
                x=df[feature], 
                name=feature, 
                marker=dict(color="steelblue"),
                nbinsx=60  # increase bin count for smoother histograms
            ), 
            row=row, col=col
        )

    fig.update_layout(
        title="Feature Distributions",
        template="plotly_white",
        height=height,
        width=width,
        showlegend=False,
        font=dict(size=22),  # larger font for clarity
        margin=dict(l=40, r=40, t=80, b=40)
    )

    if show_plot:
        fig.show(config={"displayModeBar": True, "displaylogo": False})

    fig.write_html("saved/visualization/feature_distribution_histograms.html", full_html=True, include_plotlyjs="cdn")


def visualize_feature_pairplot(df, show_plot=True):
    # Enhanced resolution: larger figure, larger marker size, larger font
    fig = px.scatter_matrix(
        df,
        dimensions=df.columns,
        title="Pair Plot of Features",
        color_discrete_sequence=["#2E7FBA"],
        height=1800,
        width=1800
    )

    fig.update_traces(diagonal_visible=True, marker=dict(size=7, opacity=0.7))
    fig.update_layout(
        template="plotly_white",
        font=dict(size=22),
        margin=dict(l=40, r=40, t=80, b=40)
    )

    if show_plot:
        fig.show(config={"displayModeBar": True, "displaylogo": False})

    fig.write_html("saved/visualization/feature_pairplot.html", full_html=True, include_plotlyjs="cdn")

def visualize_scatterplot_bivariate(df, x, y="Strength", show_plot=True, trendline=None):
    # Enhanced resolution: larger figure, larger marker, larger font
    fig = px.scatter(
        df, x=x, y=y,
        template="plotly_white", opacity=0.7, trendline=trendline,
        height=900, width=1600
    )

    fig.update_traces(marker=dict(size=12, opacity=0.8))
    fig.update_layout(
        title=f"{x} vs {y}",
        font=dict(size=22),
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    if show_plot:
        fig.show(config={"displayModeBar": True, "displaylogo": False})

    fig.write_html(f"saved/visualization/{x}_vs_{y}.html", full_html=True, include_plotlyjs="cdn")


def visualize_correlation_heatmap(df, show_plot=True):
    # Enhanced resolution: larger figure, larger font, larger cell text
    corr_matrix = df.drop(columns=["Age_Category"], errors="ignore").corr().round(2)

    # Remove 'titlefont' from colorbar dict, as it is not a valid property
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="Blues",
            zmin=-1, zmax=1,
            colorbar=dict(title="Correlation", tickfont=dict(size=20)),
            text=corr_matrix.values,
            texttemplate="%{text}",
            hoverinfo="z"
        )
    )

    fig.update_layout(
        title="Correlation Heatmap with Values (Excluding Age_Category)",
        width=1800, height=1400,
        template="plotly_white",
        font=dict(size=22),
        margin=dict(l=40, r=40, t=80, b=40)
    )

    # Increase annotation font size for heatmap
    for d in fig.data:
        if hasattr(d, "textfont"):
            d.textfont = dict(size=20)

    if show_plot:
        fig.show(config={"displayModeBar": True, "displaylogo": False})

    fig.write_html("saved/visualization/correlation_heatmap.html", full_html=True, include_plotlyjs="cdn")