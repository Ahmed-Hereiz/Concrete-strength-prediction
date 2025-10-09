import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

os.makedirs("saved/visualization", exist_ok=True)


def visualize_feature_distribution(
    df, features=None, show_plot=False, name=""
):
    """
    Plots a grid of feature histograms using seaborn/matplotlib.
    The grid size is determined dynamically based on the number of features.
    If features is None, uses all columns of df.
    """
    if features is None:
        features = list(df.columns)
    else:
        features = list(features)
    n = len(features)
    # Dynamically determine grid size (aim for a nearly square grid)
    ncols = min(4, n) if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig_width = max(4 * ncols, 8)
    fig_height = max(3 * nrows, 6)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), constrained_layout=True)
    # Flatten axes for easy indexing, even if nrows or ncols == 1
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    for i, feature in enumerate(features):
        ax = axes[i]
        sns.histplot(
            df[feature].dropna(),
            bins=30,
            color="#4B8BBE",
            edgecolor="white",
            alpha=0.85,
            ax=ax
        )
        ax.set_title(f"{feature}", fontsize=10, fontweight="bold", pad=10)
        # ax.set_xlabel(f"{feature}", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
    # Hide any unused subplots
    for j in range(n, nrows * ncols):
        fig.delaxes(axes[j])
    # fig.suptitle("Feature Distributions", fontsize=22, fontweight="bold", y=0.98)
    plt.savefig(f"saved/visualization/feature_distribution_grid_{name}.png", dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close(fig)

def visualize_feature_pairplot(df, show_plot=False):
    # Use seaborn pairplot with reduced marker size and simpler histograms
    g = sns.pairplot(df, diag_kind='hist', plot_kws={'color': '#2E7FBA', 'alpha': 0.5, 's': 10})
    g.fig.suptitle("Pair Plot of Features", fontsize=12, y=1.02)
    
    # Adjust axis labels with units
    for ax in g.axes.flatten():
        if ax.get_xlabel():
            ax.set_xlabel(f"{ax.get_xlabel()} ")
        if ax.get_ylabel():
            ax.set_ylabel(f"{ax.get_ylabel()} ")
    
    # Save the pairplot
    g.savefig("saved/visualization/feature_pairplot.png", dpi=300, bbox_inches='tight')
    plt.close(g.fig)

def visualize_scatterplot_bivariate(df, x, y="concrete CS", show_plot=False, trendline=None,
                                    unit_x="",unit_y=""):
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    ax.scatter(df[x], df[y], color='steelblue', alpha=0.5, s=30, edgecolor='black')  # Reduced marker size
    
    if trendline == 'ols':
        sns.regplot(x=df[x], y=df[y], ax=ax, scatter=False, color='red', line_kws={'linewidth': 1})
    
    ax.set_title(f"Relationship between {x} and {y}", fontsize=10)
    ax.set_xlabel(f"{x} {unit_x}")
    ax.set_ylabel(f"{y} {unit_y}")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(f"saved/visualization/{x}_vs_{y}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def visualize_correlation_heatmap(df, show_plot=False):
    import os
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        print("Not enough numeric columns to compute correlation heatmap.")
        return

    corr_matrix = numeric_df.corr().round(2)

    n_vars = len(corr_matrix.columns)
    # Make the plot much larger for better appearance
    fig_width = max(2.2 * n_vars, 16)
    fig_height = max(1.6 * n_vars, 12)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    # Set annotation font size to be large and bold
    annot_fontsize = max(22, int(320 / max(n_vars, 1)))

    # Draw the heatmap
    try:
        heatmap = sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=-1,
            vmax=1,
            annot_kws={"size": annot_fontsize, "weight": "bold"},
            cbar_kws={'label': 'Correlation'},
            linewidths=2.5,
            linecolor='white',
            square=True,
            ax=ax
        )
    except Exception as e:
        print(f"Error drawing heatmap: {e}")
        plt.close(fig)
        return

    # Make axis tick labels very large and bold
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha='right',
        fontsize=22,
        weight='bold'
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=22,
        weight='bold'
    )

    # Make colorbar label larger and bold
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('Correlation', fontsize=24, weight='bold')

    # Make the title very large and bold
    ax.set_title("Correlation Heatmap", fontsize=36, fontweight="bold", pad=24)

    os.makedirs("saved/visualization", exist_ok=True)
    try:
        plt.savefig("saved/visualization/correlation_heatmap.png", dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error saving heatmap: {e}")
    if show_plot:
        plt.show()
    plt.close(fig)