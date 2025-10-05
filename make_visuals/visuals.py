import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Ensure the save directory exists
os.makedirs("saved/visualization", exist_ok=True)

def visualize_feature_distribution(df, show_plot=False):
    features = df.columns
    for feature in features:
        # Create a single plot for each feature
        fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
        ax.hist(df[feature], bins=30, color='steelblue', edgecolor='black')  # Reduced bins for faster rendering
        ax.set_title(feature, fontsize=10)
        ax.set_xlabel(f"{feature} ")
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle='--', alpha=0.5)  # Simplified grid
        
        # Save individual histogram
        plt.savefig(f"saved/visualization/histogram_{feature}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close immediately to free memory

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

def visualize_scatterplot_bivariate(df, x, y="Strength", show_plot=False, trendline=None):
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    ax.scatter(df[x], df[y], color='steelblue', alpha=0.5, s=30, edgecolor='black')  # Reduced marker size
    
    if trendline == 'ols':
        sns.regplot(x=df[x], y=df[y], ax=ax, scatter=False, color='red', line_kws={'linewidth': 1})
    
    ax.set_title(f"{x} vs {y}", fontsize=10)
    ax.set_xlabel(f"{x} ")
    ax.set_ylabel(f"{y} ")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(f"saved/visualization/{x}_vs_{y}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def visualize_correlation_heatmap(df, show_plot=False):
    corr_matrix = df.drop(columns=["Age_Category"], errors="ignore").corr().round(2)
    
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Blues", vmin=-1, vmax=1,
                annot_kws={"size": 8}, cbar_kws={'label': 'Correlation'}, ax=ax)
    
    ax.set_title("Correlation Heatmap", fontsize=10)
    
    plt.savefig("saved/visualization/correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close(fig)