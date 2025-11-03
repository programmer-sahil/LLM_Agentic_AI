"""
===============================================================
File: plotting.py
Author: SK Sahil
Project: Bachelor's Thesis – From Transformers to Agents
Module: Utility - Plotting Tools
===============================================================
Description:
Reusable plotting functions for generating charts and visualizations
for LLM performance comparison and analysis.
===============================================================
"""

import matplotlib.pyplot as plt


def plot_bar_chart(models, values, title, ylabel, output_path):
    """Creates a labeled bar chart."""
    plt.figure(figsize=(7, 4))
    plt.bar(models, values, color=["#90CAF9", "#A5D6A7"])
    plt.title(title, fontsize=12, fontweight="bold")
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved bar chart: {output_path}")


def plot_scatter(x, y, xlabel, ylabel, title, output_path):
    """Plots a scatter diagram for correlation visualization."""
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, color="#42A5F5", edgecolor="black")
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved scatter plot: {output_path}")


if __name__ == "__main__":
    plot_bar_chart(["TinyLlama", "Phi-2"], [210.35, 9.84],
                   "Latency Comparison", "Latency (s)",
                   "results/visualizations/sample_bar_chart.png")
