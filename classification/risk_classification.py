"""
Risk Classification and Radial Visualization for Pumps
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

# ***Risk Classification***

# This function assigns a risk level (High, Medium, Low)
# based on the quartiles of the `predicted_days` column:
# - Q1: High Risk
# - Q2: Medium Risk
# - Q3+: Low Risk
def classify_risk(df):
    q25 = df["predicted_days"].quantile(0.25)
    q75 = df["predicted_days"].quantile(0.75)

    def assign_risk(x):
        if x <= q25:
            return "High Risk"
        elif x <= q75:
            return "Medium Risk"
        else:
            return "Low Risk"

    df["Risk Group"] = df["predicted_days"].apply(assign_risk)
    return df

# ***Radial Chart Visualization***

# Creates a radar-style chart to visualize the distribution
# of pumps across different risk groups. The chart uses a
# customizable colormap for aesthetics.
def create_risk_radial_chart(df, title, output_filename, colormap=plt.cm.rainbow):
    df = classify_risk(df)
    risk_counts = df["Risk Group"].value_counts().reindex(["High Risk", "Medium Risk", "Low Risk"]).fillna(0).astype(int)

    labels = risk_counts.index.tolist()
    stats = risk_counts.values.tolist()
    num_vars = len(labels)

    # Prepare angles and wrap around to close the circle
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    stats += stats[:1]
    angles += angles[:1]

    # Initialize polar plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Plot and fill chart
    ax.plot(angles, stats, linewidth=2, linestyle='solid')
    ax.fill(angles, stats, color='plum', alpha=0.25)

    # Add color segments and text labels
    for i in range(num_vars):
        ax.plot(angles[i:i+2], stats[i:i+2], linewidth=6, solid_capstyle='round', color=colormap(i / num_vars))
        ax.text(
            angles[i], stats[i] + max(stats)*0.05, str(stats[i]),
            fontsize=12, fontweight='bold', ha='center', va='center', color='navy'
        )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])

    # Save chart
    plt.title(title, fontsize=14, y=1.08, weight='bold')
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()

# ***Load Prediction Files***

# Load model output files for train, test, and blind datasets.
# Ensure these files are generated in previous steps.
# Define the base path relative to the current script
# Define the base path relative to the current script
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

# Define file paths for the datasets
file_paths = {
    "train": os.path.join(base_path, "train_predictions.csv"),
    "test": os.path.join(base_path, "test_predictions.csv"),
    "blind": os.path.join(base_path, "blind_predictions.csv"),
}

# Load the datasets into DataFrames
datasets = {key: pd.read_csv(path) for key, path in file_paths.items()}

# Assign DataFrames to variables for easier access
train_df = datasets["train"]
test_df = datasets["test"]
blind_df = datasets["blind"]


# ***Generate and Save Radial Charts***

# Creates radar-style risk charts for each dataset and saves
# them as PNG files in the specified directory.
train_chart = os.path.join(base_path, "train_risk_radial_chart.png")
test_chart = os.path.join(base_path, "test_risk_radial_chart.png")
blind_chart = os.path.join(base_path, "blind_risk_radial_chart.png")

create_risk_radial_chart(train_df, "Train Set Risk Groups", train_chart, colormap=plt.cm.Blues)
create_risk_radial_chart(test_df, "Test Set Risk Groups", test_chart, colormap=plt.cm.Purples)
create_risk_radial_chart(blind_df, "Blind Set Risk Groups", blind_chart, colormap=plt.cm.plasma)

print("Radial charts saved to:")
print(train_chart)
print(test_chart)
print(blind_chart)

# ***Display Radial Charts***

# Automatically opens and displays the saved risk charts
# for train, test, and blind datasets.
for path, label in zip(
    [train_chart, test_chart, blind_chart],
    ["Train Set Risk Distribution", "Test Set Risk Distribution", "Blind Set Risk Distribution"]
):
    plt.figure(figsize=(7, 7))
    plt.imshow(mpimg.imread(path))
    plt.axis('off')
    plt.title(label)
    plt.tight_layout()
    plt.show()
