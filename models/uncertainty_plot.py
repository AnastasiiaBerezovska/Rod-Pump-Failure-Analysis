# src/uncertainty_plot.py
"""
Generate uncertainty intervals using bootstrapped Random Survival Forests.
Assumes you have already trained a model and saved train_predictions.csv.

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sksurv.ensemble import RandomSurvivalForest

# ****************** Load Train Predictions ******************
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(ROOT_DIR, "data", "train_predictions.csv")
df_train = pd.read_csv(data_path)

# Simulate the original train labels (for demo purposes)
y_train = pd.DataFrame({
    "duration": df_train.get("actual_days", np.random.randint(100, 1000, size=len(df_train))),
    "event": np.random.randint(0, 2, size=len(df_train))
})

# Placeholder: simulate preprocessed features (you should load or recompute X_train_proc)
X_train_proc = df_train.drop(columns=["predicted_days"], errors='ignore')
y_train_struct = np.array(list(zip(y_train['event'].astype(bool), y_train['duration'])),
                          dtype=[('event', 'bool'), ('duration', 'f8')])

# ****************** Bootstrapped RSF Models ******************
n_bootstraps = 100
rng = np.random.RandomState(42)
train_preds_bootstrap = []

for _ in range(n_bootstraps):
    indices = rng.choice(len(X_train_proc), size=len(X_train_proc), replace=True)
    X_sample = X_train_proc.iloc[indices].values
    y_sample = y_train_struct[indices]

    model = RandomSurvivalForest(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=rng
    )
    model.fit(X_sample, y_sample)
    train_preds_bootstrap.append(model.predict(X_train_proc))

train_preds_bootstrap = np.array(train_preds_bootstrap)
pred_train = df_train['predicted_days'].values
pred_train_q25 = np.percentile(train_preds_bootstrap, 25, axis=0)
pred_train_q75 = np.percentile(train_preds_bootstrap, 75, axis=0)

# ****************** Plot Prediction Intervals ******************
sorted_indices = np.argsort(y_train['duration'].values)

plt.figure(figsize=(10, 6))
plt.errorbar(
    np.arange(len(pred_train)),
    pred_train[sorted_indices],
    yerr=[
        pred_train[sorted_indices] - pred_train_q25[sorted_indices],
        pred_train_q75[sorted_indices] - pred_train[sorted_indices]
    ],
    fmt='o',
    ecolor='gray',
    alpha=0.5,
    capsize=3,
    label='Predicted (with uncertainty)'
)
plt.plot(np.arange(len(pred_train)), y_train['duration'].values[sorted_indices], 'r-', label='Actual')
plt.xlabel("Sample Index")
plt.ylabel("Lifetime (days)")
plt.title("Prediction Intervals (Bootstrapped) vs Actual (Train Set)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("train_uncertainty_intervals.png")
plt.show()
