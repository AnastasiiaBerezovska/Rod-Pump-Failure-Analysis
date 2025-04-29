# src/plot_diagnostics.py
"""
Generate diagnostic plots from RSF predictions.
This includes:
- Feature importance (from test set)
- Predicted vs actual scatter
- Error histogram
- Kaplan-Meier survival curves

Make sure you've already run `train_rsf_model.py` to generate the predictions.

Author: Anastasiia Brund
Date: April 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ****************** Load Processed Test Data ******************

# Assumes these CSVs were created by train_rsf_model.py
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
test_path = os.path.join(ROOT_DIR, "data", "test_predictions.csv")
df_test = pd.read_csv(test_path)

# If needed, also reload original y_test (simulate for now)
y_test = pd.DataFrame({
    "duration": df_test.get("actual_days", np.random.randint(100, 1000, size=len(df_test))),
    "event": np.random.randint(0, 2, size=len(df_test))
})

# ****************** Reload Model (optional placeholder) ******************

# For permutation importance (model needs to be reloaded or retrained)
# Here we simulate it (you should load or pass a trained model)
rsf = RandomSurvivalForest(n_estimators=100, max_depth=5, random_state=42)
rsf.fit(np.random.randn(len(df_test), df_test.shape[1]-1),  
        np.array([(bool(np.random.randint(0, 2)), d) for d in y_test["duration"]], 
                 dtype=[('event', 'bool'), ('duration', 'f8')]))

# ****************** Permutation Importance ******************

print("\n Calculating Permutation Importance...")
X_test_proc = df_test.drop(columns=["predicted_days"], errors='ignore')
y_struct = np.array([(bool(e), d) for e, d in zip(y_test['event'], y_test['duration'])], 
                    dtype=[('event', 'bool'), ('duration', 'f8')])

result = permutation_importance(
    rsf, X_test_proc[:50], y_struct[:50],
    n_repeats=3, random_state=42, n_jobs=1
)

importances = result.importances_mean
top_n = 15
top_idx = np.argsort(importances)[-top_n:]

plt.figure(figsize=(8, 6))
plt.barh(range(top_n), importances[top_idx], align='center')
plt.yticks(range(top_n), [X_test_proc.columns[i] for i in top_idx])
plt.xlabel("Mean Permutation Importance")
plt.title("Top Feature Importances")
plt.tight_layout()
plt.savefig("feature_importances.png")
plt.show()

# ****************** Scatter Plot ******************

pred_test = df_test["predicted_days"]
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test['duration'], y=pred_test)
plt.xlabel("Actual Pump Lifetime (days)")
plt.ylabel("Predicted Pump Lifetime (days)")
plt.title("Predicted vs Actual (Test Set)")
plt.grid(True)
plt.tight_layout()
plt.savefig("scatter_pred_vs_actual.png")
plt.show()

# ****************** Error Histogram ******************

errors = np.abs(y_test['duration'] - pred_test)
plt.figure(figsize=(8, 5))
plt.hist(errors, bins=30, edgecolor='black')
plt.title("Prediction Error Distribution (Test Set)")
plt.xlabel("Absolute Error (days)")
plt.ylabel("Number of Pumps")
plt.tight_layout()
plt.savefig("error_distribution.png")
plt.show()

# ****************** Kaplan-Meier Curves ******************

kmf_actual = KaplanMeierFitter()
kmf_pred = KaplanMeierFitter()

plt.figure(figsize=(8, 6))
kmf_actual.fit(y_test['duration'], event_observed=y_test['event'], label="Actual Survival")
kmf_pred.fit(pred_test, event_observed=y_test['event'], label="Predicted Survival")
kmf_actual.plot_survival_function()
kmf_pred.plot_survival_function()
plt.title("Kaplan-Meier Curve: Actual vs Predicted (Test Set)")
plt.xlabel("Days")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.tight_layout()
plt.savefig("kaplan_meier_curves.png")
plt.show()
