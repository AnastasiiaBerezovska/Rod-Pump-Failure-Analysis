"""
Pump Lifetime Prediction using Random Survival Forest

This script predicts the expected lifetime of rod pumps using survival analysis techniques.

**Important**: 
- The code was originally developed using a proprietary dataset from ConocoPhillips.
- The dataset cannot be shared publicly.
- Users must supply their own dataset with similar structure (dates, lifetime, failure events).
- This script includes all processing steps except for bootstrapping, which is implemented in a separate file.

Author: Anastasiia Brund
"""

# *** Imports ***
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Survival modeling
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance

# Survival analysis visualization
from lifelines import KaplanMeierFitter

# *** STEP 1: Load Dataset ***
# TODO: Update the path below to point to your own dataset
csv_path = os.path.join("path", "to", "your", "dataset.csv")
df = pd.read_csv(csv_path)

# *** STEP 2: Preprocess Dates ***
df['lifetime_start'] = pd.to_datetime(df['lifetime_start'], errors='coerce')
df['lifetime_end'] = pd.to_datetime(df['lifetime_end'], errors='coerce')
df['FAILSTART'] = pd.to_datetime(df['FAILSTART'], errors='coerce')

# Calculate lifetime duration and event occurrence
df['duration'] = (df['lifetime_end'] - df['lifetime_start']).dt.days
df['event'] = df['FAILSTART'].notna().astype(int)

# *** STEP 3: Time-Based Split ***
# Split the data into Train, Test, and Blind sets based on end date
latest_date = df['lifetime_end'].max()
blind_cutoff = latest_date.replace(day=1)
test_cutoff = blind_cutoff - pd.DateOffset(months=6)

train_mask = df['lifetime_end'] < test_cutoff
test_mask = (df['lifetime_end'] >= test_cutoff) & (df['lifetime_end'] < blind_cutoff)
blind_mask = df['lifetime_end'] >= blind_cutoff

df_train = df[train_mask].copy()
df_test = df[test_mask].copy()
df_blind = df[blind_mask].copy()

print(f"Dataset split: Train={len(df_train)}, Test={len(df_test)}, Blind={len(df_blind)}")

# *** STEP 4: Clean and Select Features ***
# Drop columns with too many missing values
missing_fraction = df_train.isnull().mean()
columns_to_keep = missing_fraction[missing_fraction < 0.5].index.tolist()

df_train = df_train[columns_to_keep].select_dtypes(include=['float64', 'int64'])
df_test = df_test[df_train.columns]
df_blind = df_blind[df_train.columns]

# Drop constant columns
constant_cols = df_train.columns[df_train.nunique() <= 1]
df_train = df_train.drop(columns=constant_cols)
df_test = df_test.drop(columns=constant_cols)
df_blind = df_blind.drop(columns=constant_cols)

# Separate features and targets
X_train = df_train.drop(columns=['duration', 'event'], errors='ignore')
X_test = df_test.drop(columns=['duration', 'event'], errors='ignore')
X_blind = df_blind.drop(columns=['duration', 'event'], errors='ignore')

y_train = df.loc[train_mask, ['event', 'duration']]
y_test = df.loc[test_mask, ['event', 'duration']]
y_blind = df.loc[blind_mask, ['event', 'duration']]

# Cap extreme duration values
y_train['duration'] = np.minimum(y_train['duration'], 2000)
y_test['duration'] = np.minimum(y_test['duration'], 2000)
y_blind['duration'] = np.minimum(y_blind['duration'], 2000)

# Convert to structured arrays for survival models
y_train_struct = np.array(list(zip(y_train['event'].astype(bool), y_train['duration'])),
                          dtype=[('event', 'bool'), ('duration', 'f8')])
y_test_struct = np.array(list(zip(y_test['event'].astype(bool), y_test['duration'])),
                         dtype=[('event', 'bool'), ('duration', 'f8')])
y_blind_struct = np.array(list(zip(y_blind['event'].astype(bool), y_blind['duration'])),
                          dtype=[('event', 'bool'), ('duration', 'f8')])

# *** STEP 5: Preprocessing Pipeline ***
preprocessor = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler()
)
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)
X_blind_proc = preprocessor.transform(X_blind)

# *** STEP 6: Model Training ***
rsf = RandomSurvivalForest(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)
rsf.fit(X_train_proc, y_train_struct)

# *** STEP 7: Evaluation Function ***
def evaluate_predictions(name, X_proc, y, y_struct):
    preds = rsf.predict(X_proc)
    actual = y['duration'].values
    mae = np.mean(np.abs(actual - preds))
    med_ae = np.median(np.abs(actual - preds))
    c_idx = concordance_index_censored(y_struct['event'], y_struct['duration'], preds)[0]
    
    print(f"\n{name} Set Evaluation:")
    print(f"MAE: {mae:.2f} days, Median AE: {med_ae:.2f} days, C-Index: {c_idx:.4f}")
    return preds

# *** STEP 8: Evaluate on Datasets ***
pred_train = evaluate_predictions("Train", X_train_proc, y_train, y_train_struct)
pred_test = evaluate_predictions("Test", X_test_proc, y_test, y_test_struct)
pred_blind = evaluate_predictions("Blind", X_blind_proc, y_blind, y_blind_struct)

# *** STEP 9: Save Predictions ***
df_test_output = df_test.copy()
df_test_output["actual_days"] = y_test['duration']
df_test_output["predicted_days"] = pred_test
df_test_output.to_csv("test_predictions.csv", index=False)

df_blind_output = df_blind.copy()
df_blind_output["actual_days"] = y_blind['duration']
df_blind_output["predicted_days"] = pred_blind
df_blind_output.to_csv("blind_predictions.csv", index=False)

# *** STEP 10: Feature Importance ***
print("\nCalculating Permutation Importance...")
result = permutation_importance(
    rsf, X_test_proc[:50], y_test_struct[:50],
    n_repeats=3, random_state=42, n_jobs=1
)

importances = result.importances_mean
top_n = 15
top_idx = np.argsort(importances)[-top_n:]

plt.figure(figsize=(8, 6))
plt.barh(range(top_n), importances[top_idx], align='center')
plt.yticks(range(top_n), [X_train.columns[i] for i in top_idx])
plt.xlabel("Mean Permutation Importance")
plt.title("Top Feature Importances")
plt.tight_layout()
plt.savefig("feature_importances.png")
plt.show()

# *** STEP 11: Visualizations ***
# 1. Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test['duration'].values, y=pred_test)
plt.xlabel("Actual Pump Lifetime (days)")
plt.ylabel("Predicted Pump Lifetime (days)")
plt.title("Predicted vs Actual (Test Set)")
plt.grid(True)
plt.tight_layout()
plt.savefig("scatter_pred_vs_actual.png")
plt.show()

# 2. Histogram of Prediction Error
errors = np.abs(y_test['duration'].values - pred_test)
plt.figure(figsize=(8, 5))
plt.hist(errors, bins=30, edgecolor='black')
plt.title("Prediction Error Distribution (Test Set)")
plt.xlabel("Absolute Error (days)")
plt.ylabel("Number of Pumps")
plt.tight_layout()
plt.savefig("error_distribution.png")
plt.show()

# 3. Kaplan-Meier Curves
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

# *** STEP 12: Bootstrapped Prediction Intervals (Train Set) ***
n_bootstraps = 100
rng = np.random.RandomState(42)
train_preds_bootstrap = []

for _ in range(n_bootstraps):
    indices = rng.choice(len(X_train_proc), size=len(X_train_proc), replace=True)
    X_sample = X_train_proc[indices]
    y_sample = y_train_struct[indices]

    model = RandomSurvivalForest(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=rng
    )
    model.fit(X_sample, y_sample)
    train_preds_bootstrap.append(model.predict(X_train_proc))

train_preds_bootstrap = np.array(train_preds_bootstrap)
pred_train_q25 = np.percentile(train_preds_bootstrap, 25, axis=0)
pred_train_q75 = np.percentile(train_preds_bootstrap, 75, axis=0)

# Plot prediction intervals
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
