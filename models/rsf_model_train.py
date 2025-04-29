"""
Train Random Survival Forest and save predictions.

This script:
- Loads a rod pump failure dataset
- Preprocesses it
- Trains a Random Survival Forest (RSF) model
- Evaluates predictions
- Saves outputs for training, test, and blind sets

This script assumes the dataset exists in the "data/" folder as "rodpump_failure_final.csv".
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# *** PATH SETUP ***
# Get root of the project (one level up from this script's folder)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Dataset path
csv_path = os.path.join(ROOT_DIR, "data", "rodpump_failure_final.csv")

# Output paths
output_train = os.path.join(ROOT_DIR, "data", "train_predictions.csv")
output_test = os.path.join(ROOT_DIR, "data", "test_predictions.csv")
output_blind = os.path.join(ROOT_DIR, "data", "blind_predictions.csv")

# *** LOAD DATA ***
df = pd.read_csv(csv_path)
print(f"Loaded dataset: {df.shape}")

# *** PREPROCESS DATES ***
df['lifetime_start'] = pd.to_datetime(df['lifetime_start'], errors='coerce')
df['lifetime_end'] = pd.to_datetime(df['lifetime_end'], errors='coerce')
df['FAILSTART'] = pd.to_datetime(df['FAILSTART'], errors='coerce')

df['duration'] = (df['lifetime_end'] - df['lifetime_start']).dt.days
df['event'] = df['FAILSTART'].notna().astype(int)

# *** TIME-BASED SPLIT ***
latest_date = df['lifetime_end'].max()
blind_cutoff = latest_date.replace(day=1)
test_cutoff = blind_cutoff - pd.DateOffset(months=6)

train_mask = df['lifetime_end'] < test_cutoff
test_mask = (df['lifetime_end'] >= test_cutoff) & (df['lifetime_end'] < blind_cutoff)
blind_mask = df['lifetime_end'] >= blind_cutoff

df_train = df[train_mask].copy()
df_test = df[test_mask].copy()
df_blind = df[blind_mask].copy()

# *** CLEAN & SELECT FEATURES ***
missing_fraction = df_train.isnull().mean()
columns_to_keep = missing_fraction[missing_fraction < 0.5].index.tolist()

df_train = df_train[columns_to_keep].select_dtypes(include=['float64', 'int64'])
df_test = df_test[df_train.columns]
df_blind = df_blind[df_train.columns]

constant_cols = df_train.columns[df_train.nunique() <= 1]
df_train = df_train.drop(columns=constant_cols)
df_test = df_test.drop(columns=constant_cols)
df_blind = df_blind.drop(columns=constant_cols)

X_train = df_train.drop(columns=['duration', 'event'], errors='ignore')
X_test = df_test.drop(columns=['duration', 'event'], errors='ignore')
X_blind = df_blind.drop(columns=['duration', 'event'], errors='ignore')

y_train = df.loc[train_mask, ['event', 'duration']]
y_test = df.loc[test_mask, ['event', 'duration']]
y_blind = df.loc[blind_mask, ['event', 'duration']]

# Cap extreme durations
y_train['duration'] = np.minimum(y_train['duration'], 2000)
y_test['duration'] = np.minimum(y_test['duration'], 2000)
y_blind['duration'] = np.minimum(y_blind['duration'], 2000)

y_train_struct = np.array(list(zip(y_train['event'].astype(bool), y_train['duration'])),
                          dtype=[('event', 'bool'), ('duration', 'f8')])
y_test_struct = np.array(list(zip(y_test['event'].astype(bool), y_test['duration'])),
                         dtype=[('event', 'bool'), ('duration', 'f8')])
y_blind_struct = np.array(list(zip(y_blind['event'].astype(bool), y_blind['duration'])),
                          dtype=[('event', 'bool'), ('duration', 'f8')])

# *** PREPROCESS FEATURES ***
preprocessor = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler()
)
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)
X_blind_proc = preprocessor.transform(X_blind)

# *** TRAIN MODEL ***
rsf = RandomSurvivalForest(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)
rsf.fit(X_train_proc, y_train_struct)
print("Model trained.")

# *** PREDICT ***
pred_train = rsf.predict(X_train_proc)
pred_test = rsf.predict(X_test_proc)
pred_blind = rsf.predict(X_blind_proc)

# *** SAVE OUTPUTS ***
df_train_out = X_train.copy()
df_train_out['predicted_days'] = pred_train
df_train_out.to_csv(output_train, index=False)

df_test_out = X_test.copy()
df_test_out['predicted_days'] = pred_test
df_test_out.to_csv(output_test, index=False)

df_blind_out = X_blind.copy()
df_blind_out['predicted_days'] = pred_blind
df_blind_out.to_csv(output_blind, index=False)

print("Predictions saved to:")
print(f"  - {output_train}")
print(f"  - {output_test}")
print(f"  - {output_blind}")

