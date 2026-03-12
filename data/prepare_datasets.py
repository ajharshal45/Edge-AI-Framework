"""
EdgeAI Framework — Dataset Preparation
========================================
Converts three raw public datasets into the feature + target
format required by the framework, saves them as CSV, then
retrains all 9 models on the real data.

Run:
    python data/prepare_datasets.py
"""

import os
import sys
import subprocess
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════
#  STEP 1: Healthcare (processed.cleveland.data)
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 50)
print("STEP 1: Preparing Healthcare Dataset")
print("=" * 50)

cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
        'restecg', 'thalach', 'exang', 'oldpeak',
        'slope', 'ca', 'thal', 'num']

df = pd.read_csv('data/processed.cleveland.data', names=cols, na_values='?')
print(f"  Loaded: {len(df)} rows")

df = df.dropna()
print(f"  After dropping NaN: {len(df)} rows")

healthcare = pd.DataFrame()
healthcare['heart_rate'] = df['thalach']
healthcare['blood_pressure'] = df['trestbps']
healthcare['temperature'] = 36.0 + (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min()) * 4.0
healthcare['spo2'] = 90 + (df['ca'].max() - df['ca']) / (df['ca'].max() - df['ca'].min()) * 10.0
healthcare['activity_level'] = df['slope']
healthcare['target'] = (df['num'] > 0).astype(int)

healthcare.to_csv('data/healthcare.csv', index=False)

t0 = (healthcare['target'] == 0).sum()
t1 = (healthcare['target'] == 1).sum()
print(f"  Shape: {healthcare.shape}")
print(f"  Target: 0={t0} ({t0/len(healthcare)*100:.1f}%) | 1={t1} ({t1/len(healthcare)*100:.1f}%)")
print(f"  Saved: data/healthcare.csv")


# ══════════════════════════════════════════════════════════
#  STEP 2: Smart City (Metro_Interstate_Traffic_Volume.csv)
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 50)
print("STEP 2: Preparing Smart City Dataset")
print("=" * 50)

df = pd.read_csv('data/Metro_Interstate_Traffic_Volume.csv')
print(f"  Loaded: {len(df)} rows")
print(f"  Columns: {list(df.columns)}")

# De-duplicate
df = df.drop_duplicates(subset=['date_time'], keep='first').reset_index(drop=True)
print(f"  After de-dup: {len(df)} rows")

# Extract hour
df['time_of_day'] = pd.to_datetime(df['date_time']).dt.hour

# vehicle_count: normalize traffic_volume to 0-500
vol_min, vol_max = df['traffic_volume'].min(), df['traffic_volume'].max()
df['vehicle_count'] = ((df['traffic_volume'] - vol_min) / (vol_max - vol_min) * 500).round(0).astype(int)

# avg_speed: normalize temp (Kelvin) to 20-120
t_min, t_max = df['temp'].min(), df['temp'].max()
df['avg_speed'] = (20.0 + (df['temp'] - t_min) / (t_max - t_min) * 100.0).round(1)

# traffic_density: clouds_all / 100
df['traffic_density'] = (df['clouds_all'] / 100.0).round(3)

# weather mapping
weather_map = {'Clear': 1, 'Clouds': 2, 'Rain': 3, 'Snow': 4}
df['weather'] = df['weather_main'].map(weather_map).fillna(2).astype(int)

# TARGET: rush hour based (NOT traffic volume)
rush_hours = [7, 8, 9, 16, 17, 18, 19]
df['target'] = df['time_of_day'].isin(rush_hours).astype(int)

smartcity = df[['vehicle_count', 'avg_speed', 'traffic_density',
                'weather', 'target']].copy()

smartcity.to_csv('data/smartcity.csv', index=False)

t0 = (smartcity['target'] == 0).sum()
t1 = (smartcity['target'] == 1).sum()
print(f"  Shape: {smartcity.shape}")
print(f"  Target: 0={t0} ({t0/len(smartcity)*100:.1f}%) | 1={t1} ({t1/len(smartcity)*100:.1f}%)")
print(f"  Saved: data/smartcity.csv")


# ══════════════════════════════════════════════════════════
#  STEP 3: Environment (AirQualityUCI.csv)
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 50)
print("STEP 3: Preparing Environment Dataset")
print("=" * 50)

df = pd.read_csv('data/AirQualityUCI.csv', sep=';', decimal=',')
print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Drop empty trailing columns
df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
df = df.dropna(how='all').reset_index(drop=True)

# Replace -200 with NaN
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].replace(-200, np.nan).replace(-200.0, np.nan)

# Drop NaN rows
before = len(df)
df = df.dropna().reset_index(drop=True)
print(f"  Dropped {before - len(df)} rows with missing → {len(df)} remain")

# TARGET: PT08.S1(CO) > 60th percentile → 1, else 0
co_sensor = df['PT08.S1(CO)']
threshold = co_sensor.quantile(0.60)
print(f"  PT08.S1(CO) 60th percentile threshold: {threshold}")

environment = pd.DataFrame()
environment['temperature'] = df['T'].round(1)
environment['humidity'] = df['RH'].round(1)
environment['pm25'] = df['PT08.S4(NO2)'].round(1)
environment['co2'] = df['PT08.S2(NMHC)'].round(1)
environment['target'] = (co_sensor > threshold).astype(int)

# PT08.S1(CO) is NOT included as a feature — only used for target

environment.to_csv('data/environment.csv', index=False)

t0 = (environment['target'] == 0).sum()
t1 = (environment['target'] == 1).sum()
print(f"  Shape: {environment.shape}")
print(f"  Target: 0={t0} ({t0/len(environment)*100:.1f}%) | 1={t1} ({t1/len(environment)*100:.1f}%)")
print(f"  Saved: data/environment.csv")


# ══════════════════════════════════════════════════════════
#  STEP 4: Summary + Retrain
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 50)
print("DATASET PREPARATION COMPLETE")
print("=" * 50)
print("Now retraining all models on real data...")

result = subprocess.run(
    [sys.executable, '-m', 'edgeai.models.trainer'],
    capture_output=True, text=True
)
print(result.stdout)
if result.stderr:
    print(result.stderr)
