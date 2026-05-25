import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')

print("🌍 TARGET: GRAND UNIFICATION (Lakes + Dams)")
path_dams = 'data/processed/phase2_regional_L8.csv'
path_lakes = 'data/processed/tn_phase1_lakes_L8.csv'

if not os.path.exists(path_dams) or not os.path.exists(path_lakes):
    print("❌ CRITICAL ERROR: Missing one or both dataset files.")
    exit()

# Load and Combine Both Datasets
df_dams = pd.read_csv(path_dams)
df_lakes = pd.read_csv(path_lakes)
df = pd.concat([df_dams, df_lakes], ignore_index=True)

# Self-Healing Cleaner
df = df.dropna(subset=['Target_NDCI'])
df = df.replace([np.inf, -np.inf], np.nan)

ideal_features = [
    'Month', 'Air_Temp_C', 'Surface_Temp_C', 'Rainfall_mm', 'Solar_Radiation', 'Wind_Speed',
    'Lagged_NDTI', 'Lagged_Air_Temp_C', 'Lagged_Surface_Temp_C', 'Lagged_Wind_Speed'
]

valid_features = [col for col in ideal_features if col in df.columns and not df[col].isna().all()]
df[valid_features] = df[valid_features].fillna(df[valid_features].mean(numeric_only=True))
df = df.dropna(subset=valid_features)

print(f"✅ Training on {len(df)} MASSIVE combined observations...")

X = df[valid_features]
y = df['Target_NDCI']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

mlr_model = LinearRegression().fit(X_train, y_train)
svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.01).fit(X_train, y_train)
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42).fit(X_train, y_train)

pred_ensemble = (mlr_model.predict(X_test) + svr_model.predict(X_test) + xgb_model.predict(X_test)) / 3.0

def get_metrics(y_true, y_pred): return r2_score(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred))

r2_ens, rmse_ens = get_metrics(y_test, pred_ensemble)

print("-" * 55)
print(f"{'🏆 COMBINED ENSEMBLE':<30} | {r2_ens:>8.4f} | {rmse_ens:>8.4f}")
print("=========================================================\n")