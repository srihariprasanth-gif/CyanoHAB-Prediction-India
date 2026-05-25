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

# Suppress minor warnings for a clean terminal output
warnings.filterwarnings('ignore')

# ==========================================
# 1. LOAD THE DATA
# ==========================================
csv_path = 'data/processed/tn_phase1_lakes_L8.csv'

if not os.path.exists(csv_path):
    print(f"❌ CRITICAL ERROR: Could not find {csv_path}.")
    exit()

df = pd.read_csv(csv_path)

# ==========================================
# 2. THE SELF-HEALING CLEANING BLOCK
# ==========================================
print("🧹 Initiating automated data cleaning and ghost-column detection...")

# A. Drop rows missing the target variable (can't train without an answer)
df = df.dropna(subset=['Target_NDCI'])

# B. Convert weird "Infinity" math errors into standard NaNs
df = df.replace([np.inf, -np.inf], np.nan)

# C. Define the ideal features we WANT to use
ideal_features = [
    'Month', 'Air_Temp_C', 'Surface_Temp_C', 'Rainfall_mm', 'Solar_Radiation', 'Wind_Speed',
    'Lagged_NDTI', 'Lagged_Air_Temp_C', 'Lagged_Surface_Temp_C', 'Lagged_Wind_Speed'
]

# D. AUTO-DETECT GHOST COLUMNS: Only keep features that actually have data
valid_features = []
for col in ideal_features:
    if col in df.columns and not df[col].isna().all():
        valid_features.append(col)
    else:
        print(f"   ⚠️ WARNING: Ghost Column Detected & Removed -> '{col}'")

# E. Try to fill missing weather data with averages ONLY for valid features
df[valid_features] = df[valid_features].fillna(df[valid_features].mean(numeric_only=True))

# F. Final safety check: drop any rows that somehow still have NaNs
df = df.dropna(subset=valid_features)

if len(df) == 0:
    print("\n❌ CRITICAL ERROR: Dataset is still empty after cleaning. Extraction failed completely.")
    exit()

print(f"✅ Data rigorously cleaned! Training on {len(df)} pristine observations using {len(valid_features)} features...\n")

# ==========================================
# 3. PREPROCESSING & SPLIT
# ==========================================
X = df[valid_features]
y = df['Target_NDCI']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Lock the random seed to 42 to guarantee 100% reproducibility 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ==========================================
# 4. INITIALIZE & TRAIN THE ALGORITHMS
# ==========================================
print("🧠 Training Individual Models and Building Ensemble...")
print("---------------------------------------------------------")

mlr_model = LinearRegression()
svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.01)
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)

# Train all three models
mlr_model.fit(X_train, y_train)
svr_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Generate predictions
pred_mlr = mlr_model.predict(X_test)
pred_svr = svr_model.predict(X_test)
pred_xgb = xgb_model.predict(X_test)

# 🔬 THE ENSEMBLE FUSION (Average the predictions)
pred_ensemble = (pred_mlr + pred_svr + pred_xgb) / 3.0

# ==========================================
# 5. CALCULATE METRICS
# ==========================================
def get_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, rmse

r2_mlr, rmse_mlr = get_metrics(y_test, pred_mlr)
r2_svr, rmse_svr = get_metrics(y_test, pred_svr)
r2_xgb, rmse_xgb = get_metrics(y_test, pred_xgb)
r2_ens, rmse_ens = get_metrics(y_test, pred_ensemble)

# ==========================================
# 6. DISPLAY ACADEMIC BENCHMARK TABLE
# ==========================================
print(f"{'Model Framework':<30} | {'R² Score':<10} | {'RMSE':<10}")
print("-" * 55)
print(f"{'1. Multiple Linear Regression':<30} | {r2_mlr:>8.4f} | {rmse_mlr:>8.4f}")
print(f"{'2. Support Vector Regression':<30} | {r2_svr:>8.4f} | {rmse_svr:>8.4f}")
print(f"{'3. XGBoost (Gradient Boosting)':<30} | {r2_xgb:>8.4f} | {rmse_xgb:>8.4f}")
print("-" * 55)
print(f"{'🏆 STATISTICAL ENSEMBLE':<30} | {r2_ens:>8.4f} | {rmse_ens:>8.4f}")
print("=========================================================\n")