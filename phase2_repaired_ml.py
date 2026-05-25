import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ==========================================
# 1. LOAD THE REPAIRED DATASET
# ==========================================
print("🔬 Loading Repaired Master Dataset (with 100% feature matrix)...")
df = pd.read_csv('data/processed/master_hab_dataset_repaired.csv')

# ⚠️ CRITICAL UPDATE: We are NO LONGER dropping Solar_Radiation, Evaporation, or Runoff.
# We are only dropping the string identifier and the mathematically entangled optical bands.
cols_to_drop = [
    'Lake_Name', 'Chl_A_Proxy', 
    'NDWI_Shallow_Index', 'Turbidity_NDTI'
]
df_clean = df.drop(columns=cols_to_drop)

# ==========================================
# 2. DATA PREPARATION & IMPUTATION
# ==========================================
print("🩹 Imputing minor telemetry gaps using KNN...")
imputer = KNNImputer(n_neighbors=5)
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])

X = df_clean.drop(columns=['Target_NDCI'])
y = df_clean['Target_NDCI']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale for SVR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. MULTI-MODEL BENCHMARK SUITE
# ==========================================
models = {
    "XGBoost": XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
    "Support Vector Regression (SVR)": SVR(kernel='rbf', C=10, epsilon=0.01)
}

results = []
print("\n🏋️ Training models on full 16-parameter matrix...")

for name, model in models.items():
    if name == "Support Vector Regression (SVR)":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    results.append({"Model": name, "R-squared": r2, "RMSE": rmse, "MAE": mae})

results_df = pd.DataFrame(results)
print("\n📊 UPGRADED MODEL PERFORMANCE COMPARISON TABLE:")
print(results_df.to_string(index=False))

# ==========================================
# 4. VISUALIZE NEW FEATURE IMPORTANCE (XGBoost)
# ==========================================
print("\n📈 Generating Updated Feature Importance Chart...")
xgb_final = models["XGBoost"]
importance = xgb_final.feature_importances_
feature_names = X.columns
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(by='Importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color='darkmagenta')
plt.title('True Environmental Drivers (Including Full Hydrological Flow)', fontweight='bold')
plt.xlabel('Relative Importance')
plt.tight_layout()

# Save Outputs
os.makedirs('data/processed/presentation_graphs', exist_ok=True)
plt.savefig('data/processed/presentation_graphs/9_Upgraded_Feature_Importance.png', dpi=300)
print("✅ Upgraded Feature Importance chart saved to: data/processed/presentation_graphs/9_Upgraded_Feature_Importance.png")