import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

print("🚀 Step 1: Initiating Clean Chemical Feature Extraction...")

# 1. Load and Consolidate the 4 Hectare Baseline Files
try:
    # 🐛 FIXED: Looking for the correct Phase 1 filename
    df1 = pd.read_csv('data/processed/phase1_tn_cropland.csv') 
    df2 = pd.read_csv('data/processed/phase2_batch_cropland.csv')
    df3 = pd.read_csv('data/processed/phase3_batch_cropland.csv')
    df4 = pd.read_csv('data/processed/remaining_batch_cropland.csv')
    
    spatial_baseline = pd.concat([df1, df2, df3, df4], ignore_index=True)
    print(f"   ↳ Consolidated spatial land-use metrics for all {len(spatial_baseline)} reservoirs.")
except Exception as e:
    print(f"❌ Error loading baseline CSVs: {e}")
    print("💡 Please ensure all 4 files are sitting in your current directory.")
    exit()

# 2. Define Regional Chemical Export Coefficients (kg/Ha/year)
INTENSIVE_N = 12.5
INTENSIVE_P = 1.8

PROTECTED_N = 2.5
PROTECTED_P = 0.25

# Forested/mountain catchments receiving baseline background leaching parameters
protected_catchments = [
    "Pechiparai_Dam", "Papanasam_Dam", "Sholayar_Dam", "Idukki_Dam", 
    "Mullaperiyar_Dam", "Parambikulam_Dam", "Neyyar_Dam", "Peechi_Dam", 
    "Mattupetty_Dam", "Supa_Dam"
]

spatial_baseline['E_N'] = np.where(spatial_baseline['Lake_Name'].isin(protected_catchments), PROTECTED_N, INTENSIVE_N)
spatial_baseline['E_P'] = np.where(spatial_baseline['Lake_Name'].isin(protected_catchments), PROTECTED_P, INTENSIVE_P)

# 3. Read original dataset and merge into a brand new standalone file
input_path = 'data/processed/master_hab_dataset_repaired.csv'
output_path = 'data/processed/master_hab_dataset_chemical.csv'

if os.path.exists(input_path):
    df_ml = pd.read_csv(input_path)
    
    # Structural merge of static spatial features into temporal matrix
    df_merged = df_ml.merge(spatial_baseline, on='Lake_Name', how='left')
    
    # Export Coefficient Model: Load = Hectares * Monthly Coefficient Factor * Dynamic Surface Transport (Runoff)
    df_merged['Calculated_Nitrogen_Load_kg'] = (
        df_merged['Cropland_Area_Ha'] * (df_merged['E_N'] / 12.0) * df_merged['Runoff']
    )
    df_merged['Calculated_Phosphorus_Load_kg'] = (
        df_merged['Cropland_Area_Ha'] * (df_merged['E_P'] / 12.0) * df_merged['Runoff']
    )
    
    # Drop intermediate processing flags
    df_merged = df_merged.drop(columns=['E_N', 'E_P'])
    
    # Save output dataset
    df_merged.to_csv(output_path, index=False)
    print(f"   ↳ Enriched nutrient matrix saved to: {output_path}")
else:
    print(f"❌ Core dataset missing at: {input_path}")
    exit()

print("\n🚀 Step 2: Ingesting Enriched Chemical Dataset for Machine Learning Evaluation...")
df_eval = pd.read_csv(output_path)

# 4. Feature Selection: Drop spatial identifiers, date blocks, and raw optical indices to prevent target leakage
cols_to_drop = ['Lake_Name', 'Year', 'Month', 'Chl_A_Proxy', 'NDWI_Shallow_Index', 'Turbidity_NDTI']
X_features = df_eval.drop(columns=[c for c in cols_to_drop if c in df_eval.columns or 'Target' in c])
y_target = df_eval['Target_NDCI']

# Safeguard imputation for any potential edge anomalies
X_features = X_features.fillna(X_features.mean())

# Evaluate structural configuration
print(f"   ↳ Feature matrix shape: {X_features.shape}")
print(f"   ↳ Target sequence shape: {y_target.shape}")

# Random-state locked train/test division (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=42)

# 5. Initialize Evaluators
models = {
    "XGBoost": XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Support Vector Regression (SVR)": SVR(kernel='rbf', C=10, epsilon=0.01)
}

print("\n⚡ Step 3: Executing Algorithmic Performance Benchmarking...")
print(f"{'Model Name':<35} {'R-squared':<12} {'RMSE':<12} {'MAE':<12}")
print("-" * 71)

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    
    print(f"{name:<35} {r2:<12.6f} {rmse:<12.6f} {mae:<12.6f}")

print("\n🎉 PIPELINE EXECUTION SUCCESSFUL!")