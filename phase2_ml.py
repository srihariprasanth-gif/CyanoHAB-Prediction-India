import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
# ... (rest of your imports stay the same)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ==========================================
# 1. LOAD & CLEAN DATASET
# ==========================================
print("🔬 Loading Master Dataset for Performance Upgrades...")
df = pd.read_csv('data/processed/master_hab_dataset.csv')

# Drop the identical/entangled optical bands and missing variables
cols_to_drop = [
    'Solar_Radiation', 'Evaporation', 'Runoff', 
    'Chl_A_Proxy', 'Lake_Name', 
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

# SVR requires feature scaling (Z-score normalization) to converge properly
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

print("\n🏋️ Training models and calculating metrics...")
for name, model in models.items():
    # Use scaled data for SVR, unscaled for tree-based models
    if name == "Support Vector Regression (SVR)":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
    # Calculate performance metrics
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    
    results.append({
        "Model": name,
        "R-squared": r2,
        "RMSE": rmse,
        "MAE": mae
    })

# Convert results into a neat dataframe for comparison
results_df = pd.DataFrame(results)
print("\n📊 MASTER MODEL PERFORMANCE COMPARISON TABLE:")
print(results_df.to_string(index=False))

# ==========================================
# ==========================================
# 4. VISUALIZATION OF ACCURACY METRICS
# ==========================================
sns.set_theme(style="whitegrid")
fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot R-squared (higher is better) as a bar chart
color = 'teal'
ax1.set_ylabel('R-squared (Higher is Better)', color=color, fontweight='bold')
# Updated to fix the seaborn warning by adding hue='Model' and legend=False
sns.barplot(data=results_df, x='Model', y='R-squared', hue='Model', ax=ax1, palette='mako', alpha=0.8, legend=False)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 1.0)

# Create a secondary axis to plot RMSE error margin (lower is better) as a line plot
ax2 = ax1.twinx()
color = 'crimson'
ax2.set_ylabel('RMSE Error (Lower is Better)', color=color, fontweight='bold')
sns.lineplot(data=results_df, x='Model', y='RMSE', ax=ax2, color=color, marker="o", linewidth=3, markersize=10)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Cyanobacterial Model Comparison: Telemetry Benchmarking', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save image
os.makedirs('data/processed/presentation_graphs', exist_ok=True)
plt.savefig('data/processed/presentation_graphs/8_Model_Comparison_Benchmark.png', dpi=300)
print("\n✅ Multi-model benchmark chart saved to: data/processed/presentation_graphs/8_Model_Comparison_Benchmark.png")