import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

print("🚀 Initiating SHAP (SHapley Additive exPlanations) Analysis...")

# 1. Load the Enriched Dataset
df = pd.read_csv('data/processed/master_hab_dataset_chemical.csv')

# Isolate predictive features and target
cols_to_drop = ['Lake_Name', 'Year', 'Month', 'Chl_A_Proxy', 'NDWI_Shallow_Index', 'Turbidity_NDTI']
X = df.drop(columns=[c for c in cols_to_drop if c in df.columns or 'Target' in c])
y = df['Target_NDCI']

X = X.fillna(X.mean())

# 2. Train the Supreme XGBoost Model
# We train on the full dataset here to get the complete global interpretability of all data points
model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X, y)

# 3. Compute SHAP Values
print("⏳ Cracking the Black Box: Calculating Shapley values for all data points...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 4. Generate the SHAP Summary Plot
# This creates the classic beeswarm plot used in top-tier academic journals
plt.figure(figsize=(10, 8))
plt.title('SHAP Summary: Drivers of Cyanobacterial Bloom Intensity', fontsize=14, fontweight='bold', pad=20)

shap.summary_plot(shap_values, X, show=False, plot_size=(10, 8))

# Save the plot for your thesis
os.makedirs('data/processed/presentation_graphs', exist_ok=True)
shap_path = 'data/processed/presentation_graphs/13_SHAP_Summary_Plot.png'
plt.tight_layout()
plt.savefig(shap_path, dpi=300, bbox_inches='tight')

print(f"\n✅ SHAP Analysis Complete! Graph saved to: {shap_path}")