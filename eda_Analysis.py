import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. LOAD THE MASTER DATASET
# ==========================================
file_path = 'data/processed/master_hab_dataset.csv'
print(f"🔬 Loading God Dataset from {file_path}...")
df = pd.read_csv(file_path)

# ==========================================
# 2. MISSING DATA DIAGNOSTICS
# ==========================================
print("\n📊 DATASET SHAPE:", df.shape)
print("\n⚠️ MISSING VALUES PER FEATURE:")
missing_data = df.isnull().sum()
print(missing_data[missing_data > 0])

# ==========================================
# 3. BIOLOGICAL CORRELATION MATRIX
# ==========================================
print("\n🧬 Generating Correlation Matrix...")

# Isolate only the numerical columns for the math
numeric_df = df.select_dtypes(include=[np.number])

# Calculate the correlation (Pearson)
corr_matrix = numeric_df.corr()

# Set up the visualization
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, 
            annot=True,          # Shows the actual correlation numbers
            fmt=".2f",           # Rounds to 2 decimal places
            cmap='coolwarm',     # Red = Positive Correlation, Blue = Negative
            center=0, 
            linewidths=0.5, 
            cbar_kws={"shrink": .8})

plt.title('Cyanobacterial HAB Drivers: Feature Correlation Matrix (30 Dams)', fontsize=16)
plt.tight_layout()

# Save the plot
save_path = 'data/processed/correlation_heatmap_v2.png'
plt.savefig(save_path, dpi=300)
print(f"✅ High-resolution heatmap saved to: {save_path}")