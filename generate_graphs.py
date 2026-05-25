import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Create a folder for the presentation images
os.makedirs('data/processed/presentation_graphs', exist_ok=True)

# 2. Load the Phase 2 Master Dataset
print("📊 Loading Phase 2 Master Dataset...")
df = pd.read_csv('data/processed/master_hab_dataset.csv')

# Drop the NaN rows for plotting purposes
df_clean = df.dropna(subset=['Target_NDCI', 'Dynamic_Soil_Moisture', 'Air_Temp_C'])

# Set global visual style for academic presentation
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# ==========================================
# GRAPH 1: The Primary Driver (Soil Moisture vs Algae)
# ==========================================
print("📈 Generating Soil Moisture Scatter Plot...")
plt.figure(figsize=(10, 6))
sns.regplot(data=df_clean, x='Dynamic_Soil_Moisture', y='Target_NDCI', 
            scatter_kws={'alpha':0.3, 'color':'teal'}, line_kws={'color':'darkorange', 'linewidth':3})
plt.title('Impact of Agricultural Runoff Proxy on Cyanobacteria Density', fontsize=14, fontweight='bold')
plt.xlabel('Dynamic Soil Moisture (Volumetric Water Content)')
plt.ylabel('Target Algal Biomass (NDCI)')
plt.tight_layout()
plt.savefig('data/processed/presentation_graphs/1_Soil_Moisture_Driver.png', dpi=300)
plt.close()

# ==========================================
# GRAPH 2: The Regional Limiter (Soil pH Boxplot)
# ==========================================
print("📉 Generating Soil pH Boxplot...")
# Create categorical bins for pH to make a clean boxplot
df_clean['pH_Category'] = pd.cut(df_clean['Static_Soil_pH'], bins=[0, 6.5, 7.5, 9.0], labels=['Acidic (<6.5)', 'Neutral (6.5-7.5)', 'Alkaline (>7.5)'])
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_clean, x='pH_Category', y='Target_NDCI', palette='coolwarm')
plt.title('Regional Suppression of Blooms by Alkaline Geology', fontsize=14, fontweight='bold')
plt.xlabel('Static Soil pH')
plt.ylabel('Target Algal Biomass (NDCI)')
plt.tight_layout()
plt.savefig('data/processed/presentation_graphs/2_Soil_pH_Limiter.png', dpi=300)
plt.close()

# ==========================================
# GRAPH 3: Seasonal Timeline (Month vs Algae)
# ==========================================
print("📅 Generating Seasonal Boxplot...")
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_clean, x='Month', y='Target_NDCI', palette='viridis')
plt.title('Seasonal Distribution of Cyanobacterial Blooms (2016-2026)', fontsize=14, fontweight='bold')
plt.xlabel('Month of the Year')
plt.ylabel('Target Algal Biomass (NDCI)')
plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.tight_layout()
plt.savefig('data/processed/presentation_graphs/3_Seasonal_Trend.png', dpi=300)
plt.close()

# ==========================================
# GRAPH 4: Temperature Impact
# ==========================================
print("🌡️ Generating Air Temperature Scatter Plot...")
plt.figure(figsize=(10, 6))
sns.regplot(data=df_clean, x='Air_Temp_C', y='Target_NDCI', 
            scatter_kws={'alpha':0.3, 'color':'crimson'}, line_kws={'color':'black', 'linewidth':3})
plt.title('Impact of Ambient Air Temperature on Bloom Formation', fontsize=14, fontweight='bold')
plt.xlabel('Air Temperature (°C)')
plt.ylabel('Target Algal Biomass (NDCI)')
plt.tight_layout()
plt.savefig('data/processed/presentation_graphs/4_Temperature_Impact.png', dpi=300)
plt.close()

print("\n✅ All presentation graphs successfully generated in: data/processed/presentation_graphs/")