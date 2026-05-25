import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Ensure the output directory exists
os.makedirs('data/processed/presentation_graphs', exist_ok=True)

# 2. Load the Phase 2 Master Dataset
print("📊 Loading Master Dataset for remaining parameters...")
df = pd.read_csv('data/processed/master_hab_dataset.csv')

# Drop NaN rows for the specific columns we are plotting
df_clean = df.dropna(subset=['Target_NDCI', 'Rainfall_mm', 'Wind_Speed', 'Dewpoint_Temp_C'])

# Set global visual style for academic presentation
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# ==========================================
# GRAPH 5: Rainfall Impact (Precipitation vs Algae)
# ==========================================
print("🌧️ Generating Rainfall Scatter Plot...")
plt.figure(figsize=(10, 6))
sns.regplot(data=df_clean, x='Rainfall_mm', y='Target_NDCI', 
            scatter_kws={'alpha':0.3, 'color':'royalblue'}, line_kws={'color':'black', 'linewidth':3})
plt.title('Impact of Total Monthly Rainfall on Cyanobacterial Density', fontsize=14, fontweight='bold')
plt.xlabel('Rainfall (mm)')
plt.ylabel('Target Algal Biomass (NDCI)')
plt.tight_layout()
plt.savefig('data/processed/presentation_graphs/5_Rainfall_Impact.png', dpi=300)
plt.close()

# ==========================================
# GRAPH 6: Wind Speed Impact (Water Column Mixing)
# ==========================================
print("💨 Generating Wind Speed Scatter Plot...")
plt.figure(figsize=(10, 6))
sns.regplot(data=df_clean, x='Wind_Speed', y='Target_NDCI', 
            scatter_kws={'alpha':0.3, 'color':'purple'}, line_kws={'color':'darkorange', 'linewidth':3})
plt.title('Impact of Wind Speed on Cyanobacterial Stratification', fontsize=14, fontweight='bold')
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('Target Algal Biomass (NDCI)')
plt.tight_layout()
plt.savefig('data/processed/presentation_graphs/6_Wind_Speed_Impact.png', dpi=300)
plt.close()

# ==========================================
# GRAPH 7: Dewpoint Temperature Impact (Humidity Proxy)
# ==========================================
print("🌡️ Generating Dewpoint Temperature Scatter Plot...")
plt.figure(figsize=(10, 6))
sns.regplot(data=df_clean, x='Dewpoint_Temp_C', y='Target_NDCI', 
            scatter_kws={'alpha':0.3, 'color':'forestgreen'}, line_kws={'color':'crimson', 'linewidth':3})
plt.title('Impact of Dewpoint Temperature (Humidity Proxy) on Bloom Conditions', fontsize=14, fontweight='bold')
plt.xlabel('Dewpoint Temperature (°C)')
plt.ylabel('Target Algal Biomass (NDCI)')
plt.tight_layout()
plt.savefig('data/processed/presentation_graphs/7_Dewpoint_Impact.png', dpi=300)
plt.close()

print("\n✅ Remaining parameter graphs successfully generated in: data/processed/presentation_graphs/")