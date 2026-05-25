import os
import pandas as pd
import numpy as np

print("🚀 Initiating Export Coefficient Model (ECM) Pipeline...")

# 1. Load and Consolidate the 4 Hectare Baseline Files using your exact terminal filenames
try:
    df1 = pd.read_csv('phase1_tn_cropland.csv') # Matches your first terminal run output
    df2 = pd.read_csv('phase2_batch_cropland.csv')
    df3 = pd.read_csv('phase3_batch_cropland.csv')
    df4 = pd.read_csv('remaining_batch_cropland.csv')
    
    spatial_baseline = pd.concat([df1, df2, df3, df4], ignore_index=True)
    print(f"   ↳ Successfully consolidated spatial baselines for all {len(spatial_baseline)} reservoirs.")
except Exception as e:
    print(f"❌ Error loading baseline CSVs: {e}")
    print("💡 Please verify that 'phase2_catchment.csv' exists in your root folder.")
    exit()

# 2. Define Regional Chemical Export Coefficients (kg/Ha/year)
INTENSIVE_N = 12.5
INTENSIVE_P = 1.8

PROTECTED_N = 2.5
PROTECTED_P = 0.25

# Dictionary mapping specific forested/mountain catchments to protected coefficients
protected_catchments = [
    "Pechiparai_Dam", "Papanasam_Dam", "Sholayar_Dam", "Idukki_Dam", 
    "Mullaperiyar_Dam", "Parambikulam_Dam", "Neyyar_Dam", "Peechi_Dam", 
    "Mattupetty_Dam", "Supa_Dam"
]

# Apply the coefficients to each dam based on its ecological zoning
spatial_baseline['E_N'] = np.where(spatial_baseline['Lake_Name'].isin(protected_catchments), PROTECTED_N, INTENSIVE_N)
spatial_baseline['E_P'] = np.where(spatial_baseline['Lake_Name'].isin(protected_catchments), PROTECTED_P, INTENSIVE_P)

# 3. Process the Machine Learning Datasets
target_files = [
    'data/processed/master_hab_dataset_repaired.csv',
    'data/processed/master_hab_dataset_v3.csv'
]

for file_path in target_files:
    if os.path.exists(file_path):
        print(f"\n⚡ Injecting dynamic chemical parameters into: {file_path}")
        df_ml = pd.read_csv(file_path)
        
        # Merge the spatial cropland capacity into the time-series matrix
        df_merged = df_ml.merge(spatial_baseline, on='Lake_Name', how='left')
        
        # ECM Core Equation: Load = Area (Ha) * Coefficient (kg/Ha/yr / 12 months) * Hydrological Transport Runoff
        df_merged['Calculated_Nitrogen_Load_kg'] = (
            df_merged['Cropland_Area_Ha'] * (df_merged['E_N'] / 12.0) * df_merged['Runoff']
        )
        df_merged['Calculated_Phosphorus_Load_kg'] = (
            df_merged['Cropland_Area_Ha'] * (df_merged['E_P'] / 12.0) * df_merged['Runoff']
        )
        
        # Clean up temporary coefficient tracking columns
        df_merged = df_merged.drop(columns=['E_N', 'E_P'])
        
        # Overwrite with the newly enriched dataset
        df_merged.to_csv(file_path, index=False)
        print(f"   ✅ Successfully added 'Calculated_Nitrogen_Load_kg' and 'Calculated_Phosphorus_Load_kg'")
    else:
        print(f"⚠️ Target file not found, skipping: {file_path}")

print("\n🎉 CHEMICAL FEATURE ENGINEERING COMPLETE! Ready for model training.")