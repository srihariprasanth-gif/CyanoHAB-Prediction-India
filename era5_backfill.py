import ee
import pandas as pd
import numpy as np

# 1. Initialize Earth Engine
print("🚀 Initializing ERA5 Data Rescue Protocol...")
try:
    ee.Initialize(project='euphoric-anchor-496506-p4')
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='euphoric-anchor-496506-p4')

# 2. Load the broken dataset
df = pd.read_csv('data/processed/master_hab_dataset.csv')
era5_col = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")

# Dam coordinates dictionary
phase2_regional_dams = {
    "Mettur_Dam": [77.80, 11.80], "Bhavanisagar_Dam": [77.12, 11.47],
    "Amaravathi_Dam": [77.26, 10.43], "Vaigai_Dam": [77.58, 10.05],
    "Pechiparai_Dam": [77.31, 8.43], "Papanasam_Dam": [77.30, 8.70],
    "Sholayar_Dam": [76.75, 10.30], "Aliyar_Dam": [76.96, 10.48],
    "Krishnagiri_Dam": [78.20, 12.48], "Sathanur_Dam": [78.85, 12.18],
    "Manimuthar_Dam": [77.41, 8.61], "Idukki_Dam": [76.97, 9.84], 
    "Mullaperiyar_Dam": [77.14, 9.53], "Malampuzha_Dam": [76.68, 10.83], 
    "Banasura_Sagar": [75.95, 11.66], "Parambikulam_Dam": [76.78, 10.39], 
    "Neyyar_Dam": [77.15, 8.53], "Peechi_Dam": [76.32, 10.53], 
    "Mattupetty_Dam": [77.12, 10.10], "KRS_Dam": [76.57, 12.42], 
    "Kabini_Dam": [76.35, 11.97], "Hemavathi_Dam": [76.03, 12.77], 
    "Tungabhadra_Dam": [76.33, 15.24], "Almatti_Dam": [75.89, 16.33], 
    "Supa_Dam": [74.52, 15.26], "Bhadra_Dam": [75.62, 13.70],
    "Nagarjuna_Sagar": [79.31, 16.57], "Srisailam_Dam": [78.87, 16.08],
    "Somasila_Dam": [79.30, 14.48], "Kandaleru_Dam": [79.60, 14.33]
}

def get_val(img, band, geom):
    try:
        val = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=10000).get(band).getInfo()
        return val if val is not None else np.nan
    except:
        return np.nan

print("⏳ Pinging Earth Engine API to recover missing Runoff, Evaporation, and Solar Radiation...")

# 3. Iterate and backfill
for index, row in df.iterrows():
    if pd.isna(row['Solar_Radiation']) or pd.isna(row['Runoff']):
        lake = row['Lake_Name']
        year = int(row['Year'])
        month = int(row['Month'])
        
        coords = phase2_regional_dams[lake]
        water_zone = ee.Geometry.Point(coords).buffer(1000)
        
        start_date = ee.Date.fromYMD(year, month, 1)
        end_date = start_date.advance(1, 'month')
        
        curr_era5 = era5_col.filterDate(start_date, end_date).first()
        
        # Extract using the correct '_sum' nomenclature
        solar_rad = get_val(curr_era5, 'surface_solar_radiation_downwards_sum', water_zone)
        raw_evap = get_val(curr_era5, 'total_evaporation_sum', water_zone)
        runoff = get_val(curr_era5, 'surface_runoff_sum', water_zone)
        
        df.at[index, 'Solar_Radiation'] = solar_rad
        df.at[index, 'Evaporation'] = abs(raw_evap) if not np.isnan(raw_evap) else np.nan
        df.at[index, 'Runoff'] = runoff
        
        print(f"   ↳ Backfilled {lake} | {year}-{month:02d}")

# 4. Save the repaired dataset
df.to_csv('data/processed/master_hab_dataset_repaired.csv', index=False)
print("\n✅ DATA RESCUE COMPLETE! Saved as 'master_hab_dataset_repaired.csv'")