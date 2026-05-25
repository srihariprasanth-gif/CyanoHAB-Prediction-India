import ee
import pandas as pd
import numpy as np
import datetime
import os
import time

# ==========================================
# 1. AUTHENTICATION & INITIALIZATION
# ==========================================
print("🚀 Initializing Track B: Landsat-8 Unified Pipeline...")
try:
    ee.Initialize(project='euphoric-anchor-496506-p4')
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='euphoric-anchor-496506-p4')
print("✅ Connected to Project: euphoric-anchor-496506-p4\n")

# ==========================================
# 2. SPATIAL GEOMETRY (The 11 Phase 1 Lakes)
# ==========================================
tn_phase1_lakes = {
    "Poondi_Reservoir": [79.86, 13.18],
    "Cholavaram_Lake": [80.14, 13.23],
    "Red_Hills_Lake": [80.16, 13.16],
    "Chembarambakkam_Lake": [80.05, 13.01],
    "Thervoy_Kandigai_Lake": [80.01, 13.36],
    "Veeranam_Lake": [79.55, 11.35],
    "Willingdon_Lake": [79.62, 11.48],
    "Pykara_Lake": [76.59, 11.45],
    "Porthimund_Lake": [76.57, 11.39],
    "Avalanche_Lake": [76.58, 11.31],
    "Emerald_Lake": [76.61, 11.32]
}

# ==========================================
# 3. LANDSAT-8 EXTRACTION FUNCTION
# ==========================================
def extract_l8_data(lake_name, coords, year, month):
    point_geom = ee.Geometry.Point(coords)
    water_zone = point_geom.buffer(500) 
    
    start_date = ee.Date.fromYMD(year, month, 1)
    end_date = start_date.advance(1, 'month')
    lag_start = start_date.advance(-1, 'month')
    lag_end = start_date

    # Landsat 8 Collection 2 Tier 1 Level 2 (Surface Reflectance + Surface Temp)
    l8_col = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(water_zone).filter(ee.Filter.lt('CLOUD_COVER', 20))
    era5_col = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
    chirps_col = ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD")

    def get_val(img, band, scale):
        if img is None: return np.nan
        try:
            val = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=water_zone, scale=scale, maxPixels=1e6).get(band).getInfo()
            return val if val is not None else np.nan
        except:
            return np.nan

    try:
        # --- CURRENT MONTH ---
        curr_l8 = l8_col.filterDate(start_date, end_date).median()
        
        # Calculate NDCI proxy (Red vs NIR)
        nir = get_val(curr_l8, 'SR_B5', 30)
        red = get_val(curr_l8, 'SR_B4', 30)
        target_ndci = (nir - red) / (nir + red) if nir is not None and red is not None else np.nan
        
        # Exact Thermal Surface Temperature from the same satellite
        raw_temp = get_val(curr_l8, 'ST_B10', 30)
        surf_temp_c = (raw_temp * 0.00341802 + 149.0 - 273.15) if not np.isnan(raw_temp) else np.nan

        # Wind & Rain
        curr_era5 = era5_col.filterDate(start_date, end_date).first()
        curr_rain = chirps_col.filterDate(start_date, end_date).select('precipitation').sum()
        
        air_temp = get_val(curr_era5, 'temperature_2m', 10000)
        air_temp_c = (air_temp - 273.15) if not np.isnan(air_temp) else np.nan
        u_wind = get_val(curr_era5, 'u_component_of_wind_10m', 10000)
        v_wind = get_val(curr_era5, 'v_component_of_wind_10m', 10000)
        wind_speed = np.sqrt(u_wind**2 + v_wind**2) if not np.isnan(u_wind) else np.nan
        rain_mm = get_val(curr_rain, 'precipitation', 5000)

        # --- LAGGED MONTH ---
        lag_l8 = l8_col.filterDate(lag_start, lag_end).median()
        lag_green = get_val(lag_l8, 'SR_B3', 30)
        lag_red = get_val(lag_l8, 'SR_B4', 30)
        lag_ndti = (lag_red - lag_green) / (lag_red + lag_green) if lag_red is not None and lag_green is not None else np.nan
        
        lag_raw_temp = get_val(lag_l8, 'ST_B10', 30)
        lag_surf_temp_c = (lag_raw_temp * 0.00341802 + 149.0 - 273.15) if not np.isnan(lag_raw_temp) else surf_temp_c

        lag_era5 = era5_col.filterDate(lag_start, lag_end).first()
        lag_air = get_val(lag_era5, 'temperature_2m', 10000)
        lag_air_c = (lag_air - 273.15) if not np.isnan(lag_air) else air_temp_c
        
        lag_u = get_val(lag_era5, 'u_component_of_wind_10m', 10000)
        lag_v = get_val(lag_era5, 'v_component_of_wind_10m', 10000)
        lag_wind = np.sqrt(lag_u**2 + lag_v**2) if not np.isnan(lag_u) else wind_speed

        if np.isnan(target_ndci):
            return None

        return {
            'Lake_Name': lake_name, 'Year': year, 'Month': month,
            'Target_NDCI': target_ndci, 'Air_Temp_C': air_temp_c, 'Surface_Temp_C': surf_temp_c, 
            'Rainfall_mm': rain_mm, 'Wind_Speed': wind_speed, 'Lagged_NDTI': lag_ndti, 
            'Lagged_Air_Temp_C': lag_air_c, 'Lagged_Surface_Temp_C': lag_surf_temp_c, 'Lagged_Wind_Speed': lag_wind
        }
    except Exception as e:
        return None

# ==========================================
# 4. COMPILER LOOP
# ==========================================
master_data = []
target_years = [2024, 2025, 2026]
start_time = time.time()

print(f"📡 Beginning Track B (Landsat-8) Extraction for 11 lakes...")

for lake_name, coords in tn_phase1_lakes.items():
    print(f"🌊 Scanning {lake_name}...")
    for yr in target_years:
        for mn in range(1, 13): 
            if yr == 2026 and mn > 5: break
            time.sleep(1.5) 
            row = extract_l8_data(lake_name, coords, yr, mn)
            if row:
                master_data.append(row)
                print(f"   ↳ {yr}-{mn:02d}: Data secured.")

# ==========================================
# 5. SAFE EXPORT
# ==========================================
os.makedirs('data/processed', exist_ok=True)
df = pd.DataFrame(master_data)

if not df.empty:
    file_path = 'data/processed/tn_phase1_lakes_L8.csv'
    df.to_csv(file_path, index=False)
    print(f"\n🎉 TRACK B COMPLETE! Total Rows: {len(df)} | Saved to: {file_path}")
else:
    print("\n❌ Error: No valid data could be compiled.")