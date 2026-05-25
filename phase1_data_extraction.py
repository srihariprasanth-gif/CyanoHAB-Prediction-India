import ee
import pandas as pd
import numpy as np
import os
import time

# ==========================================
# 1. AUTHENTICATION & INITIALIZATION
# ==========================================
print("🚀 PHASE 1: Initializing Regional Landsat-8 Extraction...")
try:
    ee.Initialize(project='euphoric-anchor-496506-p4')
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='euphoric-anchor-496506-p4')
print("✅ Connected to Google Earth Engine.\n")

# ==========================================
# 2. THE 30-DAM REGIONAL ARRAY
# ==========================================
phase2_regional_dams = {
    # Tamil Nadu
    "Mettur_Dam": [77.80, 11.80], "Bhavanisagar_Dam": [77.12, 11.47],
    "Amaravathi_Dam": [77.26, 10.43], "Vaigai_Dam": [77.58, 10.05],
    "Pechiparai_Dam": [77.31, 8.43], "Papanasam_Dam": [77.30, 8.70],
    "Sholayar_Dam": [76.75, 10.30], "Aliyar_Dam": [76.96, 10.48],
    "Krishnagiri_Dam": [78.20, 12.48], "Sathanur_Dam": [78.85, 12.18],
    "Manimuthar_Dam": [77.41, 8.61],
    # Kerala
    "Idukki_Dam": [76.97, 9.84], "Mullaperiyar_Dam": [77.14, 9.53],
    "Malampuzha_Dam": [76.68, 10.83], "Banasura_Sagar": [75.95, 11.66],
    "Parambikulam_Dam": [76.78, 10.39], "Neyyar_Dam": [77.15, 8.53],
    "Peechi_Dam": [76.32, 10.53], "Mattupetty_Dam": [77.12, 10.10],
    # Karnataka
    "KRS_Dam": [76.57, 12.42], "Kabini_Dam": [76.35, 11.97],
    "Hemavathi_Dam": [76.03, 12.77], "Tungabhadra_Dam": [76.33, 15.24],
    "Almatti_Dam": [75.89, 16.33], "Supa_Dam": [74.52, 15.26],
    "Bhadra_Dam": [75.62, 13.70],
    # Andhra/Telangana
    "Nagarjuna_Sagar": [79.31, 16.57], "Srisailam_Dam": [78.87, 16.08],
    "Somasila_Dam": [79.30, 14.48], "Kandaleru_Dam": [79.60, 14.33]
}

# ==========================================
# 3. EXTRACTION FUNCTION
# ==========================================
def extract_l8_data(lake_name, coords, year, month):
    point_geom = ee.Geometry.Point(coords)
    water_zone = point_geom.buffer(1000) # Increased to 1km for Mega-Dams
    
    start_date = ee.Date.fromYMD(year, month, 1)
    end_date = start_date.advance(1, 'month')
    lag_start = start_date.advance(-1, 'month')
    lag_end = start_date

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
        curr_l8 = l8_col.filterDate(start_date, end_date).median()
        nir = get_val(curr_l8, 'SR_B5', 30)
        red = get_val(curr_l8, 'SR_B4', 30)
        target_ndci = (nir - red) / (nir + red) if nir is not None and red is not None else np.nan
        
        raw_temp = get_val(curr_l8, 'ST_B10', 30)
        surf_temp_c = (raw_temp * 0.00341802 + 149.0 - 273.15) if not np.isnan(raw_temp) else np.nan

        curr_era5 = era5_col.filterDate(start_date, end_date).first()
        curr_rain = chirps_col.filterDate(start_date, end_date).select('precipitation').sum()
        
        air_temp = get_val(curr_era5, 'temperature_2m', 10000)
        air_temp_c = (air_temp - 273.15) if not np.isnan(air_temp) else np.nan
        u_wind = get_val(curr_era5, 'u_component_of_wind_10m', 10000)
        v_wind = get_val(curr_era5, 'v_component_of_wind_10m', 10000)
        wind_speed = np.sqrt(u_wind**2 + v_wind**2) if not np.isnan(u_wind) else np.nan
        rain_mm = get_val(curr_rain, 'precipitation', 5000)

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

print(f"📡 Scanning 30 Mega-Dams across South India...")

for lake_name, coords in phase2_regional_dams.items():
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
    file_path = 'data/processed/phase2_regional_L8.csv'
    df.to_csv(file_path, index=False)
    print(f"\n🎉 PHASE 1 COMPLETE! Total Rows: {len(df)} | Saved to: {file_path}")
else:
    print("\n❌ Error: No valid data could be compiled.")