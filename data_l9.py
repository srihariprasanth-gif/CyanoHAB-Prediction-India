import ee
import pandas as pd
import numpy as np
import os
import time

# ==========================================
# 1. AUTHENTICATION & INITIALIZATION
# ==========================================
print("🚀 PHASE 3 (V3.0): Initializing Landsat 8 + 9 Fusion Extraction...")
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
    "Mettur_Dam": [77.80, 11.80], "Bhavanisagar_Dam": [77.12, 11.47],
    "Amaravathi_Dam": [77.26, 10.43], "Vaigai_Dam": [77.58, 10.05],
    "Pechiparai_Dam": [77.31, 8.43], "Papanasam_Dam": [77.30, 8.70],
    "Sholayar_Dam": [76.75, 10.30], "Aliyar_Dam": [76.96, 10.48],
    "Krishnagiri_Dam": [78.20, 12.48], "Sathanur_Dam": [78.85, 12.18],
    "Manimuthar_Dam": [77.41, 8.61],
    "Idukki_Dam": [76.97, 9.84], "Mullaperiyar_Dam": [77.14, 9.53],
    "Malampuzha_Dam": [76.68, 10.83], "Banasura_Sagar": [75.95, 11.66],
    "Parambikulam_Dam": [76.78, 10.39], "Neyyar_Dam": [77.15, 8.53],
    "Peechi_Dam": [76.32, 10.53], "Mattupetty_Dam": [77.12, 10.10],
    "KRS_Dam": [76.57, 12.42], "Kabini_Dam": [76.35, 11.97],
    "Hemavathi_Dam": [76.03, 12.77], "Tungabhadra_Dam": [76.33, 15.24],
    "Almatti_Dam": [75.89, 16.33], "Supa_Dam": [74.52, 15.26],
    "Bhadra_Dam": [75.62, 13.70],
    "Nagarjuna_Sagar": [79.31, 16.57], "Srisailam_Dam": [78.87, 16.08],
    "Somasila_Dam": [79.30, 14.48], "Kandaleru_Dam": [79.60, 14.33]
}

# ==========================================
# 3. EXTRACTION ENGINE (L8 + L9 FUSION)
# ==========================================
soil_clay_img = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select('b0')
soil_ph_img = ee.Image("OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02").select('b0')

def get_static_soil(water_zone):
    try:
        clay = soil_clay_img.reduceRegion(reducer=ee.Reducer.mean(), geometry=water_zone, scale=250).get('b0').getInfo()
        ph = soil_ph_img.reduceRegion(reducer=ee.Reducer.mean(), geometry=water_zone, scale=250).get('b0').getInfo()
        return clay if clay is not None else np.nan, (ph / 10.0) if ph is not None else np.nan
    except:
        return np.nan, np.nan

def extract_v3_fusion_data(lake_name, coords, year, month, static_clay, static_ph):
    point_geom = ee.Geometry.Point(coords)
    water_zone = point_geom.buffer(1000) 
    
    start_date = ee.Date.fromYMD(year, month, 1)
    end_date = start_date.advance(1, 'month')

    # 🛰️ THE TWIN SATELLITE MERGE
    l8_col = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(water_zone).filter(ee.Filter.lt('CLOUD_COVER', 20))
    l9_col = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(water_zone).filter(ee.Filter.lt('CLOUD_COVER', 20))
    merged_optical_col = l8_col.merge(l9_col)

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
        # Pulling the highest quality median pixel from BOTH satellites
        curr_optical = merged_optical_col.filterDate(start_date, end_date).median()
        green = get_val(curr_optical, 'SR_B3', 30)
        red = get_val(curr_optical, 'SR_B4', 30)
        nir = get_val(curr_optical, 'SR_B5', 30)
        raw_temp = get_val(curr_optical, 'ST_B10', 30)

        target_ndci, chl_a_proxy, ndti_turbidity, ndwi_water, surf_temp_c = np.nan, np.nan, np.nan, np.nan, np.nan
        
        if nir is not None and red is not None:
            target_ndci = (nir - red) / (nir + red)
            chl_a_proxy = (nir / red) if red != 0 else np.nan
        if red is not None and green is not None:
            ndti_turbidity = (red - green) / (red + green)
        if green is not None and nir is not None:
            ndwi_water = (green - nir) / (green + nir)
        if raw_temp is not None and not np.isnan(raw_temp):
            surf_temp_c = (raw_temp * 0.00341802 + 149.0 - 273.15)

        curr_era5 = era5_col.filterDate(start_date, end_date).first()
        curr_rain = chirps_col.filterDate(start_date, end_date).select('precipitation').sum()
        
        air_temp = get_val(curr_era5, 'temperature_2m', 10000)
        dew_temp = get_val(curr_era5, 'dewpoint_temperature_2m', 10000)
        u_wind = get_val(curr_era5, 'u_component_of_wind_10m', 10000)
        v_wind = get_val(curr_era5, 'v_component_of_wind_10m', 10000)
        
        solar_rad = get_val(curr_era5, 'surface_solar_radiation_downwards', 10000)
        evap = get_val(curr_era5, 'total_evaporation', 10000)
        runoff = get_val(curr_era5, 'surface_runoff', 10000) 
        soil_moisture = get_val(curr_era5, 'volumetric_soil_water_layer_1', 10000)

        air_temp_c = (air_temp - 273.15) if not np.isnan(air_temp) else np.nan
        dewpoint_c = (dew_temp - 273.15) if not np.isnan(dew_temp) else np.nan
        wind_speed = np.sqrt(u_wind**2 + v_wind**2) if not np.isnan(u_wind) else np.nan
        rain_mm = get_val(curr_rain, 'precipitation', 5000)

        if np.isnan(target_ndci):
            return None

        return {
            'Lake_Name': lake_name, 'Year': year, 'Month': month,
            'Target_NDCI': target_ndci, 'Chl_A_Proxy': chl_a_proxy, 'Turbidity_NDTI': ndti_turbidity,
            'NDWI_Shallow_Index': ndwi_water, 'Surface_Temp_C': surf_temp_c, 
            'Air_Temp_C': air_temp_c, 'Dewpoint_Temp_C': dewpoint_c,
            'Rainfall_mm': rain_mm, 'Wind_Speed': wind_speed, 'Solar_Radiation': solar_rad,
            'Evaporation': evap, 'Runoff': runoff, 'Dynamic_Soil_Moisture': soil_moisture,
            'Static_Soil_Clay_Pct': static_clay, 'Static_Soil_pH': static_ph
        }
    except Exception as e:
        return None

# ==========================================
# 4. RUN CHECKPOINT ARCHITECTURE
# ==========================================
target_years = list(range(2016, 2027))
save_dir = 'data/processed/individual_dams_v3' # 🛡️ New directory to protect your old data
os.makedirs(save_dir, exist_ok=True)

print(f"⚡ Running L8 + L9 Fusion API Extraction...")

for lake_name, coords in phase2_regional_dams.items():
    file_path = os.path.join(save_dir, f'{lake_name}_v3.csv')
    
    if os.path.exists(file_path):
        print(f"\n⏭️ Skipping {lake_name} - Already secured.")
        continue

    print(f"\n🌊 Starting {lake_name}...")
    dam_data = [] 
    
    point_geom = ee.Geometry.Point(coords)
    water_zone = point_geom.buffer(1000)
    static_clay, static_ph = get_static_soil(water_zone)

    for yr in target_years:
        for mn in range(1, 13): 
            if yr == 2026 and mn > 5: break
            
            success = False
            attempts = 0
            while not success and attempts < 3:
                try:
                    row = extract_v3_fusion_data(lake_name, coords, yr, mn, static_clay, static_ph)
                    if row:
                        dam_data.append(row)
                        print(f"   ↳ {yr}-{mn:02d}: Secured.")
                    success = True 
                except Exception as e:
                    attempts += 1
                    error_msg = str(e).lower()
                    if "too many" in error_msg or "quota" in error_msg or "429" in error_msg:
                        print(f"   ⚠️ Rate Limit Hit! Cooling down 5s...")
                        time.sleep(5)
                    else:
                        print(f"   ❌ Skipped {yr}-{mn:02d} | Error: {e}") 
                        break 

    df = pd.DataFrame(dam_data)
    if not df.empty:
        df.to_csv(file_path, index=False)
        print(f"   💾 CHECKPOINT: {lake_name} permanently saved to {file_path}")
    else:
        print(f"   ❌ ERROR: No valid data could be compiled for {lake_name}.")

print("\n🎉 ALL DAMS PROCESSED AND SECURED INDIVIDUALLY!")