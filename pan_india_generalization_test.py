import ee
import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ==========================================
# 1. INITIALIZE EARTH ENGINE
# ==========================================
print("🚀 Initiating Pan-India Generalization Array (2021-2025)...")
try:
    ee.Initialize(project='euphoric-anchor-496506-p4')
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='euphoric-anchor-496506-p4')

# ==========================================
# 2. THE PAN-INDIA DAM DICTIONARY (15 DAMS)
# ==========================================
pan_india_dams = {
    # Northern Alpine/Sub-Himalayan
    "Bhakra_Dam": [76.432, 31.411], "Tehri_Dam": [78.480, 30.378], "Pong_Dam": [75.976, 31.968],
    # Western Arid/Semi-Arid
    "Sardar_Sarovar": [73.748, 21.831], "Ukai_Dam": [73.588, 21.246], "Jayakwadi_Dam": [75.316, 19.481],
    # Central/Gangetic Plains
    "Indira_Sagar": [76.471, 22.285], "Gandhi_Sagar": [75.561, 24.717], "Rihand_Dam": [83.002, 24.202],
    # Eastern Plateaus & Basins
    "Hirakud_Dam": [83.872, 21.526], "Maithon_Dam": [86.811, 23.784], "Panchet_Dam": [86.746, 23.693],
    # Deccan/Transition Zones
    "Koyna_Dam": [73.749, 17.401], "Ujjani_Dam": [75.118, 18.074], "Bisalpur_Dam": [75.454, 25.925]
}

# Protected list for ECM modeling (Mountain/Forested catchments)
protected_catchments = ["Tehri_Dam", "Pong_Dam", "Koyna_Dam"]

# ==========================================
# 3. EXTRACTION SETUP (GEE ASSETS)
# ==========================================
worldcover = ee.ImageCollection("ESA/WorldCover/v200").first()
cropland_mask = worldcover.eq(40)
era5_col = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
# Landsat 8 & 9 for Target NDCI
l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")

def get_val(img, band, geom, scale=10000):
    try:
        val = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=scale).get(band).getInfo()
        return val if val is not None else np.nan
    except:
        return np.nan

# ==========================================
# 4. DATA PIPELINE EXECUTION
# ==========================================
print("⏳ Extracting Spatial, Climate, and Optical Data (This may take a while)...")
results = []
years = [2021, 2022, 2023, 2024, 2025]
months = list(range(1, 13))

for lake, coords in pan_india_dams.items():
    print(f"\n📡 Processing: {lake}...")
    water_zone = ee.Geometry.Point(coords).buffer(1000)
    catchment_zone = ee.Geometry.Point(coords).buffer(10000)
    
    # 4A. Spatial Extraction (Cropland)
    cropland_area_img = cropland_mask.multiply(ee.Image.pixelArea())
    sq_meters = get_val(cropland_area_img, 'Map', catchment_zone, scale=10)
    hectares = sq_meters / 10000 if not np.isnan(sq_meters) else 0
    
    # Assign Export Coefficients
    e_n = 2.5 if lake in protected_catchments else 12.5
    e_p = 0.25 if lake in protected_catchments else 1.8

    # 4B. Temporal Extraction (Climate + NDCI)
    for year in years:
        for month in months:
            start_date = ee.Date.fromYMD(year, month, 1)
            end_date = start_date.advance(1, 'month')
            
            # Climate (ERA5)
            curr_era5 = era5_col.filterDate(start_date, end_date).first()
            if curr_era5 is None: continue
            
            temp = get_val(curr_era5, 'temperature_2m', water_zone) - 273.15
            runoff = get_val(curr_era5, 'surface_runoff_sum', water_zone)
            soil_m = get_val(curr_era5, 'volumetric_soil_water_layer_1', water_zone)
            
            # Landsat NDCI Proxy (Using simple band math for speed: (Red - Green) / (Red + Green) as a stand-in for bloom presence if no specific cyan band)
            # *Assuming standard SR bands: B4=Red, B3=Green for L8/9*
            l_col = l8.merge(l9).filterBounds(water_zone).filterDate(start_date, end_date).filter(ee.Filter.lt('CLOUD_COVER', 30))
            if l_col.size().getInfo() > 0:
                img = l_col.median()
                ndci = img.normalizedDifference(['SR_B4', 'SR_B3']).reduceRegion(reducer=ee.Reducer.mean(), geometry=water_zone, scale=30).get('nd').getInfo()
            else:
                ndci = np.nan # Will impute later
            
            # 4C. ECM Feature Engineering
            n_load = hectares * (e_n / 12.0) * (runoff if not np.isnan(runoff) else 0)
            p_load = hectares * (e_p / 12.0) * (runoff if not np.isnan(runoff) else 0)
            
            results.append({
                'Lake_Name': lake, 'Year': year, 'Month': month,
                'Surface_Temp': temp, 'Runoff': runoff, 'Dynamic_Soil_Moisture': soil_m,
                'Calculated_Nitrogen_Load_kg': n_load, 'Calculated_Phosphorus_Load_kg': p_load,
                'Target_NDCI': ndci
            })

# Save the new test dataset
df_test = pd.DataFrame(results)
# Drop completely empty rows (due to cloud cover) and fill minor gaps
df_test = df_test.dropna(subset=['Target_NDCI'])
df_test = df_test.fillna(df_test.mean(numeric_only=True))
df_test.to_csv('data/processed/pan_india_testing_dataset.csv', index=False)
print("\n✅ Pan-India Extraction Complete! Data saved.")

# ==========================================
# 5. GENERALIZATION TESTING (THE ULTIMATE TEST)
# ==========================================
print("\n🧠 Initiating Out-of-Distribution Generalization Test...")

# Load your finalized South Indian Training Dataset
df_train = pd.read_csv('data/processed/master_hab_dataset_chemical.csv')

# Align features (Ensure both datasets use the exact same columns)
features = ['Surface_Temp', 'Runoff', 'Dynamic_Soil_Moisture', 'Calculated_Nitrogen_Load_kg', 'Calculated_Phosphorus_Load_kg']

X_train = df_train[features].fillna(df_train[features].mean())
y_train = df_train['Target_NDCI']

X_test = df_test[features]
y_test = df_test['Target_NDCI']

# Train on South India
model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Predict on Pan-India
preds = model.predict(X_test)

r2 = r2_score(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print("==================================================")
print("🏆 PAN-INDIA GENERALIZATION PERFORMANCE:")
print(f"R-squared (Accuracy): {r2:.4f}")
print(f"RMSE (Error Margin):  {rmse:.4f}")
print("==================================================")
if r2 > 0.75:
    print("🌟 OUTSTANDING: Your model successfully generalized across different Indian ecological zones!")
else:
    print("⚠️ EXPECTED VARIANCE: The model experienced climate shock. We can discuss transfer learning next.")