import ee
import pandas as pd

# 1. Initialize Earth Engine
print("🚀 Initializing Phase 1: Tamil Nadu Catchment Analysis...")
try:
    ee.Initialize(project='euphoric-anchor-496506-p4')
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='euphoric-anchor-496506-p4')

# 2. Define Phase 1 Dams (Tamil Nadu - First 6)
tn_dams_phase1 = {
    "Mettur_Dam": [77.80, 11.80],
    "Bhavanisagar_Dam": [77.12, 11.47],
    "Amaravathi_Dam": [77.26, 10.43],
    "Vaigai_Dam": [77.58, 10.05],
    "Pechiparai_Dam": [77.31, 8.43],
    "Papanasam_Dam": [77.30, 8.70]
}

# 3. Load the High-Resolution ESA WorldCover Dataset (2021 Baseline)
# Class 40 strictly represents "Cropland"
worldcover = ee.ImageCollection("ESA/WorldCover/v200").first()
cropland_mask = worldcover.eq(40)

print("⏳ Pinging Earth Engine to calculate 10km agricultural buffers...")
results = []

# 4. Iterate through Phase 1 Dams
for lake, coords in tn_dams_phase1.items():
    # Create a 10km (10,000 meter) catchment zone around the dam's center
    catchment_zone = ee.Geometry.Point(coords).buffer(10000)
    
    # Calculate the physical area of the cropland pixels
    cropland_area_img = cropland_mask.multiply(ee.Image.pixelArea())
    
    # Sum the area within our specific 10km zone
    area_dict = cropland_area_img.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=catchment_zone,
        scale=10,  # 10-meter native resolution
        maxPixels=1e9
    ).getInfo()
    
    # Extract square meters (The ESA band is named 'Map') and convert to Hectares
    sq_meters = area_dict.get('Map', 0)
    hectares = sq_meters / 10000
    
    print(f"   ↳ {lake}: {hectares:.2f} Hectares of cropland detected.")
    results.append({'Lake_Name': lake, 'Cropland_Area_Ha': hectares})

# 5. Save the Phase 1 Baseline
df = pd.DataFrame(results)
df.to_csv('data/processed/phase1_tn_cropland.csv', index=False)
print("\n✅ Phase 1 Catchment Extraction Complete! Saved as 'phase1_tn_cropland.csv'")