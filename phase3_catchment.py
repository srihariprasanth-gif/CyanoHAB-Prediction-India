import ee
import pandas as pd

# 1. Initialize Earth Engine
print("🚀 Initializing Phase 3 Catchment Analysis...")
try:
    ee.Initialize(project='euphoric-anchor-496506-p4')
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='euphoric-anchor-496506-p4')

# 2. Define Phase 3 Dams (Next 6 in the array)
dams_phase3 = {
    "Mullaperiyar_Dam": [77.14, 9.53], 
    "Malampuzha_Dam": [76.68, 10.83], 
    "Banasura_Sagar": [75.95, 11.66], 
    "Parambikulam_Dam": [76.78, 10.39], 
    "Neyyar_Dam": [77.15, 8.53], 
    "Peechi_Dam": [76.32, 10.53]
}

# 3. Load the High-Resolution ESA WorldCover Dataset
worldcover = ee.ImageCollection("ESA/WorldCover/v200").first()
cropland_mask = worldcover.eq(40) # Class 40 = Cropland

print("⏳ Pinging Earth Engine to calculate 10km agricultural buffers...")
results = []

# 4. Iterate through Phase 3 Dams
for lake, coords in dams_phase3.items():
    catchment_zone = ee.Geometry.Point(coords).buffer(10000)
    
    cropland_area_img = cropland_mask.multiply(ee.Image.pixelArea())
    
    area_dict = cropland_area_img.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=catchment_zone,
        scale=10,
        maxPixels=1e9
    ).getInfo()
    
    sq_meters = area_dict.get('Map', 0)
    hectares = sq_meters / 10000
    
    print(f"   ↳ {lake}: {hectares:.2f} Hectares of cropland detected.")
    results.append({'Lake_Name': lake, 'Cropland_Area_Ha': hectares})

# 5. Save the Phase 3 Baseline
df = pd.DataFrame(results)
df.to_csv('data/processed/phase3_batch_cropland.csv', index=False)
print("\n✅ Phase 3 Catchment Extraction Complete! Saved as 'phase3_batch_cropland.csv'")