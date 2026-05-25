import ee
import pandas as pd

# 1. Initialize Earth Engine
print("🚀 Initializing Phase 2 Catchment Analysis...")
try:
    ee.Initialize(project='euphoric-anchor-496506-p4')
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='euphoric-anchor-496506-p4')

# 2. Define Phase 2 Dams (Next 6 in the array)
dams_phase2 = {
    "Sholayar_Dam": [76.75, 10.30], 
    "Aliyar_Dam": [76.96, 10.48],
    "Krishnagiri_Dam": [78.20, 12.48], 
    "Sathanur_Dam": [78.85, 12.18],
    "Manimuthar_Dam": [77.41, 8.61], 
    "Idukki_Dam": [76.97, 9.84]
}

# 3. Load the High-Resolution ESA WorldCover Dataset
worldcover = ee.ImageCollection("ESA/WorldCover/v200").first()
cropland_mask = worldcover.eq(40) # Class 40 = Cropland

print("⏳ Pinging Earth Engine to calculate 10km agricultural buffers...")
results = []

# 4. Iterate through Phase 2 Dams
for lake, coords in dams_phase2.items():
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

# 5. Save the Phase 2 Baseline
df = pd.DataFrame(results)
df.to_csv('data/processed/phase2_batch_cropland.csv', index=False)
print("\n✅ Phase 2 Catchment Extraction Complete! Saved as 'phase2_batch_cropland.csv'")