import ee
import pandas as pd

# 1. Initialize Earth Engine
print("🚀 Initializing Final Phase: Remaining Catchment Analysis...")
try:
    ee.Initialize(project='euphoric-anchor-496506-p4')
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='euphoric-anchor-496506-p4')

# 2. Define Remaining Dams (Phases 4 and 5 Combined)
remaining_dams = {
    # Phase 4 Batch
    "Mattupetty_Dam": [77.12, 10.10], 
    "KRS_Dam": [76.57, 12.42], 
    "Kabini_Dam": [76.35, 11.97], 
    "Hemavathi_Dam": [76.03, 12.77], 
    "Tungabhadra_Dam": [76.33, 15.24], 
    "Almatti_Dam": [75.89, 16.33],
    # Phase 5 Batch
    "Supa_Dam": [74.52, 15.26], 
    "Bhadra_Dam": [75.62, 13.70],
    "Nagarjuna_Sagar": [79.31, 16.57], 
    "Srisailam_Dam": [78.87, 16.08],
    "Somasila_Dam": [79.30, 14.48], 
    "Kandaleru_Dam": [79.60, 14.33]
}

# 3. Load the High-Resolution ESA WorldCover Dataset
worldcover = ee.ImageCollection("ESA/WorldCover/v200").first()
cropland_mask = worldcover.eq(40) # Class 40 = Cropland

print("⏳ Pinging Earth Engine to calculate final 10km agricultural buffers...")
results = []

# 4. Iterate through the final 12 dams
for lake, coords in remaining_dams.items():
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

# 5. Save the final dataset
df = pd.DataFrame(results)
df.to_csv('data/processed/remaining_batch_cropland.csv', index=False)
print("\n✅ Final Catchment Extraction Complete! Saved as 'remaining_batch_cropland.csv'")