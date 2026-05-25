import os
import pandas as pd

# 1. Define input directory and output file
v3_folder = 'data/processed/individual_dams_v3'
output_file = 'data/processed/master_hab_dataset_v3.csv'

# 2. Collect all the data
all_dataframes = []
print("🚀 Initiating Phase 3 Master Merge Protocol...")

for filename in os.listdir(v3_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(v3_folder, filename)
        
        # Read the rescued CSV
        df = pd.read_csv(file_path)
        all_dataframes.append(df)
        print(f"   ↳ Loaded: {filename}")

# 3. Concatenate and sort
print("\n🧬 Merging 30 dam datasets into a single matrix...")
master_v3_df = pd.concat(all_dataframes, ignore_index=True)

# Sort geographically and chronologically for neatness
if {'Lake_Name', 'Year', 'Month'}.issubset(master_v3_df.columns):
    master_v3_df = master_v3_df.sort_values(by=['Lake_Name', 'Year', 'Month'])

# 4. Save the final deep learning dataset
master_v3_df.to_csv(output_file, index=False)

# 5. Final diagnostic stats
total_rows = len(master_v3_df)
print(f"\n✅ MERGE COMPLETE! Master V3 Dataset saved as: {output_file}")
print(f"📊 Total Rows Secured for Deep Learning: {total_rows}")