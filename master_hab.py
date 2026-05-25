import pandas as pd
import glob
import os

# 1. Define the folder where your 30 CSV files are
folder_path = 'data/processed/individual_dams'
output_file = 'data/processed/master_hab_dataset.csv'

# 2. Get a list of all CSV files in that folder
all_files = glob.glob(os.path.join(folder_path, "*.csv"))

# 3. Read and merge
print(f"Merging {len(all_files)} dam files into one master dataset...")
df_list = [pd.read_csv(file) for file in all_files]
master_df = pd.concat(df_list, axis=0, ignore_index=True)

# 4. Save the master dataset
master_df.to_csv(output_file, index=False)
print(f"🎉 Master dataset saved to: {output_file}")
print(f"Total observations: {len(master_df)}")