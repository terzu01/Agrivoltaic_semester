import os
import pandas as pd
from netCDF4 import Dataset

# Define the main data directory and output file
DATA_DIR = 'data/pangaea'  # Update this to your main folder path
OUTPUT_FILE = 'extracted_crop_yield_data.csv'

# Specify the crops you want to extract
desired_crops = ['wheat']  # Add your desired crops

# Prepare to write to CSV
with open(OUTPUT_FILE, 'w') as csvfile:
    # Write header
    csvfile.write('crop,year,longitude,latitude,yield\n')

    # Process each crop type in the directory
    for crop in desired_crops:
        crop_path = os.path.join(DATA_DIR, crop)
        if os.path.isdir(crop_path):
            print(f"Processing crop: {crop}")

            # Process each .nc4 file in the current crop directory
            for file in os.listdir(crop_path):
                if file.endswith('.nc4') and os.path.isfile(os.path.join(crop_path, file)):
                    file_path = os.path.join(crop_path, file)

                    # Extract the year from the filename (assuming the format is yield_YYYY.nc4)
                    year = file.split('_')[1][:4]  # Extracts the year from the filename

                    try:
                        # Load the netCDF file
                        with Dataset(file_path, 'r') as nc:
                            # Extract relevant variables
                            lon = nc.variables['lon'][:]
                            lat = nc.variables['lat'][:]
                            yield_data = nc.variables['var'][:]  # Adjust if needed

                            # Loop through the yield data
                            for i in range(len(lat)):
                                for j in range(len(lon)):
                                    # Create a record for each grid cell
                                    record = f"{crop},{year},{lon[j]},{lat[i]},{yield_data[i, j]}\n"
                                    csvfile.write(record)

                        print(f"  Loaded data from: {file_path}")
                    except Exception as e:
                        print(f"  Error loading {file_path}: {e}")

print(f"Data extraction complete. Saved to {OUTPUT_FILE}")
