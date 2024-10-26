import os
import rasterio
import pandas as pd
import numpy as np

# Define the main data directory and output file
DATA_DIR = 'data'  # Directory containing all the crop subdirectories
OUTPUT_FILE = 'extracted_crop_data.csv'

# Specify the crops you want to extract data for
desired_crops = ['wheat', 'tomatoe']  # Replace with your desired crop names

# Open the output file and prepare to write headers
header_written = False

# Process each crop type in the directory
for crop in os.listdir(DATA_DIR):
    if crop not in desired_crops:
        continue  # Skip crops not in the desired list

    crop_path = os.path.join(DATA_DIR, crop)
    if os.path.isdir(crop_path):
        print(f"Processing crop: {crop}")
        
        # Store crop data in a dictionary
        crop_data = {}

        # Load all .tif files in the current crop directory
        for file in os.listdir(crop_path):
            if file.endswith('.tif') and os.path.isfile(os.path.join(crop_path, file)):
                variable_name = file.replace(f"{crop}_", "").replace(".tif", "")
                file_path = os.path.join(crop_path, file)

                try:
                    with rasterio.open(file_path) as src:
                        # Read data
                        data = src.read(1)
                        crop_data[variable_name] = {
                            'data': data,
                            'nodata': src.nodata,
                            'transform': src.transform,
                            'crs': src.crs
                        }
                        print(f"  Loaded variable: {variable_name}")
                except Exception as e:
                    print(f"  Error loading {file_path}: {e}")

        # Ensure all data arrays are of the same size and there is at least one dataset loaded
        if not crop_data or len(set(v['data'].shape for v in crop_data.values())) > 1:
            print("  Error: Data arrays mismatch in size or no data loaded.")
            continue

        # Generate coordinate grid using the metadata from the first file
        example_src = next(iter(crop_data.values()))
        rows, cols = example_src['data'].shape
        # Create row and column indices
        rows_indices, cols_indices = np.indices((rows, cols))
        # Compute the x and y coordinates
        xs, ys = rasterio.transform.xy(example_src['transform'], rows_indices, cols_indices, offset='center')
        xs = np.array(xs)
        ys = np.array(ys)

        # Flatten the arrays
        xs_flat = xs.flatten()
        ys_flat = ys.flatten()

        # Prepare data dictionary
        data_dict = {
            'crop': crop,
            'longitude': xs_flat,
            'latitude': ys_flat
        }

        # Process each variable
        for var, info in crop_data.items():
            data = info['data'].astype('float32')
            nodata = info['nodata']
            if nodata is not None:
                data[data == nodata] = np.nan
            data_dict[var] = data.flatten()

        # Create DataFrame
        df_crop = pd.DataFrame(data_dict)

        # Optionally filter out rows where all measurements are NaN
        measurement_vars = [col for col in df_crop.columns if col not in ('crop', 'longitude', 'latitude')]
        df_crop.dropna(subset=measurement_vars, how='all', inplace=True)

        # Write to CSV file
        if not header_written:
            df_crop.to_csv(OUTPUT_FILE, index=False, mode='w')
            header_written = True
        else:
            df_crop.to_csv(OUTPUT_FILE, index=False, mode='a', header=False)

        print(f"  Data for crop {crop} written to {OUTPUT_FILE}")

print(f"Data extraction complete. Saved to {OUTPUT_FILE}")
