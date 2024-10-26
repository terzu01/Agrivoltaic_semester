from netCDF4 import Dataset

# Path to your .nc4 file
file_path = 'data/pangaea/wheat/yield_2002.nc4'

# Open the netCDF file
with Dataset(file_path, 'r') as nc:
    print("File variables:")
    for var_name in nc.variables:
        print(f"  Variable: {var_name}")
        print(f"    Dimensions: {nc.variables[var_name].dimensions}")
        print(f"    Shape: {nc.variables[var_name].shape}")
        print(f"    Data type: {nc.variables[var_name].dtype}")
        print(f"    Attributes: {nc.variables[var_name].__dict__}")

    print("\nGlobal attributes:")
    for attr_name in nc.ncattrs():
        print(f"  {attr_name}: {nc.getncattr(attr_name)}")
