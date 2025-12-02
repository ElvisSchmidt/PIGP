# Filename: read_u_data.py

import pandas as pd
import numpy as np

# --- 1. Set file path and sheets ---
file_path = "CellData2.xlsx"  # replace with your file path
sheets = ["0h","12h","24h", "36h", "48h"]  # sheet names
time_points = [0, 12, 24, 36, 48]  # corresponding t values

# --- 2. Read space coordinates from row 3 (Excel is 1-indexed, row=3) ---
# Columns B:AM correspond to 2:39 in pandas iloc (0-indexed)
space_coords_df = pd.read_excel(file_path, sheet_name=sheets[0], header=None, usecols="B:AM", nrows=1, skiprows=2)
space_coords = space_coords_df.values.flatten()  # 1D array of 38 x coordinates

# --- 3. Loop over sheets and read row 4 data ---
u_list = []
x_list = []
t_list = []

for sheet, t in zip(sheets, time_points):
    # Read row 4 (skip first 3 rows), columns B:AM
    data_df = pd.read_excel(file_path, sheet_name=sheet, header=None, usecols="B:AM", nrows=1, skiprows=10)
    u_values = data_df.values.flatten()  # 1D array of u values
    
    # Repeat x coordinates and t
    x_values = space_coords
    t_values = np.full_like(x_values, t, dtype=float)
    
    # Append to lists
    u_list.append(u_values)
    x_list.append(x_values)
    t_list.append(t_values)

# --- 4. Concatenate all sheets into single vectors ---
u_vec = 1000* np.concatenate(u_list)
x_vec = np.concatenate(x_list)
t_vec = np.concatenate(t_list)

# --- 5. Optional: create a DataFrame ---
df = pd.DataFrame({"x": x_vec, "t": t_vec,"y": u_vec})

# --- 6. Save to CSV if needed ---
df.to_csv("CellData2.csv", index=False)

print("Data successfully read and concatenated!")
print(df.head())