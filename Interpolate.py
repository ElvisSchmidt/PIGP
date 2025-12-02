import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# Load your data
data = pd.read_csv("CellData2.csv")

# Select only rows with t = 0
d0 = data[data["t"] == 0].sort_values("x")

# Original x and y
x_data = d0["x"].values
y_data = d0["y"].values

# Define a finer x-grid (for example 200 points)
x_fine = np.linspace(x_data.min(), x_data.max(), 76*3+1)

# Build linear interpolator
f = interp1d(x_data, y_data, kind='linear', fill_value="extrapolate")

# Compute interpolated y values
y_fine = f(x_fine)

# Put in a DataFrame (optional)
interp_t0 = pd.DataFrame({
    "x": x_fine,
    "y": y_fine
})
interp_t0.to_csv("Initial_interpolated_data.csv",index=False)
print(interp_t0.head())