import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# Load data
# ------------------------------------------------
data = pd.read_csv("CellData2.csv")
sim  = pd.read_csv("solutionKPP.csv")

# Normalize column names
data.columns = data.columns.str.lower()
sim.columns  = sim.columns.str.lower()

# Target times and colors
time_colors = {
    0: "red",
    12: "green",
    24: "black",
    36: "yellow",
    48: "blue"
}

# Find matching times in data
data_times = np.sort(data["t"].unique())
data_times = [t for t in data_times if t in time_colors]

print("Times found in data:", data_times)

# ------------------------------------------------
# Plot
# ------------------------------------------------
plt.figure(figsize=(10,6))

for t0 in data_times:
    color = time_colors[t0]

    # data at this time
    d = data[np.isclose(data["t"], t0)]

    # simulation curve at this time
    s = sim[np.isclose(sim["t"], t0)]

    # Plot simulation line
    plt.plot(
        s["x"], s["y"],
        color=color,
        linewidth=2,
        label=f"Sim t={t0}"
    )

    # Plot data points
    plt.scatter(
        d["x"], d["y"],
        color=color,
        edgecolor='k',
        s=60,
        marker='o',
        label=f"Data t={t0}"
    )

plt.xlabel("x")
plt.ylabel("y")
plt.title("Simulation vs Data at Measurement Times")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
