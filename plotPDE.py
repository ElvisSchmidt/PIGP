import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import pandas as pd
######################## SIMULATED DATA####################
# Load your CSV
df = pd.read_csv("solutionKPP.csv")

# Condition 1: x = 25 + 50*n
mask_x = (df['x'] - 25) % 50 == 0

# Condition 2: t in selected times
valid_times = [24, 36, 48]
mask_t = df['t'].isin(valid_times)

# Apply filters
filtered = df[mask_x & mask_t].copy()
# Save result
filtered.to_csv("CellDataSim.csv", index=False)

print("Saved filtered data with noise to SimDatKPP.csv")

######################################################################
# 1. Load CSV: columns are x, t, y
data = np.loadtxt("solutionKPP.csv", delimiter=",", skiprows=1)  # skip header if needed
x = data[:, 0]
t = data[:, 1]
y = data[:, 2]
# 2. Get unique time steps
time_steps = np.unique(t)
print(np.max(y))
# 3. Group (x, y) by time
frames = []
for ti in time_steps:
    mask = (t == ti)
    frames.append((x[mask], y[mask]))

# 4. Setup plot
fig, ax = plt.subplots()
line, = ax.plot([], [], "-", lw=2)
ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(np.min(y), np.max(y)+0.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Animation of y = x(t)")

# 5. Update function
def update(frame):
    xi, yi = frames[frame]
    line.set_data(xi, yi)
    ax.set_title(f"Time t={time_steps[frame]:.2f}")
    return line,

# 6. Animate
ani = FuncAnimation(fig, update, frames=len(frames), blit=False, interval=10)

plt.show()


