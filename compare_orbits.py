import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths to the two CSV files
file_a = 'work/output/cartesian.csv'
file_b = 'work/output/cartesian_2body.csv'

# Read both files
csv_a = pd.read_csv(file_a)
csv_b = pd.read_csv(file_b)

# Extract position columns
cols = ['X [km]', 'Y [km]', 'Z [km]']

pos_a = csv_a[cols].to_numpy(dtype=float)
pos_b = csv_b[cols].to_numpy(dtype=float)

# Ensure same number of rows
if len(pos_a) != len(pos_b):
    n = min(len(pos_a), len(pos_b))
    pos_a = pos_a[:n]
    pos_b = pos_b[:n]

# Compute distance between the two trajectories
Delta = pos_a - pos_b
distance = np.linalg.norm(Delta, axis=1)

# Time axis in hours from the first sample
if 'epoch JD' in csv_a.columns:
    t_hours = (csv_a['epoch JD'].to_numpy(dtype=float) - csv_a['epoch JD'].iloc[0]) * 24.0
else:
    t_hours = np.arange(len(distance))

# Optional: print summary
print(f'Number of samples: {len(distance)}')
print(f'Min distance: {distance.min():.6f} km')
print(f'Max distance: {distance.max():.6f} km')
print(f'Final distance: {distance[-1]:.6f} km')

# Plot distance vs time
plt.figure(figsize=(10, 5))
plt.plot(t_hours, distance, label='Separation distance', color='tab:red')
plt.xlabel('Time [h]')
plt.ylabel('Distance [km]')

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('work/output/plots/orbit_distance.png', dpi=150)
print('Saved plot to work/output/plots/orbit_distance.png')
plt.show()
