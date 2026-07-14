"""
Plot orbital trajectory from CSV file in 3D with Earth
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from src.constants import R_EARTH

# Read the CSV file
df = pd.read_csv('work/output/cartesian.csv')

# Extract coordinates
x = df['X [km]'].values
y = df['Y [km]'].values
z = df['Z [km]'].values

# Create 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Load Earth texture image
try:
    img = np.array(Image.open("Blue_Marble_2002.png")) / 255.0
except FileNotFoundError:
    print("Warning: Blue_Marble_2002.png not found. Using solid color for Earth.")
    img = None

# Plot Earth as a sphere with texture
u = np.linspace(0, 2 * np.pi, img.shape[1] if img is not None else 50)
v = np.linspace(0, np.pi, img.shape[0] if img is not None else 50)
u, v = np.meshgrid(u, v)

x_earth = R_EARTH * np.cos(u) * np.sin(v)
y_earth = R_EARTH * np.sin(u) * np.sin(v)
z_earth = R_EARTH * np.cos(v)

# Plot Earth surface with image or solid color
ax.computed_zorder = False
if img is not None:
    ax.plot_surface(x_earth, y_earth, z_earth, 
                    rstride=1, cstride=1,
                    facecolors=img, linewidth=0, 
                    antialiased=False, shade=False, zorder=0)
else:
    ax.plot_surface(x_earth, y_earth, z_earth, color='blue', alpha=0.3, rstride=4, cstride=4)

# Plot trajectory
ax.plot(x, y, z, 'r-', linewidth=2, label='Orbital Trajectory')
ax.scatter(x[0], y[0], z[0], color='g', s=100, marker='o', label='Start')
ax.scatter(x[-1], y[-1], z[-1], color='r', s=100, marker='*', label='End')

# Set labels and title
ax.set_xlabel('X [km]')
ax.set_ylabel('Y [km]')
ax.set_zlabel('Z [km]')
ax.set_title('Orbital Trajectory with Earth')
ax.legend()

# Set equal aspect ratio for better visualization
max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
mid_x = (x.max() + x.min()) * 0.5
mid_y = (y.max() + y.min()) * 0.5
mid_z = (z.max() + z.min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.tight_layout()
plt.savefig('work/output/plots/trajectory_3d.png', dpi=150)
print("3D trajectory plot saved to: work/output/plots/trajectory_3d.png")
plt.show()
