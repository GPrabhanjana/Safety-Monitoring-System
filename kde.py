import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde

# Sample dataset (Replace with actual dataset)
data = [
    {"latitude": 12.974763013366534, "longitude": 77.58223915152485},
    {"latitude": 12.9715987, "longitude": 77.5945627},
    {"latitude": 12.927923, "longitude": 77.627108},
    {"latitude": 12.986375, "longitude": 77.536768},
    {"latitude": 12.960632, "longitude": 77.641603},
    {"latitude": 12.952579, "longitude": 77.490363},
    {"latitude": 12.891076, "longitude": 77.546123},
    {"latitude": 12.834503, "longitude": 77.573850}
]

# Convert to DataFrame
df = pd.DataFrame(data)

# Feature extraction
X = df[['latitude', 'longitude']]

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.3, min_samples=2)
df['cluster'] = dbscan.fit_predict(X_scaled)

# Plot DBSCAN results
plt.figure(figsize=(10, 8))
sns.set_style("darkgrid")
sns.scatterplot(x=df['longitude'], y=df['latitude'], hue=df['cluster'], palette='viridis', s=100, edgecolor='black')
plt.xlabel('Longitude', fontsize=12, fontweight='bold')
plt.ylabel('Latitude', fontsize=12, fontweight='bold')
plt.title('Crime Hotspots Identified by DBSCAN', fontsize=14, fontweight='bold')
plt.legend(title="Clusters")
plt.show()

# Kernel Density Estimation (KDE) Heatmap
x = df['longitude']
y = df['latitude']
kde = gaussian_kde([x, y])
x_grid, y_grid = np.meshgrid(np.linspace(x.min(), x.max(), 200), np.linspace(y.min(), y.max(), 200))
z_grid = kde(np.vstack([x_grid.ravel(), y_grid.ravel()])).reshape(x_grid.shape)

plt.figure(figsize=(10, 8))
sns.heatmap(z_grid, cmap='magma', alpha=0.75, xticklabels=False, yticklabels=False, cbar=True)
sns.scatterplot(x=x, y=y, color='cyan', s=120, edgecolor='black', label='Crime Points')
plt.xlabel('Longitude', fontsize=12, fontweight='bold')
plt.ylabel('Latitude', fontsize=12, fontweight='bold')
plt.title('Crime Density Estimation using KDE', fontsize=14, fontweight='bold')
plt.legend()
plt.show()

# Print cluster assignments
print(df)
