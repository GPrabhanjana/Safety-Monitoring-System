import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
import json
from datetime import datetime

def analyze_crime_hotspots(input_file, output_file):
    # Read JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
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
    
    # Calculate cluster centers
    cluster_centers = []
    for cluster in df['cluster'].unique():
        if cluster != -1:  # Exclude noise points
            cluster_points = df[df['cluster'] == cluster]
            center = {
                'cluster_id': int(cluster),
                'center_latitude': float(cluster_points['latitude'].mean()),
                'center_longitude': float(cluster_points['longitude'].mean()),
                'point_count': int(len(cluster_points))
            }
            cluster_centers.append(center)
    
    # Calculate density estimation
    x = df['longitude']
    y = df['latitude']
    kde = gaussian_kde([x, y])
    
    # Create grid for density estimation
    grid_points = 50
    x_grid = np.linspace(x.min(), x.max(), grid_points)
    y_grid = np.linspace(y.min(), y.max(), grid_points)
    xx, yy = np.meshgrid(x_grid, y_grid)
    z = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    
    # Create density grid points for output
    density_grid = []
    for i in range(grid_points):
        for j in range(grid_points):
            if z[i][j] > z.mean():  # Only include points with above-average density
                point = {
                    'latitude': float(yy[i][j]),
                    'longitude': float(xx[i][j]),
                    'density': float(z[i][j])
                }
                density_grid.append(point)
    
    # Prepare output data
    output_data = {
        'analysis_timestamp': datetime.now().isoformat(),
        'total_points': len(df),
        'cluster_count': len(cluster_centers),
        'cluster_centers': cluster_centers,
        'density_points': density_grid,
        'bounds': {
            'lat_min': float(df['latitude'].min()),
            'lat_max': float(df['latitude'].max()),
            'long_min': float(df['longitude'].min()),
            'long_max': float(df['longitude'].max())
        }
    }
    
    # Save results to JSON file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Plotting
    # Plot 1: DBSCAN results
    plt.figure(figsize=(10, 8))
    sns.set_style("darkgrid")
    sns.scatterplot(data=df, x='longitude', y='latitude', 
                   hue='cluster', palette='viridis', 
                   s=100, edgecolor='black')
    plt.xlabel('Longitude', fontsize=12, fontweight='bold')
    plt.ylabel('Latitude', fontsize=12, fontweight='bold')
    plt.title('Crime Hotspots Identified by DBSCAN', 
             fontsize=14, fontweight='bold')
    plt.legend(title="Clusters")
    plt.show()
    
    # Plot 2: KDE Heatmap
    plt.figure(figsize=(10, 8))
    
    # Create the heatmap first
    plt.pcolormesh(xx, yy, z, cmap='magma', alpha=0.75)
    plt.colorbar(label='Density')
    
    # Overlay the actual points
    plt.scatter(df['longitude'], df['latitude'], 
               color='cyan', s=120, edgecolor='black', 
               label='Crime Points')
    
    plt.xlabel('Longitude', fontsize=12, fontweight='bold')
    plt.ylabel('Latitude', fontsize=12, fontweight='bold')
    plt.title('Crime Density Estimation using KDE', 
             fontsize=14, fontweight='bold')
    plt.legend()
    plt.show()
    
    return output_data

# Example usage
if __name__ == "__main__":
    input_file = "geo_points.json"
    output_file = "crime_analysis_results.json"
    results = analyze_crime_hotspots(input_file, output_file)
