import json
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import folium
from folium.plugins import HeatMap
import math
from collections import Counter
from datetime import datetime
import scipy.stats as stats

def load_geo_data(file_path):
    """Load geographic points from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame for easier manipulation
    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = pd.DataFrame([data])
    
    # Convert timestamp to datetime, if not already
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
    # Extract hour from timestamp for peak hour analysis
    if 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
    
    return df

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in meters using Haversine formula."""
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    # Earth radius in meters
    radius = 6371000
    distance = radius * c
    
    return distance

def calculate_point_density(coords, k=20):
    """Calculate the local density of each point using k-nearest neighbors."""
    # Create a k-NN model
    k = min(k, len(coords)-1)
    nn = NearestNeighbors(n_neighbors=k+1)  # +1 to include the point itself
    nn.fit(coords)
    
    # Get distances to k nearest neighbors
    distances, _ = nn.kneighbors(coords)
    
    # Calculate density as inverse of average distance to k neighbors
    # Skip the first neighbor (which is the point itself)
    avg_distances = np.mean(distances[:, 1:], axis=1)
    
    # Avoid division by zero
    avg_distances = np.maximum(avg_distances, 1e-10)
    
    # Density is inverse of average distance
    density = 1.0 / avg_distances
    
    return density

def analyze_peak_hours(points_df, significance_threshold=0.5, min_points=10):
    """
    Analyze the distribution of points by hour to determine peak hours.
    
    Parameters:
    - points_df: DataFrame with an 'hour' column
    - significance_threshold: Threshold for peak significance (0.0 to 1.0)
        Higher values require more pronounced peaks to be considered significant
    - min_points: Minimum number of points required for significance testing
    
    Returns:
    - Dictionary with peak hour information
    """
    if 'hour' not in points_df.columns or len(points_df) < min_points:
        return {
            'peak_hours': [],
            'hourly_counts': {},
            'has_significant_peaks': False,
            'significance_score': 0.0
        }
    
    # Count points by hour
    hour_counts = points_df['hour'].value_counts().sort_index()
    
    # Convert to dictionary for easier JSON serialization
    hourly_counts = {int(hour): int(count) for hour, count in hour_counts.items()}
    
    # Calculate statistics for significance test
    if len(hour_counts) < 2:  # Need at least two hours for comparison
        has_significant_peaks = False
        significance_score = 0.0
    else:
        counts_array = np.array([hourly_counts.get(hour, 0) for hour in range(24)])
        max_count = np.max(counts_array)
        min_count = np.min(counts_array[counts_array > 0]) if np.any(counts_array > 0) else 0
        mean_count = np.mean(counts_array[counts_array > 0]) if np.any(counts_array > 0) else 0
        
        # Calculate coefficient of variation (higher means more variance)
        non_zero_counts = counts_array[counts_array > 0]
        if len(non_zero_counts) > 1:
            cv = np.std(non_zero_counts) / np.mean(non_zero_counts)
        else:
            cv = 0.0
            
        # Calculate peak-to-average ratio
        if mean_count > 0:
            peak_to_avg = max_count / mean_count
        else:
            peak_to_avg = 1.0
            
        # Calculate significance score (0-1)
        significance_score = min(1.0, max(0.0, (cv * peak_to_avg - 1) / 2))
        
        # Check if peaks are significant based on threshold
        has_significant_peaks = significance_score >= significance_threshold
    
    # Find peak hour(s) if significant
    if has_significant_peaks:
        threshold = 0.8 * max_count
        peak_hours = hour_counts[hour_counts >= threshold].index.tolist()
        # Format peak hours for display
        peak_hours_formatted = [f"{hour:02d}:00-{(hour+1) % 24:02d}:00" for hour in peak_hours]
    else:
        peak_hours = []
        peak_hours_formatted = []
    
    return {
        'peak_hours': peak_hours,
        'peak_hours_formatted': peak_hours_formatted,
        'hourly_counts': hourly_counts,
        'max_hour_count': int(max_count) if 'max_count' in locals() else 0,
        'has_significant_peaks': has_significant_peaks,
        'significance_score': float(significance_score)
    }

def multi_level_clustering(df, eps_values, min_samples=5, min_cluster_size=5, max_cluster_size=None, min_radius=1000, max_radius=10000):
    """
    Perform multi-level clustering with different epsilon values and size/radius constraints.
    
    Parameters:
    - df: DataFrame with latitude and longitude columns
    - eps_values: List of epsilon values from smallest (densest) to largest (sparsest)
    - min_samples: Minimum number of points required to form a dense region
    - min_cluster_size: Minimum number of points required for a valid cluster
    - max_cluster_size: Maximum number of points allowed in a cluster (None for no limit)
    - min_radius: Minimum radius of a cluster in meters (default: 500m)
    - max_radius: Maximum radius of a cluster in meters (default: 5000m)
    
    Returns:
    - DataFrame with cluster assignments
    - List of cluster information dictionaries
    """
    # Extract coordinates
    coords = df[['latitude', 'longitude']].values
    
    # Calculate point density to identify dense regions
    density = calculate_point_density(coords)
    df['density'] = density
    
    # Create copy of dataframe for working with
    working_df = df.copy()
    working_df['cluster'] = -1  # Initialize all points as noise
    
    # Track the next available cluster ID
    next_cluster_id = 0
    
    # Store all clusters
    all_clusters = []
    
    # For each epsilon value (from smallest to largest)
    for level, eps in enumerate(eps_values):
        # Get only unclustered points
        mask_unclustered = working_df['cluster'] == -1
        if not any(mask_unclustered):
            break  # All points are clustered
            
        coords_unclustered = coords[mask_unclustered]
        
        if len(coords_unclustered) < min_samples:
            break  # Not enough points left to form a cluster
        
        # Apply DBSCAN with current epsilon
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        level_labels = clusterer.fit_predict(coords_unclustered)
        
        # Process clusters found at this level
        unique_labels = set(label for label in level_labels if label != -1)
        for label in unique_labels:
            # Get indices of points in this cluster
            indices = np.where(mask_unclustered)[0][level_labels == label]
            
            # Get the cluster points
            cluster_points = working_df.iloc[indices]
            point_count = len(cluster_points)
            
            # Skip if cluster is too small
            if point_count < min_cluster_size:
                continue
                
            # Skip if cluster is too large (if max_cluster_size is specified)
            if max_cluster_size is not None and point_count > max_cluster_size:
                continue
            
            # Calculate cluster center and radius
            center_lat = cluster_points['latitude'].mean()
            center_lon = cluster_points['longitude'].mean()
            
            # Calculate maximum distance from center (radius)
            max_dist = 0
            for _, point in cluster_points.iterrows():
                dist = haversine_distance(
                    center_lat, center_lon, 
                    point['latitude'], point['longitude']
                )
                max_dist = max(max_dist, dist)
            
            # Enforce radius constraints - skip clusters with radius outside allowed range
            if max_dist < min_radius or max_dist > max_radius:
                continue
            
            # Calculate density metrics
            avg_density = cluster_points['density'].mean()
            area = np.pi * (max_dist/1000)**2  # area in square km
            density_per_area = point_count / max(area, 0.001)  # points per km²
            
            # Store cluster information
            cluster_info = {
                'cluster_id': next_cluster_id,
                'level': level,  # Which epsilon level created this cluster
                'size': point_count,
                'center_lat': center_lat,
                'center_lon': center_lon,
                'radius_meters': max_dist,
                'avg_density': avg_density,
                'density_per_area': density_per_area,
                'eps_used': eps,
                'indices': indices.tolist()
            }
            
            all_clusters.append(cluster_info)
            
            # Update working dataframe with cluster ID
            working_df.loc[indices, 'cluster'] = next_cluster_id
            
            # Increment cluster ID
            next_cluster_id += 1
    
    # Update original dataframe with final cluster assignments
    df['cluster'] = working_df['cluster']
    
    # For each cluster, store the points and analyze peak hours
    for cluster in all_clusters:
        cluster_points = df[df['cluster'] == cluster['cluster_id']]
        cluster['points'] = cluster_points
        
        # Analyze peak hours for this cluster
        peak_hour_info = analyze_peak_hours(cluster_points)
        cluster['peak_hours'] = peak_hour_info
    
    return df, all_clusters

def main(file_path, output_map='bengaluru_adaptive_clusters.html', significance_threshold=0.5, 
         min_cluster_size=3, max_cluster_size=None, min_radius=500, max_radius=5000):
    """Main function to process the geospatial data with cluster size and radius constraints."""
    # Load data
    df = load_geo_data(file_path)
    
    print(f"Processing {len(df)} geographic points...")
    
    # Define multiple epsilon levels from dense to sparse
    # Values are in coordinate units (decimal degrees)
    # Approximately: 0.005 ≈ 500m, 0.01 ≈ 1km, 0.02 ≈ 2km, 0.05 ≈ 5km
    eps_values = [0.005, 0.01, 0.02, 0.05]
    
    # Run multi-level clustering with size and radius constraints
    clustered_df, clusters = multi_level_clustering(
        df, 
        eps_values=eps_values,
        min_samples=5,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        min_radius=min_radius,
        max_radius=max_radius
    )
    
    # Print cluster statistics
    print(f"Found {len(clusters)} clusters across {len(eps_values)} density levels")
    print(f"Cluster size constraints: min={min_cluster_size}, max={max_cluster_size or 'None'}")
    print(f"Cluster radius constraints: min={min_radius}m, max={max_radius}m")
    
    # Group clusters by level
    level_counts = Counter([c['level'] for c in clusters])
    for level, count in sorted(level_counts.items()):
        level_label = ['Dense', 'Medium', 'Sparse', 'Very Sparse', 'Extremely Sparse'][min(level, 4)]
        print(f"Level {level} ({level_label}): {count} clusters, eps={eps_values[level]}")
    
    # Print individual cluster info with peak hours (only if significant)
    clusters_with_peaks = 0
    for cluster in clusters:
        has_significant_peaks = cluster['peak_hours']['has_significant_peaks']
        
        cluster_info = f"Cluster {cluster['cluster_id']} (Level {cluster['level']}): {cluster['size']} points, " \
                       f"radius: {cluster['radius_meters']:.0f}m"
        
        if has_significant_peaks:
            peak_hours_text = ", ".join(cluster['peak_hours']['peak_hours_formatted'])
            cluster_info += f", peak hours: {peak_hours_text}"
            clusters_with_peaks += 1
        
        print(cluster_info)
    
    print(f"\nClusters with significant peak hours: {clusters_with_peaks}/{len(clusters)}")
    
    # Count unclustered points
    noise_count = sum(clustered_df['cluster'] == -1)
    print(f"Unclustered points: {noise_count} ({noise_count/len(df):.1%})")
    
    # Generate and save peak hour report
    report = generate_hourly_report(clusters)
    report_file = output_map.replace('.html', '_peak_hours_report.txt')
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Peak hour report saved to {report_file}")
    
    # Visualize results
    map_vis = visualize_clusters(clustered_df, clusters, output_file=output_map)
    
    return {
        'clustered_df': clustered_df,
        'clusters': clusters,
        'map': map_vis,
        'peak_hour_report': report
    }
    """
    Perform multi-level clustering with different epsilon values.
    
    Parameters:
    - df: DataFrame with latitude and longitude columns
    - eps_values: List of epsilon values from smallest (densest) to largest (sparsest)
    - min_samples: Minimum number of points required to form a dense region
    
    Returns:
    - DataFrame with cluster assignments
    - List of cluster information dictionaries
    """
    # Extract coordinates
    coords = df[['latitude', 'longitude']].values
    
    # Calculate point density to identify dense regions
    density = calculate_point_density(coords)
    df['density'] = density
    
    # Create copy of dataframe for working with
    working_df = df.copy()
    working_df['cluster'] = -1  # Initialize all points as noise
    
    # Track the next available cluster ID
    next_cluster_id = 0
    
    # Store all clusters
    all_clusters = []
    
    # For each epsilon value (from smallest to largest)
    for level, eps in enumerate(eps_values):
        # Get only unclustered points
        mask_unclustered = working_df['cluster'] == -1
        if not any(mask_unclustered):
            break  # All points are clustered
            
        coords_unclustered = coords[mask_unclustered]
        
        if len(coords_unclustered) < min_samples:
            break  # Not enough points left to form a cluster
        
        # Apply DBSCAN with current epsilon
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        level_labels = clusterer.fit_predict(coords_unclustered)
        
        # Process clusters found at this level
        unique_labels = set(label for label in level_labels if label != -1)
        for label in unique_labels:
            # Get indices of points in this cluster
            indices = np.where(mask_unclustered)[0][level_labels == label]
            
            # Get the cluster points
            cluster_points = working_df.iloc[indices]
            
            # Calculate cluster center and radius
            center_lat = cluster_points['latitude'].mean()
            center_lon = cluster_points['longitude'].mean()
            
            # Calculate maximum distance from center (radius)
            max_dist = 0
            for _, point in cluster_points.iterrows():
                dist = haversine_distance(
                    center_lat, center_lon, 
                    point['latitude'], point['longitude']
                )
                max_dist = max(max_dist, dist)
            
            # Calculate density metrics
            avg_density = cluster_points['density'].mean()
            point_count = len(cluster_points)
            area = np.pi * (max_dist/1000)**2  # area in square km
            density_per_area = point_count / max(area, 0.001)  # points per km²
            
            # Store cluster information
            cluster_info = {
                'cluster_id': next_cluster_id,
                'level': level,  # Which epsilon level created this cluster
                'size': point_count,
                'center_lat': center_lat,
                'center_lon': center_lon,
                'radius_meters': max_dist,
                'avg_density': avg_density,
                'density_per_area': density_per_area,
                'eps_used': eps,
                'indices': indices.tolist()
            }
            
            all_clusters.append(cluster_info)
            
            # Update working dataframe with cluster ID
            working_df.loc[indices, 'cluster'] = next_cluster_id
            
            # Increment cluster ID
            next_cluster_id += 1
    
    # Update original dataframe with final cluster assignments
    df['cluster'] = working_df['cluster']
    
    # For each cluster, store the points and analyze peak hours
    for cluster in all_clusters:
        cluster_points = df[df['cluster'] == cluster['cluster_id']]
        cluster['points'] = cluster_points
        
        # Analyze peak hours for this cluster
        peak_hour_info = analyze_peak_hours(cluster_points)
        cluster['peak_hours'] = peak_hour_info
    
    return df, all_clusters

def create_hour_distribution_chart(hourly_counts, has_significant_peaks=False, peak_hours=None):
    """Generate an HTML/SVG chart for hour distribution."""
    hours = list(range(24))
    counts = [hourly_counts.get(hour, 0) for hour in hours]
    max_count = max(counts) if counts else 1
    
    # Fix: Check if max_count is zero to avoid division by zero
    if max_count == 0:
        max_count = 1  # Set to 1 to avoid division by zero
    
    # Normalize counts for display (0-100%)
    normalized = [int((count / max_count) * 100) for count in counts]
    
    # Generate SVG 
    svg_height = 150
    svg_width = 480
    bar_width = 18
    
    svg = f'<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">'
    
    # Add title indicating significance
    if has_significant_peaks:
        svg += f'<text x="{svg_width/2}" y="15" text-anchor="middle" font-size="12" font-weight="bold">Hourly Distribution (Significant Peaks Detected)</text>'
    else:
        svg += f'<text x="{svg_width/2}" y="15" text-anchor="middle" font-size="12" font-weight="bold">Hourly Distribution (No Significant Peaks)</text>'
    
    # Add bars
    for i, (hour, height_pct) in enumerate(zip(hours, normalized)):
        x = i * (bar_width + 2) + 5
        bar_height = (height_pct / 100) * (svg_height - 40)
        y = svg_height - bar_height - 25
        
        # Color peak hours differently only if significant
        is_peak = has_significant_peaks and peak_hours and hour in peak_hours
        color = "#3498db" if is_peak else "#95a5a6"
        
        # Draw bar
        svg += f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="{color}" />'
        
        # Hour label
        svg += f'<text x="{x + bar_width/2}" y="{svg_height-10}" text-anchor="middle" font-size="10">{hour}</text>'
        
        # Value label for peaks
        if is_peak:
            svg += f'<text x="{x + bar_width/2}" y="{y-5}" text-anchor="middle" font-size="10" fill="#2c3e50">{counts[i]}</text>'
    
    # X-axis labels
    svg += f'<text x="{svg_width/2}" y="{svg_height-2}" text-anchor="middle" font-size="10">Hour of Day</text>'
    
    svg += '</svg>'

    """Generate an HTML/SVG chart for hour distribution."""
    hours = list(range(24))
    counts = [hourly_counts.get(hour, 0) for hour in hours]
    max_count = max(counts) if counts else 1
    
    if max_count==0:
        max_count = 1
    
    # Normalize counts for display (0-100%)
    normalized = [int((count / max_count) * 100) for count in counts]
    
    # Generate SVG 
    svg_height = 150
    svg_width = 480
    bar_width = 18
    
    svg = f'<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">'
    
    # Add title indicating significance
    if has_significant_peaks:
        svg += f'<text x="{svg_width/2}" y="15" text-anchor="middle" font-size="12" font-weight="bold">Hourly Distribution (Significant Peaks Detected)</text>'
    else:
        svg += f'<text x="{svg_width/2}" y="15" text-anchor="middle" font-size="12" font-weight="bold">Hourly Distribution (No Significant Peaks)</text>'
    
    # Add bars
    for i, (hour, height_pct) in enumerate(zip(hours, normalized)):
        x = i * (bar_width + 2) + 5
        bar_height = (height_pct / 100) * (svg_height - 40)
        y = svg_height - bar_height - 25
        
        # Color peak hours differently only if significant
        is_peak = has_significant_peaks and peak_hours and hour in peak_hours
        color = "#3498db" if is_peak else "#95a5a6"
        
        # Draw bar
        svg += f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="{color}" />'
        
        # Hour label
        svg += f'<text x="{x + bar_width/2}" y="{svg_height-10}" text-anchor="middle" font-size="10">{hour}</text>'
        
        # Value label for peaks
        if is_peak:
            svg += f'<text x="{x + bar_width/2}" y="{y-5}" text-anchor="middle" font-size="10" fill="#2c3e50">{counts[i]}</text>'
    
    # X-axis labels
    svg += f'<text x="{svg_width/2}" y="{svg_height-2}" text-anchor="middle" font-size="10">Hour of Day</text>'
    
    svg += '</svg>'
    return svg

def visualize_clusters(df, clusters, output_file='cluster_map.html'):
    """Create an interactive map visualization of the clusters."""
    # Create a map centered on the mean coordinates
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Define colors for clusters based on their level (density)
    level_colors = {
        0: 'red',       # Densest clusters
        1: 'blue',      # Medium density
        2: 'green',     # Low density
        3: 'purple',    # Very sparse
        4: 'orange'     # Extremely sparse
    }
    
    # Add unclustered points (noise)
    noise_points = df[df['cluster'] == -1]
    for _, point in noise_points.iterrows():
        folium.CircleMarker(
            location=[point['latitude'], point['longitude']],
            radius=3,
            color='gray',
            fill=True,
            fill_opacity=0.7
        ).add_to(m)
    
    # Add clusters
    for cluster in clusters:
        cluster_id = cluster['cluster_id']
        level = cluster['level']
        color = level_colors.get(level, 'cadetblue')  # Default color if level not in dict
        
        # Get peak hour info
        has_significant_peaks = cluster['peak_hours']['has_significant_peaks']
        peak_hours = cluster['peak_hours']['peak_hours']
        
        # Format peak hours for display
        if has_significant_peaks and cluster['peak_hours']['peak_hours_formatted']:
            peak_hours_text = ", ".join(cluster['peak_hours']['peak_hours_formatted'])
        else:
            peak_hours_text = "No significant peaks detected"
        
        # Create hour distribution chart
        hour_chart = create_hour_distribution_chart(
            cluster['peak_hours']['hourly_counts'], 
            has_significant_peaks,
            peak_hours
        )
        
        # Prepare popup HTML with peak hour information
        popup_html = f"""
        <div style="width:500px;">
            <h4>Cluster {cluster_id}</h4>
            <p><b>Level:</b> {level}</p>
            <p><b>Size:</b> {cluster['size']} points</p>
            <p><b>Radius:</b> {cluster['radius_meters']:.0f}m</p>
            <p><b>Peak Hours:</b> {peak_hours_text}</p>
            <div>
                <h5>Hourly Distribution</h5>
                {hour_chart}
            </div>
        </div>
        """
        
        # Add a circle showing the cluster extent
        folium.Circle(
            location=[cluster['center_lat'], cluster['center_lon']],
            radius=cluster['radius_meters'],
            color=color,
            fill=True,
            fill_opacity=0.2,
            popup=folium.Popup(popup_html, max_width=500)
        ).add_to(m)
        
        # Add markers for points in the cluster
        for _, point in cluster['points'].iterrows():
            timestamp_str = str(point['timestamp']) if 'timestamp' in point else "N/A"
            hour_str = f"{point['hour']}:00" if 'hour' in point else "N/A"
            
            folium.CircleMarker(
                location=[point['latitude'], point['longitude']],
                radius=4,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=f"Cluster: {cluster_id}<br>Timestamp: {timestamp_str}<br>Hour: {hour_str}"
            ).add_to(m)
        
        # Add cluster info label with peak hour info only if significant
        label_html = f'<div style="font-size: 12pt; color: black; font-weight: bold;">Cluster {cluster_id}: {cluster["size"]} pts'
        
        # Only add peak hour info to label if significant
        if has_significant_peaks and cluster['peak_hours']['peak_hours_formatted']:
            label_html += f'<br><span style="font-size: 10pt;">Peak: {peak_hours_text}</span>'
        
        label_html += '</div>'
        
        folium.Marker(
            location=[cluster['center_lat'], cluster['center_lon']],
            icon=folium.DivIcon(
                icon_size=(150, 36),
                icon_anchor=(75, 18),
                html=label_html
            )
        ).add_to(m)
    
    # Add a heatmap layer to visualize density
    heat_data = [[row['latitude'], row['longitude']] for _, row in df.iterrows()]
    HeatMap(heat_data, radius=15).add_to(m)
    
    # Save the map
    m.save(output_file)
    print(f"Map saved to {output_file}")
    return m

def generate_hourly_report(clusters):
    """Generate a report of peak hours for all clusters."""
    report = "Peak Hour Analysis Report\n"
    report += "========================\n\n"
    
    # Count clusters with significant peaks
    clusters_with_peaks = [c for c in clusters if c['peak_hours']['has_significant_peaks']]
    
    report += f"Clusters with significant peak hours: {len(clusters_with_peaks)}/{len(clusters)}\n"
    
    # Overall statistics for clusters with significant peaks
    if clusters_with_peaks:
        all_peak_hours = []
        for cluster in clusters_with_peaks:
            all_peak_hours.extend(cluster['peak_hours']['peak_hours'])
        
        if all_peak_hours:
            hour_counter = Counter(all_peak_hours)
            most_common_peak = hour_counter.most_common(1)[0][0]
            most_common_peak_formatted = f"{most_common_peak:02d}:00-{(most_common_peak+1) % 24:02d}:00"
            
            report += f"Most common peak hour across clusters with significant peaks: {most_common_peak_formatted}\n\n"
    
    # Per cluster details
    report += "Cluster Peak Hours\n"
    report += "----------------\n"
    for cluster in sorted(clusters, key=lambda c: c['cluster_id']):
        has_significant_peaks = cluster['peak_hours']['has_significant_peaks']
        
        if has_significant_peaks:
            peak_hours_text = ", ".join(cluster['peak_hours']['peak_hours_formatted'])
            report += f"Cluster {cluster['cluster_id']} (Level {cluster['level']}, {cluster['size']} points): {peak_hours_text}\n"
        else:
            report += f"Cluster {cluster['cluster_id']} (Level {cluster['level']}, {cluster['size']} points): No significant peaks\n"
    
    return report

def main(file_path, output_map='bengaluru_adaptive_clusters.html', significance_threshold=0.5):
    """Main function to process the geospatial data."""
    # Load data
    df = load_geo_data(file_path)
    
    print(f"Processing {len(df)} geographic points...")
    
    # Define multiple epsilon levels from dense to sparse
    # Values are in coordinate units (decimal degrees)
    # Approximately: 0.005 ≈ 500m, 0.01 ≈ 1km, 0.02 ≈ 2km, 0.05 ≈ 5km
    eps_values = [0.005, 0.01, 0.02, 0.05]
    
    # Run multi-level clustering
    clustered_df, clusters = multi_level_clustering(
        df, 
        eps_values=eps_values,
        min_samples=5
    )
    
    # Print cluster statistics
    print(f"Found {len(clusters)} clusters across {len(eps_values)} density levels")
    
    # Group clusters by level
    level_counts = Counter([c['level'] for c in clusters])
    for level, count in sorted(level_counts.items()):
        level_label = ['Dense', 'Medium', 'Sparse', 'Very Sparse', 'Extremely Sparse'][min(level, 4)]
        print(f"Level {level} ({level_label}): {count} clusters, eps={eps_values[level]}")
    
    # Print individual cluster info with peak hours (only if significant)
    clusters_with_peaks = 0
    for cluster in clusters:
        has_significant_peaks = cluster['peak_hours']['has_significant_peaks']
        
        cluster_info = f"Cluster {cluster['cluster_id']} (Level {cluster['level']}): {cluster['size']} points, " \
                       f"radius: {cluster['radius_meters']:.0f}m"
        
        if has_significant_peaks:
            peak_hours_text = ", ".join(cluster['peak_hours']['peak_hours_formatted'])
            cluster_info += f", peak hours: {peak_hours_text}"
            clusters_with_peaks += 1
        
        print(cluster_info)
    
    print(f"\nClusters with significant peak hours: {clusters_with_peaks}/{len(clusters)}")
    
    # Count unclustered points
    noise_count = sum(clustered_df['cluster'] == -1)
    print(f"Unclustered points: {noise_count} ({noise_count/len(df):.1%})")
    
    # Generate and save peak hour report
    report = generate_hourly_report(clusters)
    report_file = output_map.replace('.html', '_peak_hours_report.txt')
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Peak hour report saved to {report_file}")
    
    # Visualize results
    map_vis = visualize_clusters(clustered_df, clusters, output_file=output_map)
    
    return {
        'clustered_df': clustered_df,
        'clusters': clusters,
        'map': map_vis,
        'peak_hour_report': report
    }

if __name__ == "__main__":
    # Example usage
    file_path = "geo_points.json"
    results = main(file_path, significance_threshold=0.5)  # Adjust threshold as needed
