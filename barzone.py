import json
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import os
from datetime import datetime
import matplotlib.cm as cm

def create_zone_time_histograms():
    print("Loading geo points data...")
    # Read the JSON file
    with open('geo_points.json', 'r') as file:
        data = json.load(file)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Convert timestamp strings to datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract just the time part (hour:minute)
    df['time'] = df['timestamp'].dt.strftime('%H:%M')
    df['hour'] = df['timestamp'].dt.hour + df['timestamp'].dt.minute/60
    
    # Create shapely Point objects for spatial analysis
    df['geometry'] = df.apply(lambda row: Point(row.longitude, row.latitude), axis=1)
    
    # Convert to GeoDataFrame
    points_gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    
    # Load zone shapefile
    print("Loading zone shapefile...")
    zones_gdf = gpd.read_file('zone.shp')
    
    # Identify correct zone name column
    possible_names = ['name', 'zone', 'region', 'district', 'ZoneName', 'ZONE_NAME']
    for col in possible_names:
        if col in zones_gdf.columns:
            zones_gdf = zones_gdf.rename(columns={col: 'zone_name'})
            break
    
    if 'zone_name' not in zones_gdf.columns:
        raise KeyError("No valid 'zone_name' column found in zone.shp")
    
    # Ensure same CRS for spatial join
    if points_gdf.crs != zones_gdf.crs:
        zones_gdf = zones_gdf.to_crs(points_gdf.crs)
    
    # Perform spatial join to assign zones to points
    print("Assigning points to zones...")
    joined_gdf = gpd.sjoin(points_gdf, zones_gdf, how="left", predicate="within")
    
    # Handle points that don't fall within any zone
    joined_gdf['zone_name'] = joined_gdf['zone_name'].fillna('Outside Bangalore')
    
    # Create output directory for plots
    output_dir = 'zone_time_histograms'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get unique zones
    unique_zones = joined_gdf['zone_name'].unique()
    num_zones = len(unique_zones)
    
    print(f"Found {num_zones} unique zones.")
    
    # Create individual histograms for each zone
    for zone in unique_zones:
        zone_data = joined_gdf[joined_gdf['zone_name'] == zone]
        
        if len(zone_data) > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(zone_data['hour'], 
                    bins=24,  # One bin per hour
                    edgecolor='black',
                    color='skyblue',
                    alpha=0.7)
            
            # Customize the plot
            plt.title(f'Distribution of Geo Points Over Time: {zone}', fontsize=14, pad=20)
            plt.xlabel('Hour of Day', fontsize=12)
            plt.ylabel('Number of Data Points', fontsize=12)
            
            # Set x-axis ticks to show hours
            plt.xticks(range(0, 24), 
                      [f'{i:02d}:00' for i in range(24)],
                      rotation=45)
            
            # Add grid for better readability
            plt.grid(True, alpha=0.3)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            # Save the plot
            safe_zone_name = zone.replace(" ", "_").replace("/", "_")
            plt.savefig(f'{output_dir}/{safe_zone_name}_histogram.png')
            plt.close()
            
            print(f"Created histogram for zone: {zone} ({len(zone_data)} points)")
    
    # Create a combined histogram with all zones
    create_combined_zone_histogram(joined_gdf, unique_zones, output_dir)
    
    # Create total histogram
    create_total_histogram(joined_gdf, output_dir)
    
    # Print summary statistics
    print_summary_statistics(joined_gdf)

def create_combined_zone_histogram(joined_gdf, unique_zones, output_dir):
    """Create a combined histogram showing all zones with different colors"""
    print("Creating combined zone histogram...")
    
    plt.figure(figsize=(16, 10))
    
    # Create a colormap for the zones
    colors = cm.rainbow(np.linspace(0, 1, len(unique_zones)))
    
    # Create a list to store histogram data for the legend
    hist_data = []
    
    # Plot histogram for each zone with a different color
    for i, zone in enumerate(unique_zones):
        zone_data = joined_gdf[joined_gdf['zone_name'] == zone]
        
        if len(zone_data) > 0:
            # Create histogram and get the histogram data
            counts, bins, patches = plt.hist(zone_data['hour'], 
                                           bins=24,  # One bin per hour
                                           edgecolor='black',
                                           color=colors[i],
                                           alpha=0.7,
                                           label=zone)
            hist_data.append((zone, len(zone_data), counts.sum()))
    
    # Customize the plot
    plt.title('Distribution of Geo Points Over Time by Zone', fontsize=16, pad=20)
    plt.xlabel('Hour of Day', fontsize=14)
    plt.ylabel('Number of Data Points', fontsize=14)
    
    # Set x-axis ticks to show hours
    plt.xticks(range(0, 24), 
              [f'{i:02d}:00' for i in range(24)],
              rotation=45)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Create a legend with smaller font size and place it outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the combined plot
    plt.savefig(f'{output_dir}/combined_zones_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a sorted data table for the zones
    hist_data.sort(key=lambda x: x[1], reverse=True)
    
    # Plot a bar chart showing the total points per zone
    plt.figure(figsize=(14, 8))
    zones = [x[0] for x in hist_data]
    counts = [x[1] for x in hist_data]
    
    # Create horizontal bar chart, sorted by count
    plt.barh(range(len(zones)), counts, color=colors)
    plt.yticks(range(len(zones)), zones)
    plt.xlabel('Number of Data Points', fontsize=14)
    plt.title('Total Data Points by Zone', fontsize=16)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    # Save the zone comparison plot
    plt.savefig(f'{output_dir}/zone_comparison.png', dpi=300)
    plt.close()

def create_total_histogram(joined_gdf, output_dir):
    """Create a histogram for all data points combined"""
    print("Creating total histogram...")
    
    plt.figure(figsize=(12, 6))
    plt.hist(joined_gdf['hour'], 
            bins=24,  # One bin per hour
            edgecolor='black',
            color='skyblue',
            alpha=0.7)
    
    # Customize the plot
    plt.title('Distribution of All Geo Points Over Time', fontsize=14, pad=20)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Number of Data Points', fontsize=12)
    
    # Set x-axis ticks to show hours
    plt.xticks(range(0, 24), 
              [f'{i:02d}:00' for i in range(24)],
              rotation=45)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{output_dir}/total_histogram.png')
    plt.close()

def print_summary_statistics(joined_gdf):
    """Print summary statistics about the data"""
    print("\nData Summary:")
    print(f"Total number of data points: {len(joined_gdf)}")
    print(f"Date range: {joined_gdf['timestamp'].min()} to {joined_gdf['timestamp'].max()}")
    
    print("\nDistribution by zone:")
    zone_counts = joined_gdf.groupby('zone_name').size().sort_values(ascending=False)
    for zone, count in zone_counts.items():
        print(f"{zone}: {count} points ({count/len(joined_gdf)*100:.1f}%)")
    
    print("\nDistribution by hour:")
    hour_counts = joined_gdf.groupby(joined_gdf['timestamp'].dt.hour).size()
    for hour, count in hour_counts.items():
        print(f"{hour:02d}:00: {count} points ({count/len(joined_gdf)*100:.1f}%)")
    
    # Calculate the busiest hour by zone
    print("\nBusiest hour by zone:")
    for zone in joined_gdf['zone_name'].unique():
        zone_data = joined_gdf[joined_gdf['zone_name'] == zone]
        if len(zone_data) > 0:
            hour_counts = zone_data.groupby(zone_data['timestamp'].dt.hour).size()
            busiest_hour = hour_counts.idxmax()
            print(f"{zone}: {busiest_hour:02d}:00 ({hour_counts[busiest_hour]} points)")

if __name__ == "__main__":
    create_zone_time_histograms()