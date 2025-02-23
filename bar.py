import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def create_geo_histogram():
    # Read the JSON file
    with open('geo_points.json', 'r') as file:
        data = json.load(file)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Convert timestamp strings to datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract just the time part (hour:minute)
    df['time'] = df['timestamp'].dt.strftime('%H:%M')
    
    # Create the histogram
    plt.figure(figsize=(12, 6))
    plt.hist(df['timestamp'].dt.hour + df['timestamp'].dt.minute/60, 
             bins=24,  # One bin per hour
             edgecolor='black',
             color='skyblue',
             alpha=0.7)
    
    # Customize the plot
    plt.title('Distribution of Geo Points Over Time', fontsize=14, pad=20)
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
    plt.savefig('geo_points_histogram.png')
    plt.close()
    
    # Print some basic statistics
    print("\nData Summary:")
    print(f"Total number of data points: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print("\nDistribution by hour:")
    print(df.groupby(df['timestamp'].dt.hour)['latitude'].count())

if __name__ == "__main__":
    create_geo_histogram()