import json
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from datetime import datetime
import os
import warnings
import matplotlib.cm as cm

warnings.filterwarnings("ignore")


def load_data():
    """Load the geo points and zone data"""
    print("Loading geo points data...")
    
    # Load geo points
    with open('geo_points.json', 'r') as f:
        geo_points = json.load(f)

    # Convert to pandas DataFrame
    points_df = pd.DataFrame(geo_points)
    
    # Convert timestamp to datetime
    points_df['timestamp'] = pd.to_datetime(points_df['timestamp'])
    
    # Sort by timestamp
    points_df = points_df.sort_values('timestamp')

    # Create shapely Point objects
    points_df['geometry'] = points_df.apply(lambda row: Point(row.longitude, row.latitude), axis=1)

    # Convert to GeoDataFrame
    points_gdf = gpd.GeoDataFrame(points_df, geometry='geometry', crs="EPSG:4326")

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

    print("Zone shapefile successfully loaded.")
    
    return points_gdf, zones_gdf


def assign_points_to_zones(points_gdf, zones_gdf):
    """Assign each point to a zone using spatial join"""
    print("Assigning points to zones...")

    # Ensure same CRS for spatial join
    if points_gdf.crs != zones_gdf.crs:
        zones_gdf = zones_gdf.to_crs(points_gdf.crs)

    # Perform spatial join
    joined_gdf = gpd.sjoin(points_gdf, zones_gdf, how="left", predicate="within")

    # Handle points that don't fall within any zone
    joined_gdf['zone_name'] = joined_gdf['zone_name'].fillna('Outside Bangalore')

    return joined_gdf


def aggregate_points_by_zone_and_time(joined_gdf, time_freq='D'):
    """Aggregate points by zone and time period"""
    print("Aggregating data by zone and time...")

    # Count points per zone per day
    zone_time_counts = joined_gdf.groupby(['zone_name', pd.Grouper(key='timestamp', freq=time_freq)]).size().reset_index(name='count')

    # Pivot to create time series for each zone
    zone_time_series = zone_time_counts.pivot(index='timestamp', columns='zone_name', values='count').fillna(0)

    return zone_time_series


def fit_arima_by_zone(zone_time_series):
    """Fit ARIMA model for each zone"""
    print("Fitting ARIMA models...")
    results = {}

    for zone_name in zone_time_series.columns:
        zone_data = zone_time_series[zone_name]

        # Skip if no data
        if zone_data.sum() == 0:
            results[zone_name] = {'status': 'No data', 'model': None, 'forecast': None}
            continue

        try:
            # Fit ARIMA model (p,d,q) = (1,1,1)
            model = ARIMA(zone_data, order=(1, 1, 1))
            model_fit = model.fit()

            # Forecast next 7 days
            forecast = model_fit.forecast(steps=7)

            results[zone_name] = {
                'status': 'success',
                'model': model_fit,
                'forecast': forecast,
                'aic': model_fit.aic,
                'bic': model_fit.bic
            }
        except Exception as e:
            results[zone_name] = {'status': f'Error: {str(e)}', 'model': None, 'forecast': None}

    return results


def plot_forecasts_combined(zone_time_series, arima_results, output_dir='zone_forecasts'):
    """Plot all zones in a single graph with different colors"""
    print("Plotting combined forecast...")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get zones with successful models
    successful_zones = [zone for zone, result in arima_results.items() 
                        if result['status'] == 'success']
    
    # Skip if no successful models
    if not successful_zones:
        print("No successful models to plot.")
        return
    
    # Create a colormap for the zones
    num_zones = len(successful_zones)
    colors = cm.rainbow(np.linspace(0, 1, num_zones))
    
    # Create figure for combined plot
    plt.figure(figsize=(16, 10))
    
    # Plot each zone with its own color
    for i, zone_name in enumerate(successful_zones):
        result = arima_results[zone_name]
        color = colors[i]
        
        # Plot historical data
        historical_data = zone_time_series[zone_name]
        plt.plot(historical_data.index, historical_data.values, 
                 color=color, linestyle='-', linewidth=1.5, 
                 label=f'{zone_name} (Historical)')
        
        # Plot forecast
        forecast = result['forecast']
        forecast_index = pd.date_range(start=historical_data.index[-1], 
                                       periods=len(forecast) + 1, freq='D')[1:]
        plt.plot(forecast_index, forecast.values, 
                 color=color, linestyle='--', linewidth=1.5,
                 label=f'{zone_name} (Forecast)')
    
    # Add labels and legend
    plt.title('Combined ARIMA Forecasts for All Zones in Bengaluru', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Point Count', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Create a legend with smaller font size and place it outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    
    # Save the combined plot
    plt.savefig(f'{output_dir}/combined_zones_forecast.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Total Forecast for Bengaluru
    plot_total_forecast(zone_time_series, output_dir)


def plot_total_forecast(zone_time_series, output_dir):
    """Plot total forecast for Bengaluru"""
    print("Plotting total forecast for Bengaluru...")
    
    total_series = zone_time_series.sum(axis=1)

    try:
        # Fit ARIMA model to total series
        total_model = ARIMA(total_series, order=(1, 1, 1))
        total_model_fit = total_model.fit()

        # Forecast next 7 days
        total_forecast = total_model_fit.forecast(steps=7)

        plt.figure(figsize=(14, 8))
        plt.plot(total_series.index, total_series.values, 'b-', 
                linewidth=2, label='Total Historical Data')

        forecast_index = pd.date_range(start=total_series.index[-1], 
                                      periods=len(total_forecast) + 1, freq='D')[1:]
        plt.plot(forecast_index, total_forecast.values, 'r--', 
                linewidth=2, label='Total Forecast')

        plt.title('Total ARIMA Forecast for Bengaluru', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Total Point Count', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add forecast values as text labels
        for i, val in enumerate(total_forecast.values):
            plt.text(forecast_index[i], val, f'{val:.0f}', 
                    fontsize=10, ha='center', va='bottom')

        # Save total forecast plot with higher resolution
        plt.savefig(f'{output_dir}/total_forecast.png', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error in Total Forecast: {e}")


def export_results(arima_results, zone_time_series, output_file='arima_results.json'):
    """Export ARIMA results to a JSON file"""
    print("Exporting results...")

    export_data = {}

    for zone_name, result in arima_results.items():
        if result['status'] == 'success':
            # Extract historical data
            historical_data = zone_time_series[zone_name].to_dict()
            historical_data_str = {str(date): count for date, count in historical_data.items()}

            # Extract forecast
            forecast = result['forecast'].to_dict()
            forecast_str = {str(date): count for date, count in forecast.items()}

            export_data[zone_name] = {
                'status': 'success',
                'historical_data': historical_data_str,
                'forecast': forecast_str,
                'metrics': {
                    'aic': result['aic'],
                    'bic': result['bic']
                }
            }
        else:
            export_data[zone_name] = {'status': result['status']}

    # Save to file
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)


def main():
    print("Starting Analysis...")

    points_gdf, zones_gdf = load_data()
    joined_gdf = assign_points_to_zones(points_gdf, zones_gdf)
    zone_time_series = aggregate_points_by_zone_and_time(joined_gdf)

    arima_results = fit_arima_by_zone(zone_time_series)

    # Use the new combined plotting function instead of the original
    plot_forecasts_combined(zone_time_series, arima_results)
    export_results(arima_results, zone_time_series)

    print("Analysis complete! Results saved.")


if __name__ == "__main__":
    main()
