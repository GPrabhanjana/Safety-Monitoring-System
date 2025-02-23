import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load and preprocess cluster data
def load_and_preprocess_cluster_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    if 'clusters' not in data:
        raise ValueError("JSON must contain a 'clusters' key")

    df = pd.DataFrame(data['clusters'])
    df['latitude'] = df['center'].apply(lambda x: x['latitude'] if isinstance(x, dict) else np.nan)
    df['longitude'] = df['center'].apply(lambda x: x['longitude'] if isinstance(x, dict) else np.nan)
    df.dropna(inplace=True)

    return df

# Load geo points data
def load_geo_points(file_path):
    with open(file_path, 'r') as file:
        points = json.load(file)
    
    df = pd.DataFrame(points)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    
    return df

# Match points to clusters
def match_points_to_clusters(df_points, df_clusters):
    cluster_centers = df_clusters[['latitude', 'longitude']].values
    points_coords = df_points[['latitude', 'longitude']].values
    
    distances = cdist(points_coords, cluster_centers)
    df_points['cluster'] = np.argmin(distances, axis=1)

    return df_points

# Aggregate points by date and cluster
def aggregate_points(df_points):
    daily_counts = df_points.groupby(['cluster', 'date']).size().reset_index(name='count')
    return daily_counts

# Predict future distribution with ARIMA
def predict_future_distribution(daily_counts, cluster_centers):
    predictions = {}
    for cluster_idx in daily_counts['cluster'].unique():
        cluster_data = daily_counts[daily_counts['cluster'] == cluster_idx]
        time_series = cluster_data.set_index('date')['count'].asfreq('D').fillna(0)
        
        model = ARIMA(time_series, order=(1, 1, 1), enforce_invertibility=False)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=10)

        predictions[cluster_idx] = {
            'forecast_dates': pd.date_range(pd.Timestamp.today(), periods=10, freq='D').strftime('%Y-%m-%d').tolist(),
            'forecast': forecast.tolist(),
            'center': cluster_centers[cluster_idx].tolist(),
            'total_points': time_series.sum()
        }

    return predictions

# Generate prediction plots
def create_prediction_plots(predictions):
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Individual Cluster Predictions', 'Cumulative Predictions'))

    for cluster_idx, pred in predictions.items():
        dates = pd.to_datetime(pred['forecast_dates'])
        fig.add_trace(go.Scatter(
            x=dates, y=pred['forecast'],
            name=f'Cluster {cluster_idx}', mode='lines+markers', line=dict(width=2)
        ), row=1, col=1)

    cumulative = pd.DataFrame([pred['forecast'] for pred in predictions.values()]).sum()
    fig.add_trace(go.Scatter(
        x=dates, y=cumulative, name='Total Crimes', line=dict(color='red', width=3)
    ), row=2, col=1)

    fig.update_layout(height=800, title_text='Crime Prediction Analysis (Real Data)', showlegend=True)
    return fig

# Main function to run the analysis
def main():
    cluster_df = load_and_preprocess_cluster_data('cluster_analysis.json')
    points_df = load_geo_points('geo_points.json')

    points_df = match_points_to_clusters(points_df, cluster_df)
    daily_counts = aggregate_points(points_df)

    cluster_centers = cluster_df[['latitude', 'longitude']].values
    predictions = predict_future_distribution(daily_counts, cluster_centers)

    prediction_plot = create_prediction_plots(predictions)
    prediction_plot.write_html('prediction_plots.html')

    print("\nClusters Processed with Real Historical Data. Check 'prediction_plots.html' for visualization.")

if __name__ == "__main__":
    main()
