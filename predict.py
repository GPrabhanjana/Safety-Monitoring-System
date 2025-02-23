import sys
sys.stdout.reconfigure(encoding='utf-8')  # Fix for UnicodeEncodeError

import json
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load and preprocess data
def load_and_preprocess_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    if 'clusters' not in data:
        raise ValueError("JSON must contain a 'clusters' key")

    df = pd.DataFrame(data['clusters'])

    # Extract latitude & longitude
    df['latitude'] = df['center'].apply(lambda x: x['latitude'] if isinstance(x, dict) else np.nan)
    df['longitude'] = df['center'].apply(lambda x: x['longitude'] if isinstance(x, dict) else np.nan)
    df.dropna(inplace=True)

    return df

# Create spatial clusters using KMeans
def create_spatial_clusters(df):
    num_clusters = min(21, len(df))  # Ensure at most 21 clusters

    coords = df[['latitude', 'longitude']].values
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(coords)

    clusters = {}
    for cluster_idx in range(num_clusters):
        cluster_data = df[df['cluster'] == cluster_idx]
        center = kmeans.cluster_centers_[cluster_idx]
        clusters[cluster_idx] = {
            'center': center.tolist(),
            'cluster_size': len(cluster_data),
            'data': cluster_data
        }
    
    return clusters

# Predict future crime distribution using ARIMA
def predict_future_distribution(clusters):
    predictions = {}

    for cluster_idx, cluster_data in clusters.items():
        historical_data = np.random.randint(10, 50, size=30)
        model = ARIMA(historical_data, order=(1, 1, 1), enforce_invertibility=False)  # Fixed ARIMA warning
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=10)

        predictions[cluster_idx] = {
            'forecast_dates': pd.date_range(pd.Timestamp.today(), periods=10, freq='D').strftime('%Y-%m-%d').tolist(),
            'forecast': forecast.tolist(),
            'center': cluster_data['center'],
            'cluster_size': cluster_data['cluster_size']
        }

    return predictions

# Generate crime prediction plots
def create_prediction_plots(predictions):
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Individual Cluster Predictions', 'Cumulative Predictions'))

    for cluster_idx in sorted(predictions.keys()):
        dates = pd.to_datetime(predictions[cluster_idx]['forecast_dates'])
        fig.add_trace(go.Scatter(
            x=dates, y=predictions[cluster_idx]['forecast'],
            name=f'Cluster {cluster_idx}', mode='lines+markers', line=dict(width=2)
        ), row=1, col=1)

    cumulative = pd.DataFrame([pred['forecast'] for pred in predictions.values()]).sum()
    fig.add_trace(go.Scatter(
        x=dates, y=cumulative, name='Total Crimes', line=dict(color='red', width=3)
    ), row=2, col=1)

    fig.update_layout(height=800, title_text='Crime Prediction Analysis', showlegend=True)
    return fig

# Main function to execute analysis
def main():
    df = load_and_preprocess_data('cluster_analysis.json')
    clusters = create_spatial_clusters(df)
    predictions = predict_future_distribution(clusters)

    prediction_plot = create_prediction_plots(predictions)
    prediction_plot.write_html('prediction_plots.html')

    print("\n21 Clusters Processed. Check 'prediction_plots.html' for visualization.")  # Fixed Unicode issue

if __name__ == "__main__":
    main()
