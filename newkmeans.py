import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class CrimeClusterAnalyzer:
    def __init__(self, n_clusters=None):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
    
    def load_data_from_json(self, json_path):
        """Load data from a JSON file"""
        with open(json_path, 'r') as f:
            data_list = json.load(f)
        return self.load_data(data_list)
        
    def load_data(self, data_list):
        df = pd.DataFrame(data_list)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        self.data = df
        return df
    
    def determine_optimal_clusters(self, n_samples):
        if self.n_clusters is None or self.n_clusters >= n_samples:
            self.n_clusters = min(max(2, n_samples // 2), 5)
        print(f"Using {self.n_clusters} clusters for {n_samples} data points")
    
    def prepare_features(self):
        features = self.data[['latitude', 'longitude', 'hour']].copy()
        scaled_features = self.scaler.fit_transform(features)
        return scaled_features
    
    def perform_clustering(self):
        if len(self.data) < 2:
            raise ValueError("Need at least 2 data points for clustering")
            
        scaled_features = self.prepare_features()
        self.determine_optimal_clusters(len(self.data))
        
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans.fit(scaled_features)
        self.data['cluster'] = self.kmeans.labels_
    
    def generate_cluster_analysis(self):
        """Generate comprehensive analysis results in a dictionary format"""
        if not hasattr(self, 'kmeans'):
            raise ValueError("Must perform clustering before analysis")

        # Transform cluster centers back to original scale
        centers = self.kmeans.cluster_centers_
        original_scale_centers = self.scaler.inverse_transform(centers)
        
        # Calculate density metrics
        density_by_cluster = {}
        for i in range(self.n_clusters):
            cluster_data = self.data[self.data['cluster'] == i]
            density = len(cluster_data) / len(self.data)
            density_by_cluster[i] = density

        # Generate cluster statistics
        clusters_info = []
        for i in range(self.n_clusters):
            cluster_data = self.data[self.data['cluster'] == i]
            cluster_info = {
                'cluster_id': i,
                'center': {
                    'latitude': float(original_scale_centers[i][0]),
                    'longitude': float(original_scale_centers[i][1]),
                    'hour': float(original_scale_centers[i][2])
                },
                'statistics': {
                    'size': int(len(cluster_data)),
                    'density': float(density_by_cluster[i]),
                    'avg_hour': float(cluster_data['hour'].mean()),
                    'peak_hour': int(cluster_data['hour'].mode().iloc[0]) if not cluster_data.empty else None,
                    'lat_std': float(cluster_data['latitude'].std()),
                    'lon_std': float(cluster_data['longitude'].std()),
                    'hour_std': float(cluster_data['hour'].std())
                },
                'bounds': {
                    'lat_min': float(cluster_data['latitude'].min()),
                    'lat_max': float(cluster_data['latitude'].max()),
                    'lon_min': float(cluster_data['longitude'].min()),
                    'lon_max': float(cluster_data['longitude'].max())
                }
            }
            clusters_info.append(cluster_info)

        # Generate global statistics
        global_stats = {
            'total_points': len(self.data),
            'n_clusters': self.n_clusters,
            'global_bounds': {
                'lat_min': float(self.data['latitude'].min()),
                'lat_max': float(self.data['latitude'].max()),
                'lon_min': float(self.data['longitude'].min()),
                'lon_max': float(self.data['longitude'].max())
            },
            'timestamp': datetime.now().isoformat()
        }

        return {
            'clusters': clusters_info,
            'global_statistics': global_stats
        }

    def save_analysis_to_json(self, output_path):
        """Save the analysis results to a JSON file"""
        analysis_results = self.generate_cluster_analysis()
        with open(output_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        print(f"Analysis results saved to {output_path}")

def main(input_json_path, output_json_path):
    try:
        # Initialize and run analysis
        analyzer = CrimeClusterAnalyzer()
        analyzer.load_data_from_json(input_json_path)
        analyzer.perform_clustering()
        analyzer.save_analysis_to_json(output_json_path)
        
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    input_path = "geo_points.json"
    output_path = "cluster_analysis.json"
    main(input_path, output_path)