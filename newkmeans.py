import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import json
from datetime import datetime
import matplotlib.pyplot as plt

class CrimeClusterAnalyzer:
    def __init__(self, kval=0):
        self.kval = kval  # If 0, use the elbow method
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


    def determine_optimal_clusters(self, max_k=None):
        """
        Determine optimal number of clusters using multiple criteria and plot results.
        """
        scaled_features = self.prepare_features()
        n_samples = len(self.data)
        
        if self.kval > 0:
            print(f"Using manually set {self.kval} clusters.")
            self.n_clusters = self.kval
            return

        lat_range = self.data['latitude'].max() - self.data['latitude'].min()
        lon_range = self.data['longitude'].max() - self.data['longitude'].min()
        
        R = 6371  # Earth's radius in km
        avg_lat = self.data['latitude'].mean()
        area = (lat_range * R) * (lon_range * R * np.cos(np.radians(avg_lat)))
        
        density = n_samples / area if area > 0 else n_samples
        min_k = max(3, int(np.sqrt(n_samples / 3)))  # Ensure at least 3 clusters
        
        if max_k is None:
            max_k = min(int(np.sqrt(n_samples)), int(area * 3))  # Higher range to explore
        
        max_k = min(max_k, n_samples - 1)
        if max_k <= min_k:
            min_k = 3
            max_k = min(15, n_samples - 1)

        if min_k >= n_samples or max_k < 3:
            print("Warning: Not enough data points for meaningful clustering")
            self.n_clusters = min(3, n_samples)
            return
        
        K_range = range(min_k, max_k + 1)
        metrics = {'distortions': [], 'silhouette_scores': [], 'calinski_scores': []}

        try:
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(scaled_features)
                
                metrics['distortions'].append(kmeans.inertia_)
                
                if len(set(labels)) > 1:
                    metrics['silhouette_scores'].append(silhouette_score(scaled_features, labels))
                    metrics['calinski_scores'].append(calinski_harabasz_score(scaled_features, labels))
                else:
                    metrics['silhouette_scores'].append(0)
                    metrics['calinski_scores'].append(0)
        except Exception as e:
            print(f"Warning: Error during metric calculation: {e}")
            self.n_clusters = min_k
            return

        def safe_normalize(x):
            x = np.array(x)
            return (x - np.min(x)) / (np.max(x) - np.min(x)) if np.min(x) != np.max(x) else np.zeros_like(x)

        norm_distortions = safe_normalize(-np.array(metrics['distortions']))
        norm_silhouette = safe_normalize(metrics['silhouette_scores'])
        norm_calinski = safe_normalize(metrics['calinski_scores'])
        
        combined_scores = (0.3 * norm_distortions + 0.4 * norm_silhouette + 0.3 * norm_calinski)
        optimal_idx = np.argmax(combined_scores)
        self.n_clusters = K_range[optimal_idx]
        
        min_points_per_cluster = 15
        if n_samples / self.n_clusters < min_points_per_cluster:
            self.n_clusters = max(min_k, n_samples // min_points_per_cluster)
        
        print(f"Selected {self.n_clusters} clusters (adjusted from previous estimation).")

        # Plot the results
        plt.figure(figsize=(10, 5))
        plt.plot(K_range, norm_distortions, label='Distortion (inertia)', marker='o')
        plt.plot(K_range, norm_silhouette, label='Silhouette Score', marker='s')
        plt.plot(K_range, norm_calinski, label='Calinski-Harabasz Score', marker='^')
        plt.plot(K_range, combined_scores, label='Final Combined Score', marker='d', linestyle='--', color='black')
        plt.axvline(self.n_clusters, color='r', linestyle='--', label=f'Optimal k = {self.n_clusters}')
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Normalized Score")
        plt.title("Determination of Optimal k")
        plt.legend()
        plt.grid(True)
        plt.show()



    def find_elbow_point(self, K_range, distortions):
        """Finds the optimal k using the elbow method"""
        diffs = np.diff(distortions)
        second_diffs = np.diff(diffs)
        
        # The elbow is where the second derivative is largest (i.e., the highest "bend")
        elbow_idx = np.argmax(second_diffs) + 2  # Offset by +2 because of diff shifts
        return elbow_idx

    def prepare_features(self):
        features = self.data[['latitude', 'longitude', 'hour']].copy()
        scaled_features = self.scaler.fit_transform(features)
        return scaled_features
    
    def perform_clustering(self):
        if len(self.data) < 2:
            raise ValueError("Need at least 2 data points for clustering")
            
        scaled_features = self.prepare_features()
        self.determine_optimal_clusters()

        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
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
            'kval_used': self.kval,  # Track the kval used
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

def main(input_json_path, output_json_path, kval=0):
    try:
        # Initialize and run analysis
        analyzer = CrimeClusterAnalyzer(kval)
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
    
    kval = 0  # Set to 0 to use elbow method, or specify a fixed k
    main(input_path, output_path, kval)
