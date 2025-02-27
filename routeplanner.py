import json
from flask import Flask, request, jsonify, render_template, redirect, url_for
import requests
import math

app = Flask(__name__)
VALHALLA_URL = "http://localhost:8002/route"

# Load cluster borders automatically
try:
    with open('cluster_borders.json') as f:
        cluster_data = json.load(f)
    print(f"Loaded {len(cluster_data)} clusters from cluster_borders.json")
except Exception as e:
    print(f"Error loading cluster_borders.json: {e}")
    cluster_data = []

@app.route('/')
def home():
    return redirect(url_for('route_mapper'))

@app.route('/clusters')
def get_clusters():
    return jsonify(cluster_data)

@app.route('/route_mapper')
def route_mapper():
    return render_template('route_mapper.html', cluster_data=cluster_data)

def calculate_square_coords(center, radius_meters):
    """Calculate four corner points of a square around a center point given a radius in meters."""
    lat, lon = center[0], center[1]
    earth_radius = 6378137  # Earth's radius in meters

    # Convert radius in meters to degrees (approximate)
    lat_offset = (radius_meters / earth_radius) * (180 / math.pi)
    lon_offset = (radius_meters / (earth_radius * math.cos(math.pi * lat / 180))) * (180 / math.pi)

    # Calculate the four corners
    square_coords = [
        [lon - lon_offset, lat - lat_offset],  # Bottom-left
        [lon + lon_offset, lat - lat_offset],  # Bottom-right
        [lon + lon_offset, lat + lat_offset],  # Top-right
        [lon - lon_offset, lat + lat_offset],  # Top-left
        [lon - lon_offset, lat - lat_offset]   # Close the polygon
    ]
    return square_coords

@app.route('/plan_route', methods=['POST'])
def plan_route():
    try:
        data = request.get_json()
        start = data.get('start')
        end = data.get('end')
        # Use avoid_polygons from frontend if provided, but we'll override with square polygons
        avoid_polygons_frontend = data.get('avoid_polygons', [])
        
        if not start or not end:
            return jsonify({"error": "Invalid input data"}), 400
        
        print(f"Planning route from {start} to {end}")
        print(f"Avoid polygons from frontend (ignored): {json.dumps(avoid_polygons_frontend, indent=2)}")
        
        # Determine which levels to avoid based on frontend polygons
        avoid_levels = []
        for poly in avoid_polygons_frontend:
            # Extract cluster_id from tooltip or assume level from polygon count
            # Since frontend sends polygons for avoided clusters, infer levels from cluster_data
            for cluster in cluster_data:
                if cluster['cluster_id'] in [c['cluster_id'] for c in clusters if 'cluster_id' in c]:
                    if cluster['level'] not in avoid_levels:
                        avoid_levels.append(cluster['level'])
        
        # If no levels inferred, default to avoiding Level 0 (red areas)
        if not avoid_levels:
            avoid_levels = [0]
        print(f"Avoiding levels: {avoid_levels}")
        
        # Construct square avoid_polygons using center and radius_meters
        avoid_polygons = []
        for cluster in cluster_data:
            if cluster.get('level') in avoid_levels:
                center = cluster.get('center')
                radius_meters = cluster.get('radius_meters')
                if center and radius_meters:
                    square_coords = calculate_square_coords(center, radius_meters)
                    avoid_polygon = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [square_coords]
                        }
                    }
                    avoid_polygons.append(avoid_polygon)
        
        print(f"Created {len(avoid_polygons)} square avoid polygons")
        
        valhalla_request = {
            "locations": [
                {"lat": start[0], "lon": start[1], "type": "break"},
                {"lat": end[0], "lon": end[1], "type": "break"}
            ],
            "costing": "bicycle",
            "avoid_polygons": avoid_polygons,
            "costing_options": {
                "bicycle": {
                    "maneuver_penalty": 1000,  # High penalty to enforce avoidance
                    "destination_penalty": 1000,  # Penalize nearing avoided areas
                    "use_highways": 0.3  # Reduce highway preference
                }
            },
            "filters": {
                "exclude_locations": True  # Ensure locations outside polygons
            }
        }
        
        
        print(f"Valhalla request: {json.dumps(valhalla_request, indent=2)}")
        
        response = requests.post(VALHALLA_URL, json=valhalla_request)
        
        if response.status_code != 200:
            print(f"Valhalla error: {response.status_code}, {response.text}")
            return jsonify({"error": f"Valhalla returned status code {response.status_code}", "details": response.text}), response.status_code
        
        route_data = response.json()
        print(f"Valhalla response: {json.dumps(route_data, indent=2)}")
        
        return jsonify(route_data)
    
    except Exception as e:
        print(f"Error planning route: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)