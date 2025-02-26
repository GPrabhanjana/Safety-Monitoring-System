from flask import Flask, render_template, request, jsonify
import json
import os
import logging
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# File paths for storing data
GEO_DATA_FILE = 'geo_points.json'
ISOCHRONES_FILE = 'isochrones.json'
HEATMAP_RESULTS_FILE = 'crime_heatmap_results.json'  # Added for heatmap

def ensure_file_exists(filepath, default_content=None):
    """Ensure that a JSON file exists; if not, create it with default content."""
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            json.dump(default_content or [], f)

def read_json_file(filepath):
    """Safely read a JSON file and return its content."""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return []
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {filepath}: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Error reading {filepath}: {str(e)}")
        return []

def write_json_file(filepath, data):
    """Safely write data to a JSON file."""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error writing to {filepath}: {str(e)}")
        return False

# Ensure required data files exist
ensure_file_exists(GEO_DATA_FILE)
ensure_file_exists(ISOCHRONES_FILE)
ensure_file_exists(HEATMAP_RESULTS_FILE, {  # Added for heatmap
    "analysis_timestamp": "",
    "total_points": 0,
    "density_points": [],
    "bounds": {
        "lat_min": 0,
        "lat_max": 0,
        "long_min": 0,
        "long_max": 0
    }
})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/points', methods=['GET'])
def get_points():
    try:
        points = read_json_file(GEO_DATA_FILE)
        return jsonify(points)
    except Exception as e:
        logger.error(f"Error fetching points: {str(e)}")
        return jsonify({'error': 'Failed to fetch points'}), 500

@app.route('/points', methods=['POST'])
def add_point():
    try:
        new_point = request.json
        if not all(field in new_point for field in ['latitude', 'longitude', 'timestamp']):
            return jsonify({'error': 'Missing required fields'}), 400
        
        points = read_json_file(GEO_DATA_FILE)
        points.append(new_point)
        
        if write_json_file(GEO_DATA_FILE, points):
            return jsonify({'message': 'Point added successfully', 'point': new_point})
        return jsonify({'error': 'Failed to save point'}), 500
    except Exception as e:
        logger.error(f"Error adding point: {str(e)}")
        return jsonify({'error': 'Failed to add point'}), 500

@app.route('/isochrones', methods=['GET'])
def get_isochrones():
    try:
        isochrones = read_json_file(ISOCHRONES_FILE)
        return jsonify(isochrones)
    except Exception as e:
        logger.error(f"Error fetching isochrones: {str(e)}")
        return jsonify({'error': 'Failed to fetch isochrones'}), 500

@app.route('/heatmap', methods=['GET'])
def get_heatmap():
    try:
        heatmap_data = read_json_file(HEATMAP_RESULTS_FILE)
        return jsonify(heatmap_data)
    except Exception as e:
        logger.error(f"Error fetching heatmap data: {str(e)}")
        return jsonify({'error': 'Failed to fetch heatmap data'}), 500

@app.route('/heatmap', methods=['POST'])
def update_heatmap():
    try:
        new_heatmap_data = request.json
        required_fields = ['analysis_timestamp', 'total_points', 'density_points', 'bounds']
        if not all(field in new_heatmap_data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        if write_json_file(HEATMAP_RESULTS_FILE, new_heatmap_data):
            return jsonify({'message': 'Heatmap data updated successfully'})
        return jsonify({'error': 'Failed to save heatmap data'}), 500
    except Exception as e:
        logger.error(f"Error updating heatmap data: {str(e)}")
        return jsonify({'error': 'Failed to update heatmap data'}), 500

if __name__ == '__main__':
    app.run(debug=True)
