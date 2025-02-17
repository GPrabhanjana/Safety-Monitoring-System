from flask import Flask, render_template, request, jsonify
import json
from datetime import datetime
import os
import logging

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# File paths
GEO_DATA_FILE = 'geo_points.json'
ISOCHRONES_FILE = 'isochrones.json'

def ensure_file_exists(filepath, default_content=None):
    """Ensure that a JSON file exists, create if it doesn't"""
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            json.dump(default_content or [], f)

def read_json_file(filepath):
    """Safely read a JSON file"""
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
    """Safely write to a JSON file"""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error writing to {filepath}: {str(e)}")
        return False

# Initialize files if they don't exist
ensure_file_exists(GEO_DATA_FILE)
ensure_file_exists(ISOCHRONES_FILE)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/points', methods=['GET'])
def get_points():
    """Get all points from geo_points.json"""
    try:
        points = read_json_file(GEO_DATA_FILE)
        return jsonify(points)
    except Exception as e:
        logger.error(f"Error fetching points: {str(e)}")
        return jsonify({'error': 'Failed to fetch points'}), 500

@app.route('/points', methods=['POST'])
def add_point():
    """Add a new point to geo_points.json"""
    try:
        new_point = request.json
        
        # Validate required fields
        required_fields = ['latitude', 'longitude', 'timestamp']
        if not all(field in new_point for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400

        # Validate data types
        if not isinstance(new_point['latitude'], (int, float)) or \
           not isinstance(new_point['longitude'], (int, float)):
            return jsonify({'error': 'Invalid coordinate format'}), 400

        # Read existing points
        points = read_json_file(GEO_DATA_FILE)
        
        # Add new point
        points.append(new_point)
        
        # Save updated points
        if write_json_file(GEO_DATA_FILE, points):
            return jsonify({'message': 'Point added successfully', 'point': new_point})
        else:
            return jsonify({'error': 'Failed to save point'}), 500

    except Exception as e:
        logger.error(f"Error adding point: {str(e)}")
        return jsonify({'error': 'Failed to add point'}), 500

@app.route('/isochrones', methods=['GET'])
def get_isochrones():
    """Get all isochrones from isochrones.json"""
    try:
        isochrones = read_json_file(ISOCHRONES_FILE)
        return jsonify(isochrones)
    except Exception as e:
        logger.error(f"Error fetching isochrones: {str(e)}")
        return jsonify({'error': 'Failed to fetch isochrones'}), 500

if __name__ == '__main__':
    app.run(debug=True)