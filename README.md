# Safety-Monitoring-System

A web application for mapping geographical points and their corresponding isochrones. The application allows users to add points on a map and visualizes isochrones (time-based travel boundaries) for these points using Valhalla routing engine.

## Features

- Interactive map interface using Leaflet.js
- Add geographical points with timestamps
- Generate and visualize isochrones for each point
- 50% transparent visualization of isochrones with color coding
- Persistent storage of points and isochrones in JSON files
- RESTful API endpoints for data management

## Prerequisites

- Python 3.7+
- Flask
- Docker (for running Valhalla)
- A running Valhalla instance in Docker

## Project Structure

```
your_project/
├── app.py                 # Main Flask application
├── isochrones.py         # Script to generate isochrones using Valhalla
├── geo_points.json       # Storage for geographical points
├── isochrones.json       # Storage for generated isochrones
└── templates/
    └── index.html        # Web interface template
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/geopoint-isochrone-mapper.git
cd geopoint-isochrone-mapper
```

2. Install Python dependencies:
```bash
pip install flask requests
```

3. Ensure you have a running Valhalla instance in Docker

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Access the web interface at `http://localhost:5000`

3. To generate isochrones for points:
```bash
python isochrones.py
```

## API Endpoints

### Points

- `GET /points`
  - Retrieve all geographical points
  - Response: Array of point objects

- `POST /points`
  - Add a new geographical point
  - Request body:
    ```json
    {
        "latitude": float,
        "longitude": float,
        "timestamp": "ISO-8601 timestamp"
    }
    ```

### Isochrones

- `GET /isochrones`
  - Retrieve all isochrones
  - Response: Array of isochrone objects with corresponding points

## Data Structure

### geo_points.json
```json
[
    {
        "latitude": 12.9686,
        "longitude": 77.5758,
        "timestamp": "2025-02-16T14:56:00.000Z"
    }
]
```

### isochrones.json
```json
[
    {
        "latitude": 12.9686,
        "longitude": 77.5758,
        "timestamp": "2025-02-16T14:56:00.000Z",
        "isochrone": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "fillColor": "#ff0000",
                        "fillOpacity": 0.33,
                        "color": "#ff0000",
                        "contour": 5,
                        "metric": "time"
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [...]
                    }
                }
            ]
        }
    }
]
```

## Acknowledgments

- [Leaflet.js](https://leafletjs.com/) for map visualization
- [Valhalla](https://github.com/valhalla/valhalla) for isochrone generation
- [Flask](https://flask.palletsprojects.com/) for the web framework
