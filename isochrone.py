import json
import requests
import os

def read_geo_points(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def get_isochrone(lat, lon, contours=[5], mode="pedestrian"):
    url = "http://localhost:8002/isochrone"
    payload = {
        "locations": [{"lat": lat, "lon": lon}],
        "costing": mode,
        "contours": [{"time": t, "color": "ff0000"} for t in contours],
        "polygons": True
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching isochrone for ({lat}, {lon}): {response.text}")
        return None

def save_isochrones(all_isochrones, file_name):
    with open(file_name, 'w') as file:
        json.dump(all_isochrones, file, indent=2)

def load_existing_isochrones(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'r') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return []
    return []

def main():
    input_file = "geo_points.json"
    output_file = "isochrones.json"
    
    geo_points = read_geo_points(input_file)
    existing_isochrones = load_existing_isochrones(output_file)
    processed_coords = {(iso['latitude'], iso['longitude']) for iso in existing_isochrones}
    
    for point in geo_points:
        if (point['latitude'], point['longitude']) in processed_coords:
            print(f"Skipping already processed point: ({point['latitude']}, {point['longitude']})")
            continue
        
        isochrone = get_isochrone(point['latitude'], point['longitude'])
        if isochrone:
            point['isochrone'] = isochrone
            existing_isochrones.append(point)
            print(f"Processed and added: ({point['latitude']}, {point['longitude']})")
    
    save_isochrones(existing_isochrones, output_file)
    print(f"Saved all isochrones to {output_file}")

if __name__ == "__main__":
    main()
