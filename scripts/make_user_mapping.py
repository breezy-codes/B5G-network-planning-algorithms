"""
%┏━┓┏━┓┏━━━┓┏┓┏━┓┏━━━┓━━━━┏┓━┏┓┏━━━┓┏━━━┓┏━━━┓━━━━┏━┓┏━┓┏━━━┓┏━━━┓
?┃┃┗┛┃┃┃┏━┓┃┃┃┃┏┛┃┏━━┛━━━━┃┃━┃┃┃┏━┓┃┃┏━━┛┃┏━┓┃━━━━┃┃┗┛┃┃┃┏━┓┃┃┏━┓┃
%┃┏┓┏┓┃┃┃━┃┃┃┗┛┛━┃┗━━┓━━━━┃┃━┃┃┃┗━━┓┃┗━━┓┃┗━┛┃━━━━┃┏┓┏┓┃┃┃━┃┃┃┗━┛┃
?┃┃┃┃┃┃┃┗━┛┃┃┏┓┃━┃┏━━┛━━━━┃┃━┃┃┗━━┓┃┃┏━━┛┃┏┓┏┛━━━━┃┃┃┃┃┃┃┗━┛┃┃┏━━┛
%┃┃┃┃┃┃┃┏━┓┃┃┃┃┗┓┃┗━━┓━━━━┃┗━┛┃┃┗━┛┃┃┗━━┓┃┃┃┗┓━━━━┃┃┃┃┃┃┃┏━┓┃┃┃━━━
?┗┛┗┛┗┛┗┛━┗┛┗┛┗━┛┗━━━┛━━━━┗━━━┛┗━━━┛┗━━━┛┗┛┗━┛━━━━┗┛┗┛┗┛┗┛━┗┛┗┛━━━
%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
?━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This script generates the user mapping based on demand points and radio units from the dataset.
It uses the Haversine formula to determine if users are within range of radio units and assigns them accordingly. The output is saved as a JSON file in the dataset directory.

If an RU's coverage radius is changed, this script must be run again to update the user mapping. The file made tells the script whether a user is within range of a radio unit and which radio units they could be assigned to.

Example usage:
    python make_user_mapping.py
This will generate the user mapping and save it to a JSON file in the dataset directory.

"""

import json, os, math, random

# Set the dataset you want to use, being either "small-dataset" or "full-dataset"
dataset = "small-dataset"

# Set base directory for data files
current_dir = os.path.dirname(os.path.abspath(__file__))  # scripts/
base_dir = os.path.dirname(current_dir)                   # project root
scenario = os.path.join(base_dir, "dataset", dataset)     # dataset directory

def load_json_data(file_name):
    """Load JSON data from a specified file."""
    with open(file_name, 'r') as file:
        return json.load(file)

def load_all_data():
    """Load all necessary JSON data from the dataset directory."""
    file_paths = {
        "radio_units": os.path.join(scenario, "radio_units.json"),
        "demand_points": os.path.join(scenario, "demand_points.json")
    }

    data = {}
    for key, path in file_paths.items():
        data[key] = load_json_data(path)
    return data

def is_within_range(lat1, lon1, lat2, lon2, radius, margin=0):
    """Calculate the distance between two points on the Earth specified in decimal degrees and check if it is within the specified radius plus a margin."""

    R = 6371e3  # Earth radius in metres
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c  # in metres
    return distance <= (radius + margin)

def generate_random_coordinate_in_square(center_lat, center_lon, size):
    """Generate a random coordinate within a square of given size (in metres) centered at (center_lat, center_lon)."""

    # Convert size from metres to degrees (approximation)
    half_size = size / 2  # half of the square size in metres
    
    # Random offsets within the square
    offset_lat = random.uniform(-half_size, half_size) / 111320  # approx metres per degree latitude
    offset_lon = random.uniform(-half_size, half_size) / (111320 * math.cos(math.radians(center_lat)))
    
    return center_lat + offset_lat, center_lon + offset_lon

# Load all JSON data
data = load_all_data()

# Extracting the data
radio_units_data = data["radio_units"]
demand_points = data["demand_points"]

# Prepare the result structure
results = []

# Generate users for each demand point and assign them to RUs
for dp in demand_points["demand_points"]:
    users = []
    
    # Generate unique users within the demand point's square
    for user_id in range(dp["users"]):
        user_lat, user_lon = generate_random_coordinate_in_square(dp["latitude"], dp["longitude"], dp["size"])
        user_info = {
            "user_id": f"{dp['demand_id']}_user_{user_id+1}",
            "latitude": user_lat,
            "longitude": user_lon,
            "assigned_ru": []
        }
        users.append(user_info)
    
    # Determine which RUs each user is within range of
    for user in users:
        for ru in radio_units_data["radio_units"]:
            if is_within_range(user["latitude"], user["longitude"], ru["latitude"], ru["longitude"], ru["range_m"], margin=10):
                user["assigned_ru"].append(ru["ru_name"])
    
    # Append user data to results
    results.extend(users)

# Output results to JSON
output_file_path = os.path.join(scenario, "user_map.json")
with open(output_file_path, 'w') as json_file:
    json.dump(results, json_file, indent=4)

print(f"JSON file created successfully at {output_file_path}.")
