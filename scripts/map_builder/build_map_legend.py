"""
%┏━┓┏━┓┏━━━┓┏━━━┓━━━━┏┓━━━┏━━━┓┏━━━┓┏━━━┓┏━┓━┏┓┏━━━┓
?┃┃┗┛┃┃┃┏━┓┃┃┏━┓┃━━━━┃┃━━━┃┏━━┛┃┏━┓┃┃┏━━┛┃┃┗┓┃┃┗┓┏┓┃
%┃┏┓┏┓┃┃┃━┃┃┃┗━┛┃━━━━┃┃━━━┃┗━━┓┃┃━┗┛┃┗━━┓┃┏┓┗┛┃━┃┃┃┃
?┃┃┃┃┃┃┃┗━┛┃┃┏━━┛━━━━┃┃━┏┓┃┏━━┛┃┃┏━┓┃┏━━┛┃┃┗┓┃┃━┃┃┃┃
%┃┃┃┃┃┃┃┏━┓┃┃┃━━━━━━━┃┗━┛┃┃┗━━┓┃┗┻━┃┃┗━━┓┃┃━┃┃┃┏┛┗┛┃
?┗┛┗┛┗┛┗┛━┗┛┗┛━━━━━━━┗━━━┛┗━━━┛┗━━━┛┗━━━┛┗┛━┗━┛┗━━━┛
%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
?━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This script generates a standalone map legend using the Folium library. The legend displays key network components — Radio Units (RUs), Distributed Units (DUs), and Centralised Units (CUs) — with interactive markers and colour coding.

The script loads data from the dataset JSON files and produces an HTML file showing the legend, which can be used alongside other map outputs for reference and clarity.

Usage:
    python build_map_legend.py

This will generate an HTML file in the `maps/` directory containing the map legend, based on the selected dataset.

Setup:
1. Set the `dataset` variable to "small-dataset" or "full-dataset" to load the appropriate JSON files.

Customisation:
- Adjust the thresholds in the `get_color_by_demand` function (in config.py) if your dataset uses different ranges of user demand.
- Switch datasets by changing the `dataset` variable.
- Update map centre coordinates and zoom level in the `create_map_with_layers` function for a different geographic area.

"""

import folium, os

from map_modules.helper_functions import add_grid_squares, add_roads_to_map, map_legend_for_legend, load_json_data, add_radio_unit_radii, add_legend_feature_group, add_existing_paths

# Set the dataset you want to use, being either "small-dataset" or "full-dataset"
dataset = "small-dataset"

# Set base directory for data files
current_dir = os.path.dirname(os.path.abspath(__file__))    # scripts/
base_dir = os.path.dirname(current_dir)                     # project root
scenario = os.path.join(base_dir, "..", "dataset", dataset) # dataset directory
map_dir = os.path.join(base_dir, "..", "maps", "legends")   # map directory
os.makedirs(map_dir, exist_ok=True)                         # Ensure the map result directory exists

def load_all_data():
    """Loads all necessary data files for the map legend."""
    file_paths = {
        "centralised_units": os.path.join(scenario, "centralised_units.json"),
        "distributed_units_existing": os.path.join(scenario, "distributed_units_exist.json"),
        "distributed_units_new": os.path.join(scenario, "distributed_units_new.json"),
        "radio_units_existing": os.path.join(scenario, "radio_units_exist.json"),
        "radio_units_new": os.path.join(scenario, "radio_units_new.json"),
        "road_nodes": os.path.join(scenario, "road_nodes.json"),
        "road_edges": os.path.join(scenario, "road_edges.json"),
        "polygon_coords": os.path.join(scenario, "region.json"),
        "ru_du_paths": os.path.join(scenario, "ru_du_path_exist_graph.json"),
        "du_cu_paths_existing": os.path.join(scenario, "du_cu_path_exist.json"),
        "du_cu_paths_new": os.path.join(scenario, "du_cu_path_new.json"),
        "grid_data": os.path.join(scenario, "demand_points.json")}

    data = {}
    for key, path in file_paths.items():
        data[key] = load_json_data(path)
    return data

data = load_all_data()
centralised_units_data = data["centralised_units"]
distributed_units_data_new = data["distributed_units_new"]
distributed_units_data_existing = data["distributed_units_existing"]
radio_units_data_existing = data["radio_units_existing"]
radio_units_data_new = data["radio_units_new"]
road_nodes_data = data["road_nodes"]
road_edges_data = data["road_edges"]
polygon_coords = data["polygon_coords"]
ru_du_paths = data["ru_du_paths"]
du_cu_paths_existing = data["du_cu_paths_existing"]
grid_data = data["grid_data"]
polygon_coords_lat_lon = [(lat, lon) for lon, lat in polygon_coords["polygon_coords"]]

def create_map_with_layers():
    """Creates a Folium map with all layers and features."""
    map_centre = [-35.351422, 150.4589332]
    # Create the map with the specified center and zoom level
    zoom_level = 15 if dataset == "small-dataset" else 14
    m = folium.Map(location=map_centre, zoom_start=zoom_level, tiles="OpenStreetMap")
    
    boundary_group = folium.FeatureGroup(name="Boundary", show=False)
    folium.Polygon(locations=polygon_coords_lat_lon, color='red', weight=1.5).add_to(boundary_group)
    boundary_group.add_to(m)
    
    add_grid_squares(grid_data, m, dataset)
    
    add_existing_paths(m, ru_du_paths, road_edges_data, "RU to DU Paths (Existing)", color="black")
    add_existing_paths(m, du_cu_paths_existing, road_edges_data, "DU to CU Paths (Existing)", color="yellow")
    
    add_legend_feature_group(centralised_units_data["centralised_units"], m, "Centralised Units", lambda _: 'black', marker_type='hexagon')    

    add_legend_feature_group(distributed_units_data_existing["distributed_units_existing"], m, "Distributed Units (Existing)", lambda _: 'blue', marker_type='square', service_attribute='status')
    add_legend_feature_group(distributed_units_data_new["distributed_units_new"], m, "Distributed Units (Candidate)", lambda _: 'green', marker_type='pentagon', service_attribute='status')   
     
    add_legend_feature_group(radio_units_data_existing["radio_units_existing"], m, "Radio Units (Existing)", lambda _: 'red', marker_type='circle')
    add_legend_feature_group(radio_units_data_new["radio_units_new"], m, "Radio Units (Candidate)", lambda _: 'purple', marker_type='circle')
    
    add_roads_to_map(road_edges_data, m)
    
    combined_radio_units_data = radio_units_data_existing["radio_units_existing"] + radio_units_data_new["radio_units_new"]
    add_radio_unit_radii(combined_radio_units_data, m, mode="base")

    folium.LayerControl().add_to(m)
    map_legend_for_legend(m)
    return m

# Create the map with all layers and features
m = create_map_with_layers()
combined_html_map_path = os.path.join(map_dir, f"{dataset}_legend.html")
m.save(combined_html_map_path)
print(f"Combined map with connections has been saved to {combined_html_map_path}")
