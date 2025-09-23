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

from map_modules.helper_functions import add_grid_squares, add_cleaned_roads_to_map, map_legend_for_legend, load_json_data

# Set the dataset you want to use, being either "small-dataset" or "full-dataset"
dataset = "small-dataset"

# Set base directory for data files
current_dir = os.path.dirname(os.path.abspath(__file__))    # scripts/
base_dir = os.path.dirname(current_dir)                     # project root
scenario = os.path.join(base_dir, "..", "dataset", dataset) # dataset directory
map_dir = os.path.join(base_dir, "..", "maps", "legends")   # map directory

def load_all_data():
    """Loads all necessary data files for the map legend."""
    file_paths = {
        "centralised_units": os.path.join(scenario, "centralised_units.json"),
        "distributed_units_existing": os.path.join(scenario, "distributed_units_exist.json"),
        "distributed_units_new": os.path.join(scenario, "distributed_units_new.json"),
        "radio_units_existing": os.path.join(scenario, "radio_units_exist.json"),
        "radio_units_new": os.path.join(scenario, "radio_units_new.json"),
        "road_nodes": os.path.join(scenario, "road_nodes.json"),
        "cleaned_road_edges": os.path.join(scenario, "road_edges.json"),
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
cleaned_road_edges_data = data["cleaned_road_edges"]
polygon_coords = data["polygon_coords"]
ru_du_paths = data["ru_du_paths"]
du_cu_paths_existing = data["du_cu_paths_existing"]
grid_data = data["grid_data"]
polygon_coords_lat_lon = [(lat, lon) for lon, lat in polygon_coords["polygon_coords"]]

def plot_path_on_map_with_curvature(map_obj, path_data, road_edges, color, weight=2):
    """ Plots paths on a Folium map using road edges for curvature."""
    pos = {}
    for path_entry in path_data:
        path = path_entry['path']
        for i in range(len(path) - 1):
            start_node = path[i]
            end_node = path[i + 1]
            
            # Find the corresponding edge in road_edges
            edge = next((e for e in road_edges if (e['from'] == start_node and e['to'] == end_node) or (e['from'] == end_node and e['to'] == start_node)), None)
            if edge:
                # Use the geometry of the edge to plot the path
                if edge['geometry']:
                    line_points = [(point['latitude'], point['longitude']) for point in edge['geometry']]
                    folium.PolyLine(locations=line_points, color=color, weight=weight).add_to(map_obj)
                else:
                    # If no geometry, fall back to direct line between nodes
                    start_pos, end_pos = pos[start_node], pos[end_node]
                    folium.PolyLine(locations=[start_pos[::-1], end_pos[::-1]], color=color, weight=weight).add_to(map_obj)

def add_du_cu_paths(map_object, du_cu_paths_data, road_edges_data, layer_name, color, weight=4):
    """Adds DU to CU paths to the map with curvature."""
    du_cu_paths_group = folium.FeatureGroup(name=layer_name, show=True)
    plot_path_on_map_with_curvature(du_cu_paths_group, du_cu_paths_data, road_edges_data['road_edges'], color, weight)
    du_cu_paths_group.add_to(map_object)

def add_feature_group(data, map_object, layer_name, get_color_function, marker_type, service_attribute=None):
    """Adds a feature group to the map with markers based on the provided data."""
    feature_group = folium.FeatureGroup(name=layer_name, show=True)
    
    for location in data:
        attribute_value = location.get(service_attribute, 'N/A') if service_attribute else 'N/A'
        marker_color = get_color_function(attribute_value)
        popup_content = f"ID: {location.get('ru_name') or location.get('du_name') or location.get('cu_name')}, Status: {attribute_value}"
        
        if marker_type == 'circle':
            folium.CircleMarker(location=[location['latitude'], location['longitude']], radius=4, color=marker_color, fill=True, fill_color=marker_color, fill_opacity=1, popup=popup_content).add_to(feature_group)
        elif marker_type == 'triangle':
            folium.RegularPolygonMarker(location=[location['latitude'], location['longitude']], number_of_sides=3, radius=7, color=marker_color, fill=True, fill_color=marker_color, fill_opacity=1, popup=popup_content).add_to(feature_group)
        elif marker_type == 'square':
            folium.RegularPolygonMarker(location=[location['latitude'], location['longitude']], number_of_sides=4, radius=8, color=marker_color, fill=True, fill_color=marker_color, fill_opacity=1, popup=popup_content).add_to(feature_group)
        elif marker_type == 'pentagon':
            folium.RegularPolygonMarker(location=[location['latitude'], location['longitude']], number_of_sides=5, radius=9, color=marker_color, fill=True, fill_color=marker_color, fill_opacity=1, popup=popup_content).add_to(feature_group)
        elif marker_type == 'hexagon':
            folium.RegularPolygonMarker(location=[location['latitude'], location['longitude']], number_of_sides=6, radius=14 if layer_name == "Centralised Units" else 8, color=marker_color, fill=True, fill_color=marker_color, fill_opacity=1, popup=popup_content).add_to(feature_group)
    
    feature_group.add_to(map_object)

# Add the radio unit radii to the map
def add_radio_unit_radii(data, map_object, default_radius=400):
    """Adds radio unit radii to the map."""
    feature_group_radii = folium.FeatureGroup(name="Radio Unit Radii", show=False)

    for location in data:
        # Use the range_m value from the JSON if present, else fall back to default
        radius_value = location.get("range_m", default_radius)
        color = 'purple'  # Same color for all radio unit radii
        circle = folium.Circle(location=[location['latitude'], location['longitude']], radius=radius_value, color=color, fill=True, fill_opacity=0.15, weight=1)
        circle.add_to(feature_group_radii)

    feature_group_radii.add_to(map_object)

def create_map_with_layers():
    """Creates a Folium map with all layers and features."""
    map_centre = [-35.351422, 150.4589332]
    # Create the map with the specified center and zoom level
    zoom_level = 15 if dataset == "small-dataset" else 14
    m = folium.Map(location=map_centre, zoom_start=zoom_level, tiles="OpenStreetMap")
    
    boundary_group = folium.FeatureGroup(name="Boundary", show=False)
    folium.Polygon(locations=polygon_coords_lat_lon, color='red', weight=1.5).add_to(boundary_group)
    boundary_group.add_to(m)
    
    # Add the user demand grid squares
    add_grid_squares(grid_data, m, dataset)
    
    # Add RU to DU paths (Existing only)
    add_du_cu_paths(m, ru_du_paths, cleaned_road_edges_data, "RU to DU Paths (Existing)", color="black")
    
    # Add DU to CU paths (Existing only)
    add_du_cu_paths(m, du_cu_paths_existing, cleaned_road_edges_data, "DU to CU Paths (Existing)", color="yellow")
    
    # Centralised units (larger hexagon markers)
    add_feature_group(centralised_units_data["centralised_units"], m, "Centralised Units", lambda _: 'black', marker_type='hexagon')    
    
    # Distributed Units - Existing and New (square and pentagon markers)
    add_feature_group(distributed_units_data_existing["distributed_units_existing"], m, "Distributed Units (Existing)", lambda _: 'blue', marker_type='square', service_attribute='status')
    add_feature_group(distributed_units_data_new["distributed_units_new"], m, "Distributed Units (Candidate)", lambda _: 'green', marker_type='pentagon', service_attribute='status')   
     
    # Radio units - Existing and New (circle and triangle markers)
    add_feature_group(radio_units_data_existing["radio_units_existing"], m, "Radio Units (Existing)", lambda _: 'red', marker_type='circle')
    add_feature_group(radio_units_data_new["radio_units_new"], m, "Radio Units (Candidate)", lambda _: 'purple', marker_type='circle')
    
    # Add road paths to the map
    add_cleaned_roads_to_map(cleaned_road_edges_data, m)
    
    # Add radio unit radii
    combined_radio_units_data = radio_units_data_existing["radio_units_existing"] + radio_units_data_new["radio_units_new"]
    add_radio_unit_radii(combined_radio_units_data, m)

    folium.LayerControl().add_to(m)
    map_legend_for_legend(m)
    return m

# Create the map with all layers and features
m = create_map_with_layers()
combined_html_map_path = os.path.join(map_dir, f"{dataset}_legend.html")
m.save(combined_html_map_path)
print(f"Combined map with connections has been saved to {combined_html_map_path}")
