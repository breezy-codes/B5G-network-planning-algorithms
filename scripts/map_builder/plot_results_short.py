"""
%┏━┓┏━┓┏━━━┓┏━━━┓━━━━┏━━━┓┏┓━┏┓┏━━━┓┏━━━┓┏━━━━┓━━━━┏━━━┓┏━━━┓┏━━━┓┏┓━┏┓┏┓━━━┏━━━━┓
?┃┃┗┛┃┃┃┏━┓┃┃┏━┓┃━━━━┃┏━┓┃┃┃━┃┃┃┏━┓┃┃┏━┓┃┃┏┓┏┓┃━━━━┃┏━┓┃┃┏━━┛┃┏━┓┃┃┃━┃┃┃┃━━━┃┏┓┏┓┃
%┃┏┓┏┓┃┃┃━┃┃┃┗━┛┃━━━━┃┗━━┓┃┗━┛┃┃┃━┃┃┃┗━┛┃┗┛┃┃┗┛━━━━┃┗━┛┃┃┗━━┓┃┗━━┓┃┃━┃┃┃┃━━━┗┛┃┃┗┛
?┃┃┃┃┃┃┃┗━┛┃┃┏━━┛━━━━┗━━┓┃┃┏━┓┃┃┃━┃┃┃┏┓┏┛━━┃┃━━━━━━┃┏┓┏┛┃┏━━┛┗━━┓┃┃┃━┃┃┃┃━┏┓━━┃┃━━
%┃┃┃┃┃┃┃┏━┓┃┃┃━━━━━━━┃┗━┛┃┃┃━┃┃┃┗━┛┃┃┃┃┗┓━┏┛┗┓━━━━━┃┃┃┗┓┃┗━━┓┃┗━┛┃┃┗━┛┃┃┗━┛┃━┏┛┗┓━
?┗┛┗┛┗┛┗┛━┗┛┗┛━━━━━━━┗━━━┛┗┛━┗┛┗━━━┛┗┛┗━┛━┗━━┛━━━━━┗┛┗━┛┗━━━┛┗━━━┛┗━━━┛┗━━━┛━┗━━┛━
%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
?━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This script generates an interactive map visualisation of the results from the Greedy or CPLEX shortest path models.

The script uses the Folium library to display selected and unselected network components (RUs, DUs) as map markers, and overlays the shortest path connections determined by the optimisation model.

Usage:
    python plot_results_short.py

This will generate an HTML file named:
    short_{dataset}.html
in the `maps/` directory, where `{dataset}` is either "small-dataset" or "full-dataset".

Setup:
1. Add the results from your algorithm to the variables at the top of the file:
   - `selected_rus`, `not_selected_rus`
   - `selected_dus`, `not_selected_dus`
2. Set the `dataset` variable to "small-dataset" or "full-dataset" to load the correct JSON files.

Customisation:
- Modify thresholds in the `get_color_by_demand` function (in config.py) if your dataset has higher or lower user numbers.
- Adjust the `dataset` variable to switch datasets.
- Change the map centre coordinates and zoom level in the `create_map_with_layers` function for different geographic regions.

"""

import folium, os

from map_modules.config import get_radio_unit_color, get_distributed_unit_color, get_centralised_unit_color
from map_modules.helper_functions import add_feature_group, add_radio_unit_radii, add_cleaned_roads_to_map, add_grid_squares, map_legend_for_results, load_json_data

# ===================================
# Place the selected devices and connections from the results file here

selected_rus = ['RU3', 'RU4', 'RU7', 'RU14', 'RU15', 'RU17', 'RU19', 'RU22', 'RU24', 'RU26', 'RU32', 'RU54', 'RU57', 'RU83', 'RU2', 'RU18', 'RU1']
not_selected_rus = ['RU8', 'RU10', 'RU11', 'RU13', 'RU16', 'RU20', 'RU21', 'RU23', 'RU25', 'RU34', 'RU35', 'RU36', 'RU38', 'RU50', 'RU51', 'RU55', 'RU58', 'RU59', 'RU82', 'RU5', 'RU12']
selected_dus = ['DU2', 'DU1']
not_selected_dus = ['DU3', 'DU4', 'DU9']

ru_to_du_connections = [('RU3', 'DU2'), ('RU4', 'DU1'), ('RU7', 'DU1'), ('RU14', 'DU2'), ('RU15', 'DU1'), ('RU17', 'DU1'), ('RU19', 'DU2'), ('RU22', 'DU1'), ('RU24', 'DU1'), ('RU26', 'DU1'), ('RU32', 'DU1'), ('RU54', 'DU2'), ('RU57', 'DU2'), ('RU83', 'DU2'), ('RU2', 'DU1'), ('RU18', 'DU1'), ('RU1', 'DU2')]
du_to_cu_connections = [('DU2', 'CU1'), ('DU1', 'CU1')]

# ===================================

# Set the dataset you want to use, being either "small-dataset" or "full-dataset"
dataset = "full-dataset"

# Set the file name for the map based on the dataset
file_name = f"short_{dataset}"

# Set base directory for data files
current_dir = os.path.dirname(os.path.abspath(__file__))    # scripts/
base_dir = os.path.dirname(current_dir)                     # project root
scenario = os.path.join(base_dir, "..", "dataset", dataset) # dataset directory
map_dir = os.path.join(base_dir, "..", "maps", "generated") # map directory

def load_all_data():
    """Loads all necessary JSON data from the dataset directory."""
    file_paths = {
        "centralised_units": os.path.join(scenario, "centralised_units.json"),
        "distributed_units": os.path.join(scenario, "distributed_units.json"),
        "radio_units": os.path.join(scenario, "radio_units.json"),
        "road_nodes": os.path.join(scenario, "road_nodes.json"),
        "cleaned_road_edges": os.path.join(scenario, "road_edges.json"),
        "polygon_coords": os.path.join(scenario, "region.json"),
        "ru_du_paths": os.path.join(scenario, "ru_du_path.json"),
        "du_cu_paths": os.path.join(scenario, "du_cu_path.json"),
        "ru_du_existing_paths": os.path.join(scenario, "ru_du_path_exist_graph.json"),
        "radio_units_existing": os.path.join(scenario, "radio_units_exist.json"),
        "grid_data": os.path.join(scenario, "demand_points.json")}

    data = {}
    for key, path in file_paths.items():
        data[key] = load_json_data(path)
    return data

data = load_all_data()
centralised_units_data = data["centralised_units"]
distributed_units_data = data["distributed_units"]
radio_units_data = data["radio_units"]
radio_units_existing_data = data["radio_units_existing"]
road_nodes_data = data["road_nodes"]
cleaned_road_edges_data = data["cleaned_road_edges"]
polygon_coords = data["polygon_coords"]
ru_du_paths = data["ru_du_paths"]
du_cu_paths = data["du_cu_paths"]
ru_du_existing_paths = data["ru_du_existing_paths"] 
grid_data = data["grid_data"]
polygon_coords_lat_lon = [(lat, lon) for lon, lat in polygon_coords["polygon_coords"]]

def plot_path_on_map_with_curvature(map_obj, path_data, road_edges, color, ru_du_connections):
    """Plots paths on the map with curvature based on road edges."""
    pos = {}
    
    for path_entry in path_data:
        path = path_entry['path']
        if len(path) >= 2:
            start_node = path[0]
            end_node = path[-1]

            if (start_node, end_node) in ru_du_connections or (end_node, start_node) in ru_du_connections:
                for i in range(len(path) - 1):
                    current_start_node = path[i]
                    current_end_node = path[i + 1]
                    
                    edge = next((e for e in road_edges if (e['from'] == current_start_node and e['to'] == current_end_node) or (e['from'] == current_end_node and e['to'] == current_start_node)), None)
                    if edge:
                        if edge['geometry']:
                            line_points = [(point['latitude'], point['longitude']) for point in edge['geometry']]
                            folium.PolyLine(locations=line_points, color=color, weight=3).add_to(map_obj)
                        else:
                            start_pos, end_pos = pos[current_start_node], pos[current_end_node]
                            folium.PolyLine(locations=[start_pos[::-1], end_pos[::-1]], color=color, weight=3).add_to(map_obj)
 
def plot_existing_ru_du_paths(map_obj, path_data, road_edges, color='black'):
    """Plots existing RU-DU paths on the map."""
    pos = {}
    
    for path_entry in path_data:
        path = path_entry['path']
        if len(path) >= 2:
            for i in range(len(path) - 1):
                current_start_node = path[i]
                current_end_node = path[i + 1]
                
                edge = next((e for e in road_edges if (e['from'] == current_start_node and e['to'] == current_end_node) or (e['from'] == current_end_node and e['to'] == current_start_node)), None)
                if edge:
                    if edge['geometry']:
                        line_points = [(point['latitude'], point['longitude']) for point in edge['geometry']]
                        folium.PolyLine(locations=line_points, color=color, weight=8).add_to(map_obj)
                    else:
                        start_pos, end_pos = pos[current_start_node], pos[current_end_node]
                        folium.PolyLine(locations=[start_pos[::-1], end_pos[::-1]], color=color, weight=3).add_to(map_obj)

def create_map_with_layers(selected_rus=None, not_selected_rus=None, selected_dus=None, not_selected_dus=None):
    """Creates a Folium map with all layers and features."""
    map_centre = [-35.351422, 150.4589332]
    # Create the map with the specified center and zoom level
    zoom_level = 15 if dataset == "small-dataset" else 14
    m = folium.Map(location=map_centre, zoom_start=zoom_level, tiles="OpenStreetMap")
    
    boundary_group = folium.FeatureGroup(name="Boundary", show=False)
    folium.Polygon(locations=polygon_coords_lat_lon, color='red', weight=1.5).add_to(boundary_group)
    add_cleaned_roads_to_map(cleaned_road_edges_data, m)
    boundary_group.add_to(m)
    
    add_feature_group(centralised_units_data["centralised_units"], m, "Centralised Units", get_centralised_unit_color, marker_type='hexagon', selected_ids=["CU_1"], not_selected_ids=[])    
    add_radio_unit_radii(radio_units_data["radio_units"], m, selected_ids=selected_rus, not_selected_ids=not_selected_rus) 
    add_feature_group(distributed_units_data["distributed_units"], m, "Selected Distributed Units", get_distributed_unit_color, marker_type='pentagon', selected_ids=selected_dus, not_selected_ids=[])
    add_feature_group(distributed_units_data["distributed_units"], m, "Unselected Distributed Units", get_distributed_unit_color, marker_type='pentagon', selected_ids=[], not_selected_ids=not_selected_dus)  
      
    add_feature_group(radio_units_data["radio_units"], m, "Selected Radio Units", get_radio_unit_color, marker_type='circle', selected_ids=selected_rus, not_selected_ids=[])
    add_feature_group(radio_units_data["radio_units"], m, "Unselected Radio Units", get_radio_unit_color, marker_type='circle', selected_ids=[], not_selected_ids=not_selected_rus)
    
    # Add Existing Radio Units as a new layer
    existing_radio_units_group = folium.FeatureGroup(name="Existing Radio Units", show=True)
    for location in radio_units_existing_data["radio_units_existing"]:
        folium.CircleMarker(location=[location['latitude'], location['longitude']], radius=4, color='purple', fill=True, fill_color='purple', fill_opacity=1, popup=f"ID: {location.get('ru_name')}, Status: existing").add_to(existing_radio_units_group)
    existing_radio_units_group.add_to(m)
    
    # Add the new RU-DU existing paths in black
    existing_ru_du_paths_group = folium.FeatureGroup(name="Existing RU-DU Paths", show=True)
    plot_existing_ru_du_paths(existing_ru_du_paths_group, ru_du_existing_paths, cleaned_road_edges_data['road_edges'], color='#BD33A3')
    existing_ru_du_paths_group.add_to(m)
    
    # Chosen RU-DU paths
    ru_du_paths_group = folium.FeatureGroup(name="RU to DU Paths", show=True)
    plot_path_on_map_with_curvature(ru_du_paths_group, ru_du_paths, cleaned_road_edges_data['road_edges'], color='black', ru_du_connections=ru_to_du_connections)
    ru_du_paths_group.add_to(m)

    # Chosen DU-CU paths
    du_cu_paths_group = folium.FeatureGroup(name="DU to CU Paths", show=True)
    plot_path_on_map_with_curvature(du_cu_paths_group, du_cu_paths, cleaned_road_edges_data['road_edges'], color='black', ru_du_connections=du_to_cu_connections)
    du_cu_paths_group.add_to(m)
    
    add_grid_squares(grid_data, m, dataset)

    folium.LayerControl().add_to(m)
    map_legend_for_results(m)
    return m

m = create_map_with_layers(selected_rus=selected_rus, not_selected_rus=not_selected_rus, selected_dus=selected_dus, not_selected_dus=not_selected_dus)

combined_html_map_path = os.path.join(map_dir, f"{file_name}.html")
m.save(combined_html_map_path)
print(f"Combined map with connections has been saved to {combined_html_map_path}")
