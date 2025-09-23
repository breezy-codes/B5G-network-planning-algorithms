"""
%┏┓━┏┓┏━━━┓┏┓━━━┏━━━┓┏━━━┓┏━━━┓━━━━━┏━━━┓┏┓━┏┓┏━┓━┏┓┏━━━┓
?┃┃━┃┃┃┏━━┛┃┃━━━┃┏━┓┃┃┏━━┛┃┏━┓┃━━━━━┃┏━━┛┃┃━┃┃┃┃┗┓┃┃┃┏━┓┃
%┃┗━┛┃┃┗━━┓┃┃━━━┃┗━┛┃┃┗━━┓┃┗━┛┃━━━━━┃┗━━┓┃┃━┃┃┃┏┓┗┛┃┃┃━┗┛
?┃┏━┓┃┃┏━━┛┃┃━┏┓┃┏━━┛┃┏━━┛┃┏┓┏┛━━━━━┃┏━━┛┃┃━┃┃┃┃┗┓┃┃┃┃━┏┓
%┃┃━┃┃┃┗━━┓┃┗━┛┃┃┃━━━┃┗━━┓┃┃┃┗┓━━━━┏┛┗┓━━┃┗━┛┃┃┃━┃┃┃┃┗━┛┃
?┗┛━┗┛┗━━━┛┗━━━┛┗┛━━━┗━━━┛┗┛┗━┛━━━━┗━━┛━━┗━━━┛┗┛━┗━┛┗━━━┛
%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
?━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This script contains helper functions for the map visualisation of results from greedy, genetic, local search, and CPLEX algorithms.

"""

import folium, json
import numpy as np
from map_modules.config import get_color_by_demand

def load_json_data(file_name):
    """Loads JSON data from a file."""
    with open(file_name, 'r') as file:
        return json.load(file)

def add_feature_group(data, map_object, layer_name, get_color_function, marker_type, selected_ids, not_selected_ids, service_attribute=None):
    """Adds a feature group to the map with markers based on the provided data."""
    feature_group = folium.FeatureGroup(name=layer_name, show=True)
    
    for location in data:
        unit_id = location.get('ru_name') or location.get('du_name') or location.get('cu_name')
        if unit_id in selected_ids:
            attribute_value = 'selected'
            marker_color = get_color_function(attribute_value)
            popup_content = f"ID: {unit_id}, Status: {attribute_value}"
        elif unit_id in not_selected_ids:
            attribute_value = 'not_selected'
            marker_color = get_color_function(attribute_value)
            popup_content = f"ID: {unit_id}, Status: {attribute_value}"
        else:
            continue
        
        if marker_type == 'circle':
            folium.CircleMarker(location=[location['latitude'], location['longitude']], radius=4, color=marker_color, fill=True, fill_color=marker_color, fill_opacity=1, popup=popup_content).add_to(feature_group)
        elif marker_type == 'triangle':
            folium.RegularPolygonMarker(location=[location['latitude'], location['longitude']], number_of_sides=3, radius=7, color=marker_color, fill=True, fill_color=marker_color, fill_opacity=1, popup=popup_content).add_to(feature_group)
        elif marker_type == 'pentagon':
            folium.RegularPolygonMarker(location=[location['latitude'], location['longitude']], number_of_sides=5, radius=9, color=marker_color, fill=True, fill_color=marker_color, fill_opacity=1, popup=popup_content).add_to(feature_group)
        elif marker_type == 'hexagon':
            folium.RegularPolygonMarker(location=[location['latitude'], location['longitude']], number_of_sides=6, radius=14, color=marker_color, fill=True, fill_color=marker_color, fill_opacity=1, popup=popup_content).add_to(feature_group)
    
    feature_group.add_to(map_object)

def add_radio_unit_radii(data, map_object, selected_ids, not_selected_ids):
    """Adds radio unit radii to the map with different colors for selected and not selected units."""
    selected_feature_group = folium.FeatureGroup(name="Selected Radio Unit Radii", show=True)
    unselected_feature_group = folium.FeatureGroup(name="Unselected Radio Unit Radii", show=False)
    
    for location in data:
        unit_id = location.get('ru_name')
        status = location.get('status', 'N/A').strip().lower()
        radius = location.get('range_m')
        color = 'purple' if unit_id in selected_ids else 'darkpurple' if unit_id in not_selected_ids else 'red'
        
        circle = folium.Circle(location=[location['latitude'], location['longitude']], radius=radius, color=color, fill=True, fill_opacity=0.05, weight=1)
        
        if unit_id in selected_ids: circle.add_to(selected_feature_group)
        elif unit_id in not_selected_ids: circle.add_to(unselected_feature_group)
    
    selected_feature_group.add_to(map_object)
    unselected_feature_group.add_to(map_object)
    
def add_cleaned_roads_to_map(road_data, map_object):
    """Adds cleaned road paths to the map."""
    feature_group = folium.FeatureGroup(name="Road Paths", show=True)  # Enable toggle
    for edge in road_data['road_edges']:
        coordinates = [[point['latitude'], point['longitude']] for point in edge['geometry']]
        folium.PolyLine(coordinates, color='blue', weight=1.5, opacity=0.8).add_to(feature_group)
    feature_group.add_to(map_object) 

def add_grid_squares(grid_data, map_object, dataset):
    """Adds grid squares to the map based on demand points."""
    feature_group = folium.FeatureGroup(name="Grid Squares", show=False)

    for location in grid_data['demand_points']:
        center_lat = location['latitude']
        center_lon = location['longitude']
        size = location.get('size', 500)
        users = location.get('users', 0)
        demand_id = location.get('demand_id', 'N/A')
        half_side = size / 2 / 111320
        half_side_lon = size / 2 / (40075000 * np.cos(np.radians(center_lat)) / 360)
        square = [[center_lat - half_side, center_lon - half_side_lon], [center_lat - half_side, center_lon + half_side_lon], [center_lat + half_side, center_lon + half_side_lon], [center_lat + half_side, center_lon - half_side_lon], [center_lat - half_side, center_lon - half_side_lon]]
        color = get_color_by_demand(users, dataset)
        folium.Polygon(locations=square,  color=color,  fill=True,  fill_opacity=0.2,).add_to(feature_group)
           
    feature_group.add_to(map_object)

def map_legend_for_results(m):
    """Adds a legend to the map with custom shapes for different units for the results."""

    legend_html = '''
    <style>
        .legend-circle { 
            width: 15px; height: 15px; border-radius: 50%; display: inline-block; 
        }
        .legend-square { 
            width: 15px; height: 15px; display: inline-block; 
        }
        .legend-line { 
            width: 15px; height: 4px; display: inline-block; 
        }
        .legend-pentagon1 {
            width: 15px; height: 15px; background-color: green;
            clip-path: polygon(50% 0%, 100% 38%, 82% 100%, 18% 100%, 0% 38%);
            display: inline-block;
        }
        .legend-pentagon2 {
            width: 15px; height: 15px; background-color: blue;
            clip-path: polygon(50% 0%, 100% 38%, 82% 100%, 18% 100%, 0% 38%);
            display: inline-block;
        }
        .legend-hexagon {
            width: 15px; height: 15px; background-color: black;
            clip-path: polygon(25% 0%, 75% 0%, 100% 50%, 75% 100%, 25% 100%, 0% 50%);
            display: inline-block;
        }
    </style>
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 180px; height: 280px; 
                background-color: white; z-index:9999; font-size:13px;
                border:2px solid grey; padding: 10px;">
    <b>Legend</b><br>
    <div><span class="legend-circle" style="background-color:red"></span> RUs (Selected)</div>
    <div><span class="legend-circle" style="background-color:#710E9D"></span> RUs (Not Selected)</div>
    <div><span class="legend-pentagon2" style="background-color:blue"></span> DUs (Selected)</div>
    <div><span class="legend-pentagon1" style="background-color:green"></span> DUs (Not Selected)</div>
    <div><span class="legend-hexagon"></span> Centralised Units</div>
    <div><span class="legend-line" style="background-color:#BD33A3"></span> Existing Paths</div>
    <div><span class="legend-line" style="background-color:black"></span> Chosen Paths</div>
    <div><span class="legend-line" style="background-color:blue"></span> Road Paths</div>
    <div><span class="legend-line" style="background-color:red"></span> Region Boundary</div>
    <div><span class="legend-square" style="background-color:green"></span> Low Demand</div>
    <div><span class="legend-square" style="background-color:orange"></span> Medium Demand</div>
    <div><span class="legend-square" style="background-color:red"></span> High Demand</div>
    </div>
    '''
    m.get_root().add_child(folium.Element(legend_html))

def map_legend_for_legend(m):
    """Adds a legend to the map with custom shapes for different units for the legend."""

    legend_html = '''
    <style>
        .legend-circle { 
            width: 15px; height: 15px; border-radius: 50%; display: inline-block; 
        }
        .legend-square { 
            width: 15px; height: 15px; display: inline-block; 
        }
        .legend-line { 
            width: 15px; height: 5px; display: inline-block; 
        }
        .legend-pentagon1 {
            width: 15px; height: 15px; background-color: green;
            clip-path: polygon(50% 0%, 100% 38%, 82% 100%, 18% 100%, 0% 38%);
            display: inline-block;
        }
        .legend-pentagon2 {
            width: 15px; height: 15px; background-color: blue;
            clip-path: polygon(50% 0%, 100% 38%, 82% 100%, 18% 100%, 0% 38%);
            display: inline-block;
        }
        .legend-hexagon {
            width: 15px; height: 15px; background-color: black;
            clip-path: polygon(25% 0%, 75% 0%, 100% 50%, 75% 100%, 25% 100%, 0% 50%);
            display: inline-block;
        }
    </style>
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 180px; height: 280px; 
                background-color: white; z-index:9999; font-size:13px;
                border:2px solid grey; padding: 10px;">
    <b>Legend</b><br>
    <div><span class="legend-circle" style="background-color:red"></span> RUs (Existing)</div>
    <div><span class="legend-circle" style="background-color:purple"></span> RUs (Candidate)</div>
    <div><span class="legend-pentagon2" style="background-color:blue"></span> DUs (Existing)</div>
    <div><span class="legend-pentagon1" style="background-color:green"></span> DUs (Candidate)</div>
    <div><span class="legend-hexagon"></span> Centralised Units</div>
    <div><span class="legend-line" style="background-color:black"></span> Existing Paths</div>
    <div><span class="legend-line" style="background-color:blue"></span> Road Paths</div>
    <div><span class="legend-line" style="background-color:red"></span> Region Boundary</div>
    <div><span class="legend-square" style="background-color:green"></span> Low Demand</div>
    <div><span class="legend-square" style="background-color:orange"></span> Medium Demand</div>
    <div><span class="legend-square" style="background-color:red"></span> High Demand</div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))