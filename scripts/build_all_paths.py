"""
%┏━━━┓┏━━━┓┏━━━━┓┏┓━┏┓━━━━┏━━┓━┏┓━┏┓┏━━┓┏┓━━━┏━━━┓┏━━━┓┏━━━┓
?┃┏━┓┃┃┏━┓┃┃┏┓┏┓┃┃┃━┃┃━━━━┃┏┓┃━┃┃━┃┃┗┫┣┛┃┃━━━┗┓┏┓┃┃┏━━┛┃┏━┓┃
%┃┗━┛┃┃┃━┃┃┗┛┃┃┗┛┃┗━┛┃━━━━┃┗┛┗┓┃┃━┃┃━┃┃━┃┃━━━━┃┃┃┃┃┗━━┓┃┗━┛┃
?┃┏━━┛┃┗━┛┃━━┃┃━━┃┏━┓┃━━━━┃┏━┓┃┃┃━┃┃━┃┃━┃┃━┏┓━┃┃┃┃┃┏━━┛┃┏┓┏┛
%┃┃━━━┃┏━┓┃━┏┛┗┓━┃┃━┃┃━━━━┃┗━┛┃┃┗━┛┃┏┫┣┓┃┗━┛┃┏┛┗┛┃┃┗━━┓┃┃┃┗┓
?┗┛━━━┗┛━┗┛━┗━━┛━┗┛━┗┛━━━━┗━━━┛┗━━━┛┗━━┛┗━━━┛┗━━━┛┗━━━┛┗┛┗━┛
%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
?━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This script is used for building all of the shortest paths between the RUs, DUs and CUs in the network. If you make modifications to the dataset files for these, such as removing or adding devices, changing locations, or moving a device from existing to new or vice versa, you will need to run this script to regenerate the paths.

The paths made in this are specifically used within the greedy algorithm to find the best solution for the network. It uses the NetworkX library to create a graph of the network and find the shortest paths between devices based on their geographical locations.

Example usage:
    python build_all_paths.py
This will generate the paths and save them to JSON files in the dataset directory.

"""

import os, json, math
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Tuple

# Set the dataset you want to use, being either "small-dataset" or "full-dataset"
dataset = "small-dataset"

def load_json_data(file_name):
    """Load JSON data from a specified file."""
    with open(file_name, 'r') as file:
        return json.load(file)
    
def save_json_data(data, file_path):
    """Save JSON data to a specified file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)   
    print(f"JSON file saved to: {file_path}")

# Set base directory relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))   # scripts/
base_dir = os.path.dirname(current_dir)                    # project root
path_data_dir = os.path.join(base_dir, "dataset", dataset) # dataset directory

@dataclass
class NetworkElement:
    id: str
    latitude: float
    longitude: float
    type: str
    
    # Connection logic
    def connect(self, other_element: 'NetworkElement'):
        raise NotImplementedError("Connect method should be implemented by subclasses.")

@dataclass
class RadioUnit(NetworkElement):
    connected_DUs: List['DistributedUnit'] = field(default_factory=list)

    def connect(self, du: 'DistributedUnit'):
        self.connected_DUs.append(du)
        du.connected_RUs.append(self)
               
@dataclass
class DistributedUnit(NetworkElement):
    connected_RUs: List['RadioUnit'] = field(default_factory=list)
    connected_CUs: List['CentralUnit'] = field(default_factory=list)
    
    def connect(self, cu: 'CentralUnit'):
        self.connected_CUs.append(cu)
        cu.connected_DUs.append(self)

@dataclass
class CentralUnit(NetworkElement):
    connected_DUs: List['DistributedUnit'] = field(default_factory=list)
    connected_to_cn: List['CoreNetwork'] = field(default_factory=list)
    
    def connect(self, cn: 'CoreNetwork'):
        self.connected_to_cn.append(cn)
        cn.connected_CUs.append(self)

@dataclass
class CoreNetwork(NetworkElement):
    connected_CUs: List['CentralUnit'] = field(default_factory=list)

@dataclass
class RoadNode:
    id: int 
    latitude: float
    longitude: float
    
@dataclass
class RoadEdge:
    from_node: RoadNode  # Starting node
    to_node: RoadNode    # Ending node
    geometry: List[dict[str, float]]  # List of points with latitude and longitude
    length: float         # Calculated length of the edge

# Define the relative paths to the files
centralised_units = os.path.join(path_data_dir, "centralised_units.json")
distributed_units = os.path.join(path_data_dir, "distributed_units.json")
radio_units = os.path.join(path_data_dir, "radio_units.json")
road_nodes = os.path.join(path_data_dir, "road_nodes.json")
road_edges = os.path.join(path_data_dir, "road_edges.json")
radio_units_exist = os.path.join(path_data_dir, "radio_units_exist.json")
radio_units_new = os.path.join(path_data_dir, "radio_units_new.json")
distributed_units_exist = os.path.join(path_data_dir, "distributed_units_exist.json")
distributed_units_new = os.path.join(path_data_dir, "distributed_units_new.json")

# Load the JSON data
centralised_units = load_json_data(centralised_units)
distributed_units = load_json_data(distributed_units)
radio_units = load_json_data(radio_units)
road_nodes = load_json_data(road_nodes)
road_edges = load_json_data(road_edges)
radio_units_exist = load_json_data(radio_units_exist)
radio_units_new = load_json_data(radio_units_new)
distributed_units_exist = load_json_data(distributed_units_exist)
distributed_units_new = load_json_data(distributed_units_new)

# Initialise all the JSON data files
def instantiate_radio_units(radio_units_data):
    return [RadioUnit(id=ru['ru_name'], latitude=ru['latitude'], longitude=ru['longitude'], type='RadioUnit') for ru in radio_units_data]

def instantiate_distributed_units(distributed_units_data):
    return [DistributedUnit(id=du['du_name'], latitude=du['latitude'], longitude=du['longitude'], type='DistributedUnit') for du in distributed_units_data]

def instantiate_central_units(central_units_data):
    return [CentralUnit(id=cu['cu_name'], latitude=cu['latitude'], longitude=cu['longitude'], type='CentralUnit') for cu in central_units_data]

def instantiate_road_nodes(road_nodes_data):
    return [RoadNode(id=rd['id'], latitude=rd['latitude'], longitude=rd['longitude']) for rd in road_nodes_data]
    
def instantiate_road_edges(road_edges_data):
    return [RoadEdge(from_node=rd['from'], to_node=rd['to'], geometry=rd['geometry'], length=rd['length']) for rd in road_edges_data]

# Create objects for each class and store them in a list
radio_unit_objects = instantiate_radio_units(radio_units['radio_units'])
distributed_unit_objects = instantiate_distributed_units(distributed_units['distributed_units'])
central_unit_objects = instantiate_central_units(centralised_units['centralised_units'])
road_node_objects = instantiate_road_nodes(road_nodes['road_nodes'])
road_edge_objects = instantiate_road_edges(road_edges['road_edges'])

# Additional JSON data instantiation
radio_unit_existing_objects = instantiate_radio_units(radio_units_exist['radio_units_existing'])
radio_unit_new_objects = instantiate_radio_units(radio_units_new['radio_units_new'])
distributed_unit_existing_objects = instantiate_distributed_units(distributed_units_exist['distributed_units_existing'])
distributed_unit_new_objects = instantiate_distributed_units(distributed_units_new['distributed_units_new'])

G = nx.Graph()

for node in road_node_objects:
    G.add_node(node.id, pos=(node.latitude, node.longitude))
    
for edge in road_edge_objects:
    G.add_edge(edge.from_node, edge.to_node, length=edge.length, geometry=edge.geometry)
    
pos = nx.get_node_attributes(G, 'pos')

def haversine(coord1, coord2):
    """Calculate the Haversine distance between two coordinates."""
    R = 6371  # Radius of the Earth in km
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

def find_nearest_road_node(device_lat, device_lon, road_nodes):
    """Find the nearest road node to a given device location."""
    nearest_node = None
    min_distance = float('inf')
    for node in road_nodes:
        distance = haversine((device_lat, device_lon), (node.latitude, node.longitude))
        if distance < min_distance:
            min_distance = distance
            nearest_node = node
    return nearest_node

def add_device_to_graph(device, device_type, road_nodes, G):
    """Add a device to the graph and connect it to the nearest road node."""
    nearest_road_node = find_nearest_road_node(device.latitude, device.longitude, road_nodes)
    G.add_node(device.id, type=device_type, pos=(device.latitude, device.longitude))
    G.add_edge(device.id, nearest_road_node.id, type='device_connection')

for device in radio_unit_objects + distributed_unit_objects + central_unit_objects:
    add_device_to_graph(device, type(device).__name__, road_node_objects, G)

def find_nearest_device(source_device, target_devices: List[NetworkElement], road_node_objects) -> Tuple[NetworkElement, float]:
    """Find the nearest device to a given source device from a list of target devices."""
    nearest_device = None
    min_distance = float('inf')
    source_location = (source_device.latitude, source_device.longitude)

    for device in target_devices:
        target_location = (device.latitude, device.longitude)
        distance = haversine(source_location, target_location)
        if distance < min_distance:
            min_distance = distance
            nearest_device = device

    return nearest_device, min_distance

def find_shortest_path(G, source_device, target_device, road_nodes):
    """Find the shortest path between two devices in the graph, considering road nodes."""

    # If source_device and target_device are passed as IDs, fetch the full objects
    if isinstance(source_device, str) or isinstance(source_device, int):
        source_device = next(dev for dev in radio_unit_objects + distributed_unit_objects + central_unit_objects if dev.id == source_device)
    if isinstance(target_device, str) or isinstance(target_device, int):
        target_device = next(dev for dev in radio_unit_objects + distributed_unit_objects + central_unit_objects if dev.id == target_device)
    
    # Check if the source and target devices are at the same location
    if (source_device.latitude == target_device.latitude) and (source_device.longitude == target_device.longitude):
        # Find the nearest road node to this location
        nearest_road_node = find_nearest_road_node(source_device.latitude, source_device.longitude, road_nodes)
        return [source_device.id, nearest_road_node.id, target_device.id], 0  # Path via road node with 0 distance

    try:
        path = nx.shortest_path(G, source=source_device.id, target=target_device.id, weight='length')
        path_length = nx.shortest_path_length(G, source=source_device.id, target=target_device.id, weight='length')
        return path, path_length
    except nx.NetworkXNoPath:
        print(f"No path found between {source_device.id} and {target_device.id}")
        return None, None

def find_valid_paths(G, source_device, target_devices, road_node_objects):
    """Find valid paths from a source device to multiple target devices, considering road nodes."""
    valid_paths = []
    
    for target_device in target_devices:
        # Check for paths, considering same-location cases
        path, path_length = find_shortest_path(G, source_device, target_device, road_node_objects)
        
        if path is not None:
            valid_paths.append((target_device, path_length, path))
    
    # Sort the paths by their length
    valid_paths.sort(key=lambda x: x[1])
    
    return valid_paths

def configure_network_paths(G, radio_units, distributed_units, central_units):
    """Configure paths between Radio Units (RUs), Distributed Units (DUs), and Central Units (CUs) in the network."""
    ru_du_paths = []  # To store RU to DU paths
    du_cu_paths = []  # To store DU to CU paths

    # Connect RUs to DUs and store the paths
    for ru in radio_units:
        valid_paths = find_valid_paths(G, ru, distributed_units, road_node_objects)
        if valid_paths:
            for du, path_length, path in sorted(valid_paths, key=lambda x: x[0].id):
                ru_du_paths.append({
                    'ru_name': ru.id,
                    'du_name': du.id,
                    'path': path
                })

    # Connect DUs to CUs and store the paths
    for du in distributed_units:
        valid_paths = find_valid_paths(G, du, central_units, road_node_objects)
        if valid_paths:
            nearest_cu, path_length, path = valid_paths[0]
            du.connect(nearest_cu)
            du_cu_paths.append({
                'du_name': du.id,
                'cu_name': nearest_cu.id,
                'path': path
            })
    return ru_du_paths, du_cu_paths

def generate_all_data():
    """Generate all data from the dataset files and return a dictionary of data."""
    # Generate all required path combinations
    ru_du_paths, du_cu_paths = configure_network_paths(G, radio_unit_objects, distributed_unit_objects, central_unit_objects)

    # RU-DU combinations
    ru_du_paths_new, _ = configure_network_paths(G, radio_unit_new_objects, distributed_unit_objects, central_unit_objects)
    ru_du_paths_exist, _ = configure_network_paths(G, radio_unit_existing_objects, distributed_unit_objects, central_unit_objects)
    ru_du_paths_exist_graph, _ = configure_network_paths(G, radio_unit_existing_objects, distributed_unit_existing_objects, central_unit_objects)

    # DU-CU combinations
    _, du_cu_paths_exist = configure_network_paths(G, radio_unit_objects, distributed_unit_existing_objects, central_unit_objects)
    _, du_cu_paths_new = configure_network_paths(G, radio_unit_objects, distributed_unit_new_objects, central_unit_objects)

    # Save all generated paths to JSON
    ru_du_paths_json = os.path.join(path_data_dir, 'ru_du_path.json')
    du_cu_paths_json = os.path.join(path_data_dir, 'du_cu_path.json')
    ru_du_paths_new_json = os.path.join(path_data_dir, 'ru_du_path_new.json')
    ru_du_paths_exist_json = os.path.join(path_data_dir, 'ru_du_path_exist.json')
    ru_du_paths_exist_graph_json = os.path.join(path_data_dir, 'ru_du_path_exist_graph.json')
    du_cu_paths_exist_json = os.path.join(path_data_dir, 'du_cu_path_exist.json')
    du_cu_paths_new_json = os.path.join(path_data_dir, 'du_cu_path_new.json')

    save_json_data(ru_du_paths, ru_du_paths_json)
    save_json_data(du_cu_paths, du_cu_paths_json)
    save_json_data(ru_du_paths_new, ru_du_paths_new_json)
    save_json_data(ru_du_paths_exist, ru_du_paths_exist_json)
    save_json_data(ru_du_paths_exist_graph, ru_du_paths_exist_graph_json)
    save_json_data(du_cu_paths_exist, du_cu_paths_exist_json)
    save_json_data(du_cu_paths_new, du_cu_paths_new_json)

generate_all_data()
print("Network paths configured and saved successfully.")