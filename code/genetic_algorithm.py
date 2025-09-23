"""
%┏━━━┓┏━━━┓┏━┓━┏┓┏━━━┓┏━━━━┓┏━━┓┏━━━┓━━━━┏━━━┓┏┓━━━┏━━━┓┏━━━┓
?┃┏━┓┃┃┏━━┛┃┃┗┓┃┃┃┏━━┛┃┏┓┏┓┃┗┫┣┛┃┏━┓┃━━━━┃┏━┓┃┃┃━━━┃┏━┓┃┃┏━┓┃
%┃┃━┗┛┃┗━━┓┃┏┓┗┛┃┃┗━━┓┗┛┃┃┗┛━┃┃━┃┃━┗┛━━━━┃┃━┃┃┃┃━━━┃┃━┗┛┃┃━┃┃
?┃┃┏━┓┃┏━━┛┃┃┗┓┃┃┃┏━━┛━━┃┃━━━┃┃━┃┃━┏┓━━━━┃┗━┛┃┃┃━┏┓┃┃┏━┓┃┃━┃┃
%┃┗┻━┃┃┗━━┓┃┃━┃┃┃┃┗━━┓━┏┛┗┓━┏┫┣┓┃┗━┛┃━━━━┃┏━┓┃┃┗━┛┃┃┗┻━┃┃┗━┛┃
?┗━━━┛┗━━━┛┗┛━┗━┛┗━━━┛━┗━━┛━┗━━┛┗━━━┛━━━━┗┛━┗┛┗━━━┛┗━━━┛┗━━━┛
%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
?━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This is the main script for running the genetic algorithm on the given dataset.
It sets up the environment, loads the necessary data, initialises the units, and runs the genetic algorithm.
It can run the algorithm with or without a time limit, depending on the requirements.

To use, set the `dataset` variable to either "small-dataset" or "full-dataset". Or load your own dataset.
You can also adjust the parameters for the genetic algorithm to change its behaviour.

If you want to use a time limit, set the `time_limit_minutes` variable to the desired number of minutes. Otherwise, set it to None.
If set to None, the algorithm will run for the specified number of iterations.

%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os, json 
from modules import config
from modules.data_classes.dataclass_CU import CentralisedUnit
from modules.data_classes.dataclass_DU import DistributedUnit
from modules.data_classes.dataclass_roads import RoadGraph
from modules.data_classes.dataclass_RU import RadioUnit
from modules.data_classes.dataclass_user import User
from modules.algorithms.genetic_algorithm import save_genetic_algorithm

# Set the dataset you want to use, being either "small-dataset" or "full-dataset"
dataset = "small-dataset"

# Default run configuration
run = 1
max_iterations = 250
num_neighbours = 10
num_mutations = 3
num_solutions = 20
crossover_interval = 50
elite_pop = 2
num_clones = 2
crossover_start = 2
crossover_end = None
save_interval = 500
time_limit_minutes = None

# Create a unique run ID based on the parameters
config_str = f'conf_{num_solutions}_{num_neighbours}_{num_mutations}_{crossover_interval}_{elite_pop}_{num_clones}'
run_id = f"{run}_{config.RUN_ID}_{config_str}"

# Predefined best settings for small dataset - use these for best results
# Uncomment this block to override the defaults above
"""
run = 1
max_iterations = 1000
num_neighbours = 15
num_mutations = 3
num_solutions = 25
crossover_interval = 100
elite_pop = 2
num_clones = 2
crossover_start = 1
crossover_end = None
save_interval = 500
time_limit_minutes = None

config_str = f'conf_{num_solutions}_{num_neighbours}_{num_mutations}_{crossover_interval}_{elite_pop}_{num_clones}'
run_id = f"{run}_{config.RUN_ID}_{config_str}"
"""

# Set the output directory and paths for the scenario and results
# The scenario directory matches up to the dataset which is set above
# The results directory is set to the results folder

# Set base directory relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))  # code/
base_dir = os.path.dirname(current_dir)                   # project root
 
# Dataset path
scenario = os.path.join(base_dir, "dataset", dataset)

# Results + log path
results_dir = os.path.join(current_dir, "results")
output_dir = results_dir 

greedy_solution_path = os.path.join(results_dir, f"greedy_short_{dataset}_{config.RUN_ID}_all.json")

with open(os.path.join(scenario, 'radio_units_new.json')) as ru_new_file: ru_data_new = json.load(ru_new_file)
with open(os.path.join(scenario, 'radio_units_exist.json')) as ru_existing_file: ru_data_existing = json.load(ru_existing_file)
with open(os.path.join(scenario, 'distributed_units_new.json')) as du_new_file: du_data_new = json.load(du_new_file)
with open(os.path.join(scenario, 'distributed_units_exist.json')) as du_existing_file: du_data_existing = json.load(du_existing_file)
with open(os.path.join(scenario, 'centralised_units.json')) as cu_file: cu_data = json.load(cu_file)
with open(os.path.join(scenario, 'ru_du_path_new.json')) as ru_du_new_file: ru_du_new_paths = json.load(ru_du_new_file)
with open(os.path.join(scenario, 'ru_du_path_exist.json')) as ru_du_existing_file: ru_du_existing_paths = json.load(ru_du_existing_file)
with open(os.path.join(scenario, 'ru_du_path_exist_graph.json')) as ru_du_existing_file_graph: ru_du_existing_paths_graph = json.load(ru_du_existing_file_graph)
with open(os.path.join(scenario, 'du_cu_path_new.json')) as du_cu_new_file: du_cu_paths_new = json.load(du_cu_new_file)
with open(os.path.join(scenario, 'du_cu_path_exist.json')) as du_cu_existing_file: du_cu_paths_existing = json.load(du_cu_existing_file)
with open(os.path.join(scenario, 'road_distances.json')) as road_file: road_distances = json.load(road_file)
with open(os.path.join(scenario, 'user_map.json')) as users_file: user_ru_mapping = json.load(users_file)
with open(os.path.join(scenario, 'road_edges.json')) as road_edges_file: road_edges = json.load(road_edges_file)['road_edges']

DistributedUnit.initialise_units(du_data_new, du_data_existing) # Initialise DUs
DU_names_new = DistributedUnit.DUs_new            # New DUs
DU_names_existing = DistributedUnit.DUs_existing  # Existing DUs
DUs = {**DU_names_new, **DU_names_existing}       # Combine both new and existing DUs
DU_names = list(DUs.keys()) # All DUs

RUs = RadioUnit.initialise_units(ru_data_new, ru_data_existing) # Initialise RUs
RU_names_new = RadioUnit.RUs_new            # New RUs
RU_names_existing = RadioUnit.RUs_existing  # Existing RUs
RUs = {**RU_names_new, **RU_names_existing} # Combine both new and existing RUs
RU_names = list(RUs.keys()) # All RUs

CentralisedUnit.initialise_units(cu_data)
CUs = CentralisedUnit.CUs   # Initialise CUs
CU_names = list(CUs.keys()) # All CUs

users = User.initialise_users(user_ru_mapping, RUs) # Initialise users and map them to RUs

graph = RoadGraph(road_edges) # Create a road graph from the road edges

# Connect RUs, DUs, and CUs to the nearest road nodes
for ru in RUs.values():
    graph.connect_device_to_road(ru.name, (ru.latitude, ru.longitude))

for du in DUs.values():
    graph.connect_device_to_road(du.name, (du.latitude, du.longitude))

for cu in CUs.values():
    graph.connect_device_to_road(cu.name, (cu.latitude, cu.longitude))

# Add paths to the graph
graph.mark_existing_path(ru_du_existing_paths_graph)
graph.mark_existing_path(du_cu_paths_existing)

# Precompute shortest paths for the graph
graph.precompute_shortest_paths()

if __name__ == "__main__":
    save_genetic_algorithm(greedy_solution_path, RUs, DUs, CUs, users, output_dir, graph, scenario, run_id, max_iterations=max_iterations, num_neighbours=num_neighbours, num_mutations=num_mutations,  num_solutions=num_solutions, crossover_interval=crossover_interval, save_interval=save_interval,elite_pop=elite_pop, num_clones=num_clones, time_limit_minutes=time_limit_minutes)