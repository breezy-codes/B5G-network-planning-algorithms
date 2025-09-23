"""
%┏━━━┓┏━━━┓┏┓━━━┏━━━┓┏━┓┏━┓━━━━┏━━━┓┏┓━┏┓┏━━━┓┏━━━┓┏━━━━┓
?┃┏━┓┃┃┏━┓┃┃┃━━━┃┏━━┛┗┓┗┛┏┛━━━━┃┏━┓┃┃┃━┃┃┃┏━┓┃┃┏━┓┃┃┏┓┏┓┃
%┃┃━┗┛┃┗━┛┃┃┃━━━┃┗━━┓━┗┓┏┛━━━━━┃┗━━┓┃┗━┛┃┃┃━┃┃┃┗━┛┃┗┛┃┃┗┛
?┃┃━┏┓┃┏━━┛┃┃━┏┓┃┏━━┛━┏┛┗┓━━━━━┗━━┓┃┃┏━┓┃┃┃━┃┃┃┏┓┏┛━━┃┃━━
%┃┗━┛┃┃┃━━━┃┗━┛┃┃┗━━┓┏┛┏┓┗┓━━━━┃┗━┛┃┃┃━┃┃┃┗━┛┃┃┃┃┗┓━┏┛┗┓━
?┗━━━┛┗┛━━━┗━━━┛┗━━━┛┗━┛┗━┛━━━━┗━━━┛┗┛━┗┛┗━━━┛┗┛┗━┛━┗━━┛━
%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
?━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This is the code for running the shortest path model using CPLEX.
It is the coded version of the mathematical model described in the paper.

You can run this model on both the small and full datasets.

This script will save the CPLEX log into the end of the results file so its combined with the results.

At the end it also tidies up the generated files from CPLEX, and moves the .pulp and .sol file into a directory called cplex-generated-files within the results directory.

NOTE - 
In order to run this, you will need CPLEX installed and the path to the CPLEX executable set in the CPLEX_CMD class in the pulp library.

"""

import json, os
from pulp import *
from collections import defaultdict
from modules import config
from modules.save_cplex_results import save_ilp_results_to_file, tidy_working_directory

coverage_low = config.COVERAGE_THRESHOLD
coverage_high = min(coverage_low + 0.05, 1.0) # Sets the coverage high to be 5% more than the coverage low, but not more than 100%

# Set the dataset you want to use, being either "small-dataset" or "full-dataset"
dataset = "small-dataset"

# Create a unique run ID based on the parameters, this is used to identify the run in the results
run_id = f"short_{dataset}_{coverage_low}_{config.UM_VALUE}_{config.MR_VALUE}_{config.MD_VALUE}_{config.CR_VALUE}"

# Set base directory relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))  # code/
base_dir = os.path.dirname(current_dir)                   # project root
 
# Dataset path
scenario = os.path.join(base_dir, "dataset", dataset)

# Results + log path
results_dir = os.path.join(current_dir, "results")
cplex_files_dir = os.path.join(results_dir, "cplex-generated-files")
os.makedirs(cplex_files_dir, exist_ok=True)

log_path = os.path.join(results_dir, "cplex_log.log")

with open(log_path, 'w') as f:
    f.write("")

username = os.getlogin()  # gets the current user
# Make sure to update the path to your CPLEX installation
cplex_path = f"/home/{username}/CPLEX_Studio2212/cplex/bin/x86-64_linux/cplex"

cplex_solver = CPLEX_CMD(
    path=cplex_path,
    msg=True,
    logPath=f'{log_path}',
    #timeLimit=700,
    #warmStart=True,
    keepFiles=True,
    options=[
        'set mip display 4',
        'set mip tolerances mipgap 0.5',
        'set mip strategy variableselect 3', 
    ],
)

with open(os.path.join(scenario, 'radio_units_new.json')) as ru_new_file: ru_data_new = json.load(ru_new_file)
with open(os.path.join(scenario, 'radio_units_exist.json')) as ru_existing_file: ru_data_existing = json.load(ru_existing_file)
with open(os.path.join(scenario, 'distributed_units_new.json')) as du_new_file: du_data_new = json.load(du_new_file)
with open(os.path.join(scenario, 'distributed_units_exist.json')) as du_existing_file: du_data_existing = json.load(du_existing_file)
with open(os.path.join(scenario, 'centralised_units.json')) as cu_file: cu_data = json.load(cu_file)
with open(os.path.join(scenario, 'ru_du_path_new.json')) as ru_du_new_file: ru_du_new_paths = json.load(ru_du_new_file)
with open(os.path.join(scenario, 'ru_du_path_exist.json')) as ru_du_existing_file: ru_du_existing_paths = json.load(ru_du_existing_file)
with open(os.path.join(scenario, 'ru_du_path_exist_graph.json')) as ru_du_existing_graph_file: ru_du_existing_paths_graph = json.load(ru_du_existing_graph_file)
with open(os.path.join(scenario, 'du_cu_path_new.json')) as du_cu_new_file: du_cu_paths_new = json.load(du_cu_new_file)
with open(os.path.join(scenario, 'du_cu_path_exist.json')) as du_cu_existing_file: du_cu_paths_existing = json.load(du_cu_existing_file)
with open(os.path.join(scenario, 'road_distances.json')) as road_file: road_distances = json.load(road_file)
with open(os.path.join(scenario, 'user_map.json')) as users_file: user_ru_mapping = json.load(users_file)
with open(os.path.join(scenario, 'ru_du_existing_mappings.json')) as existing_mappings_file: existing_mappings = json.load(existing_mappings_file)

# Extract RU, DU, CU data
RUs_new = {ru['ru_name']: {'RC': ru['capacity_bandwidth']} for ru in ru_data_new['radio_units_new']}
RUs_existing = {ru['ru_name']: {'RC': ru['capacity_bandwidth']} for ru in ru_data_existing['radio_units_existing']}

CUs = {cu['cu_name']: {'CC': cu['capacity_bandwidth'], 'CP': cu['capacity_ports']} for cu in cu_data['centralised_units']}

# Combine new and existing DUs
DUs_new = {du['du_name']: {'DC': du['capacity_bandwidth'], 'DP': du['capacity_ports']} for du in du_data_new['distributed_units_new']}
DUs_existing = {du['du_name']: {'DC': du['capacity_bandwidth'], 'DP': du['capacity_ports']} for du in du_data_existing['distributed_units_existing']}

# Merge both new and existing DUs into one dictionary
DUs = {**DUs_new, **DUs_existing}
RUs = {**RUs_new, **RUs_existing}

# Create sets for each device type
RU_names_new = list(RUs_new.keys())  # New RUs
RU_names_existing = list(RUs_existing.keys())  # Existing RUs
RU_names = list(RUs.keys())  # All RUs (new + existing)

DU_names_new = list(DUs_new.keys())  # new DUs
DU_names_existing = list(DUs_existing.keys())  # existing DUs
DU_names = list(DUs.keys())  # All DUs (new + existing)

CU_names = list(CUs.keys())

# User to RU Mapping and adding coverage rate
ur_map = {user['user_id']: user['assigned_ru'] for user in user_ru_mapping}
user_ids = ur_map.keys()    # List of user IDs
total_users = len(user_ids) # Total number of users
UC_low = coverage_low * total_users    # Minimum number of users to cover
UC_high = coverage_high * total_users  # Maximum number of users to cover
ur_rng = {u: [r for r in RU_names if r in ur_map[u]] for u in user_ids}

# Convert the road distances into a dictionary with both directions
road_distance_dict = {}
for entry in road_distances:
    road_distance_dict[(entry['from'], entry['to'])] = entry['length']
    road_distance_dict[(entry['to'], entry['from'])] = entry['length']

# Store the segments and their costs
KS = defaultdict(int)

def add_path_segments_to_costs(path, cost=config.KY_COST):
    """Add path segments to the cost dictionary with the specified cost."""
    for i in range(len(path) - 1):
        s = (path[i], path[i + 1])
        reverse_segment = (path[i + 1], path[i])
        # Check both segment and reverse_segment
        if s in road_distance_dict:
            KS[s] = road_distance_dict[s] * cost
        elif reverse_segment in road_distance_dict:
            KS[reverse_segment] = road_distance_dict[reverse_segment] * cost
            
for conn in ru_du_new_paths: add_path_segments_to_costs(conn['path'], cost=config.KY_COST)

for conn in du_cu_paths_new:
    add_path_segments_to_costs(conn['path'], cost=config.KY_COST)

# First add NEW paths using KY
for conn in ru_du_existing_paths:
    ru_name = conn['ru_name']
    du_name = conn['du_name']
    path = conn['path']

    if du_name not in existing_mappings or ru_name not in existing_mappings[du_name]:
        add_path_segments_to_costs(path, cost=config.KY_COST)

# Then override EXISTING paths using KM
for conn in ru_du_existing_paths:
    ru_name = conn['ru_name']
    du_name = conn['du_name']
    path = conn['path']

    if du_name in existing_mappings and ru_name in existing_mappings[du_name]:
        add_path_segments_to_costs(path, cost=config.KM_COST)
        
for conn in du_cu_paths_existing:
    add_path_segments_to_costs(conn['path'], cost=config.KM_COST)

# ============== Decision Variables ==============

# Decision variable A is α for α_{u,r} alpha
# Binary decision variable for if a user is connected to a RU
A = LpVariable.dicts("A", ((u, r) for u in user_ids for r in ur_map[u]), cat=LpBinary)

# Decision variable B is β for β_{r,d} beta
# Binary decision variable for if a RU is connected to a DU
B = LpVariable.dicts("B", [(r, d) for r in RU_names for d in DU_names], cat=LpBinary)

# Decision variable C is γ for γ_{d,c} gamma
# Binary decision variable for if a DU is connected to a CU
C = LpVariable.dicts("C", [(d, c) for d in DU_names for c in CU_names], cat=LpBinary)

# Decision variable D is δ for δ_{r} delta
# Binary decision variable for if a RU is placed
D = LpVariable.dicts("D", RU_names, cat=LpBinary)

# Decision variable E is ε for ε_{d} zeta
# Binary decision variable for if a DU is placed
E = LpVariable.dicts("E", DU_names, cat=LpBinary)

# Decision variable I is ξ for ξ_{s} xi
# Binary decision variable for if a segment is used
I = LpVariable.dicts("I", KS.keys(), cat=LpBinary)

# Decision variable J is ϖ for ϖ_{r} varpi
# Integer decision variable for the number of RUs at each location
J = LpVariable.dicts("J", RU_names, lowBound=0, upBound=config.MR_VALUE, cat='Integer')

# Decision variable K is μ for μ_{r,d} mu
# Integer decision variable for the maximum number of RUs at each location
K = LpVariable.dicts("K", [(r, d) for r in RU_names for d in DU_names], lowBound=0, upBound=config.MR_VALUE, cat="Integer")

# Decision variable L is θ for θ_{d} theta
# Integer decision variable for the number of DUs at each location
L = LpVariable.dicts("L", DU_names, lowBound=0, upBound=config.MD_VALUE, cat="Integer")

# Decision variable M is η for η_{d,c} eta
# Integer decision variable for the number of DUs at each location
M = LpVariable.dicts("M", [(d, c) for d in DU_names for c in CU_names], lowBound=0, upBound=config.MD_VALUE, cat="Integer")

# Decision variable N is ν for ν_{d,c} nu
# Integer decision variable for scaling H_{d,c} to the value of L_{d}
N = LpVariable.dicts("N", [(d, c) for d in DU_names for c in CU_names], lowBound=0, cat='Integer')

# ============= Objective Function ==============

model = LpProblem(f"{run_id}", LpMinimize)

Segment_cost = lpSum(I[s] * KS[s] for s in KS.keys())
RU_cost = lpSum(J[r] * config.KV_COST for r in RU_names)
DU_cost = ((lpSum(E[d] * config.KW_COST for d in DUs_new) + (lpSum(L[d] - E[d]) * config.KX_COST for d in DU_names)) + (lpSum(K[(r, d)] * config.KL_COST for r in RU_names for d in DU_names)))
CU_cost = lpSum(M[(d, c)] * config.KB_COST for d in DU_names for c in CU_names)

# Other Tracking Variables
used_RU_capacity = {r: lpSum(A[(u, r)] * config.UM_VALUE for u in user_ids if r in ur_map[u]) for r in RU_names}
used_DU_capacity_bandwidth = {d: lpSum(K[(r, d)] * RUs[r]['RC'] for r in RU_names) for d in DU_names}
used_DU_capacity_ports = {d: lpSum(K[(r, d)] for r in RU_names) for d in DU_names}

# Objective function as the sum of the individual cost components
model += Segment_cost + RU_cost + DU_cost + CU_cost, "Total_Cost"

# ============= Constraints ==============

# User and Coverage Requirements
for u in user_ids:
    if ur_rng[u]:
        # Ensure that the user is connected to at least one RU
        model += lpSum(A[(u, r)] for r in ur_rng[u]) <= 1, f"{u}_connectivity"
        
        # Ensure that the RU is activated if it's covering the user
        for r in ur_rng[u]:
            model += A[(u, r)] <= D[r], f"{u}_{r}_activation"

# Minimum User Coverage Constraint, and upper bound
model += lpSum(lpSum(A[(u, r)] for r in ur_rng[u]) for u in user_ids) >= UC_low, f"min_UM_{r}_{u}"
model += lpSum(lpSum(A[(u, r)] for r in ur_rng[u]) for u in user_ids) <= UC_high, f"max_UM_{r}_{u}"

# Device Activation Requirements
# RU to DU connectivity and activation constraints
for r in RU_names:
    model += lpSum(B[(r, d)] for d in DU_names) == D[r], f"{r}_activation"
    for d in DU_names:
        model += B[(r, d)] <= E[d], f"{r}_{d}_activation"

# DU to CU connectivity constraints
for d in DU_names:
    model += lpSum(C[(d, c)] for c in CU_names) == E[d], f"{d}_activation"
    for c in CU_names:
        model += C[(d, c)] <= L[d], f"{d}_{c}_activation"

# RUs per location Requirements
for r in RU_names:
    for d in DU_names:
        model += lpSum(K[(r, d)] for d in DU_names) == J[r], f"set_{r}_{d}_K"
        model += K[(r, d)] <= config.MR_VALUE * B[(r, d)], f"{r}_{d}_K_upperbound_B"

# Dus per location Requirements
for d in DU_names:
    for c in CU_names:
        model += lpSum(M[(d, c)] for c in CU_names) == L[d], f"set_{d}_M"
        model += M[(d, c)] <= config.MD_VALUE * C[(d, c)], f"{d}_{c}_M_upperbound_B"

# Device Capacity Requirements
for r in RU_names:
    model += lpSum(A[(u, r)] * config.UM_VALUE for u in user_ids if r in ur_map[u]) <= RUs[r]['RC'] * J[r], f"{r}_capacity"

# DU bandwidth and port capacity constraints
for d in DU_names:
    model += lpSum(K[(r, d)] * RUs[r]['RC'] for r in RU_names) <= M[(d,c)] * DUs[d]['DC'], f"{d}_bandwidth_multiple_DUs"
    model += lpSum(K[(r, d)] for r in RU_names) <= M[(d,c)] * DUs[d]['DP'], f"{d}_ports_multiple_DUs"

# Fibre Connection Requirements
# Fibre Connection Constraints (RU to DU)
for r in RU_names:
    model += lpSum(K[(r, d)] * RUs[r]['RC'] for d in DU_names) <= config.FC_VALUE * J[r], f"fibre_{r}"

# Fibre Connection Constraints (DU to CU)
for d in DU_names:
    for c in CU_names:
        model += M[(d, c)] <= N[(d, c)] * config.CR_VALUE, f"{d}_{c}_M"
        model += lpSum(K[(r, d)] * RUs[r]['RC'] for r in RU_names) <= config.FC_VALUE * N[(d, c)], f"{d}_{c}_bandwidth_H"

# RU-DU segment (h) usage constraints
for conn in ru_du_new_paths + ru_du_existing_paths:
    ru_name = conn['ru_name']
    du_name = conn['du_name']
    path = conn['path']

    for i in range(len(path) - 1):
        s = (path[i], path[i + 1])
        if s in KS:
            model += I[s] >= B[(ru_name, du_name)], f"seg_activation_{s}_{ru_name}_{du_name}"


# DU-CU segment (h) usage constraints new DUs and existing DUs
for conn in du_cu_paths_new + du_cu_paths_existing:
    du_name = conn['du_name']
    cu_name = conn['cu_name']
    path = conn['path']

    for i in range(len(path) - 1):
        s = (path[i], path[i + 1])
        if s in KS:
            model += I[s] >= C[(du_name, cu_name)], f"seg_activation_{s}_{du_name}_{cu_name}"

# ============= Solve the Model ==============

model.solve(cplex_solver)

# ============= Extract Results ==============

segment_cost_value = round(sum(value(I[s]) * KS[s] for s in KS))
RU_installation_cost_value = round(value(RU_cost))
DU_cost_value = round(value(DU_cost))
CU_cost_value = round(value(CU_cost))
total_cost = value(model.objective)
num_selected_RUs = round(sum(value(D[r]) for r in RU_names))
num_selected_DUs = round(sum(value(E[d]) for d in DU_names))
total_segment_distance = round(sum(value(I[s]) * road_distance_dict[s] for s in KS))
covered_users = sum(1 for u in user_ids if any(A[(u, r)].value() == 1 for r in ur_rng[u]))
user_coverage_percent = (covered_users / total_users) * 100
fibre_connections_ru_du = {d: sum(round(value(K[(r, d)])) for r in RU_names if value(D[r]) > 0) for d in DU_names}
fibre_connections_du_cu = {c: sum(round(value(M[(d, c)])) for d in DU_names if value(E[d]) > 0) for c in CU_names}
bandwidth_per_du = {d: fibre_connections_du_cu[c] * config.FC_VALUE for d in DU_names}
total_fibres_per_cu = {c: fibre_connections_du_cu[c] for c in CU_names}
fibre_cost_ru_du = sum(fibre_connections_ru_du[d] * config.KL_COST for d in DU_names)
fibre_cost_du_cu = sum(fibre_connections_du_cu[c] * config.KB_COST for c in CU_names)

selected_rus, selected_dus, ru_to_du_connections, du_to_cu_connections = [], [], [], []
for r in RU_names:
    for d in DU_names:
        if B[(r, d)].value() == 1:
            if r not in selected_rus: selected_rus.append(r)
            if d not in selected_dus: selected_dus.append(d)
            ru_to_du_connections.append((r, d))

for d in DU_names:
    for c in CU_names:
        if C[(d, c)].value() == 1:
            if d not in selected_dus: selected_dus.append(d)
            du_to_cu_connections.append((d, c))

not_selected_rus = [r for r in RU_names if r not in selected_rus]
not_selected_dus = [d for d in DU_names if d not in selected_dus]
used_ru_capacity = {r: round(value(used_RU_capacity[r])) for r in RU_names if value(D[r]) > 0}
total_ru_capacity = {r: RUs[r]['RC'] for r in RU_names if value(D[r]) > 0}
used_du_capacity = {d: {'bandwidth': round(value(used_DU_capacity_bandwidth[d])), 'ports': round(value(used_DU_capacity_ports[d]))} for d in DU_names if round(value(E[d])) > 0}
total_du_capacity = {d: {'bandwidth': DUs[d]['DC'], 'ports': DUs[d]['DP']} for d in DU_names if round(value(E[d])) > 0}
num_rus_with_n_cells = {n: sum(1 for r in RU_names if round(value(J[r])) == n and value(D[r]) > 0) for n in range(1, config.MR_VALUE + 1)}
num_dus_with_n_cells = {n: sum(1 for d in DU_names if round(value(L[d])) == n and value(E[d]) > 0)for n in range(1, config.MD_VALUE + 1)}

results = {
    "segment_cost_value": segment_cost_value,
    "RU_installation_cost_value": RU_installation_cost_value,
    "DU_cost_value": DU_cost_value,
    "CU_cost_value": CU_cost_value,
    "total_cost": total_cost,
    "num_selected_RUs": num_selected_RUs,
    "num_selected_DUs": num_selected_DUs,
    "total_segment_distance": total_segment_distance,
    "selected_rus": selected_rus,
    "not_selected_rus": not_selected_rus,
    "selected_dus": selected_dus,
    "not_selected_dus": not_selected_dus,
    "ru_to_du_connections": ru_to_du_connections,
    "du_to_cu_connections": du_to_cu_connections,
    "used_ru_capacity": used_ru_capacity,
    "total_ru_capacity": total_ru_capacity,
    "used_du_capacity": used_du_capacity,
    "total_du_capacity": total_du_capacity,
    "user_coverage_percent": user_coverage_percent,
    "total_users": total_users,
    "fibre_connections_ru_du": fibre_connections_ru_du,
    "fibre_connections_du_cu": fibre_connections_du_cu,
    "bandwidth_per_du": bandwidth_per_du,
    "total_fibres_per_cu": total_fibres_per_cu,
    "fibre_cost_ru_du": fibre_cost_ru_du,
    "fibre_cost_du_cu": fibre_cost_du_cu,
}

# Save the results to a file
file_path = os.path.join(results_dir, f"{run_id}.txt")

save_ilp_results_to_file(results, file_path, log_path=log_path, num_rus_with_n_cells=num_rus_with_n_cells, num_dus_with_n_cells=num_dus_with_n_cells, total_users=total_users, RU_names=RU_names, CU_names=CU_names, K=K, N=N, value=value, L=L, DUs_new=DUs_new, J=J, covered_users=covered_users, UC_low=UC_low, total_fibres_per_cu=total_fibres_per_cu)

# Tidy up the working directory by moving CPLEX files to a separate directory
tidy_working_directory(log_path, current_dir, run_id, cplex_files_dir)