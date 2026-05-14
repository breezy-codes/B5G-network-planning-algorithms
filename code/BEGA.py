"""
%┏━━━┓┏┓━━┏┓┏━━━┓┏━━━┓┏━━━┓━━━━┏━┓┏━┓┏━━┓┏┓━━━┏━━━┓
?┃┏━┓┃┃┗┓┏┛┃┃┏━┓┃┃┏━┓┃┗┓┏┓┃━━━━┃┃┗┛┃┃┗┫┣┛┃┃━━━┃┏━┓┃
%┃┗━┛┃┗┓┗┛┏┛┃┃━┗┛┃┃━┃┃━┃┃┃┃━━━━┃┏┓┏┓┃━┃┃━┃┃━━━┃┗━┛┃
?┃┏━━┛━┗┓┏┛━┃┃┏━┓┃┗━┛┃━┃┃┃┃━━━━┃┃┃┃┃┃━┃┃━┃┃━┏┓┃┏━━┛
%┃┃━━━━━┃┃━━┃┗┻━┃┃┏━┓┃┏┛┗┛┃━━━━┃┃┃┃┃┃┏┫┣┓┃┗━┛┃┃┃━━━
?┗┛━━━━━┗┛━━┗━━━┛┗┛━┗┛┗━━━┛━━━━┗┛┗┛┗┛┗━━┛┗━━━┛┗┛━━━
%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
?━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This contains the script for running the binary-encoded genetic algorithm using the out of the box solver PYGAD. It uses the greedy solutions as the initial population and then evolves the solutions based on the defined fitness function, which incorporates both the objective and the constraints of the problem.

"""

import os, json, pygad, math
from modules import config
import numpy as np
from collections import defaultdict
from modules.save_cplex_results import save_ilp_results_to_file

# Set the dataset you want to use, being either "small-dataset" or "full-dataset"
dataset = "small-dataset"

run_id = f"BEGA_{dataset}"

base_dir = os.path.dirname(os.path.abspath(__file__))
scenario = os.path.join(base_dir, "..", "dataset", dataset)
results_dir = os.path.join(base_dir, "results", "BEGA", dataset)
os.makedirs(results_dir, exist_ok=True) # Create results directory if it doesn't exist
output_dir = results_dir 

greedy_data = json.load(open(os.path.join(results_dir, "greedy", dataset, f"greedy_short_{dataset}_2000_0.95_3_3_10_all.json")))

log_path = os.path.join(results_dir, "cplex_log.log")

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
RU_names_new = list(RUs_new.keys())            # New RUs
RU_names_existing = list(RUs_existing.keys())  # Existing RUs
RU_names = list(RUs.keys())                    # All RUs (new + existing)

DU_names_new = list(DUs_new.keys())            # new DUs
DU_names_existing = list(DUs_existing.keys())  # existing DUs
DU_names = list(DUs.keys())                    # All DUs (new + existing)

CU_names = list(CUs.keys())                    # All CUs

# User to RU Mapping and adding coverage rate
coverage_low = config.COVERAGE_THRESHOLD
coverage_high = min(coverage_low + 0.05, 1.0)
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

# % =================================================================
# % BUILD INITIAL POPULATION FROM GREEDY SOLUTIONS
# % =================================================================

def build_solution_from_greedy(greedy_sol):
    """Build a GA solution vector from a greedy solution dictionary. Where the decision variables are set up for those devices."""
    solution = np.zeros(num_genes)

    # RU info
    for ru in greedy_sol["ru_info"]:
        r = ru["name"]

        # B[r,d]
        d = ru.get("connected_du")
        if d:
            solution[B_idx[(r, d)]] = 1

        # A[u,r]
        for u in ru.get("connected_users", []):
            if (u, r) in A_idx:
                solution[A_idx[(u, r)]] = 1

    # DU info
    for du in greedy_sol["du_info"]:
        d = du["name"]

        # C[d,c]
        c = du.get("connected_cu")
        if c:
            solution[C_idx[(d, c)]] = 1

    return solution

def build_initial_population(greedy_json, population_size):
    """Build an initial population for the GA using the provided greedy solutions. It cycles through the greedy solutions if there are fewer than the desired population size."""
    population = []

    greedy_keys = list(greedy_json.keys())

    for i in range(population_size):
        key = greedy_keys[i % len(greedy_keys)]
        sol = build_solution_from_greedy(greedy_json[key])

        population.append(sol)

    return population

# % =================================================================
# % BUILD GENE SPACE AND INDEX MAPPINGS FOR DECISION VARIABLES
# % =================================================================

A_keys = [(u, r) for u in user_ids for r in ur_map[u]] # user → RU
B_keys = [(r, d) for r in RU_names for d in DU_names]  # RU → DU
C_keys = [(d, c) for d in DU_names for c in CU_names]  # DU → CU

offset = 0
def assign_indices(keys):
    global offset
    idx = {k: i + offset for i, k in enumerate(keys)}
    offset += len(keys)
    return idx

A_idx = assign_indices(A_keys)
B_idx = assign_indices(B_keys)
C_idx = assign_indices(C_keys)

num_genes = offset

gene_space = (
    [[0, 1]] * len(A_keys) +
    [[0, 1]] * len(B_keys) +
    [[0, 1]] * len(C_keys)
)

def get_A(solution, u, r): 
    return int(round(solution[A_idx[(u, r)]]))

def get_B(solution, r, d):
    return int(round(solution[B_idx[(r, d)]]))

def get_C(solution, d, c):
    return int(round(solution[C_idx[(d, c)]]))

# % =================================================================
# % DERIVED VALUES (D, E, J, L, N) NEEDED FOR CONSTRAINTS AND OBJECTIVE
# % =================================================================

def compute_D(solution):
    """Compute D[r] for each RU r, which indicates whether RU r is active (1) or not (0) based on user connections."""
    return {
        r: int(any(get_A(solution, u, r) for u in user_ids if r in ur_map[u]))
        for r in RU_names
    }

def compute_E(solution):
    """Compute E[d] for each DU d, which indicates whether DU d is active (1) or not (0) based on RU connections."""
    return {
        d: int(any(get_B(solution, r, d) for r in RU_names))
        for d in DU_names
    }

def get_I(solution, s):
    """Check if a segment s is active in the solution based on the RU → DU and DU → CU paths."""
    # check RU → DU paths
    for conn in ru_du_new_paths + ru_du_existing_paths:
        if get_B(solution, conn['ru_name'], conn['du_name']):
            path = conn['path']
            for i in range(len(path) - 1):
                if (path[i], path[i+1]) == s:
                    return 1

    # check DU → CU paths
    for conn in du_cu_paths_new + du_cu_paths_existing:
        if get_C(solution, conn['du_name'], conn['cu_name']):
            path = conn['path']
            for i in range(len(path) - 1):
                if (path[i], path[i+1]) == s:
                    return 1

    return 0

def compute_J(solution):
    """Compute J[r] for each RU r, which represents the load on RU r."""
    return {
        r: (
            math.ceil(
                sum(get_A(solution, u, r) * config.UM_VALUE for u in user_ids if r in ur_map[u])
                / RUs[r]['RC']
            )
        ) if any(get_A(solution, u, r) for u in user_ids if r in ur_map[u]) else 0
        for r in RU_names
    }

def compute_L(solution):
    """Compute L[d] for each DU d, which represents the load on DU d."""
    return {
        d: (
            math.ceil(
                sum(get_B(solution, r, d) * RUs[r]['RC'] for r in RU_names)
                / DUs[d]['DC']
            )
        ) if any(get_B(solution, r, d) for r in RU_names) else 0
        for d in DU_names
    }

def compute_du_load(solution):
    """Compute the total bandwidth load on each DU d based on the RUs connected to it."""
    J = compute_J(solution)

    return {
        d: sum(
            J[r] * get_B(solution, r, d) * RUs[r]['RC']
            for r in RU_names
        )
        for d in DU_names
    }

def compute_N(solution):
    """Compute N[d,c] for each DU-CU pair, which represents the load on the DU-CU link."""
    du_load = compute_du_load(solution)

    return {
        (d, c): (
            max(
                math.ceil(du_load[d] / config.FC_VALUE) if du_load[d] > 0 else 0,
                math.ceil(1 / config.CR_VALUE) if config.CR_VALUE > 0 else 0
            )
            if get_C(solution, d, c) else 0
        )
        for d in DU_names
        for c in CU_names
    }

# % =================================================================
# % SET FITNESS FUNCTION TO COVER THE CONSTRAINTS AND OBJECTIVE
# % =================================================================

def user_constraints_penalty(solution):
    """Calculate penalties for user coverage and connection constraints for constraints 1-4"""
    penalty = 0
    D = compute_D(solution)

    #? Constraints 1-3: User coverage and connection rules
    for u in user_ids:
        if ur_rng[u]:
            #% Constraint 2: User must be connected to at most 1 RU, and only active RUs (A[u,r] ≤ 1)
            val = sum(get_A(solution, u, r) for r in ur_rng[u])
            if val > 1:
                penalty += 1e8

            #% Constraint 3: User can only connect to active RUs (A[u,r] ≤ D[r])
            for r in ur_rng[u]:
                if get_A(solution, u, r) > D[r]:
                    penalty += 1e4

    #% Constraint 4: Total users connected must be between UC_low and UC_high
    total_users_connected = sum(sum(get_A(solution, u, r) for r in ur_rng[u]) for u in user_ids)

    if total_users_connected < UC_low:
        penalty += (UC_low - total_users_connected) * 1e4

    if total_users_connected > UC_high:
        penalty += (total_users_connected - UC_high) * 1e4

    return penalty

def ru_du_constraints_penalty(solution):
    """Calculate penalties for RU-DU and DU-CU connection constraints (constraints 5-17)"""
    penalty = 0
    D = compute_D(solution)
    E = compute_E(solution)
    J = compute_J(solution)
    L = compute_L(solution)
    N = compute_N(solution)

    #? Constraint 5 & 6: RU ↔ DU activation
    for r in RU_names:
        #% Constraint 5: RU must connect to exactly one DU if active (sum_d B[r,d] == D[r])
        val = sum(get_B(solution, r, d) for d in DU_names)
        penalty += abs(val - D[r]) * 1e8

        for d in DU_names:
            #% Constraint 6: RU can only connect to active DUs (B[r,d] ≤ E[d])
            if get_B(solution, r, d) > E[d]:
                penalty += 1e8

    #? Constraint 7: DU ↔ CU activation
    for d in DU_names:
        #% Constraint 7: DU must connect to exactly one CU if active (sum_c C[d,c] == E[d])
        val = sum(get_C(solution, d, c) for c in CU_names)
        penalty += abs(val - E[d]) * 1e8

    #? Constraint 8 & 9: RU allocation consistency
    for r in RU_names:
        #% Constraint 8: Total RU allocation across DUs must equal J[r] (sum_d K[r,d] == J[r])
        total_assigned = sum(J[r] * get_B(solution, r, d) for d in DU_names)
        penalty += abs(total_assigned - J[r]) * 1e2

    for r in RU_names:
        for d in DU_names:
            #% Constraint 9: RU allocation only allowed if RU-DU link is active (K[r,d] ≤ MR * B[r,d])
            if J[r] > config.MR_VALUE * get_B(solution, r, d):
                penalty += 1e4

    #? Constraint 10 & 11: DU allocation consistency
    for d in DU_names:
        #% Constraint 10: Total DU allocation across CUs must equal L[d] (sum_c M[d,c] == L[d])
        total_assigned = sum(N[(d, c)] for c in CU_names)
        penalty += abs(total_assigned - L[d]) * 1e2

        for c in CU_names:
            #% Constraint 11: DU allocation only allowed if DU-CU link is active (M[d,c] ≤ MD * C[d,c])
            if N[(d, c)] > config.MD_VALUE * get_C(solution, d, c):
                penalty += 1e2

    #? Constraint 12: RU capacity
    for r in RU_names:
        #% Constraint 12: Total user demand assigned to RU r must not exceed its capacity (sum_u A[u,r] * UM ≤ RC * J[r])
        used = sum(get_A(solution, u, r) * config.UM_VALUE for u in user_ids if r in ur_map[u])
        cap = RUs[r]['RC'] * J[r]
        if used > cap:
            penalty += 1e3 + (used - cap) * 1e3

    #? Constraint 13 & 14: DU capacity (bandwidth and ports)
    for d in DU_names:
        #% Constraint 13: DU bandwidth capacity (sum_r K[r,d] * RC ≤ total DU bandwidth capacity)
        bw = sum((J[r] * get_B(solution, r, d)) * RUs[r]['RC'] for r in RU_names)

        #% Constraint 14: DU port capacity (sum_r K[r,d] ≤ total DU port capacity)
        ports = sum(J[r] * get_B(solution, r, d) for r in RU_names)

        bw_capacity = L[d] * DUs[d]['DC']
        port_capacity = L[d] * DUs[d]['DP']

        if bw > bw_capacity:
            penalty += (bw - bw_capacity) * 1e6

        if ports > port_capacity:
            penalty += (ports - port_capacity) * 1e6

    #? Constraint 15: Fibre capacity on RU → DU links
    for r in RU_names:
        #% Constraint 15: Total fibre connections from RU r to DUs must not exceed FC * J[r]
        fibre = sum(get_B(solution, r, d) * RUs[r]['RC'] for d in DU_names)
        limit = config.FC_VALUE * J[r]

        if fibre > limit:
            penalty += (fibre - limit) * 1e6


    #? Constraint 16 & 17: Fibre DU → CU
    du_load = compute_du_load(solution)
    for d in DU_names:
        bw = du_load[d]

        for c in CU_names:
            #% Constraint 16: Total fibre connections from DU d to CUs must not exceed FC * N[d,c]
            if get_C(solution, d, c) > N[(d, c)] * config.CR_VALUE:
                penalty += (get_C(solution, d, c) - N[(d, c)] * config.CR_VALUE) * 1e6

            #% Constraint 17: Bandwidth on DU-CU link must not exceed FC * N[d,c]
            if get_C(solution, d, c) == 1:
                if bw > config.FC_VALUE * N[(d, c)]:
                    penalty += (bw - config.FC_VALUE * N[(d, c)]) * 1e6

    return penalty

def fitness_func(ga_instance, solution, sol_idx):
    """Calculate the fitness of a solution based on the total cost (segment + RU + DU + CU) and penalties for constraint violations."""
    # Derived values
    E = compute_E(solution)
    J = compute_J(solution)
    L = compute_L(solution)
    N = compute_N(solution)

    # Segment cost (build from active paths)
    Segment_cost = 0
    for conn in ru_du_new_paths + ru_du_existing_paths:
        r, d, path = conn['ru_name'], conn['du_name'], conn['path']
        if get_B(solution, r, d):
            for i in range(len(path) - 1):
                s = (path[i], path[i+1])
                if s in KS:
                    Segment_cost += KS[s]

    for conn in du_cu_paths_new + du_cu_paths_existing:
        d, c, path = conn['du_name'], conn['cu_name'], conn['path']
        if get_C(solution, d, c):
            for i in range(len(path) - 1):
                s = (path[i], path[i+1])
                if s in KS:
                    Segment_cost += KS[s]

    # RU cost
    RU_cost = sum(J[r] * config.KV_COST for r in RU_names)

    # DU cost
    DU_cost = (
        sum(E[d] * config.KW_COST for d in DUs_new)
        + sum((L[d] - E[d]) * config.KX_COST for d in DU_names)
        + sum(
            (J[r] if get_B(solution, r, d) else 0) * config.KL_COST
            for r in RU_names for d in DU_names
        )
    )

    # CU cost
    CU_cost = sum(
        N[(d, c)] * config.KB_COST
        for d in DU_names for c in CU_names
    )

    total_cost = Segment_cost + RU_cost + DU_cost + CU_cost

    # Constraints → Penalty
    penalty = 0
    penalty += user_constraints_penalty(solution)
    penalty += ru_du_constraints_penalty(solution)
    return - (total_cost + penalty)

# % =================================================================
# % RUN THE GA ALGORITHM
# % =================================================================

def on_generation(ga):
    """Callback function to print the best solution and its fitness at the end of each generation."""
    gen = ga.generations_completed
    best_sol, best_fit, _ = ga.best_solution()

    print(f"Generation {gen}")
    print(f"Best fitness: {best_fit}, Active RUs: {sum(compute_D(best_sol).values())}, Active DUs: {sum(compute_E(best_sol).values())}, Covered Users: {sum(get_A(best_sol, u, r) for u in user_ids for r in ur_rng[u])}/{total_users}")
    print("-" * 40)


generations = 10
parents = 4
population_size = 10

initial_population = build_initial_population(greedy_data, population_size)

ga_instance = pygad.GA(
    #? General GA parameters
    num_generations=generations,
    fitness_func=fitness_func,
    sol_per_pop=population_size,
    initial_population=initial_population,
    on_generation=on_generation,

    #? Parent information
    num_parents_mating=parents,
    parent_selection_type="sss",

    #? mutation information
    #mutation_type="adaptive",
    #mutation_probability=[0.3, 0.1],

    #? gene information
    num_genes=num_genes,
    gene_space=gene_space,
    gene_type=int,

    stop_criteria=["saturate_50"],
)

ga_instance.run()
ga_instance.plot_fitness()
solution, solution_fitness, _ = ga_instance.best_solution()

# % =================================================================
# % EXTRACT RESULTS AND METRICS
# % =================================================================

class Var:
    def __init__(self, val):
        self._val = val
    def value(self):
        return self._val

D = compute_D(solution)
E = compute_E(solution)
J = compute_J(solution)
L = compute_L(solution)
N = compute_N(solution)

A_wrap = {(u, r): Var(get_A(solution, u, r)) for u in user_ids for r in ur_map[u]}
B_wrap = {(r, d): Var(get_B(solution, r, d)) for r in RU_names for d in DU_names}
C_wrap = {(d, c): Var(get_C(solution, d, c)) for d in DU_names for c in CU_names}
D_wrap = {r: Var(D[r]) for r in RU_names}
E_wrap = {d: Var(E[d]) for d in DU_names}
J_wrap = {r: Var(J[r]) for r in RU_names}
L_wrap = {d: Var(L[d]) for d in DU_names}
K_wrap = {
    (r, d): Var(J[r] if get_B(solution, r, d) else 0)
    for r in RU_names for d in DU_names
}
N_wrap = {
    (d, c): Var(N[(d, c)])
    for d in DU_names for c in CU_names
}

def value(x):
    return x.value() if hasattr(x, "value") else x

segment_cost_value = round(sum(get_I(solution, s) * KS[s] for s in KS))
total_segment_distance = round(sum(get_I(solution, s) * road_distance_dict[s] for s in KS))
RU_installation_cost_value = round(sum(value(J_wrap[r]) * config.KV_COST for r in RU_names))

DU_cost_value = round(
    sum(value(E_wrap[d]) * config.KW_COST for d in DUs_new)
    + sum(max(0, value(L_wrap[d]) - value(E_wrap[d])) * config.KX_COST for d in DU_names)
    + sum(
        (value(J_wrap[r]) if value(B_wrap[(r, d)]) else 0) * config.KL_COST
        for r in RU_names for d in DU_names
    )
)

CU_cost_value = sum(
    N[(d, c)] * config.KB_COST
    for d in DU_names for c in CU_names
)

total_cost = segment_cost_value + RU_installation_cost_value + DU_cost_value + CU_cost_value

num_selected_RUs = sum(value(D_wrap[r]) for r in RU_names)
num_selected_DUs = sum(value(E_wrap[d]) for d in DU_names)
selected_rus = [r for r in RU_names if value(D_wrap[r]) == 1]
selected_dus = [d for d in DU_names if value(E_wrap[d]) == 1]
not_selected_rus = [r for r in RU_names if value(D_wrap[r]) == 0]
not_selected_dus = [d for d in DU_names if value(E_wrap[d]) == 0]
covered_users = sum(1 for u in user_ids if any(value(A_wrap[(u, r)]) == 1 for r in ur_rng[u]))
user_coverage_percent = (covered_users / total_users) * 100

ru_to_du_connections = [
    (r, d)
    for r in RU_names for d in DU_names
    if value(B_wrap[(r, d)]) == 1 and value(D_wrap[r]) == 1 and value(E_wrap[d]) == 1
]

du_to_cu_connections = [
    (d, c)
    for d in DU_names for c in CU_names
    if value(C_wrap[(d, c)]) == 1 and value(E_wrap[d]) == 1
]

fibre_connections_ru_du = {
    d: round(sum(value(K_wrap[(r, d)]) for r in RU_names if value(D_wrap[r]) == 1))
    for d in DU_names
}

fibre_connections_du_cu = {
    c: round(sum(value(C_wrap[(d, c)]) for d in DU_names))
    for c in CU_names
}

bandwidth_per_du = {
    d: sum(
        (value(J_wrap[r]) if value(B_wrap[(r, d)]) else 0) * RUs[r]['RC']
        for r in RU_names if value(D_wrap[r]) == 1
    )
    for d in DU_names if value(E_wrap[d]) == 1
}

total_fibres_per_cu = {c: fibre_connections_du_cu[c] for c in CU_names}
fibre_cost_ru_du = sum(fibre_connections_ru_du[d] * config.KL_COST for d in DU_names)
fibre_cost_du_cu = sum(fibre_connections_du_cu[c] * config.KB_COST for c in CU_names)

used_ru_capacity = {
    r: sum(value(A_wrap[(u, r)]) * config.UM_VALUE for u in user_ids if r in ur_map[u])
    for r in selected_rus
}

total_ru_capacity = {r: RUs[r]['RC'] for r in selected_rus}

used_du_capacity = {
    d: {
        'bandwidth': round(sum(
            value(K_wrap[(r, d)]) * RUs[r]['RC']
            for r in RU_names if value(D_wrap[r]) == 1
        )),
        'ports': round(sum(
            value(K_wrap[(r, d)])
            for r in RU_names if value(D_wrap[r]) == 1
        ))
    }
    for d in selected_dus
}

total_du_capacity = {
    d: {
        'bandwidth': DUs[d]['DC'] * value(L_wrap[d]),
        'ports': DUs[d]['DP'] * value(L_wrap[d])
    }
    for d in selected_dus
}

num_rus_with_n_cells = {
    n: sum(1 for r in selected_rus if value(J_wrap[r]) == n)
    for n in range(1, config.MR_VALUE + 1)
}

num_dus_with_n_cells = {
    n: sum(1 for d in selected_dus if value(L_wrap[d]) == n)
    for n in range(1, config.MD_VALUE + 1)
}

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

file_path = os.path.join(results_dir, f"{run_id}.txt")

save_ilp_results_to_file(
    results, file_path,
    log_path=log_path,
    num_rus_with_n_cells=num_rus_with_n_cells,
    num_dus_with_n_cells=num_dus_with_n_cells,
    total_users=total_users,
    RU_names=RU_names,
    CU_names=CU_names,
    K=K_wrap,
    N=N_wrap,
    value=value,
    L=L_wrap,
    DUs_new=DUs_new,
    J=J_wrap,
    covered_users=covered_users,
    UC_low=UC_low,
    total_fibres_per_cu=total_fibres_per_cu
)

print("Total cost:", total_cost)
print("Selected RUs:", num_selected_RUs)
print("Selected DUs:", num_selected_DUs)
print("Penalty:", -solution_fitness - total_cost)
print("Covered users:", covered_users, "/", total_users)
print("total RU used capacity:", sum(used_ru_capacity[r] for r in selected_rus), "/", sum(total_ru_capacity[r] for r in selected_rus))