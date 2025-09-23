"""
%┏━━━┓┏━━━┓┏━━━┓┏┓━┏┓┏┓━━━┏━━━┓┏━━━━┓┏━━┓┏━━━┓┏━┓━┏┓━━━━┏━┓┏━┓┏━━━┓┏━━━┓┏┓━┏┓┏┓━━━┏━━━┓
%┃┏━┓┃┃┏━┓┃┃┏━┓┃┃┃━┃┃┃┃━━━┃┏━┓┃┃┏┓┏┓┃┗┫┣┛┃┏━┓┃┃┃┗┓┃┃━━━━┃┃┗┛┃┃┃┏━┓┃┗┓┏┓┃┃┃━┃┃┃┃━━━┃┏━━┛
%┃┗━┛┃┃┃━┃┃┃┗━┛┃┃┃━┃┃┃┃━━━┃┃━┃┃┗┛┃┃┗┛━┃┃━┃┃━┃┃┃┏┓┗┛┃━━━━┃┏┓┏┓┃┃┃━┃┃━┃┃┃┃┃┃━┃┃┃┃━━━┃┗━━┓
%┃┏━━┛┃┃━┃┃┃┏━━┛┃┃━┃┃┃┃━┏┓┃┗━┛┃━━┃┃━━━┃┃━┃┃━┃┃┃┃┗┓┃┃━━━━┃┃┃┃┃┃┃┃━┃┃━┃┃┃┃┃┃━┃┃┃┃━┏┓┃┏━━┛
%┃┃━━━┃┗━┛┃┃┃━━━┃┗━┛┃┃┗━┛┃┃┏━┓┃━┏┛┗┓━┏┫┣┓┃┗━┛┃┃┃━┃┃┃━━━━┃┃┃┃┃┃┃┗━┛┃┏┛┗┛┃┃┗━┛┃┃┗━┛┃┃┗━━┓
%┗┛━━━┗━━━┛┗┛━━━┗━━━┛┗━━━┛┗┛━┗┛━┗━━┛━┗━━┛┗━━━┛┗┛━┗━┛━━━━┗┛┗┛┗┛┗━━━┛┗━━━┛┗━━━┛┗━━━┛┗━━━┛
%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from collections import defaultdict
import json, os, random

from modules.logger import loggers
from modules.data_classes.dataclass_solution import Solution

# % =================================================================
# % POPULATION SETTERS
# % =================================================================

def initialise_best_solution(best_solution_path, RUs, DUs, CUs, users, graph, scenario) -> Solution:
    """Initialise the best solution from the JSON file containing the best solution data."""
    with open(best_solution_path, 'r') as file:
        best_solution_data = json.load(file)

    if not isinstance(best_solution_data, dict):
        raise ValueError("The best solution data is not a valid dictionary.")

    solution = Solution(
        RUs=(RUs),
        DUs=(DUs),
        CUs=(CUs),
        users=(users),
        graph=(graph),
    )

    solution.set_solutions() # Set the initial state of the solution
    reset_states(RUs, DUs, CUs, users, graph) # Reset the states of RUs, DUs, CUs, and users
    set_device_states(RUs, DUs, CUs, users, best_solution_data)  # Set the state of RUs, DUs, and CUs
    
    solution_dict = {}

    solution_dict['ru_to_du_connections'] = {ru.name: ru.connected_du for ru in RUs.values() if ru.is_selected and ru.connected_du}
    solution_dict['du_to_cu_connections'] = {du.name: du.connected_cu for du in DUs.values() if du.is_selected and du.connected_cu}

    initialise_road_usage(solution_dict, graph, scenario, RUs, DUs) # Update road segment usage counts based on RU-DU and DU-CU paths    

    solution.update_model() # Update the solution model with the new states and connections

    if not solution.is_feasible():
        loggers['logger'].error("The best solution is not feasible.")
        return None

    loggers['logger'].info("Best solution initialised successfully.")
    return solution

def create_empty_solution(RUs, DUs, CUs, users, graph) -> Solution:
    """Create an initial Solution object with fresh copies of all components and reset state."""
    RUs_copy = {ru_name: ru.make_copy() for ru_name, ru in RUs.items()}
    DUs_copy = {du_name: du.make_copy() for du_name, du in DUs.items()}
    CUs_copy = {cu_name: cu.make_copy() for cu_name, cu in CUs.items()}
    users_copy = {user_name: user.make_copy() for user_name, user in users.items()}
    graph_copy = graph.make_copy()

    reset_states(RUs_copy, DUs_copy, CUs_copy, users_copy, graph_copy)

    solution = Solution(
        RUs=RUs_copy,
        DUs=DUs_copy,
        CUs=CUs_copy,
        users=users_copy,
        graph=graph_copy
    )

    solution.set_solutions() # Set the initial state of the solution
    return solution

def initialise_multiple_individuals(all_solution_path, RUs, DUs, CUs, users, graph, scenario) -> dict[str, Solution]:
    """Initialise multiple solutions from a JSON file with named entries (e.g., greedy1, greedy2, ...)."""
    with open(all_solution_path, 'r') as file:
        all_solution_data = json.load(file)

    if not isinstance(all_solution_data, dict):
        raise ValueError("The best solution data must be a dictionary of named solutions.")

    solutions = {}

    # Iterate through each named solution in the JSON data
    for name, best_solution_data in all_solution_data.items():

        # Create a new Solution object with the provided RUs, DUs, CUs, users, and graph
        solution = create_empty_solution(RUs, DUs, CUs, users, graph)
        RUs_copy, DUs_copy, CUs_copy, users_copy, graph_copy = (solution.RUs, solution.DUs, solution.CUs, solution.users, solution.graph)

        # Set the initial state of the solution
        set_device_states(RUs_copy, DUs_copy, CUs_copy, users_copy, best_solution_data)

        solution_dict = {
            'ru_to_du_connections': {ru.name: ru.connected_du for ru in RUs_copy.values() if ru.is_selected and ru.connected_du},
            'du_to_cu_connections': {du.name: du.connected_cu for du in DUs_copy.values() if du.is_selected and du.connected_cu},
        }

        # Initialise road usage based on the connections in the solution
        initialise_road_usage(solution_dict, graph_copy, scenario, RUs_copy, DUs_copy)
        solution.update_model() # Update the solution model with the new states and connections

        if not solution.is_feasible():
            loggers['logger'].warning(f"Solution '{name}' is not feasible and will be skipped.")
            continue

        solutions[name] = solution

    return solutions

# % =================================================================
# % INITIALISE PATHINGS
# % =================================================================

def load_paths(scenario, file_names):
    """Load and combine JSON path data from multiple files."""
    paths = []
    for file_name in file_names:
        file_path = os.path.join(scenario, file_name)
        try:
            with open(file_path) as file:
                paths.extend(json.load(file))
        except FileNotFoundError:
            loggers['Path_logger'].warning(f"File not found: {file_path}")
    return paths

def process_connections(solution, connections_key, paths, graph, entity_dict, entity_type):
    """Process connections (RU-DU or DU-CU), update segment usage, and store them in the respective entity (RU/DU)."""
    for source_name, target_name in solution.get(connections_key, {}).items():
        path_entry = next((entry for entry in paths if entry[f"{entity_type}_name"] == source_name and entry[f"{'du' if entity_type == 'ru' else 'cu'}_name"] == target_name), None)
        
        if path_entry:
            path = path_entry['path']
            segments = update_segment_usage(path, graph, **{f"{entity_type}_name": source_name})  # Dynamically pass ru_name or du_name
            if source_name in entity_dict:
                entity_dict[source_name].path_segments = segments  # Store segments in RU/DU
        else:
            loggers['Path_logger'].warning(f"No path found for {entity_type.upper()} {source_name} -> {target_name}")

def initialise_road_usage(solution, graph, scenario, RUs, DUs):
    """Processes initial population by updating road segment usage counts based on RU-DU and DU-CU paths."""
    
    # Load paths from JSON files
    ru_du_paths = load_paths(scenario, ['ru_du_path_new.json', 'ru_du_path_exist_graph.json'])
    du_cu_paths = load_paths(scenario, ['du_cu_path_new.json', 'du_cu_path_exist.json'])
   
    process_connections(solution, 'ru_to_du_connections', ru_du_paths, graph, RUs, entity_type='ru') # Process RU-DU connections
    process_connections(solution, 'du_to_cu_connections', du_cu_paths, graph, DUs, entity_type='du') # Process DU-CU connections

def update_segment_usage(path, graph, ru_name=None, du_name=None):
    """Updates the usage count for one direction of each segment along a path and returns the list of used segments."""
    used_segments = []

    for i in range(len(path) - 1):
        from_node, to_node = path[i], path[i + 1]

        # Only update the forward direction
        segment = graph.segments.get((from_node, to_node))

        if segment:
            segment.increase_usage() # Increment usage count for the segment
            used_segments.append(segment) # Store the segment in the list of used segments

            # Add RU or DU to the segment's associated RUs or DUs
            if ru_name and ru_name not in segment.associated_rus:
                segment.add_associated_ru(ru_name)
            if du_name and du_name not in segment.associated_dus:
                segment.add_associated_du(du_name)
        else:
            loggers['Path_logger'].warning(f"update_segment_usage: No segment found for {from_node} -> {to_node}. Skipping.")

    graph.update_road_results() # Update the graph's road results after modifying segment usage
    return used_segments

# % =================================================================
# % INITIALISE DEVICE STATES
# % =================================================================

def reset_states(RUs, DUs, CUs, users, graph):
    """Reset the state of RUs, DUs, and CUs to ensure a clean slate."""
    for user in users.values():
        user.reset_user()
    for ru in RUs.values():
        ru.reset_RadioUnit()
    for du in DUs.values():
        du.reset_DistributedUnit()
    for cu in CUs.values():
        cu.reset_CentralisedUnit()
    graph.reset_segments()

def set_ru_states(RUs, users, solution_data):
    """Set the state of RUs based on the solution data."""
    selected_rus = solution_data.get("selected_RUs", [])
    RU_info = solution_data.get("ru_info", [])
    
    for ru_data in RU_info:
        ru_name = ru_data.get("name")
        if not ru_name or ru_name not in RUs:
            loggers['logger'].warning(f"RU {ru_name} not found in RUs dictionary.")
            continue
        
        # Create a copy of the RU instance to avoid modifying the original
        ru_instance = RUs[ru_name] 
        is_selected = ru_name in selected_rus
        connected_users = [user for user in users.values() if user.user_id in ru_data.get("connected_users", [])]
        connected_du = ru_data.get("connected_du")
        
        # Set the state of the RU instance
        ru_instance.set_state(
            num_rus=ru_data.get("num_rus", 0),
            connected_users=connected_users,
            connected_du=connected_du,
            is_selected=is_selected,
            used_capacity=ru_data.get("used_capacity", 0)
        )

def set_du_states(DUs, RUs, solution_data):
    """Set the state of DUs based on the solution data."""
    selected_dus = solution_data.get("selected_DUs", [])
    DU_info = solution_data.get("du_info", [])
    
    for du_data in DU_info:
        du_name = du_data.get("name")
        if du_name not in DUs:
            loggers['DU_logger'].warning(f"DU {du_name} not found in DUs dictionary.")
            continue
        
        # Create a copy of the DU instance to avoid modifying the original
        du_instance = DUs[du_name]
        is_selected = du_name in selected_dus
        connected_rus = [ru_name for ru_name in du_data.get("connected_rus", []) if ru_name in RUs]
        connected_cu = du_data.get("connected_cu")
        
        # Set the state of the DU instance
        du_instance.set_state(
            is_selected=is_selected,
            used_bandwidth=du_data.get("used_bandwidth", 0),
            num_dus=du_data.get("num_dus", 0),
            used_ports=du_data.get("used_ports", 0),
            fibre_ru_du=du_data.get("fibre_ru_du", 0),
            fibre_du_cu=du_data.get("fibre_du_cu", 0),
            connected_rus=connected_rus,
            connected_cu=connected_cu,
        )

def set_cu_states(CUs, DUs, solution_data):
    """Set the state of CUs based on the solution data."""
    selected_cus = solution_data.get("selected_CUs", [])
    CU_info = solution_data.get("cu_info", [])
    
    for cu_data in CU_info:
        cu_name = cu_data.get("name")
        if cu_name not in CUs:
            loggers['CU_logger'].warning(f"CU {cu_name} not found in CUs dictionary.")
            continue
        
        # Create a copy of the CU instance to avoid modifying the original
        cu_instance = CUs[cu_name]
        is_selected = cu_name in selected_cus
        connected_dus = [du_name for du_name in cu_data.get("connected_dus", []) if du_name in DUs]
        
        # Set the state of the CU instance
        cu_instance.set_state(
            connected_dus=connected_dus,
            used_bandwidth=cu_data.get("used_bandwidth", 0),
            used_ports=cu_data.get("used_ports", 0),
            total_fibres=cu_data.get("total_fibres", 0),
            is_selected=is_selected,
        )

def set_device_states(RUs, DUs, CUs, users, solution_data):
    """Set the state of RUs, DUs, and CUs based on the solution data."""
    set_ru_states(RUs, users, solution_data)
    set_du_states(DUs, RUs, solution_data)
    set_cu_states(CUs, DUs, solution_data)

# % =================================================================
# % CROSSOVER FUNCTIONS
# % =================================================================

def dfs(node, ordered_path, visited, adj):
    """Depth-first search to order nodes in a path."""
    ordered_path.append(node)
    visited.add(node)
    for neighbour in adj[node]:
        if neighbour not in visited:
            dfs(neighbour, ordered_path, visited, adj)

def reconstruct_ordered_path_from_segments(segments, start_hint=None, log_prefix="Path"):
    """Reconstruct ordered node path from a list of RoadSegment objects."""
    if not segments:
        return []

    edges = [(seg.from_node, seg.to_node) for seg in segments]

    # Build adjacency map
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Always use the first segment's from_node
    start_node = start_hint if start_hint is not None else segments[0].from_node

    # Check if the start node exists in the adjacency map
    if start_node not in adj:
        loggers['Path_logger'].warning(f"{log_prefix}: Start node {start_node} not found in adjacency")
        return []

    ordered_path, visited = [], set()
    dfs(start_node, ordered_path, visited, adj)
    return ordered_path

def crossover_population(parent1: Solution, parent2: Solution, RUs, DUs, CUs, users, graph):
    """
    Perform crossover between two parent solutions at the DU level to create a new child solution.
    - Half the DUs are taken from one parent, half from the other.
    - RUs are assigned according to their DU assignments in the selected parents.
    - If an RU is assigned to both, randomly pick one DU assignment.
    - Users are assigned as in the parent, but coverage is fixed after.
    """
    
    # Create an empty solution to fill with crossover data
    child_solution = create_empty_solution(RUs, DUs, CUs, users, graph)
    RUs_copy, DUs_copy, CUs_copy, users_copy, graph_copy = (child_solution.RUs, child_solution.DUs, child_solution.CUs, child_solution.users, child_solution.graph)

    # Get selected DU names from each parent
    selected_du_p1 = [du_name for du_name, du in parent1.DUs.items() if du.is_selected]
    selected_du_p2 = [du_name for du_name, du in parent2.DUs.items() if du.is_selected]

    # Shuffle and split each list in half
    random.shuffle(selected_du_p1)
    random.shuffle(selected_du_p2)
    half_p1 = int(len(selected_du_p1) / 2)
    half_p2 = int(len(selected_du_p2) / 2)

    du_from_p1 = set(selected_du_p1[:half_p1])
    du_from_p2 = set(selected_du_p2[:half_p2])

    # Avoid overlap: if a DU is in both, randomly keep in either p1 or p2
    du_origin_map = {}
    handled_overlap = set()

    overlap = du_from_p1 & du_from_p2
    for du_name in overlap:
        if random.choice([True, False]):
            du_from_p2.remove(du_name)
            du_origin_map[du_name] = ("p1", "p2")
        else:
            du_from_p1.remove(du_name)
            du_origin_map[du_name] = ("p2", "p1")
        handled_overlap.add(du_name)

    # Mark non-overlapping DUs — only if not already added
    for du_name in du_from_p1:
        if du_name not in handled_overlap:
            du_origin_map[du_name] = ("p1", None)

    for du_name in du_from_p2:
        if du_name not in handled_overlap:
            du_origin_map[du_name] = ("p2", None)

    # Collect selected DU assignments from both parents
    du_assignments = {}

    for du_name, (selected_from, _) in du_origin_map.items():
        if selected_from == "p1" and du_name in parent1.DUs:
            du_assignments[du_name] = parent1.DUs[du_name]
        elif selected_from == "p2" and du_name in parent2.DUs:
            du_assignments[du_name] = parent2.DUs[du_name]

    # Set DUs as selected and activate them, then add them to their parent CU
    for du_name, du_parent in du_assignments.items():
        cu_name = du_parent.connected_cu
        cu_obj = CUs_copy.get(cu_name)
        du_child = DUs_copy[du_name]
        du_child.activate_du(cu_obj)

        segment_objs = getattr(du_parent, "path_segments", [])
        ordered_path = reconstruct_ordered_path_from_segments(segment_objs, start_hint=du_name, log_prefix=f"DU Path: {du_name} → {cu_name}")

        if ordered_path:
            child_solution.process_best_path(best_path=ordered_path, du_name=du_name, target=du_child, log_prefix=f"DU Path: {du_name} → {cu_name}")
        else:
            loggers['Path_logger'].warning(f"DU {du_name} could not reconstruct path to CU {cu_name}")

        # Update the CU connection
        if cu_name and cu_name in CUs_copy:
            cu_child = CUs_copy[cu_name]
            cu_child.is_selected = True
            cu_child.add_du(DUs_copy[du_name])

    # Collect RU assignments only from the chosen parent DUs
    ru_du_map = {}
    ru_from_p1 = set()
    ru_from_p2 = set()

    for du_name, (selected_from, _) in du_origin_map.items():
        if selected_from == "p1" and du_name in parent1.DUs:
            du = parent1.DUs[du_name]
            for ru_name in du.connected_rus:
                if ru_name in parent1.RUs and parent1.RUs[ru_name].is_selected:
                    ru_du_map.setdefault(ru_name, []).append(du_name)
                    ru_from_p1.add(ru_name)

        elif selected_from == "p2" and du_name in parent2.DUs:
            du = parent2.DUs[du_name]
            for ru_name in du.connected_rus:
                if ru_name in parent2.RUs and parent2.RUs[ru_name].is_selected:
                    ru_du_map.setdefault(ru_name, []).append(du_name)
                    ru_from_p2.add(ru_name)

    # Assign RUs to DUs in the child, resolving conflicts randomly
    for ru_name, du_list in ru_du_map.items():
        if not du_list:
            continue
        chosen_du = random.choice(du_list) if len(du_list) > 1 else du_list[0]
        ru_child = RUs_copy[ru_name]
        du_child = DUs_copy[chosen_du]
        ru_child.activate_ru(du_child)
        du_child.add_ru(ru_child)
        ru_child.connected_du = chosen_du

        # Get parent based on DU origin
        selected_from, _ = du_origin_map.get(chosen_du, (None, None))
        parent = parent1 if selected_from == "p1" else parent2 if selected_from == "p2" else None

        segment_objs = []
        if parent:
            parent_ru = parent.RUs.get(ru_name)
            if parent_ru and getattr(parent_ru, "path_segments", []):
                segment_objs = parent_ru.path_segments
            else:
                parent_du = parent.DUs.get(chosen_du)
                du_segments = getattr(parent_du, "path_segments", [])
                segment_objs = [seg for seg in du_segments if ru_name in getattr(seg, "associated_rus", [])]

            ordered_path = reconstruct_ordered_path_from_segments(segment_objs, start_hint=ru_name, log_prefix=f"RU Path: {ru_name} → {chosen_du}")

            if ordered_path:
                child_solution.process_best_path(best_path=ordered_path, ru_name=ru_name, target=ru_child, log_prefix=f"RU Path: {ru_name} → {chosen_du}")
            else:
                loggers['Path_logger'].warning(f"Failed to construct ordered path for RU {ru_name}, falling back to recompute")
                child_solution.add_ru_du_path(ru_name, chosen_du)
        else:
            loggers['Path_logger'].warning(f"No parent found for RU {ru_name} → {chosen_du}, falling back to recompute")
            child_solution.add_ru_du_path(ru_name, chosen_du)

    # Assign users to RUs in the child, respecting parent assignments as much as possible
    assigned_users = set()

    for ru_name, ru_child in RUs_copy.items():
        if not ru_child.is_selected:
            continue

        users_from_p1 = parent1.RUs[ru_name].connected_users if ru_name in parent1.RUs else []
        users_from_p2 = parent2.RUs[ru_name].connected_users if ru_name in parent2.RUs else []

        if users_from_p1 and users_from_p2:
            chosen_users = users_from_p1 if random.choice([True, False]) else users_from_p2
        elif users_from_p1:
            chosen_users = users_from_p1
        else:
            chosen_users = users_from_p2

        for user in chosen_users:
            user_id = user.user_id if hasattr(user, 'user_id') else user
            if user_id in assigned_users:
                continue  # Skip if user already assigned

            user_obj = users_copy.get(user_id, None)
            if user_obj and ru_child.true_false_can_add_user(user_obj):
                ru_child.add_user(user_obj)
                user_obj.allocate_to_ru(ru_child)
                assigned_users.add(user_id)

    # Finalise solution
    child_solution.update_model()  # Update the child solution model
    child_solution.set_solutions() # Set the initial state of the child solution
    child_solution.ensure_coverage_requirement() # Ensure coverage requirement is met
    return child_solution