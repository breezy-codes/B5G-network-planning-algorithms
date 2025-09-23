"""
%┏━━━┓┏━━━┓┏━━━┓┏━━━┓━━━━┏━┓┏━┓┏━━━┓┏━━━┓┏┓━┏┓┏┓━━━┏━━━┓
?┃┏━┓┃┃┏━┓┃┃┏━┓┃┗┓┏┓┃━━━━┃┃┗┛┃┃┃┏━┓┃┗┓┏┓┃┃┃━┃┃┃┃━━━┃┏━━┛
%┃┗━┛┃┃┃━┃┃┃┃━┃┃━┃┃┃┃━━━━┃┏┓┏┓┃┃┃━┃┃━┃┃┃┃┃┃━┃┃┃┃━━━┃┗━━┓
?┃┏┓┏┛┃┃━┃┃┃┗━┛┃━┃┃┃┃━━━━┃┃┃┃┃┃┃┃━┃┃━┃┃┃┃┃┃━┃┃┃┃━┏┓┃┏━━┛
%┃┃┃┗┓┃┗━┛┃┃┏━┓┃┏┛┗┛┃━━━━┃┃┃┃┃┃┃┗━┛┃┏┛┗┛┃┃┗━┛┃┃┗━┛┃┃┗━━┓
?┗┛┗━┛┗━━━┛┗┛━┗┛┗━━━┛━━━━┗┛┗┛┗┛┗━━━┛┗━━━┛┗━━━┛┗━━━┛┗━━━┛
%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
?━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from modules.logger import loggers
from modules import config
from modules.haversine import haversine

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, Tuple, List, Optional, Set
from itertools import combinations
import igraph as ig
from igraph._igraph import InternalError

@dataclass
class RoadSegment:
    from_node: str          # Name of the starting node
    to_node: str            # Name of the ending node
    length: float           # Length of the road segment
    cost: float             # Cost of the road segment
    usage_count: int = 0    # Usage count of the segment, default is 0
    associated_rus: Set[str] = field(default_factory=set)            # Set of RUs associated with the segment
    associated_dus: Set[str] = field(default_factory=set)            # Set of RUs and DUs associated with the segment
    solution: Optional['Solution'] = field(default=None, init=False) # Solution the user is part of

    def set_solution(self, solution):
        """Set the solution the user is part of."""
        self.solution = solution
    
    def make_copy(self):
        """Create a deep copy of the RoadSegment, excluding the solution attribute."""
        copy_segment = RoadSegment(
            from_node=self.from_node,
            to_node=self.to_node,
            length=self.length,
            cost=self.cost,
            usage_count=self.usage_count,
            associated_rus=self.associated_rus,
            associated_dus=self.associated_dus
        )
        return copy_segment
    
    def __str__(self):
        """String representation of the RoadSegment object."""
        return (
            f"Road Segment: {self.from_node} to {self.to_node}\n"
            f"  Length: {self.length:.2f} km\n"
            f"  Cost: {self.cost:.2f}\n"
            f"  Usage Count: {self.usage_count}\n"
            f"  Associated RUs: {self.associated_rus}\n"
            f"  Associated DUs: {self.associated_dus}"
        )

    def increase_usage(self):
        """Increases the usage count of the road segment by 1."""
        self.usage_count += 1
        return self.usage_count

    def decrease_usage(self):
        """Decreases the usage count of the road segment by 1, ensuring it does not go below 0."""
        if self.usage_count > 0:
            self.usage_count -= 1
        return self.usage_count

    def update_cost(self, new_cost):
        """Update the cost of the road segment."""
        self.cost = new_cost

    def add_associated_ru(self, ru_name):
        """Adds an RU to the associated set."""
        self.associated_rus.add(ru_name)

    def remove_associated_ru(self, ru_name):
        """Removes an RU from the associated set."""
        self.associated_rus.discard(ru_name)

    def add_associated_du(self, du_name):
        """Adds a DU to the associated set."""
        self.associated_dus.add(du_name)

    def remove_associated_du(self, du_name):
        """Removes a DU from the associated set."""
        self.associated_dus.discard(du_name)
        
@dataclass
class RoadGraph:
    graph: ig.Graph                                      # iGraph object representing the road network
    KS: defaultdict                                      # Cost mapping for road segments
    segments: Dict[Tuple[str, str], 'RoadSegment']       # Dictionary mapping segments to RoadSegment
    node_positions: Dict[str, Tuple[float, float]]       # Mapping node ID to (lat, lon)
    total_road_cost: float                               # Sum of used road segment costs
    total_road_distance: float                           # Sum of used road segment lengths
    solution: Optional['Solution'] = field(default=None, init=False)    # Solution the road graph is part of
    shortest_paths: Optional[Dict[str, Dict[str, List[str]]]] = None    # Precomputed shortest paths
    shortest_path_lengths: Optional[Dict[str, Dict[str, float]]] = None # Precomputed shortest path lengths
    vs_names: List[str] = field(default_factory=list, init=False)           # Cached list of vertex names
    name_to_index: Dict[str, int] = field(default_factory=dict, init=False) # Cached name-to-index map

    #* --- CONSTANTS FROM CONFIG --- *#
    KY = config.KY_COST     # Cost associated with a new road segment
    KM = config.KM_COST     # Cost associated with an existing road segment

    def __init__(self, road_edges):
        self.graph = ig.Graph(directed=False)
        self.KS = defaultdict(int)
        self.segments = {}
        self.node_positions = {}
        self.total_road_cost = 0
        self.total_road_distance = 0
        self._build_graph(road_edges)

    def set_solution(self, solution):
        """Set the solution the road graph is part of."""
        self.solution = solution

    def make_copy(self):
        """Create a deep copy of the RoadGraph, excluding the solution attribute."""
        copy_graph = RoadGraph([])
        copy_graph.graph = self.graph
        copy_graph.KS = self.KS
        copy_graph.segments = {k: v.make_copy() for k, v in self.segments.items()}
        copy_graph.node_positions = self.node_positions
        copy_graph.total_road_cost = getattr(self, "total_road_cost", 0)
        copy_graph.total_road_distance = getattr(self, "total_road_distance", 0)
        copy_graph.shortest_paths = self.shortest_paths
        copy_graph.shortest_path_lengths = self.shortest_path_lengths
        copy_graph.vs_names = self.vs_names
        copy_graph.name_to_index = self.name_to_index

        return copy_graph

    def reset_segments_count(self):
        """Reset the usage count of all segments to 0."""
        for segment in self.segments.values():
            segment.usage_count = 0

    def add_segment_into_use(self, segment, ru_name: Optional[str] = None, du_name: Optional[str] = None):
        """Increment the usage count of a specific segment and optionally associate RU/DU names."""
        if isinstance(segment, tuple):
            key = segment
        elif hasattr(segment, 'from_node') and hasattr(segment, 'to_node'):
            key = (segment.from_node, segment.to_node)
        else:
            raise ValueError(f"Invalid segment format: {segment}")

        # Ensure the segment exists in the graph
        if key in self.segments:
            seg = self.segments[key] # Get the segment object
            seg.increase_usage()     # Increment usage count
            if ru_name is not None:
                seg.add_associated_ru(ru_name) # Add RU association
            if du_name is not None:
                seg.add_associated_du(du_name) # Add DU association
        else:
            raise KeyError(f"Segment key {key} not found in graph.")

    def reset_segments(self):
        """Reset the usage count and used_in_mst attribute for all segments."""
        for seg in self.segments.values():
            seg.usage_count = 0
            seg.associated_rus.clear()
            seg.associated_dus.clear()
        self.update_road_results()

    def calc_total_road_usage(self):
        """Computes total road cost and distance for segments with usage_count ≥ 1, summing only one direction."""
        counted_segments = set()
        total_cost = 0
        total_distance = 0

        for (u, v), seg in self.segments.items():
            if seg.usage_count >= 1 and (v, u) not in counted_segments:
                total_cost += seg.cost
                total_distance += seg.length
                counted_segments.add((u, v))

        self.total_road_cost = total_cost
        self.total_road_distance = total_distance
        return total_cost, total_distance

    def update_segments_usage(self, segments: List[RoadSegment], ru_name: Optional[str] = None, du_name: Optional[str] = None, increase: bool = True):
        """Updates usage count of segments and manages RU/DU associations."""
        affected = []
        get_segment = self.segments.get  # Use cached get method for speed

        # Use method binding for set operations
        add = set.add
        discard = set.discard

        for seg in segments:
            if not isinstance(seg, RoadSegment): continue

            key = (seg.from_node, seg.to_node)
            local = get_segment(key)
            if local is None: continue

            # Usage update
            (local.increase_usage if increase else local.decrease_usage)()

            # RU association
            if ru_name:
                (add if increase else discard)(local.associated_rus, ru_name)

            # DU association
            if du_name:
                (add if increase else discard)(local.associated_dus, du_name)

            affected.append(local)

    def update_road_results(self):
        """Updates the total road cost and distance based on current segment usage."""
        self.calc_total_road_usage()

    # % =================================================================
    # % GRAPH BUILDING FUNCTIONS
    # % =================================================================

    def _build_graph(self, road_edges):
        """Builds the graph using iGraph from road edge data."""
        node_set = set()  # Set to track unique node IDs
        raw_edges = []    # Store edges as (u, v) string tuples
        edge_costs = []   # Store costs for each edge
        edge_lengths = [] # Store lengths for each edge

        for edge in road_edges:
            u, v = edge['from'], edge['to']
            from_pos = (edge['geometry'][0]['latitude'], edge['geometry'][0]['longitude'])
            to_pos = (edge['geometry'][-1]['latitude'], edge['geometry'][-1]['longitude'])

            # Ensure both nodes are unique and have positions
            self.node_positions[u] = from_pos
            self.node_positions[v] = to_pos

            # Calculate length and cost
            length = round(edge['length'])
            cost = length * self.KY

            # Create a RoadSegment object and add it to the segments dictionary
            segment = RoadSegment(u, v, length, cost)
            self.segments[(u, v)] = segment
            self.segments[(v, u)] = segment

            # Update the total road cost and distance
            self.KS[(u, v)] = cost
            self.KS[(v, u)] = cost

            raw_edges.append((u, v))
            edge_costs.append(cost)
            edge_lengths.append(length)
            node_set.update([u, v])

        # Add only new vertices to the graph
        existing = set(self.graph.vs['name']) if 'name' in self.graph.vs.attributes() else set()
        new_nodes = list(node_set - existing)
        if new_nodes:
            # Add new vertices to the graph
            self.graph.add_vertices(len(new_nodes))
            for i, name in enumerate(new_nodes):
                self.graph.vs[self.graph.vcount() - len(new_nodes) + i]['name'] = name

        # Build name-to-index map and convert edges to index form
        name_to_index = {v['name']: v.index for v in self.graph.vs}
        indexed_edges = [(name_to_index[u], name_to_index[v]) for u, v in raw_edges]

        # Add edges to the graph
        self.graph.add_edges(indexed_edges)
        self.graph.es['cost'] = edge_costs
        self.graph.es['length'] = edge_lengths
        self.build_name_to_index()

    def mark_existing_path(self, existing_paths):
        """Marks the existing paths as existing and updates the cost of the segments."""
        if 'name' not in self.graph.vs.attributes():
            raise ValueError("Vertex names are not assigned in the graph.")

        # Create a mapping from node names to their indices
        name_to_index = {v['name']: v.index for v in self.graph.vs}

        # Iterate through existing paths and update costs
        for path in existing_paths:
            for i in range(len(path['path']) - 1):
                u_name, v_name = path['path'][i], path['path'][i + 1]
                u_idx = name_to_index.get(u_name)
                v_idx = name_to_index.get(v_name)

                if u_idx is None or v_idx is None:
                    continue  # Skip if either node not in graph

                try:
                    # Get the edge ID for the existing path
                    eid = self.graph.get_eid(u_idx, v_idx)
                except ig._igraph.InternalError:
                    continue  # No such edge

                # Update the cost of the edge
                original_length = self.graph.es[eid]['length']
                new_cost = original_length * self.KM

                # Update the segment cost in the segments dictionary
                self.graph.es[eid]['cost'] = new_cost

                self.KS[(u_name, v_name)] = new_cost
                self.KS[(v_name, u_name)] = new_cost

                # Update the segment in the segments dictionary
                if (u_name, v_name) in self.segments:
                    self.segments[(u_name, v_name)].update_cost(new_cost)
                if (v_name, u_name) in self.segments:
                    self.segments[(v_name, u_name)].update_cost(new_cost)

    def is_device(self, node_id):
        """Checks if a node ID corresponds to a device (RU, DU, or CU)."""
        return isinstance(node_id, str) and (node_id.startswith("RU") or node_id.startswith("DU") or node_id.startswith("CU"))

    def connect_device_to_road(self, device_id, device_pos):
        """Connects a device node to the nearest road node with 0-length, 0-cost segment.
        Ensures the device is only connected to a true road node, not another device.
        """

        # Only consider non-device nodes as candidates for connection
        road_nodes = [node for node in self.node_positions if not self.is_device(node)]
        if not road_nodes:
            raise ValueError("No road nodes available to connect the device.")

        # Find nearest road node by haversine distance
        nearest_node = min(((node, self.node_positions[node]) for node in road_nodes),key=lambda item: haversine(device_pos, item[1]))[0]

        # Ensure 'name' attribute exists on vertices
        if 'name' not in self.graph.vs.attributes():
            self.graph.vs['name'] = [None] * self.graph.vcount()

        existing_names = set(self.graph.vs['name'])

        # Add missing vertices (either device or nearest node)
        to_add = []
        if device_id not in existing_names:
            to_add.append(device_id)
            #loggers['Path_logger'].info(f"Connected device {device_id} to nearest road node {nearest_node}")
        if nearest_node not in existing_names:
            to_add.append(nearest_node)

        if to_add:
            self.graph.add_vertices(to_add)  # this sets 'name' correctly

        # Resolve internal vertex indices
        try:
            u_idx = self.graph.vs.find(name=device_id).index
            v_idx = self.graph.vs.find(name=nearest_node).index
        except ValueError as e:
            raise ValueError(f"Vertex resolution failed after addition: {e}")

        # Add edge if it doesn't already exist
        try:
            eid = self.graph.get_eid(u_idx, v_idx)
        except (ValueError, InternalError):
            self.graph.add_edge(u_idx, v_idx)
            eid = self.graph.get_eid(u_idx, v_idx)

        # Set segment attributes
        self.graph.es[eid]['length'] = 0
        self.graph.es[eid]['cost'] = 0

        # Update the KS mapping and RoadSegment
        segment = RoadSegment(device_id, nearest_node, 0, 0)
        self.segments[(device_id, nearest_node)] = segment
        self.segments[(nearest_node, device_id)] = segment
        self.KS[(device_id, nearest_node)] = 0
        self.KS[(nearest_node, device_id)] = 0
        self.node_positions[device_id] = device_pos

        self.build_name_to_index()  # Rebuild cache after modification

    def build_name_to_index(self):
        """Builds and caches the name→index map and vertex name list."""
        if 'name' not in self.graph.vs.attributes():
            self.vs_names = []
            self.name_to_index = {}
            return

        self.vs_names = self.graph.vs['name']
        self.name_to_index = {name: idx for idx, name in enumerate(self.vs_names)}

    # % =================================================================
    # % PATH COMPUTING FUNCTIONS
    # % =================================================================
    
    def precompute_shortest_paths(self):
        """Precomputes and caches shortest paths and path lengths between all named nodes using igraph."""

        # Update name caches if not already done
        self.vs_names = self.vs_names
        self.name_to_index = self.name_to_index

        self.shortest_paths = {}
        self.shortest_path_lengths = {}

        for from_name in self.vs_names:
            from_idx = self.name_to_index[from_name]
            self.shortest_paths[from_name] = {}
            self.shortest_path_lengths[from_name] = {}

            # Compute all paths and distances from this source
            dists = self.graph.shortest_paths_dijkstra(source=from_idx, weights='cost')[0]
            paths = self.graph.get_shortest_paths(from_idx, to=None, weights='cost', output='vpath')

            for j, to_name in enumerate(self.vs_names):
                if from_name == to_name:
                    continue

                path = paths[j]
                if not path:
                    continue  # Unreachable

                name_path = [self.vs_names[k] for k in path]
                self.shortest_paths[from_name][to_name] = name_path
                self.shortest_path_lengths[from_name][to_name] = dists[j]

    # % =================================================================
    # % PATH MANAGEMENT FUNCTIONS
    # % =================================================================

    def get_excluded_edges(self) -> Set[Tuple[str, str]]:
        """Returns a set of edges to exclude, where usage_count >= 1."""
        return {(seg.from_node, seg.to_node) for seg in self.segments.values() if seg.usage_count >= 1}

    def single_path_cost(self, path: List[str], excluded_edges: Set[Tuple[str, str]] = set(), current_best_cost: float = float('inf')) -> float:
        """Computes the cost of a path, only counting NEW segments (not in excluded_edges). Terminates early if over best."""
        total = 0
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            if edge not in excluded_edges:
                total += self.KS.get(edge, 0)
                if total >= current_best_cost:
                    return float('inf')
        return total

    def compute_best_path(self, source, target, excluded_edges: Set[Tuple[str, str]] = set(), current_best_cost=float('inf'), current_best_path=None):
        """Finds the best path from source to target, considering excluded edges and current best path/cost."""
        source_id = getattr(source, 'name', source)
        target_id = getattr(target, 'name', target)

        path, path_cost = self.compute_dijkstra_path(source_id, target_id, excluded_edges)
        if path and path_cost < current_best_cost:
            return path, path_cost
        return current_best_path, current_best_cost

    def compute_dijkstra_path(self, source_id: str, target_id: str, excluded_edges: Set[Tuple[str, str]]):
        """Computes Dijkstra's path using segment costs; excluded edges get 0 cost (cheaper)."""

        # Early exit if source and target are the same
        if source_id == target_id:
            return [source_id], 0.0

        # Use cached vs_names and name_to_index
        name_to_index = self.name_to_index
        vs_names = self.vs_names
        get_eid = self.graph.get_eid

        source_idx = name_to_index.get(source_id)
        target_idx = name_to_index.get(target_id)
        if source_idx is None or target_idx is None:
            return None, float('inf')

        # Build a set of excluded edge indices (existing edges to be set to cost 0)
        excluded_eids = set()
        for u, v in excluded_edges:
            try:
                eid = get_eid(name_to_index[u], name_to_index[v], directed=False)
                excluded_eids.add(eid)
            except (KeyError, ig._igraph.InternalError):
                continue

        # Prepare edge weights
        costs = self.graph.es['cost']
        if not excluded_eids:
            weights = costs
        else:
            weights = list(costs)
            for eid in excluded_eids:
                weights[eid] = 0

        # Use igraph's get_shortest_paths with both vertex and edge outputs
        path_indices = self.graph.get_shortest_paths(source_idx, to=target_idx, weights=weights, output='vpath')[0]
        path_eids = self.graph.get_shortest_paths(source_idx, to=target_idx, weights=weights, output='epath')[0]

        if not path_indices:
            return None, float('inf')

        # Convert vertex indices to node names
        path = [vs_names[i] for i in path_indices]

        # Compute total cost from edge IDs
        path_cost = sum(weights[eid] for eid in path_eids)

        return path, path_cost

    def build_allowed_subgraph(self, allowed_edges: Set[Tuple[str, str]]) -> ig.Graph:
        """Returns a subgraph containing only the allowed edges."""
        name_to_index = self.name_to_index  # Cached mapping
        get_eid = self.graph.get_eid

        allowed_edge_ids = []
        for u, v in allowed_edges:
            u_idx = name_to_index.get(u)
            v_idx = name_to_index.get(v)
            if u_idx is None or v_idx is None:
                continue
            try:
                eid = get_eid(u_idx, v_idx, directed=False)
                allowed_edge_ids.append(eid)
            except ig._igraph.InternalError:
                continue  # Edge doesn't exist

        return self.graph.subgraph_edges(allowed_edge_ids, delete_vertices=False)

    def compute_allowed_best_path(self, source, target, allowed_subgraph: ig.Graph, current_best_cost=float('inf'), current_best_path=None):
        """Finds best path from source to target in allowed_subgraph. Returns existing if no improvement."""

        src_id = getattr(source, 'name', source)
        tgt_id = getattr(target, 'name', target)

        # Cache lookup
        try:
            v_names = allowed_subgraph.vs_names
            name_to_index = allowed_subgraph.name_to_index
        except AttributeError:
            if 'name' not in allowed_subgraph.vs.attributes(): return current_best_path, current_best_cost
            v_names = allowed_subgraph.vs['name']
            name_to_index = {name: idx for idx, name in enumerate(v_names)}
            allowed_subgraph.vs_names = v_names
            allowed_subgraph.name_to_index = name_to_index

        src_idx = name_to_index.get(src_id)
        tgt_idx = name_to_index.get(tgt_id)
        if src_idx is None or tgt_idx is None:
            return current_best_path, current_best_cost

        try:
            indices = allowed_subgraph.get_shortest_paths(src_idx, to=tgt_idx, weights='cost', output='epath')[0]
            if not indices: return current_best_path, current_best_cost

            path_vertices = allowed_subgraph.get_shortest_paths(src_idx, to=tgt_idx, weights='cost', output='vpath')[0]
            path_names = [v_names[i] for i in path_vertices]

            # Fast cost lookup from edge attributes
            edge_costs = allowed_subgraph.es['cost']
            cost = 0
            for eid in indices:
                seg_cost = edge_costs[eid]
                cost += seg_cost
                if cost >= current_best_cost:
                    return current_best_path, current_best_cost  # Prune early

            return path_names, cost

        except Exception as e:
            loggers['Path_logger'].warning(f"compute_allowed_best_path failed: {e}")
            return current_best_path, current_best_cost

    # % =================================================================
    # % MST BUILDING FUNCTIONS
    # % =================================================================

    def build_mst(self, selected_RUs, selected_DUs, selected_CUs):
        """Applies the Minimum steiner tree using road segments. Returns used segments, MST subgraph, and total cost."""
        if not self.shortest_paths or not self.shortest_path_lengths:
            raise ValueError("Shortest paths have not been precomputed. Call precompute_shortest_paths() first.")
    
        required_nodes = list(selected_RUs | selected_DUs | selected_CUs)

        node_indices = {name: i for i, name in enumerate(required_nodes)}
        reverse_indices = {i: name for name, i in node_indices.items()}

        shortest_paths = self.shortest_paths
        shortest_path_lengths = self.shortest_path_lengths
        segments = self.segments
        name_to_index = self.name_to_index

        edges = []
        weights = []
        path_lookup = {}

        # Generate edges for meta-graph using cached paths
        for u, v in combinations(required_nodes, 2):
            path_dict = shortest_paths.get(u)
            if not path_dict or v not in path_dict:
                continue
            path = path_dict[v]
            cost = shortest_path_lengths[u][v]
            i, j = node_indices[u], node_indices[v]
            edges.append((i, j))
            weights.append(cost)
            path_lookup[(u, v)] = path

        # Create meta-graph and MST
        meta_graph = ig.Graph(n=len(required_nodes), edges=edges)
        meta_graph.es['weight'] = weights
        mst_tree = meta_graph.spanning_tree(weights=meta_graph.es['weight'])

        # Extract MST edges from original graph
        mst_edges = set()
        get_path = path_lookup.get

        for e in mst_tree.es:
            u = reverse_indices[e.source]
            v = reverse_indices[e.target]
            path = get_path((u, v)) or get_path((v, u))
            if path:
                mst_edges.update((path[i], path[i + 1]) for i in range(len(path) - 1))

        seen_segments = set()
        final_edges = []
        used_segments = []
        total_segment_cost = 0
        total_distance = 0

        log_warn = loggers['device_logger'].warning
        graph = self.graph

        for u, v in mst_edges:
            if (u, v) in seen_segments or (v, u) in seen_segments:
                continue
            seen_segments.add((u, v))

            segment = segments.get((u, v)) or segments.get((v, u))
            if segment:
                cost = segment.cost
                length = segment.length
                used_segments.append(segment)
            else:
                # Use cached index-based lookup
                u_idx = name_to_index.get(u)
                v_idx = name_to_index.get(v)
                if u_idx is None or v_idx is None:
                    continue

                eid = graph.get_eid(u_idx, v_idx, directed=False, error=False)
                if eid == -1:
                    log_warn(f"build_mst: No segment or edge found for {u} to {v}.")
                    continue

                cost = graph.es[eid]['cost']
                length = graph.es[eid]['length']
                log_warn(f"build_mst: No segment found for edge {u} to {v}.")

            final_edges.append((u, v, cost, length))
            total_segment_cost += cost
            total_distance += length

        # Construct MST subgraph with original node names
        node_list = list({u for u, _, _, _ in final_edges} | {v for _, v, _, _ in final_edges})
        mst_subgraph = ig.Graph(directed=False)
        mst_subgraph.add_vertices(node_list)

        name_to_idx = {name: idx for idx, name in enumerate(node_list)}
        indexed_edges = [(name_to_idx[u], name_to_idx[v]) for u, v, _, _ in final_edges]

        mst_subgraph.add_edges(indexed_edges)
        mst_subgraph.es['cost'] = [c for _, _, c, _ in final_edges]
        mst_subgraph.es['length'] = [l for _, _, _, l in final_edges]
        mst_subgraph.name_to_index = name_to_idx
        mst_subgraph.vs_names = node_list

        return used_segments, mst_subgraph, total_segment_cost

    def find_best_du_mst(self, selected_RUs, selected_CU, candidate_DUs, min_dus):
        """
        Finds the best set of DUs (from candidate_DUs) to add to selected_RUs + selected_CU
        such that the MST cost is minimised. Only considers DU combinations of size == min_dus.
        Returns used segments, MST subgraph, best DUs and best cost.
        """

        if not self.shortest_paths or not self.shortest_path_lengths:
            raise ValueError("Shortest paths have not been precomputed.")

        def get_name(n):
            return getattr(n, 'name', n)

        base_nodes = set(get_name(n) for n in selected_RUs)
        if isinstance(selected_CU, set):
            base_nodes.update(get_name(n) for n in selected_CU)
        else:
            base_nodes.add(get_name(selected_CU))
        base_nodes = list(base_nodes)

        du_names = [get_name(du) for du in candidate_DUs]
        min_dus = max(1, min_dus)
        best_cost = float('inf')
        best_du_set = None
        best_segments = None
        best_mst_subgraph = None

        current_selected_dus = set(get_name(du) for du in candidate_DUs if getattr(du, "is_selected", False))

        # Only consider DU combinations of size == min_dus
        if len(du_names) < min_dus:
            loggers['Path_logger'].warning(f"Not enough candidate DUs to meet min_dus={min_dus}")
            return [], None, []

        # Only try combinations of exactly min_dus DUs
        for du_combo in combinations(du_names, min_dus):
            loggers['Path_logger'].info(f"Evaluating DU combination: {du_combo}, of size {min_dus}")
            used_segments, mst_subgraph, total_cost = self.build_mst(
                selected_RUs=set(base_nodes),
                selected_DUs=set(du_combo),
                selected_CUs={get_name(selected_CU)} if not isinstance(selected_CU, set) else set(get_name(n) for n in selected_CU)
            )

            if total_cost < best_cost:
                best_cost = total_cost
                best_du_set = du_combo
                best_segments = used_segments
                best_mst_subgraph = mst_subgraph
            elif total_cost == best_cost:
                if best_du_set is not None:
                    overlap_new = len(current_selected_dus.intersection(du_combo))
                    overlap_old = len(current_selected_dus.intersection(best_du_set))
                    if overlap_new < overlap_old:
                        best_du_set = du_combo
                        best_segments = used_segments
                        best_mst_subgraph = mst_subgraph
                else:
                    best_du_set = du_combo
                    best_segments = used_segments
                    best_mst_subgraph = mst_subgraph

        if best_du_set is None:
            return [], None, []
        
        loggers['Path_logger'].info(f"Best DU set: {best_du_set} with MST cost {best_cost}")

        return best_segments, best_mst_subgraph, best_du_set, best_cost
