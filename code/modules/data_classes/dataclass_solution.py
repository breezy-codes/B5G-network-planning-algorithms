"""
%┏━━━┓┏━━━┓┏┓━━━┏┓━┏┓┏━━━━┓┏━━┓┏━━━┓┏━┓━┏┓━━━━┏━┓┏━┓┏━━━┓┏━━━┓┏┓━┏┓┏┓━━━┏━━━┓
?┃┏━┓┃┃┏━┓┃┃┃━━━┃┃━┃┃┃┏┓┏┓┃┗┫┣┛┃┏━┓┃┃┃┗┓┃┃━━━━┃┃┗┛┃┃┃┏━┓┃┗┓┏┓┃┃┃━┃┃┃┃━━━┃┏━━┛
%┃┗━━┓┃┃━┃┃┃┃━━━┃┃━┃┃┗┛┃┃┗┛━┃┃━┃┃━┃┃┃┏┓┗┛┃━━━━┃┏┓┏┓┃┃┃━┃┃━┃┃┃┃┃┃━┃┃┃┃━━━┃┗━━┓
?┗━━┓┃┃┃━┃┃┃┃━┏┓┃┃━┃┃━━┃┃━━━┃┃━┃┃━┃┃┃┃┗┓┃┃━━━━┃┃┃┃┃┃┃┃━┃┃━┃┃┃┃┃┃━┃┃┃┃━┏┓┃┏━━┛
%┃┗━┛┃┃┗━┛┃┃┗━┛┃┃┗━┛┃━┏┛┗┓━┏┫┣┓┃┗━┛┃┃┃━┃┃┃━━━━┃┃┃┃┃┃┃┗━┛┃┏┛┗┛┃┃┗━┛┃┃┗━┛┃┃┗━━┓
?┗━━━┛┗━━━┛┗━━━┛┗━━━┛━┗━━┛━┗━━┛┗━━━┛┗┛━┗━┛━━━━┗┛┗┛┗┛┗━━━┛┗━━━┛┗━━━┛┗━━━┛┗━━━┛
%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
?━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from dataclasses import dataclass, field
import math, random
from collections import defaultdict

from modules.logger import loggers
from modules.logger import log_begin, log_end, log_fail, log_fail_no_exception
from modules import config

@dataclass
class Solution:
    RUs: dict = field(default_factory=dict)   # Dictionary of RUs
    DUs: dict = field(default_factory=dict)   # Dictionary of DUs
    CUs: dict = field(default_factory=dict)   # Dictionary of CUs
    users: dict = field(default_factory=dict) # Dictionary of Users
    graph: dict = field(default_factory=dict) # Graph object for roads and segments

    total_cost: float = field(default=0)                        # Total cost of the solution
    ru_to_du_connections: dict = field(default_factory=dict)    # Connections from RUs to DUs
    du_to_cu_connections: dict = field(default_factory=dict)    # Connections from DUs to CUs
    selected_RUs: set = field(default_factory=set)              # Selected RUs in the solution
    selected_DUs: set = field(default_factory=set)              # Selected DUs in the solution
    assigned_users: set = field(default_factory=set)            # Assigned users in the solution
    unassigned_users: set = field(default_factory=set)          # Unassigned users in the solution
    coverage_percentage: float = field(default=0)               # Coverage percentage of the solution
    cost_of_RUs: float = field(default=0)           # Cost of RUs in the solution
    cost_of_DUs: float = field(default=0)           # Cost of DUs in the solution
    cost_of_CUs: float = field(default=0)           # Cost of CUs in the solution
    total_segment_cost: float = field(default=0)    # Total cost of road segments in the solution
    total_distance_used: float = field(default=0)   # Total distance used in the solution

    #* ---- CONSTANTS FROM CONFIG ---- *#
    RU_BASE_BANDWIDTH = config.RU_BASE_BANDWIDTH  # Base bandwidth per RU
    CR_VALUE = config.CR_VALUE                    # Max fibre connections between DU and CU
    FC_VALUE = config.FC_VALUE                    # Bandwidth per fibre
    MD_VALUE = config.MD_VALUE                    # Max DUs per location
    UM_VALUE = config.UM_VALUE                    # Bandwidth required per user

    @property
    def UC(self):
        """Return the number of users that can be covered by the solution."""
        return math.ceil(len(self.users) * config.COVERAGE_THRESHOLD)

    def __deepcopy__(self):
        """Create a deep copy of the solution."""
        RUs = {name: ru.make_copy() for name, ru in self.RUs.items()}
        DUs = {name: du.make_copy() for name, du in self.DUs.items()}
        CUs = {name: cu.make_copy() for name, cu in self.CUs.items()}
        users = {name: user.make_copy() for name, user in self.users.items()}
        graph = self.graph.make_copy()
        solution = Solution(RUs, DUs, CUs, users, graph)
        solution.set_solutions()
        return solution
    
    def set_solutions(self):
        """Set the solution for each device in the solution."""
        for group in [self.RUs, self.DUs, self.CUs, self.users]:
            for obj in group.values():
                obj.set_solution(self)

    # % =================================================================
    # % RETRIEVE OBJECTS OR NAMES
    # % =================================================================

    def get_user_assignments(self):
        """Get the assigned and unassigned users in the solution. This doesn't update the state."""
        assigned_users = {user.user_id: user.assigned_ru for user in self.users.values() if user.assigned_ru}
        unassigned_users = {user.user_id: None for user in self.users.values() if not user.assigned_ru}
        return assigned_users, unassigned_users
    
    def get_coverage_percentage(self):
        """Calculate the coverage percentage of the solution, updating user states if need be."""
        total_users = len(self.users)
        assigned_users = {user.user_id: user.assigned_ru for user in self.users.values() if user.assigned_ru}
        coverage_percentage = len(assigned_users) / total_users * 100 if total_users else 0
        return coverage_percentage
    
    def get_selected_RUs(self):
        """Get the selected RU names (set) in the solution."""
        return {ru.name for ru in self.RUs.values() if getattr(ru, "is_selected", False)}

    def get_selected_RU_objects(self):
        """Get the selected RU objects in the solution."""
        return [ru for ru in self.RUs.values() if getattr(ru, "is_selected", False)]

    def get_selected_DUs(self):
        """Get the selected DU names (set) in the solution."""
        return {du.name for du in self.DUs.values() if getattr(du, "is_selected", False)}

    def get_selected_DU_objects(self):
        """Get the selected DU objects in the solution."""
        return [du for du in self.DUs.values() if getattr(du, "is_selected", False)]

    def get_selected_CUs(self):
        """Get the selected CU names (set) in the solution."""
        return {cu.name for cu in self.CUs.values() if getattr(cu, "is_selected", False)}

    def get_user(self, user):
        """Get a user by ID or return the user object if already an object."""
        if isinstance(user, str):
            return self.users.get(user, None)
        return user

    def get_ru(self, ru): 
        """Get a RU by name or return the RU object if already an object."""
        if isinstance(ru, str):
            return self.RUs.get(ru)
        return ru

    def get_du(self, du): 
        """Get a DU by name or return the DU object if already an object."""
        if isinstance(du, str):
            return self.DUs.get(du)
        return du

    def get_cu(self, cu): 
        """Get a CU by name or return the CU object if already an object."""
        if isinstance(cu, str):
            return self.CUs.get(cu)
        return cu
    
    def get_cost_components(self):
        """Compute the total cost components of the solution."""
        ru_cost = sum(ru.total_cost for ru in self.RUs.values() if ru.is_selected)
        du_cost = sum(du.total_cost for du in self.DUs.values() if du.is_selected)
        cu_cost = sum(cu.total_cost for cu in self.CUs.values() if cu.is_selected)
        segment_cost = self.graph.total_road_cost
        return ru_cost, du_cost, cu_cost, segment_cost
    
    def get_total_costs(self):
        """Compute the total costs used in the solution."""
        ru_cost, du_cost, cu_cost, segment_cost = self.get_cost_components()
        total_cost_reg = ru_cost + du_cost + cu_cost + segment_cost
        return total_cost_reg
    
    def get_total_distances(self):
        """Compute the total distances used in the solution."""
        total_distance_used = self.graph.total_road_distance
        return total_distance_used
    
    def get_just_distance_cost(self):
        """Get the distance used in the solution."""
        total_road_cost = self.graph.total_road_cost
        return total_road_cost

    def get_min_dus_needed(self):
        """Calculate the minimum number of DUs needed to cover all assigned users based on total RU or user capacity (whichever is largest).
        If total_ru_capacity is a perfect multiple of max_capacity_per_du (e.g., exactly 2 DUs, 3 DUs, etc.), require one extra DU than minimum needed."""
        
        total_ru_capacity = sum(self.RU_BASE_BANDWIDTH * ru.num_rus for ru in self.RUs.values() if ru.is_selected)
        max_capacity_per_du = (self.CR_VALUE * self.FC_VALUE) * self.MD_VALUE

        if max_capacity_per_du == 0:
            return 0
        
        min_dus = math.ceil(total_ru_capacity / max_capacity_per_du)

        # If total_ru_capacity is a perfect multiple of max_capacity_per_du, add one extra DU
        if total_ru_capacity != 0 and total_ru_capacity % max_capacity_per_du == 0:
            min_dus += 1

        return min_dus

    # % =================================================================
    # % UPDATE FUNCTIONS
    # % =================================================================

    def update_ru_to_du_connections(self):
        """Get and set the RU to DU connections in the solution."""
        self.ru_to_du_connections = {ru.name: ru.connected_du for ru in self.RUs.values() if ru.is_selected and ru.connected_du}
    
    def update_du_to_cu_connections(self):
        """Get and set the DU to CU connections in the solution."""
        self.du_to_cu_connections = {du.name: du.connected_cu for du in self.DUs.values() if du.is_selected and du.connected_cu}

    def update_user_assignments(self):
        """Efficiently update assigned and unassigned users in the solution."""
        assigned = {}
        unassigned = {}
        for user in self.users.values():
            if user.assigned_ru:
                assigned[user.user_id] = user.assigned_ru
            else:
                unassigned[user.user_id] = None
        self.assigned_users = assigned
        self.unassigned_users = unassigned
    
    def update_coverage_percentage(self):
        """Calculate the coverage percentage of the solution, updating user states if need be."""
        total_users = len(self.users)
        self.update_user_assignments()
        self.coverage_percentage = len(self.assigned_users) / total_users * 100 if total_users else 0
    
    def update_RU_states(self):
        """Update the selected RUs in the solution."""
        self.selected_RUs = {ru.name for ru in self.RUs.values() if ru.is_selected}

    def update_DU_states(self):
        """Update the selected DUs in the solution."""
        self.selected_DUs = {du.name for du in self.DUs.values() if du.is_selected}

    def update_CU_states(self):
        """Update the selected CUs in the solution."""
        self.selected_CUs = {cu.name for cu in self.CUs.values() if cu.is_selected}

    def update_total_costs(self):
        """Update the total costs of the solution."""
        self.cost_of_RUs = sum(ru.total_cost for ru in self.RUs.values() if ru.is_selected)
        self.cost_of_DUs = sum(du.total_cost for du in self.DUs.values() if du.is_selected)
        self.cost_of_CUs = sum(cu.total_cost for cu in self.CUs.values() if cu.is_selected)
        self.total_segment_cost = self.graph.total_road_cost
        self.total_distance_used = self.graph.total_road_distance
        self.total_cost = self.cost_of_RUs + self.cost_of_DUs + self.cost_of_CUs + self.total_segment_cost      

    def update_model(self):
        """Update the model by recalculating the costs and connections."""
        for ru in self.RUs.values():
            ru.update_ru()

        for du in self.DUs.values():
            du.update_du()

        for cu in self.CUs.values():
            cu.update_cu()

        for user in self.users.values():
            user.update_user()

        self.graph.update_road_results()
        self.update_coverage_percentage()
        self.update_total_costs()

    # % =================================================================
    # % GENERAL HEURISTIC FUNCTIONS
    # % =================================================================
    def evaluate_fitness(self, cover_penalty_base=1000):
        """Evaluates fitness with a tunable coverage penalty multiplier."""
        #ru_cost, du_cost, cu_cost, segment_cost = self.get_cost_components()
        total_cost_reg = self.get_total_costs()
        #distance_used,= self.get_total_distances()
        assigned_users, unassigned_users = self.get_user_assignments()

        #distance_penalty = distance_used ** 2
        coverage_diff = max(0, (self.UC - len(assigned_users)) * cover_penalty_base ** 2)

        #DU_penalty = len(self.get_selected_DU_objects()) ** 75
        #RU_penalty = len(self.get_selected_RU_objects()) ** 25

        fitness_value = coverage_diff + total_cost_reg
        return fitness_value

    def fitness(sol):
        """Evaluate the fitness of a solution."""
        return sol.evaluate_fitness()
    
    # % =================================================================
    # % MUTATE FUNCTION
    # % =================================================================

    def run_mutation(self, num_mutations=1):
        """Run up to num_mutations mutations, never applying rebuild_paths or mutate_du twice in a row."""
        self.applied_mutations = []
        mutation_methods = [
            self.mutate_ru,
            self.mutate_ru,
            self.swap_rus,
            #self.mutate_du,
            self.mutate_users,
            #self.mutate_user_count,
            #self.mutate_ru_du_assignments,
            self.rebuild_paths,
            #self.swap_ru_path,
            #self.swap_du_path,
        ]

        mutations_applied = 0
        last_mutation = None
        while mutations_applied < num_mutations:
            mutation = random.choice(mutation_methods)
            # Prevent rebuild_paths or swap_rus from being applied twice in a row
            if last_mutation in [self.rebuild_paths, self.swap_rus] and mutation == last_mutation:
                continue
            mutation()
            self.applied_mutations.append(mutation.__name__)
            self.update_model()
            last_mutation = mutation
            mutations_applied += 1

    def mutate_individual(self, solution, num_mutations):
        """Mutate an individual solution."""
        solution.run_mutation(num_mutations)
        return solution
    
    # % =================================================================
    # % FEASIBILITY CHECKS
    # % =================================================================

    def is_feasible(self):
        """Check if the solution is feasible.
        Most of thse checks are ones that involve violating the constraints. Like an RU can't be selected and have 0 num RUs.
        """
        feasible = True

        #? This does a series of basic RU checks
        for ru_name, ru in self.RUs.items():
            if ru.is_selected and not ru.connected_du:
                loggers['feasible_logger'].error(f"RU {ru_name} is selected but has no connected DU.")
                feasible = False
            if ru.is_selected and ru.num_rus == 0:
                loggers['feasible_logger'].error(f"RU {ru_name} is selected but has no RUs.")
                feasible = False

        #? This does a series of basic DU checks
        for du_name, du in self.DUs.items():
            if du.is_selected and not du.connected_cu:
                loggers['feasible_logger'].error(f"DU {du_name} is selected but has no connected CU.")
                feasible = False
            if du.is_selected and du.fibre_du_cu == 0:
                loggers['feasible_logger'].error(f"DU {du_name} is selected but has no CUs.")
                feasible = False

        #? This does a series of basic CU checks
        for cu_name, cu in self.CUs.items():
            if cu.is_selected and not cu.connected_dus:
                loggers['feasible_logger'].error(f"CU {cu_name} is selected but has no connected DUs.")
                feasible = False
            if cu.is_selected and cu.total_fibres == 0:
                loggers['feasible_logger'].error(f"CU {cu_name} is selected but has no fibres.")
                feasible = False

        #? This checks if the total sum of capacity from the RUs matches the total number of users assigned
        total_ru_users = sum(ru.used_capacity // self.UM_VALUE for ru in self.RUs.values() if ru.is_selected)
        assigned_users_count = len(self.assigned_users)
        if total_ru_users != assigned_users_count:
            loggers['feasible_logger'].error(f"Total RU users ({total_ru_users}) does not match assigned users ({assigned_users_count}).")
            feasible = False

        #? This checks if number of devices mach up with number of ports used
        total_RUs = sum(ru.num_rus for ru in self.RUs.values() if ru.is_selected)
        total_ports = sum(du.used_ports for du in self.DUs.values() if du.is_selected)
        total_fibre_ru_du = sum(du.fibre_ru_du for du in self.DUs.values() if du.is_selected)
        if total_RUs != total_ports:
            loggers['feasible_logger'].error(f"Total RUs ({total_RUs}) does not match total ports ({total_ports}).")
            feasible = False
        if total_RUs != total_fibre_ru_du:
            loggers['feasible_logger'].error(f"Total RUs ({total_RUs}) does not match total fibre RU-DU ({total_fibre_ru_du}).")
            feasible = False

        #? Check RU/DU Fibre Mismatches
        expected_fibres = sum(self.RUs[ru_name].num_rus for ru_name in du.connected_rus if ru_name in self.RUs)
        if du.fibre_ru_du != expected_fibres:
            loggers['feasible_logger'].error(f"DU {du_name} fibre RU-DU mismatch: {du.fibre_ru_du} != {total_RUs}.")
            feasible = False
        if du.used_ports != expected_fibres:
            loggers['feasible_logger'].error(f"DU {du_name} port mismatch: {du.used_ports} != {expected_fibres}.")
            feasible = False

        #? Check DU/CU Fibre Mismatches
        expected_cu_fibres = sum(self.DUs[du_name].fibre_du_cu for du_name in cu.connected_dus if du_name in self.DUs)
        if cu.total_fibres != expected_cu_fibres:
            loggers['feasible_logger'].error(f"CU {cu_name} fibre DU-CU mismatch: {cu.total_fibres} != {expected_cu_fibres}.")
            feasible = False

        #? This checks if the total sum of capacity from the RUs matches the total sum of used capacity from the DUs
        total_ru_bw = sum(ru.base_capacity_bandwidth * ru.num_rus for ru in self.RUs.values() if ru.is_selected)
        total_du_bw = sum(du.used_bandwidth for du in self.DUs.values() if du.is_selected)
        if total_ru_bw != total_du_bw:
            loggers['feasible_logger'].error(f"RU bandwidth ({total_ru_bw}) does not match DU bandwidth ({total_du_bw}).")
            feasible = False
        
        if feasible:
            #loggers['feasible_logger'].debug("Solution is feasible.")
            pass
        else:
            loggers['feasible_logger'].error("Solution is NOT feasible ---------------------------------------------------")

        return feasible

    # % =================================================================
    # % RU MUTATIONS
    # % =================================================================

    def mutate_ru(self):
        """Mutate a random number of RUs (1-5) by toggling their active state, then adjust coverage."""
        func = "mutate_ru"
        all_rus = list(self.RUs.values()) # Get all RUs in the solution
        if not all_rus: return            # Early exit if no RUs

        num_to_mutate = random.randint(1, min(2, len(all_rus))) # Randomly choose how many RUs to mutate (1-3, or less if not enough RUs)
        rus_to_mutate = random.sample(all_rus, num_to_mutate)   # Randomly select RUs to mutate

        #log_begin(loggers['mutate_logger'], func, f"Toggling {num_to_mutate} RUs: {[ru.name for ru in rus_to_mutate]}")

        for ru in rus_to_mutate:
            ru_name = ru.name
            if ru.is_selected:
                #loggers['mutate_logger'].info(f"{func}: Deactivating RU {ru_name}.")
                self.deactivate_ru(ru_name) # Deactivate the RU
            else:
                #loggers['mutate_logger'].info(f"{func}: Activating RU {ru_name}.")
                self.activate_ru(ru_name)   # Activate the RU

        self.ensure_coverage_requirement() # Ensure coverage requirement is met after mutation
        #log_end(loggers['mutate_logger'], func, f"Mutated {num_to_mutate} RUs.")

    def swap_rus(self):
        """Activate a random RU, then deactivate an active RU whose users can be reassigned."""
        func = "swap_rus"
        num_to_swap = 1

        selected_rus = self.get_selected_RU_objects()
        unselected_rus = [ru for ru in self.RUs.values() if not ru.is_selected]

        if len(unselected_rus) < num_to_swap or len(selected_rus) <= num_to_swap:
            loggers['mutate_logger'].warning(f"{func}: Not enough RUs to swap, num selected is {len(selected_rus)}, num unselected is {len(unselected_rus)}.")
            return

        # Activate new RU(s)
        rus_to_activate = random.sample(unselected_rus, num_to_swap)
        activated_names = {ru.name for ru in rus_to_activate}
        for ru in rus_to_activate:
            self.activate_ru(ru.name)

        # Index users by current RU
        ru_to_users = defaultdict(list)
        for user in self.users.values():
            if user.assigned_ru:
                ru_to_users[user.assigned_ru].append(user)

        # Build map: user to potential reassignment RUs
        user_candidates = {}
        for user in self.users.values():
            if not user.assigned_ru:
                continue
            potentials = []
            for ru_name in user.potential_rus:
                if ru_name == user.assigned_ru:
                    continue
                ru = self.get_ru(ru_name)
                if ru and user.is_potential_ru(ru) and ru.true_false_can_add_user(user):
                    potentials.append(ru_name)
            if potentials:
                user_candidates[user.user_id] = potentials

        # Invert to RU list so users it can accept
        ru_to_candidate_users = defaultdict(list)
        for user_name, ru_list in user_candidates.items():
            for ru in ru_list:
                ru_to_candidate_users[ru].append(user_name)

        # Find best RUs to deactivate
        ru_deactivation_stats = []
        for ru in selected_rus:
            if ru.name in activated_names:
                continue
            assigned_users = ru_to_users.get(ru.name, [])
            if not assigned_users:
                continue

            reassignable = 0
            unassignable = 0

            for user in assigned_users:
                user_options = user_candidates.get(user.user_id, [])
                can_reassign = any(user.user_id in ru_to_candidate_users.get(potential_ru, []) for potential_ru in user_options)

                if can_reassign:
                    reassignable += 1
                else:
                    unassignable += 1

            ru_deactivation_stats.append((ru, reassignable, unassignable))

        if not ru_deactivation_stats:
            return

        # Sort by fewest unassignable, then most reassignable
        ru_deactivation_stats.sort(key=lambda x: (x[2], -x[1]))
        rus_to_deactivate = [ru for ru, _, _ in ru_deactivation_stats[:num_to_swap]]

        for ru in rus_to_deactivate:
            self.deactivate_ru(ru.name)
        
        self.ensure_coverage_requirement() # Ensure coverage requirement is met after mutation

    def mutate_ru_du_assignments(self):
        """Mutate the RU-DU assignments by reassigning RUs to DUs."""
        func = "mutate_ru_du_assignments"
        selected_rus = self.get_selected_RU_objects() # Get the currently selected RUs
        
        # Check if there are any selected RUs to mutate
        if not selected_rus: 
            return

        num_rus_to_mutate = random.randint(1, min(5, len(selected_rus))) # Randomly select 1-5 RUs to mutate
        rus_to_mutate = random.sample(selected_rus, num_rus_to_mutate)   # Randomly select RUs to mutate

        #log_begin(loggers['mutate_logger'], func, f"Mutating RU-DU assignments for {num_rus_to_mutate} RUs.")       

        for ru in rus_to_mutate:
            self.reassign_ru_du_swap(ru) # Reassign RU to a different DU

        #log_end(loggers['mutate_logger'], func, f"Mutated RU-DU assignments for {num_rus_to_mutate} RUs.")

    def ensure_coverage_requirement(self):
        """Assign as many unassigned users as possible to already-selected RUs (no new RUs), stopping at UC."""
        func = "ensure_coverage_requirement"
        assigned_users, unassigned_users = self.get_user_assignments()
        if not unassigned_users:
            return

        # Only consider RUs that are already selected and can take at least one more user
        ru_list = [ru for ru in self.RUs.values() if ru.is_selected and ru.true_false_can_add_user(None)]
        if not ru_list:
            return

        user_objs = [self.get_user(uid) for uid in unassigned_users]
        assigned_count = 0
        UC_limit = self.UC
        current_assigned = len(assigned_users)

        for user in user_objs:
            if current_assigned + assigned_count > UC_limit:
                break
            potential_rus = [ru for ru in ru_list if user.is_potential_ru(ru) and ru.true_false_can_add_user(user)]
            for ru in potential_rus:
                if self.assign_user(user, ru):
                    assigned_count += 1
                    #loggers['mutate_logger'].info(f"{func}: Assigned user {user.user_id} to RU {ru.name}.")
                    break

        #loggers['mutate_logger'].info(f"{func}: Assigned {assigned_count} users to existing RUs (Total now: {current_assigned + assigned_count}).")
        self.update_user_assignments()

    # % =================================================================
    # % DU MUTATIONS
    # % =================================================================

    def mutate_du(self):
        """Mutate DUs by activating, deactivating, or swapping a random selection of DUs."""
        func = "mutate_du"
        min_dus_needed = self.get_min_dus_needed()           # Calculate the minimum DUs needed
        selected_DUs_set = self.get_selected_DUs()           # Get the currently selected DUs
        num_selected_dus = len(selected_DUs_set)             # Count the number of selected DUs
        available_dus = self.DUs.keys() - selected_DUs_set   # Get the available DUs that are not currently selected

        # If the number of selected DUs is equal to the minimum needed, swap or activate DU
        if num_selected_dus == min_dus_needed:
            # Prefer swap if possible, else activate a DU
            if available_dus and random.random() < 0.4:
                du_to_activate = random.choice(list(available_dus))
                #log_begin(loggers['mutate_logger'], func, f"Activating DU: {du_to_activate}")
                self.activate_du(du_to_activate)
                #log_end(loggers['mutate_logger'], func, f"Activated DU: {du_to_activate}")
            else:
                # Swap: deactivate one DU and activate another
                if selected_DUs_set and available_dus:
                    du_to_deactivate = random.choice(tuple(selected_DUs_set))
                    du_to_activate = random.choice(list(available_dus))
                    #log_begin(loggers['mutate_logger'], func, f"Swapping DU: deactivate {du_to_deactivate}, activate {du_to_activate}")
                    self.activate_du(du_to_activate)
                    self.deactivate_du(du_to_deactivate)
                    #log_end(loggers['mutate_logger'], func, f"Swapped DU: deactivated {du_to_deactivate}, activated {du_to_activate}")

        # If the number of selected DUs is greater than the minimum needed, randomly activate or deactivate a DU
        else:
            if random.random() < 0.5 and available_dus:
                du_to_activate = random.choice(list(available_dus))
                #log_begin(loggers['mutate_logger'], func, f"Activating DU: {du_to_activate}")
                self.activate_du(du_to_activate)
                #log_end(loggers['mutate_logger'], func, f"Activated DU: {du_to_activate}")
            else:
                du_to_deactivate = random.choice(tuple(selected_DUs_set))
                #log_begin(loggers['mutate_logger'], func, f"Deactivating DU: {du_to_deactivate}")
                self.deactivate_du(du_to_deactivate)
                #log_end(loggers['mutate_logger'], func, f"Deactivated DU: {du_to_deactivate}")

    # % =================================================================
    # % USER MUTATIONS
    # % =================================================================

    def mutate_users(self):
        """Mutate users by reassigning, adding, or removing a random selection of users."""
        func = "mutate_users"
        num_users_to_mutate = random.randint(3, 10) # Randomly choose how many users to mutate (3-10)
        user_list = random.sample(list(self.users.values()), min(len(self.users), num_users_to_mutate)) # Randomly select users to mutate
        mutation_type = random.choice(['reassign', 'add', 'remove']) # Randomly choose a mutation type

        #log_begin(loggers['mutate_logger'], func, f"Mutating {num_users_to_mutate} users with type: {mutation_type}")
        mutated_count = sum(self.process_user_mutation(user, mutation_type) for user in user_list)
        #log_end(loggers['mutate_logger'], func, f"Mutated {mutated_count} users with type: {mutation_type}")

    def mutate_user_count(self):
        """Mutate the number of users by reassigning, adding, or removing users."""
        func = "mutate_user_count"
        assigned_users, unassigned_users = self.get_user_assignments() # Get the current user assignments
        current_count = len(assigned_users)                            # Count of currently assigned users

        UC_target_min = int(self.UC * 0.97)      # Minimum target for user count
        UC_target_max = int(self.UC * 1.02)      # Maximum target for user count
        UC_expanded_target = int(self.UC * 1.05) # Expanded target for user count

        #log_begin(loggers['mutate_logger'], func, f"Current assigned: {current_count}, UC: {self.UC}")

        # Determine the mutation type based on the current user count and target ranges
        mutation_type = self.get_mutation_type(current_count, UC_target_min, UC_target_max, UC_expanded_target)

        if mutation_type == 'reassign':
            #loggers['mutate_logger'].info(f"{func}: Coverage in range, skipping mutation.")
            return

        num_users_to_mutate = abs(current_count - self.UC) # Calculate how many users to mutate based on the difference from the target
        target_users = assigned_users if mutation_type == 'remove' else unassigned_users # Get the target users based on the mutation type
        self.mutate_user_assignment(target_users, mutation_type, num_users_to_mutate)    # Mutate the user assignments based on the mutation type and number of users to mutate
        #log_end(loggers['mutate_logger'], func, f"Mutated {num_users_to_mutate} users with type: {mutation_type}")

    def process_user_mutation(self, user, mutation_type):
        """Process a user mutation by reassigning, adding, or removing a user."""
        func = "process_user_mutation"
        #log_begin(loggers['mutate_logger'], func, f"Processing user {user} with type: {mutation_type}")
        try:
            if mutation_type == 'reassign' and user.is_assigned:
                # Reassign the user to a different RU
                if self.reassign_user(user):
                    #log_end(loggers['mutate_logger'], func, f"User {user.user_id} reassigned.")
                    return True

            elif mutation_type == 'add' and not user.is_assigned:
                # Add the user to a random RU
                for ru in self.RUs.values():
                    if self.assign_user(user, ru):
                        #log_end(loggers['mutate_logger'], func, f"User {user.user_id} assigned to RU {ru.name}.")
                        return True

            elif mutation_type == 'remove' and user.is_assigned:
                # Remove the user from their assigned RU
                if self.remove_user(user):
                    #log_end(loggers['mutate_logger'], func, f"User {user.user_id} removed.")
                    return True
            
        except Exception as e:
            log_fail(loggers['mutate_logger'], "process_user_mutation", e)

        return False

    def mutate_user_assignment(self, users, mutation_type, num_users_to_mutate):
        """Mutate a random selection of users by reassigning, adding, or removing them."""
        func = "mutate_user_assignment"
        mutated = sum(self.process_user_mutation(self.get_user(user), mutation_type) for user in random.sample(list(users), min(len(users), num_users_to_mutate)))
        #loggers['mutate_logger'].info(f"{func}: {mutated} users mutated via {mutation_type}")

    def get_mutation_type(self, current_count, UC_target_min, UC_target_max, UC_expanded_target):
        """Determine the mutation type based on the current user count and target ranges."""
        if current_count >= UC_target_max:
            return 'remove'
        elif current_count <= UC_target_min or current_count < UC_expanded_target:
            return 'add'
        return 'reassign'

    # % =================================================================
    # % MUTATE PATHS
    # % =================================================================

    def swap_ru_path(self):
        """Pick at random an RU and remove its old path, then compute a new path to the DU."""
        func = "swap_ru_path"
        selected_RUs_set = self.get_selected_RUs() # Get the currently selected RUs

        # Check if there are any selected RUs to swap paths
        if not selected_RUs_set:
            #loggers['Path_logger'].info(f"{func}: No selected RUs to swap paths.")
            return

        #log_begin(loggers['Path_logger'], func, f"Swapping RU-DU paths for 1 RU from {len(selected_RUs_set)} selected RUs.")

        ru_name = random.choice(list(selected_RUs_set)) # Get a random RU name from the selected RUs
        ru = self.get_ru(ru_name) # Get the RU object by name
        if not ru or not ru.connected_du:
            #loggers['Path_logger'].info(f"{func}: RU {ru_name} has no connected DU.")
            return

        du = self.get_du(ru.connected_du) # Get the DU object connected to the RU
        if not du:
            #loggers['Path_logger'].info(f"{func}: DU {ru.connected_du} not found for RU {ru_name}.")
            return

        self.remove_ru_du_path(ru_name) # Remove the old path for the RU

        # Compute a new path
        best_path, _ = self.graph.compute_best_path(ru, du)
        if best_path:
            self.process_best_path(best_path, ru_name=ru.name, target=ru, log_prefix="swap_ru_path", skip_update=False)
            #loggers['Path_logger'].info(f"{func}: Swapped path for RU {ru_name} to DU {du.name}.")

        #log_end(loggers['Path_logger'], func, f"Swapped RU-DU paths for 1 RU from {len(selected_RUs_set)} selected RUs.")

    def swap_du_path(self):
        """Pick at random a DU and remove its old path, then compute a new path to the CU."""
        func = "swap_du_path"
        selected_DUs_set = self.get_selected_DUs() # Get the currently selected DUs

        # Check if there are any selected DUs to swap paths
        if not selected_DUs_set:
            #loggers['Path_logger'].info(f"{func}: No selected DUs to swap paths.")
            return

        #log_begin(loggers['Path_logger'], func, f"Swapping DU-CU path for 1 DU from {len(selected_DUs_set)} selected DUs.")

        du_name = random.choice(list(selected_DUs_set)) # Get a random DU name from the selected DUs
        du = self.get_du(du_name) # Get the DU object by name
        if not du or not du.connected_cu:
            #loggers['Path_logger'].info(f"{func}: DU {du_name} has no connected CU.")
            return

        cu = self.get_cu(du.connected_cu) # Get the CU object connected to the DU
        if not cu:
            #loggers['Path_logger'].info(f"{func}: CU {du.connected_cu} not found for DU {du_name}.")
            return

        self.remove_du_cu_path(du_name) # Remove the old path for the DU

        # Compute a new path
        best_path, _ = self.graph.compute_best_path(du, cu)
        if best_path:
            self.process_best_path(best_path, du_name=du.name, target=du, log_prefix="swap_DU_path")
            #loggers['Path_logger'].info(f"{func}: Swapped path for DU {du_name} to CU {cu.name}.")
        
        #log_end(loggers['Path_logger'], func, f"Swapped DU-CU path for 1 DU from {len(selected_DUs_set)} selected DUs.")

    def rebuild_paths(self):
        """Efficiently rebuild all RU-DU and DU-CU paths using MST subgraph."""
        func = "rebuild_paths"

        #log_begin(loggers['Path_logger'], func, "Rebuilding paths for selected RUs and DUs.")
        # Pre-filter selected nodes
        selected_RUs = [ru for ru in self.RUs.values() if ru.is_selected]
        selected_DUs = [du for du in self.DUs.values() if du.is_selected]
        selected_CUs = [cu for cu in self.CUs.values() if cu.is_selected]

        # Clear stored path segments
        for ru in selected_RUs:
            ru.path_segments.clear()
        for du in selected_DUs:
            du.path_segments.clear()

        # Clear segment usage (reset usage_count, associated_rus, associated_dus, etc.)
        self.graph.reset_segments()

        # Build MST and subgraph
        ru_names = {ru.name for ru in selected_RUs}
        du_names = {du.name for du in selected_DUs}
        cu_names = {cu.name for cu in selected_CUs}

        # Build the allowed subgraph for the selected RUs, DUs, and CUs
        _, allowed_subgraph, _  = self.graph.build_mst(ru_names, du_names, cu_names)
        
        # DU/CU lookup maps
        du_map = {du.name: du for du in selected_DUs}
        cu_map = {cu.name: cu for cu in selected_CUs}

        # Precompute DU connections for RUs and CU connections for DUs
        ru_du_pairs = [(ru, du_map.get(ru.connected_du)) for ru in selected_RUs]
        du_cu_pairs = [(du, cu_map.get(du.connected_cu)) for du in selected_DUs]

        # Rebuild RU-DU paths (skip if DU missing)
        for ru, du in ru_du_pairs:
            if not du:
                continue
            best_path, _ = self.graph.compute_allowed_best_path(ru, du, allowed_subgraph)
            self.process_best_path(best_path, ru_name=ru.name, target=ru, log_prefix="rebuild_paths RU", skip_update=True)

        # Rebuild DU-CU paths (skip if CU missing)
        for du, cu in du_cu_pairs:
            if not cu:
                continue
            best_path, _ = self.graph.compute_allowed_best_path(du, cu, allowed_subgraph)
            self.process_best_path(best_path, du_name=du.name, target=du, log_prefix="rebuild_paths DU", skip_update=True)

        self.graph.update_road_results()
        #log_end(loggers['Path_logger'], func, "Rebuilt paths for selected RUs and DUs.")

    # % =================================================================
    # % RU ACTIVATION
    # % =================================================================

    def find_best_target(self, source, targets: dict, constraint_fn=None):
        """Find the best target node and path from a source node."""
        best_path, best_cost, best_target = None, float('inf'), None
        constraint_fn = constraint_fn or (lambda _: True)  # Avoid branching in loop
        compute_path = self.graph.compute_best_path        # Use the graph's compute_best_path method

        # Iterate through all targets and compute paths
        for target in targets.values():
            if not constraint_fn(target): continue

            # Compute the path and cost from source to target
            path, cost = compute_path(source, target)

            # If the path is feasible and has a lower cost, update the best found
            if path and cost < best_cost:
                best_path, best_cost, best_target = path, cost, target

        return best_path, best_target

    def find_best_du_for_ru(self, ru_to_activate):
        """Find the best DU for an RU based on the path cost, activating a DU if needed."""
        func = "find_best_du_for_ru"
        ru = self.get_ru(ru_to_activate) # Get the RU object by name
        if not ru:
            loggers['Path_logger'].warning(f"{func}: RU {ru_to_activate} not found.")
            return None, None

        # Constraint for selecting DUs
        constraint = lambda du: du.is_selected and du.true_false_can_accept_ru(ru)
        path, du = self.find_best_target(ru, self.DUs, constraint)

        # If no DU found, try to activate one and search again
        if not du or not path:
            # Try to activate a DU that can accept this RU
            for du_candidate in self.DUs.values():
                if not du_candidate.is_selected and du_candidate.true_false_can_accept_ru(ru):
                    if self.activate_du(du_candidate.name):
                        # Try again with the newly activated DU
                        path, du = self.find_best_target(ru, self.DUs, constraint)
                        if du and path:
                            break

        return path, du

    def activate_ru(self, ru_to_activate):
        """Activate an RU by finding the best DU, ensuring a feasible path, and updating the solution."""
        func = "activate_ru"
        ru = self.RUs.get(ru_to_activate) # Get the RU object by name

        # Check if the RU exists
        if not ru:
            loggers['device_logger'].error(f"{func}: RU {ru_to_activate} not found.")
            return False

        # Check if the RU is already active
        if ru.is_selected:
            #loggers['device_logger'].info(f"{func}: RU {ru.name} is already active.")
            return True  # Already active

        best_path, best_du = self.find_best_du_for_ru(ru_to_activate) # Find the best DU for the RU

        # If no suitable DU is found, log and return
        if not best_path or not best_du:
            loggers['device_logger'].warning(f"{func}: No suitable DU found for RU {ru_to_activate}, DUs active are {[du.name for du in self.get_selected_DU_objects()]}.")
            return False

        try:
            #log_begin(loggers['device_logger'], func, f"Activating RU: {ru.name} with DU: {best_du.name}")
            ru.activate_ru(best_du) # Activate the RU with the best DU
            best_du.add_ru(ru)      # Add the RU to the DU's list of connected RUs
            self.add_ru_du_path(ru.name, best_du.name) # Add the RU-DU path to the graph
            #log_end(loggers['device_logger'], func, f"Activated RU: {ru.name} with DU: {best_du.name}")
            return True

        except Exception as e:
            #log_fail(loggers['device_logger'], func, e, f"RU: {ru.name}, DU: {best_du.name}")
            ru.deactivate_ru() # Reset the RU if activation fails
            return False

    # % =================================================================
    # % RU DEACTIVATION
    # % =================================================================

    def deactivate_ru(self, ru_to_deactivate):
        """Deactivate an RU while ensuring the solution remains feasible."""
        func = "deactivate_ru"
        ru = self.get_ru(ru_to_deactivate) # Get the RU object by name

        # Check if the RU exists and is selected
        if not ru or not ru.is_selected:  
            return False

        #log_begin(loggers['device_logger'], func, f"Deactivating RU: {ru.name}")
        try:
            connected_du = ru.connected_du # Get the DU connected to the RU
            # Check if RU is connected to a DU
            if not connected_du:
                raise ValueError(f"deactivate_ru: RU {ru_to_deactivate} has no connected DU.")

            du = self.DUs.get(connected_du) if isinstance(connected_du, str) else connected_du # Get the DU object by name or directly if already an object
            if not du:
                #loggers['device_logger'].error(f"{func}: Connected DU {connected_du} not found for RU {ru_to_deactivate}.")
                raise ValueError(f"deactivate_ru: Connected DU not found for RU {ru_to_deactivate}.")

            # Reassign or remove each connected user
            for user_id in ru.connected_users[:]:
                user = self.get_user(user_id) # Get the user object by ID

                # If user is not assigned to any other RU, remove them
                if user and not self.reassign_user(user):
                    self.remove_user(user)

            self.remove_ru_du_path(ru_to_deactivate) # Remove the RU-DU path from the graph
            du.remove_ru(ru)     # Remove the RU from the DU's list of connected RUs
            ru.reset_RadioUnit() # Reset the RU state

            # Final check to confirm deactivation was successful
            if ru.connected_du or ru.name in du.connected_rus or ru.is_selected:
                raise ValueError(f"deactivate_ru: RU {ru_to_deactivate} still active after removal.")
            
            #log_end(loggers['device_logger'], func, f"Deactivated RU: {ru.name}")
            return True

        #loggers['device_logger'].error(f"{func}: Failed to deactivate RU {ru_to_deactivate}. Error: {e}")
        except Exception:
            return False

    # % =================================================================
    # % RU REASSIGNMENT
    # % =================================================================

    def reassign_ru_du_swap(self, ru):
        """Reassign an RU to a new DU if and only if it improves the path cost and the DU can accept the RU."""
        func = "reassign_ru_du_swap"
        current_du = self.get_du(ru.connected_du) # Get the DU currently connected to the RU

        # Check if the RU is connected to a DU
        if not current_du:
            #loggers['device_logger'].warning(f"{func}: RU {ru.name} has no connected DU.")
            return False

        #log_begin(loggers['device_logger'], func, f"Reassigning RU: {ru.name} from DU: {current_du.name}")

        # Initialise variables to track the best DU and path found
        best_du, best_path, best_cost = None, None, float('inf')
        current_path, current_cost = self.graph.compute_best_path(ru, current_du)

        # First, try selected DUs (except current)
        for du in self.get_selected_DU_objects():
            if du.name == current_du.name:
                continue
            if not du.true_false_can_accept_ru(ru):
                continue
            path, cost = self.graph.compute_best_path(ru, du) # Compute the path and cost to the DU
            if path and cost < best_cost:
                best_du, best_path, best_cost = du, path, cost

        # Only scan unselected DUs
        if best_cost >= current_cost:
            for du in self.DUs.values():
                # Skip if DU is selected or cannot accept RU
                if du.is_selected or not du.true_false_can_accept_ru(ru):
                    continue
                path, cost = self.graph.compute_best_path(ru, du) # Compute the path and cost to the DU
                if path and cost < best_cost:
                    best_du, best_path, best_cost = du, path, cost

        if best_cost < current_cost and best_du and best_path:
            if not best_du.true_false_can_accept_ru(ru):
                return False
            if not best_du.is_selected:
                #loggers['device_logger'].info(f"{func}: Activating DU {best_du.name} for RU {ru.name}.")
                if not self.activate_du(best_du.name):
                    #loggers['device_logger'].error(f"{func}: Failed to activate DU {best_du.name}.")
                    return False

            if self.reassign_ru_to_another_du(ru, current_du):
                #log_end(loggers['device_logger'], func, f"Reassigned RU: {ru.name} to DU: {best_du.name}.")
                return True
            else:
                #log_fail(loggers['device_logger'], func, f"Failed to reassign RU: {ru.name} to DU: {best_du.name}.")
                self.restore_failed_rus(current_du, [ru])

        #log_fail(loggers['device_logger'], func, f"Failed to reassign RU: {ru.name} to a better DU.")
        return False

    # % =================================================================
    # % DU ACTIVATION
    # % =================================================================

    def find_best_cu_for_du(self, du_to_activate):
        """Find the best CU for a DU based on the path cost."""
        func = "find_best_cu_for_du"
        du = self.get_du(du_to_activate) # Get the DU object by name
        if not du:
            #loggers['Path_logger'].warning(f"{func}: DU {du_to_activate} not found.")
            return None, None

        return self.find_best_target(du, self.CUs)

    def activate_and_connect_du(self, du, best_cu):
        """Activate the DU and connect it to the best CU."""
        func = "activate_and_connect_du"
        try:
            #log_begin(loggers['device_logger'], func, f"Activating DU {du.name} with CU {best_cu.name}.")
            du.activate_du(best_cu)        # Activate the DU with the best CU
            du.connected_cu = best_cu.name # Update the DU's connected CU

            self.add_du_cu_path(du.name, best_cu.name) # Add the DU-CU path to the solution
            best_cu.add_du(du)                         # Add the DU to the CU's list of connected DUs
            #loggers['device_logger'].info(f"{func}: Connected DU {du.name} to CU {best_cu.name}.")

            # Final check to confirm activation was successful
            if not du.connected_cu or du.connected_cu != best_cu.name:
                raise ValueError(f"DU {du.name} not connected to CU {best_cu.name} after activation.")

            #log_end(loggers['device_logger'], func, f"Activated DU {du.name} with CU {best_cu.name}.")
            return True
        
        except Exception as e:
            #log_fail(loggers['device_logger'], func, e, f"DU: {du.name}, CU: {best_cu.name}")
            du.deactivate_du()     # Reset the DU state if activation fails
            du.connected_cu = None # Ensure the DU's connected CU is reset
            return False
        
    def activate_du(self, du_to_activate):
        """Activate a DU by finding the best CU and ensuring a feasible connection."""
        func = "activate_du"
        du = self.DUs[du_to_activate] # Get the DU object by name
        #log_begin(loggers['device_logger'], func, f"Activating DU: {du_to_activate}")

        if du.connected_cu:
            #loggers['device_logger'].info(f"{func}: DU {du_to_activate} already active.")
            return False

        best_path, best_cu = self.find_best_cu_for_du(du_to_activate)

        # If no suitable CU is found, log and return
        if not best_path or not best_cu:
            #log_fail_no_exception(loggers['device_logger'], func, f"activate_du: No suitable CU found for DU {du_to_activate}.")
            return False

        # Check if the DU can accept the CU
        if self.activate_and_connect_du(du, best_cu):
            #log_end(loggers['device_logger'], func, f"Activated DU: {du_to_activate} with CU: {best_cu.name}")
            return True
        else:
            log_fail(loggers['device_logger'], func, f"Failed to activate DU: {du_to_activate} with CU: {best_cu.name}")
            return False

    # % =================================================================
    # % DU DEACTIVATION
    # % =================================================================

    def reassign_ru_to_specific_du(self, ru, from_du, to_du):
        """Reassign an RU from one specific DU to another specific DU."""
        func = "reassign_ru_to_specific_du"

        # Resolve DU and RU objects only once
        from_du_obj = self.get_du(from_du)
        to_du_obj = self.get_du(to_du)
        ru_obj = self.get_ru(ru)

        # skip if not selected or not correct DU
        if not (from_du_obj and to_du_obj and ru_obj):
            return False
        if not from_du_obj.is_selected or ru_obj.connected_du != from_du_obj.name:
            return False
        if not to_du_obj.is_selected or not to_du_obj.true_false_can_accept_ru(ru_obj):
            return False

        original_du_name = ru_obj.connected_du

        try:
            # Add RU to target DU and update allocation
            to_du_obj.add_ru(ru_obj)
            ru_obj.allocate_to_du(to_du_obj)

            # Remove old path and add new path in one go
            self.remove_ru_du_path(ru_obj.name)
            self.add_ru_du_path(ru_obj.name, to_du_obj.name)

            # Remove RU from original DU (skip if already removed)
            if ru_obj.name in from_du_obj.connected_rus:
                from_du_obj.remove_ru(ru_obj)

            # Confirm reassignment
            if ru_obj.connected_du != to_du_obj.name:
                raise ValueError(f"{func}: RU {ru_obj.name} not reassigned to DU {to_du_obj.name}.")

            return True

        except Exception:
            # restore to original DU if needed
            if ru_obj.connected_du != original_du_name:
                orig_du = self.get_du(original_du_name)
                if orig_du:
                    orig_du.add_ru(ru_obj)
                    ru_obj.allocate_to_du(orig_du)
                    self.add_ru_du_path(ru_obj.name, orig_du.name)
            return False

    def reassign_ru_to_another_du(self, ru, du_to_deactivate):
        """Try to reassign an RU to another available DU."""
        func = "reassign_ru_to_another_du"

        # Resolve DU and RU objects only once
        du_to_go = self.DUs[du_to_deactivate] if isinstance(du_to_deactivate, str) else du_to_deactivate
        ru_to_reassign = self.RUs[ru] if isinstance(ru, str) else ru

        # Check if the DU to deactivate is selected
        if not du_to_go.is_selected:
            return False

        reassigned = False  # Track if any reassignment worked

        # Pre-filter candidate DUs: skip DU to deactivate, skip unselected, skip those that can't accept RU
        candidate_dus = [
            candidate_du for candidate_du_name, candidate_du in self.DUs.items()
            if candidate_du_name != du_to_go.name and candidate_du.is_selected and candidate_du.true_false_can_accept_ru(ru_to_reassign)
        ]

        # Iterate over candidate DUs to try reassignment
        for candidate_du in candidate_dus:
            #log_begin(loggers['device_logger'], func, f"Reassigning RU {ru_to_reassign.name} to DU {candidate_du.name}.")

            # Use targeted move
            if self.reassign_ru_to_specific_du(ru_to_reassign, du_to_go, candidate_du):
                #log_end(loggers['device_logger'], func, f"Reassigned RU {ru_to_reassign.name} to DU {candidate_du.name}.")
                reassigned = True
                break

        return reassigned

    def restore_failed_rus(self, du, failed_rus):
        """Restore RUs that couldn't be reassigned to the original DU."""
        func = "restore_failed_rus"

        # Ensure DU is still usable for restores (selected and has CU).
        if not du.is_selected:
            #loggers['device_logger'].warning(f"{func}: DU {du.name} is not selected; cannot restore RUs.")
            raise ValueError(f"{func}: DU {du.name} is not selected; cannot restore RUs.")

        for ru in failed_rus:
            # If RU is already back on this DU, just ensure the path exists and continue
            if getattr(ru, "connected_du", None) == du.name:
                #loggers['device_logger'].info(f"{func}: RU {ru.name} already attached to DU {du.name}; ensuring path exists.")
                try:
                    # If your remove is idempotent, you can skip; otherwise, only add if missing.
                    self.add_ru_du_path(ru.name, du.name)
                except Exception:
                    # If add fails because path already exists, ignore.
                    pass
                continue

            # Ensure DU can actually take the RU back
            if not du.true_false_can_accept_ru(ru):
                #loggers['device_logger'].warning(f"{func}: DU {du.name} cannot accept RU {ru.name}.")
                raise ValueError(f"{func}: DU {du.name} cannot accept RU {ru.name}.")

            # Clean up any stale association/path the RU might still have
            try:
                # Remove any existing RU–DU path
                self.remove_ru_du_path(ru.name)
            except Exception:
                # Ignore if no path to remove
                pass

            # If RU is recorded on a different DU, detach it cleanly
            if getattr(ru, "connected_du", None) and ru.connected_du != du.name:
                old_du = self.get_du(ru.connected_du)
                if old_du:
                    try:
                        old_du.remove_ru(ru)
                    except Exception:
                        # If already absent from old DU list, ignore
                        pass
                try:
                    ru.deallocate_from_du()
                except Exception:
                    # If already deallocated, ignore
                    pass

            du.add_ru(ru)                         # Add the RU to the DU's list of connected RUs
            ru.allocate_to_du(du)                 # Allocate the RU to the DU
            self.add_ru_du_path(ru.name, du.name) # Add the RU-DU path to the solution

            #loggers['device_logger'].info(f"{func}: Restored RU {ru.name} to DU {du.name}.")

        du.update_du() # Update the DU state after restoring RUs

    def remove_du_connections(self, du):
        """Remove DU-to-CU connections before deactivation."""
        func = "remove_du_connections"
        #log_begin(loggers['device_logger'], func, f"Removing DU connections for DU {du.name}")
        if du.connected_cu:
            best_cu = self.get_cu(du.connected_cu)  # Get the CU connected to the DU
            self.remove_du_cu_path(du.name)         # Remove the DU-CU path from the solution
            best_cu.remove_du(du)                   # Remove the DU from the CU's list of connected DUs
            du.reset_DistributedUnit()              # Reset the DU state
            #log_end(loggers['device_logger'], func, f"Removed DU connections for DU {du.name} and CU {best_cu.name}.")

    def deactivate_du(self, du_to_deactivate):
        """Deactivate a DU while ensuring the solution remains feasible."""
        func = "deactivate_du"
        du = self.DUs[du_to_deactivate] # Get the DU object by name
        #log_begin(loggers['device_logger'], func, f"Deactivating DU: {du_to_deactivate}")

        # Check if the DU exists and is currently selected
        if not du.is_selected:
            #log_fail_no_exception(loggers['device_logger'], func, f"{func}: DU {du_to_deactivate} is not currently active.")
            return False

        try:
            # Check if the DU has any connected RUs
            if not du.connected_cu:
                #log_fail_no_exception(loggers['device_logger'], func, f"{func}: DU {du_to_deactivate} has no connected CU.")
                return False

            connected_rus = [self.get_ru(ru_name) for ru_name in du.connected_rus]
            #loggers['device_logger'].info(f"{func}: DU {du_to_deactivate} has {len(connected_rus)} RUs to reassign.")

            failed_rus = []

            for ru in connected_rus:
                #loggers['device_logger'].info(f"{func}: Attempting to reassign RU {ru.name} with type to another DU.")

                # Try to reassign the RU to another DU
                if not self.reassign_ru_to_another_du(ru, du_to_deactivate):
                    #loggers['device_logger'].warning(f"{func}: Failed to reassign RU {ru.name} from DU {du_to_deactivate}.")
                    failed_rus.append(ru) # If reassigning fails, add to the failed list

            if failed_rus:
                self.restore_failed_rus(du, failed_rus)
                #log_fail_no_exception(loggers['device_logger'], func, f"Failed to reassign {len(failed_rus)} RUs from DU {du_to_deactivate}.")
                return False

            self.remove_du_connections(du) # Remove DU-to-CU connections before deactivation
            #log_end(loggers['device_logger'], func, f"Deactivated DU: {du_to_deactivate}")
            return True

        except Exception as e:
            #log_fail(loggers['device_logger'], func, e, f"Error deactivating DU: {du_to_deactivate}")
            return False


    # % =================================================================
    # % USER ASSIGNMENT AND REASSIGNMENT
    # % =================================================================

    def rollback_user_assignment(self, user, previous_ru):
        """Rollback changes if user reassignment fails."""
        func = "rollback_user_assignment"
        #log_begin(loggers['device_logger'], func, f"Rolling back user assignment for user {user.user_id}.")
        user = self.get_user(user.user_id)      # Get the user object by ID
        previous_ru = self.get_ru(previous_ru)  # Get the previous RU object by name
        if previous_ru:
            user.allocate_to_ru(previous_ru)    # Reallocate the user to the previous RU
            previous_ru.add_user(user)          # Add the user back to the previous RU's list of connected users
            #log_end(loggers['device_logger'], func, f"Rolled back user assignment for user {user.user_id}, now reconnected to RU {previous_ru.name}.")

    def assign_user(self, user, ru):
        """Assign a user to an RU while ensuring the solution remains feasible."""
        func = "assign_user"
        user_to_assign = self.get_user(user) # Get the user object by ID
        ru_to_assign = self.get_ru(ru)       # Get the RU object by name

        #log_begin(loggers['device_logger'], func, f"Assigning user {user_to_assign.user_id} to RU {ru_to_assign.name}.")

        previous_ru = self.get_ru(user_to_assign.assigned_ru) # Get the previous RU the user was assigned to, if any

        try:
            if not user_to_assign.is_potential_ru(ru_to_assign):
                #log_fail_no_exception(loggers['device_logger'], func, f"RU {ru_to_assign.name} is not in potential RUs for user {user_to_assign.user_id}.")
                return False

            if not ru_to_assign.true_false_can_add_user(user_to_assign):
                #log_fail_no_exception(loggers['device_logger'], func, f"RU {ru_to_assign.name} cannot accept user {user_to_assign.user_id}.")
                return False

            user_to_assign.allocate_to_ru(ru_to_assign) # Allocate the user to the RU
            ru_to_assign.add_user(user_to_assign)       # Add the user to the RU's list of connected users

            #log_end(loggers['device_logger'], func, f"Assigned user {user_to_assign.user_id} to RU {ru_to_assign.name}.")
            return True

        except Exception as e:
            #log_fail(loggers['device_logger'], func, e, f"Error assigning user {user_to_assign.user_id} to RU {ru_to_assign.name}.")
            if previous_ru: self.rollback_user_assignment(user_to_assign, previous_ru) # Rollback if assignment fails
            return False

    def remove_user(self, user):
        """Remove a user from their assigned RU while ensuring the solution remains feasible."""
        func = "remove_user"
        user_to_remove = self.get_user(user)                  # Get the user object by ID
        previous_ru = self.get_ru(user_to_remove.assigned_ru) # Get the previous RU the user was assigned to, if any
        #log_begin(loggers['device_logger'], func, f"Removing user {user_to_remove.user_id} from RU {previous_ru.name if previous_ru else 'Unknown'}.")

        try:
            if not previous_ru:
                #log_fail_no_exception(loggers['device_logger'], func, f"User {user_to_remove.user_id} is not assigned to any RU.")
                raise ValueError(f"remove_user: User {user_to_remove.user_id} is not assigned to any RU.")
            
            previous_ru.remove_user(user_to_remove) # Remove the user from the RU's list of connected users
            user_to_remove.deallocate_from_ru()     # Deallocate the user from the RU

            #log_end(loggers['device_logger'], func, f"Removed user {user_to_remove.user_id} from RU {previous_ru.name}.")
            return True

        except Exception as e:
            #log_fail(loggers['device_logger'], func, e, f"Error removing user {user_to_remove.user_id} from RU {previous_ru.name if previous_ru else 'Unknown'}.")
            self.rollback_user_assignment(user_to_remove, previous_ru) # Rollback if removal fails
            return False

    def reassign_user(self, user):
        """Reassign a user to a new RU while ensuring the solution remains feasible."""
        func = "reassign_user"
        user_to_reassign = self.get_user(user)                  # Get the user object by ID
        original_ru = self.get_ru(user_to_reassign.assigned_ru) # Get the original RU the user was assigned to, if any
        #log_begin(loggers['device_logger'], func, f"Reassigning user {user_to_reassign.user_id} from RU {original_ru.name if original_ru else 'None'}.")

        if original_ru: original_ru.remove_user(user_to_reassign) # Remove the user from the original RU's list of connected users
        user_to_reassign.deallocate_from_ru()                     # Deallocate the user from the original RU

        try:
            # Check if the user has potential RUs to reassign to
            potential_rus = [self.get_ru(ru_name) for ru_name in user_to_reassign.potential_rus]

            # Loop through potential RUs to find a suitable one
            for ru in potential_rus:
                if ru.name == (original_ru.name if original_ru else None):
                    continue

                if ru and user_to_reassign.is_potential_ru(ru) and ru.true_false_can_add_user(user_to_reassign):
                    # Assign the user to the new RU
                    if self.assign_user(user_to_reassign, ru):
                        #log_end(loggers['device_logger'], func, f"Reassigned user {user_to_reassign.user_id} to RU {ru.name}.")
                        return True

            raise ValueError(f"reassign_user: No suitable RU found for user {user_to_reassign.user_id}.")

        except Exception as e:
            #log_fail(loggers['device_logger'], func, e, f"Error reassigning user {user_to_reassign.user_id}.")
            self.rollback_user_assignment(user_to_reassign, original_ru) # Rollback if reassignment fails
            return False
        
    # % =================================================================
    # % PROCESS PATH BUILDING
    # % =================================================================

    def process_best_path(self, best_path, ru_name=None, du_name=None, target=None, log_prefix="Path", skip_update=False):
        """Efficiently process the best path by updating segment usage and storing segments in target."""
        if not best_path:
            loggers['Path_logger'].warning(f"{log_prefix}: No valid path found.")
            return

        graph_segments = self.graph.segments
        segment_objects = []
        seen_segments = set()

        # Precompute path pairs and avoid repeated lookups
        path_pairs = zip(best_path, best_path[1:])
        for from_node, to_node in path_pairs:
            key = (from_node, to_node)
            reverse_key = (to_node, from_node)
            # Only process if not seen
            if key in seen_segments or reverse_key in seen_segments:
                continue
            segment = graph_segments.get(key)
            if not segment:
                segment = graph_segments.get(reverse_key)
            if segment:
                segment_objects.append(segment)
                seen_segments.add(key)
                seen_segments.add(reverse_key)

        if not segment_objects:
            return

        target.path_segments = segment_objects

        # Update segment usage only once
        self.graph.update_segments_usage(segment_objects, ru_name=ru_name, du_name=du_name, increase=True)

        if not skip_update:
            self.graph.update_road_results()

        # Use setdefault for associated_rus/dus to avoid repeated attribute lookups
        if ru_name or du_name:
            for segment in segment_objects:
                if ru_name:
                    segment.associated_rus.add(ru_name)
                if du_name:
                    segment.associated_dus.add(du_name)

    # % =================================================================
    # % ADD/REMOVE RU DU PATHING
    # % =================================================================

    def add_ru_du_path(self, ru_name, du_name):
        """Build a path between an RU and DU and update segment usage."""
        func = "add_ru_du_path"
        ru = self.RUs.get(ru_name) # Get the RU object by name
        du = self.DUs.get(du_name) # Get the DU object by name

        if not ru or not du:
            #log_fail_no_exception(loggers['Path_logger'], func, f"{func}: RU {ru_name} or DU {du_name} not found.")
            return

        #log_begin(loggers['Path_logger'], func, f"Finding best path between RU {ru.name} and DU {du.name}.")
        excluded_edges = self.graph.get_excluded_edges() # Get excluded edges from the graph
        best_path, best_cost = self.graph.compute_best_path(ru, du, excluded_edges=excluded_edges) # Compute the best path and cost

        # Process the best path by updating segment usage and storing segments in RU
        self.process_best_path(best_path, ru_name=ru.name, du_name=None, target=ru, log_prefix="add_ru_du_path")
        #log_end(loggers['Path_logger'], func, f"Added path between RU {ru.name} and DU {du.name}.")

    def remove_ru_du_path(self, ru_name):
        """Remove a path between an RU and DU and update segment usage."""
        func = "remove_ru_du_path"
        ru = self.RUs.get(ru_name) # Get the RU object by name

        # Check if the RU exists and has a path
        if not ru or not ru.path_segments:
            #loggers['Path_logger'].warning(f"remove_ru_du_path: No path segments stored for RU {ru_name}.")
            return
        
        #log_begin(loggers['Path_logger'], func, f"Removing path between RU {ru.name} and DU.")
        self.graph.update_segments_usage(ru.path_segments, ru_name=ru.name, du_name=None, increase=False) # Update segment usage count
        self.graph.update_road_results() # Update the road results after modifying segment usage
        ru.path_segments.clear()         # Clear the path segments stored in the RU
        #log_end(loggers['Path_logger'], "remove_ru_du_path", f"Removed path between RU {ru.name} and DU.")

    # % =================================================================
    # % ADD/REMOVE DU CU PATHING
    # % =================================================================

    def add_du_cu_path(self, du_name, cu_name):
        """Build a path between a DU and CU and update segment usage."""
        func = "add_du_cu_path"
        du = self.DUs.get(du_name) # Get the DU object by name
        cu = self.CUs.get(cu_name) # Get the CU object by name

        # Check if both DU and CU exist
        if not du or not cu:
            #loggers['Path_logger'].warning(f"add_du_cu_path: DU {du_name} or CU {cu_name} not found.")
            return

        #log_begin(loggers['Path_logger'], func, f"Finding best path between DU {du.name} and CU {cu.name}.")
        excluded_edges = self.graph.get_excluded_edges() # Get excluded edges from the graph
        best_path, best_cost = self.graph.compute_best_path(du, cu, excluded_edges=excluded_edges) # Compute the best path and cost

        self.process_best_path(best_path, du_name=du.name, target=du, log_prefix="add_du_cu_path") # Process the best path by updating segment usage and storing segments in DU
        #log_end(loggers['Path_logger'], func, f"Added path between DU {du.name} and CU {cu.name}.")

    def remove_du_cu_path(self, du_name):
        """Remove a path between a DU and CU and update segment usage."""
        func = "remove_du_cu_path"
        du = self.DUs.get(du_name) # Get the DU object by name

        # Check if the DU exists and has a path
        if not du or not du.path_segments:
            #loggers['Path_logger'].warning(f"remove_du_cu_path: No path segments stored for DU {du_name}.")
            return
        
        #log_begin(loggers['Path_logger'], func, f"Removing path between DU {du.name} and CU.")
        self.graph.update_segments_usage(du.path_segments, ru_name=None, du_name=du.name, increase=False) # Update segment usage count
        self.graph.update_road_results() # Update the road results after modifying segment usage
        du.path_segments.clear()         # Clear the path segments stored in the DU
        #log_end(loggers['Path_logger'], func, f"Removed path between DU {du.name} and CU.")

    # % =================================================================
    # % FIND BEST DUS FOR STEINER TREE
    # % =================================================================

    def traversing_tree_algorithm(self):
        """Use a Steiner tree approach to optimally select DUs and build RU-DU-CU paths. This method ensures all selected RUs are connected to selected CUs via an optimal set of DUs."""
        func = "traversing_tree_algorithm"

        # Determine minimum DUs needed
        min_dus = self.get_min_dus_needed()

        #log_begin(loggers['feasible_logger'], func, f"Starting Steiner tree reconfiguration with minimum {min_dus} DUs needed.")

        # Use graph to find best DUs for Steiner tree
        selected_RUs = [ru for ru in self.RUs.values() if ru.is_selected]
        selected_CUs = [cu for cu in self.CUs.values() if cu.is_selected]
        selected_DUs = [du for du in self.DUs.values() if du.is_selected]
        all_DUs = [du for du in self.DUs.values()]
        ru_names = {ru.name for ru in selected_RUs}
        cu_names = {cu.name for cu in selected_CUs}

        # Remove all RU-DU and DU-CU paths (like rebuild_paths)
        for ru in selected_RUs:
            ru.path_segments.clear()
        for du in selected_DUs:
            du.path_segments.clear()

        # find_best_du_steiner returns used_segments, mst_subgraph, du_set
        _, best_mst_subgraph, du_set, best_cost = self.graph.find_best_du_mst(ru_names, cu_names, all_DUs, min_dus)

        # Ensure du_set is a set of DU names (strings)
        if du_set and len(du_set) > 0 and not isinstance(next(iter(du_set)), str):
            du_set = {du.name for du in du_set}

        # Assume only one CU exists; get it directly
        best_cu = next(iter(self.CUs.values()))
        best_cu.reset_CentralisedUnit()

        # Activate/deactivate DUs based on selection
        # Assume only one CU exists; get it directly
        best_cu = next(iter(self.CUs.values()))
        for du in self.DUs.values():
            if du.name in du_set:
                du.reset_DistributedUnit()      # Always reset before activation
                du.activate_du(best_cu)         # Activate the DU with the best CU
                du.connected_cu = best_cu.name  # Update the DU's connected CU
                best_cu.add_du(du)              # Add the DU to the CU's list of connected DUs
            else:
                if du.is_selected:
                    du.reset_DistributedUnit()  # Reset DU state

        self.graph.reset_segments() # Reset all graph segments usage

        # Reconnect all RUs to the chosen DUs, distributing RUs across DUs
        chosen_du_objs = [self.DUs[name] for name in du_set if name in self.DUs and self.DUs[name].is_selected]
        chosen_du_names = {du.name for du in chosen_du_objs}

        # Distribute RUs across DUs, filling each DU before moving to the next
        # Step 1: Try to allocate every RU to one of the chosen DUs
        allocated_rus = set()
        for ru in selected_RUs:
            allocated = False
            for du in chosen_du_objs:
                if du.true_false_can_accept_ru(ru):
                    # Deallocate old DU connection if it exists
                    if ru.connected_du and ru.connected_du in self.DUs:
                        ru.deallocate_from_du()
                    
                    # Assign to this DU
                    du.add_ru(ru)
                    ru.allocate_to_du(du)
                    allocated_rus.add(ru.name)

                    # Build RU→DU path
                    best_path, _ = self.graph.compute_allowed_best_path(ru, du, best_mst_subgraph)
                    self.process_best_path(best_path, ru_name=ru.name, target=ru, log_prefix="traversing_tree_algorithm RU", skip_update=True)
                    allocated = True
                    break  # stop once RU is successfully placed in a DU

        # Step 2: Deactivate unallocated RUs
        for ru in selected_RUs:
            if ru.name not in allocated_rus:
                # Remove all users from this RU
                for user_id in ru.connected_users[:]:
                    user = self.get_user(user_id)
                    if user:
                        self.remove_user(user.user_id)
                ru.reset_RadioUnit()
                loggers['feasible_logger'].info(f"{func}: Deactivated RU {ru.name} as it could not be allocated to any DU during Steiner tree reconfiguration.")

        # Rebuild DU-CU paths for chosen DUs using mst_subgraph
        cu_map = {cu.name: cu for cu in selected_CUs}
        for du in chosen_du_objs:
            cu = cu_map.get(du.connected_cu)
            if cu:
                best_path, _ = self.graph.compute_allowed_best_path(du, cu, best_mst_subgraph)
                if best_path:
                    self.process_best_path(best_path, du_name=du.name, target=du, log_prefix="traversing_tree_algorithm DU", skip_update=True)

        self.graph.update_road_results() # Final update after all path processing
        self.update_model()              # Update the model to reflect changes
        #log_end(loggers['feasible_logger'], func, f"Completed Steiner tree reconfiguration with {len(du_set)} DUs.")
