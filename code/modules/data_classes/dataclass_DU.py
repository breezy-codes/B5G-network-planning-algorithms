"""
%┏━━━┓┏┓━┏┓━━━━┏━┓┏━┓┏━━━┓┏━━━┓┏┓━┏┓┏┓━━━┏━━━┓
?┗┓┏┓┃┃┃━┃┃━━━━┃┃┗┛┃┃┃┏━┓┃┗┓┏┓┃┃┃━┃┃┃┃━━━┃┏━━┛
%━┃┃┃┃┃┃━┃┃━━━━┃┏┓┏┓┃┃┃━┃┃━┃┃┃┃┃┃━┃┃┃┃━━━┃┗━━┓
?━┃┃┃┃┃┃━┃┃━━━━┃┃┃┃┃┃┃┃━┃┃━┃┃┃┃┃┃━┃┃┃┃━┏┓┃┏━━┛
%┏┛┗┛┃┃┗━┛┃━━━━┃┃┃┃┃┃┃┗━┛┃┏┛┗┛┃┃┗━┛┃┃┗━┛┃┃┗━━┓
?┗━━━┛┗━━━┛━━━━┗┛┗┛┗┛┗━━━┛┗━━━┛┗━━━┛┗━━━┛┗━━━┛
%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
?━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from dataclasses import dataclass, field
import math
from typing import List, Optional

from modules.logger import loggers
from modules import config

@dataclass
class DistributedUnit:
    #* ---- SET IN JSON ---- *#
    name: str             # Unique identifier for the DU    
    latitude: float = 0   # Latitude of the DU
    longitude: float = 0  # Longitude of the DU
    num_dus: int = 0      # Number of DUs initialisation

    #* ---- SET IN INIT ---- *#
    connected_rus: List[str] = field(default_factory=list)           # List of RUs connected to this DU
    connected_cu: str = field(default=None, init=False)              # The CU connected to this DU
    used_ports: int = field(default=0, init=False)                   # Ports currently in use
    total_capacity_ports: int = field(default=0, init=False)         # Total port capacity of the DU
    used_bandwidth: int = field(default=0, init=False)               # Current bandwidth demand from connected RUs
    capacity_bandwidth: int = field(default=0, init=False)           # Total bandwidth capacity of the DU
    total_capacity_bandwidth: int = field(default=0, init=False)     # Total bandwidth capacity of the DU    
    fibre_ru_du: int = field(default=0, init=False)                  # Fibre connections from RUs to this DU
    fibre_du_cu: int = field(default=0, init=False)                  # Fibre connections from this DU to a CU
    max_fibre_du_cu: int = field(default=0, init=False)              # Maximum fibre connections to a CU
    total_cost: int = field(default=0, init=False)                   # Total cost of the DU
    is_selected: bool = field(default=False, init=False)             # Selection flag
    path_segments: List[str] = field(default_factory=list)           # Store segment names instead of RoadSegment objects

    solution: Optional['Solution'] = field(default=None, init=False) # Solution the user is part of

    #* ---- STATIC VARIABLES ---- *#
    DUs_new = {}             # Dictionary of new DUs
    DUs_existing = {}        # Dictionary of existing DUs
   
    #* ---- CONSTANTS FROM CONFIG ---- *#
    KW = config.KW_COST      # Cost of placing an initial DU
    KX = config.KX_COST      # Cost of placing additional DUs at the same location
    KL = config.KL_COST      # Cost per fibre connection at the DU from RUs
    FC = config.FC_VALUE     # Fibre capacity per fibre connection to a CU
    MD = config.MD_VALUE     # Maximum number of DUs that can be placed at a location
    CR = config.CR_VALUE     # Maximum fibre connections between a DU and CU
    MR = config.MR_VALUE     # Maximum number of RUs allowed at a location

    RU_base_capacity = config.RU_BASE_BANDWIDTH          # Bandwidth capacity per RU
    base_capacity_bandwidth = config.DU_BASE_BANDWIDTH   # Base bandwidth capacity of the DU
    capacity_ports = config.DU_TOTAL_PORTS               # Total port capacity of the DU

    @classmethod
    def initialise_units(cls, du_data_new, du_data_existing):
        """Initialise the class-level dictionaries for new and existing DUs."""
        def create_du_dict(du_list):
            return {
                du['du_name']: cls(
                    name=du['du_name'],
                    latitude=du['latitude'],
                    longitude=du['longitude']
                ) for du in du_list
            }
        
        cls.DUs_new = create_du_dict(du_data_new['distributed_units_new'])
        cls.DUs_existing = create_du_dict(du_data_existing['distributed_units_existing'])

    def __str__(self):
        return(
            f"DU Name: {self.name}\n"
            f"  Latitude: {self.latitude}\n"
            f"  Longitude: {self.longitude}\n"
            f"  Base Capacity Bandwidth: {self.base_capacity_bandwidth}\n"
            f"  Capacity Ports: {self.capacity_ports}\n"
            f"  Number of DUs: {self.num_dus}\n"
            f"  Number of Connected RUs: {len(self.connected_rus)}\n"
            f"  Connected RUs: {self.connected_rus}\n"
            f"  Connected CU: {self.connected_cu if self.connected_cu else None}\n"
            f"  Used Ports: {self.used_ports}\n"
            f"  Total Capacity Ports: {self.total_capacity_ports}\n"
            f"  Used Bandwidth: {self.used_bandwidth}\n"
            f"  Capacity Bandwidth: {self.capacity_bandwidth}\n"
            f"  Total Capacity Bandwidth: {self.total_capacity_bandwidth}\n"
            f"  Fibre RU-DU: {self.fibre_ru_du}\n"
            f"  Fibre DU-CU: {self.fibre_du_cu}\n"
            f"  Max Fibre DU-CU: {self.max_fibre_du_cu}\n"
            f"  Total Cost: {self.total_cost}\n"
            f"  Is Selected: {self.is_selected}\n"
            f"  Path Segments: {self.path_segments}\n"
        )

    def set_solution(self, solution):
        """Set the solution the user is part of."""
        self.solution = solution

    def make_copy(self):
        """Create a deep copy of the DU, excluding the solution attribute."""
        copy_du = DistributedUnit(
            name=self.name,
            latitude=self.latitude,
            longitude=self.longitude,
            num_dus=self.num_dus,
            connected_rus=self.connected_rus.copy(),
            path_segments=self.path_segments.copy()
        )
        copy_du.base_capacity_bandwidth = self.base_capacity_bandwidth
        copy_du.capacity_ports = self.capacity_ports
        copy_du.used_ports = self.used_ports
        copy_du.total_capacity_ports = self.total_capacity_ports
        copy_du.used_bandwidth = self.used_bandwidth
        copy_du.capacity_bandwidth = self.capacity_bandwidth
        copy_du.total_capacity_bandwidth = self.total_capacity_bandwidth
        copy_du.connected_cu = self.connected_cu
        copy_du.fibre_ru_du = self.fibre_ru_du
        copy_du.fibre_du_cu = self.fibre_du_cu
        copy_du.max_fibre_du_cu = self.max_fibre_du_cu
        copy_du.total_cost = self.total_cost
        copy_du.is_selected = self.is_selected
        return copy_du
    
    def update_cost(self):
        """Calculate and update the total cost of the DU."""
        if not self.is_selected:
            return 0

        cost = self.KW + max(0, self.num_dus - 1) * self.KX + self.fibre_ru_du * self.KL
        self.total_cost = cost
        return cost
    
    def calculate_capacity_bandwidth(self):
        """Calculate the total bandwidth capacity of this DU."""
        self.capacity_bandwidth = self.fibre_du_cu * 40000

    def calculate_total_capacity_bandwidth(self):
        """Calculate the total bandwidth capacity of this DU."""
        self.total_capacity_bandwidth = self.fibre_du_cu * self.FC

    def calculate_max_fibre_du_cu(self):
        """Calculate the maximum number of fibre connections to a CU."""
        self.max_fibre_du_cu = self.num_dus * self.CR

    def calculate_total_capacity_ports(self):
        """Calculate the total number of ports on this DU."""
        self.total_capacity_ports = self.capacity_ports * self.num_dus

    def calculate_used_capacity(self):
        """Calculate the used bandwidth capacity of this DU."""
        self.used_bandwidth = sum(self.solution.RUs[ru].num_rus * self.solution.RUs[ru].base_capacity_bandwidth for ru in self.connected_rus)

    def calculate_fibre_ru_du(self):
        """Calculate the total number of fibre connections from RUs to this DU."""
        self.fibre_ru_du = sum(self.solution.RUs[ru].num_rus for ru in self.connected_rus)

    def calculate_fibre_du_cu(self):
        """Calculate the total number of fibre connections from this DU to a CU."""
        if self.used_bandwidth == 0 and self.num_dus == 0:
            self.fibre_du_cu = 0  # No demand, no fibres required
            return self.fibre_du_cu

        self.calculate_used_capacity()
        required_fibres = (self.used_bandwidth + self.FC - 1) // self.FC  # Ceiling division
        self.fibre_du_cu = max(1, required_fibres)  # Ensure at least one fibre if DU is active
        return self.fibre_du_cu

    def calculate_used_ports(self):
        """Calculate the total number of used ports on this DU."""
        self.used_ports = self.fibre_ru_du

    def check_selection(self):
        """Check if this DU is selected based on its connected RUs and CUs."""
        self.is_selected = bool(self.connected_rus or self.connected_cu)

    def recalculate_number_of_dus(self):
        """Recalculate the number of DUs based on total capacity, available fibre connections, and MD_VALUE limit."""
        max_capacity_bandwidth = self.base_capacity_bandwidth # Bandwidth capacity per DU
        max_capacity_ports = self.capacity_ports              # Port capacity per DU
        max_capacity_fibre_du_cu = self.CR                    # Fibre capacity per DU

        # Calculate how many DUs are needed to satisfy each constraint
        required_dus_bandwidth = math.ceil(self.total_capacity_bandwidth / max_capacity_bandwidth)
        required_dus_ports = math.ceil(self.used_ports / max_capacity_ports) if max_capacity_ports > 0 else self.num_dus
        required_dus_fibre = math.ceil(self.fibre_du_cu / max_capacity_fibre_du_cu)

        # Determine the maximum number of DUs required across constraints
        min_required_dus = max(required_dus_bandwidth, required_dus_ports, required_dus_fibre)
        new_du_count = min(min_required_dus, self.MD)   # Ensure we do not exceed the MD limit
        new_du_count = max(1, new_du_count)             # Ensure at least one DU remains

        # Update the DU count only if it has changed
        if new_du_count != self.num_dus:
            self.num_dus = new_du_count
            self.update_du()
            
            if self.connected_cu:
                self.calculate_fibre_du_cu()
            
            #loggers['DU_logger'].info(f"recalculate_number_of_dus: DU {self.name} adjusted to {self.num_dus} DUs.")

    def update_du(self):
        """Update the DU's state based on connected RUs and CUs, and calculate the total cost."""
        self.recalculate_number_of_dus()
        self.calculate_total_capacity_bandwidth()
        self.calculate_max_fibre_du_cu()
        self.calculate_total_capacity_ports()
        self.calculate_used_capacity()
        self.calculate_fibre_ru_du()
        self.calculate_used_ports()
        self.calculate_fibre_du_cu()
        self.update_cost()

    def reset_DistributedUnit(self):
        """Reset this DU to its initial state by disconnecting all connected RUs and CUs."""
        self.path_segments.clear()     # Ensure no segments remain
        self.connected_rus.clear()     # Clear connected RUs
        self.connected_cu = None       # Clear connected CU
        self.used_bandwidth = 0        # Reset used bandwidth
        self.max_fibre_du_cu = 0       # Reset maximum fibre connections to CU
        self.fibre_ru_du = 0           # Reset fibre connections from RUs to this DU
        self.num_dus = 0               # Reset number of DUs
        self.fibre_du_cu = 0           # Reset fibre connections to CU
        self.used_ports = 0            # Reset used ports
        self.is_selected = False       # Reset selection status
        self.update_du()               # Update the DU state

    def activate_du(self, new_cu):
        """Activate the DU by setting the number of DUs to 1 and to selected, also connects to the DU."""
        self.num_dus = 1                 # Activate the DU
        self.is_selected = True          # Set state to selected
        self.connected_cu = new_cu.name  # Connect to the CU
        self.update_du()                 # Update the DU state

    def deactivate_du(self):
        """Deactivate the DU by resetting the state."""
        self.is_selected = False      # Set state to not selected
        self.reset_DistributedUnit()  # Reset the DU state

    # % =================================================================
    # % MOVING INTO RU FUNCTIONS NOW
    # % =================================================================

    def can_accept_ru(self, ru) -> tuple[bool, bool]:
        """Check if this DU can accept an RU, possibly with expansion (via fibres or new DU)."""

        # Case 0: DU is inactive
        if self.num_dus == 0 and not self.is_selected:
            return False, False

        current_fibre_usage = self.calculate_fibre_du_cu()
        self.used_bandwidth = sum(self.solution.RUs[r].num_rus * self.solution.RUs[r].base_capacity_bandwidth for r in self.connected_rus)
        ru_bw = ru.capacity_bandwidth

        # Capacity is proportional to current fibre count (40k per fibre)
        current_capacity = current_fibre_usage * 40000
        max_fibres = self.num_dus * 10

        # Case 1: Can fit with current fibres
        if (self.used_bandwidth + ru_bw) <= current_capacity and (current_fibre_usage + 1) <= max_fibres:
            #loggers['DU_logger'].info(f"can_accept_ru: DU {self.name} can accept RU {ru.name} directly (used={self.used_bandwidth}, cap={current_capacity}, fibres={current_fibre_usage}/{max_fibres}, num_dus={self.num_dus}).")
            return True, False

        # Case 2a: Still has spare fibre slots in current DU
        if (current_fibre_usage < max_fibres) and ((self.used_bandwidth + ru_bw) <= ((current_fibre_usage + 1) * 40000)):
            #loggers['DU_logger'].info(f"can_accept_ru: DU {self.name} can accept RU {ru.name} by adding a fibre (used={self.used_bandwidth}, new_cap={(current_fibre_usage + 1) * 40000}, fibres={current_fibre_usage + 1}/{max_fibres}, num_dus={self.num_dus}).")
            return False, True

        # Case 2b: Need to add another DU (if allowed)
        if self.num_dus < self.MD:
            new_total_capacity = (self.num_dus + 1) * self.base_capacity_bandwidth
            new_max_fibres = (self.num_dus + 1) * 10
            if (self.used_bandwidth + ru_bw) <= new_total_capacity and (current_fibre_usage + 1) <= new_max_fibres:
                #loggers['DU_logger'].info(f"can_accept_ru: DU {self.name} can accept RU {ru.name} with DU expansion (used={self.used_bandwidth}, new_cap={new_total_capacity}, fibres={current_fibre_usage}/{new_max_fibres}, num_dus={self.num_dus}).")
                return False, True

        # Case 3: Cannot fit
        #loggers['DU_logger'].info(f"can_accept_ru: DU {self.name} cannot accept RU {ru.name}; used={self.used_bandwidth}, cap={current_capacity}, fibres={current_fibre_usage}/{max_fibres}, num_dus={self.num_dus}.")
        return False, False

    def true_false_can_accept_ru(self, ru) -> bool:
        """Simplified version: True if RU can fit now, with a fibre, or after DU expansion."""

        # Case 0: DU is inactive
        if self.num_dus == 0:
            return False

        current_fibre_usage = self.calculate_fibre_du_cu()
        self.used_bandwidth = sum(self.solution.RUs[r].num_rus * self.solution.RUs[r].base_capacity_bandwidth for r in self.connected_rus)
        ru_bw = ru.capacity_bandwidth

        # Capacity is proportional to current fibre count (40k per fibre)
        current_capacity = current_fibre_usage * 40000
        max_fibres = self.num_dus * 10

        # Case 1: Can fit with current fibres
        if (self.used_bandwidth + ru_bw) <= current_capacity and (current_fibre_usage + 1) <= max_fibres:
            return True

        # Case 2a: Add another fibre within this DU
        if (current_fibre_usage < max_fibres) and ((self.used_bandwidth + ru_bw) <= ((current_fibre_usage + 1) * 40000)):
            return True

        # Case 2b: Need to add another DU (if allowed)
        if self.num_dus < self.MD:
            new_total_capacity = (self.num_dus + 1) * self.base_capacity_bandwidth
            new_max_fibres = (self.num_dus + 1) * 10
            if (self.used_bandwidth + ru_bw) <= new_total_capacity and (current_fibre_usage + 1) <= new_max_fibres:
                return True

        return False

    def add_ru(self, ru):
        """Add an RU to this DU, delete the RU to update its own state."""
        can_accept, needs_expansion = self.can_accept_ru(ru) # Ensure the DU can accept this RU

        if can_accept:  # If DU can accept RU without expansion
            pass
        elif needs_expansion:  # If DU needs expansion
            self.num_dus += 1
            #loggers['DU_logger'].info(f"add_ru: DU {self.name} expanded to {self.num_dus} DUs to accommodate RU {ru.name}.")
        else:
            #loggers['DU_logger'].error(f"add_ru: DU {self.name} cannot accept RU {ru.name}; over capacity, with used bandwidth {self.used_bandwidth} and total capacity {self.total_capacity_bandwidth} and fibre connections {self.fibre_du_cu} and max fibre {self.max_fibre_du_cu}, num_dus {self.num_dus}.")
            raise ValueError(f"DU {self.name} cannot accept RU {ru.name}; over capacity, with used bandwidth {self.used_bandwidth} and total capacity {self.total_capacity_bandwidth} and fibre connections {self.fibre_du_cu} and max fibre {self.max_fibre_du_cu}, num_dus {self.num_dus}.")

        self.connected_rus.append(ru.name)  # Add RU to DU's tracking
        self.recalculate_number_of_dus()    # Recalculate DUs based on new state
        self.calculate_capacity_bandwidth() # Calculate the new capacity
        self.update_du()                    # Update DU state

    def remove_ru(self, ru):
        """Remove an RU from this DU, ensuring only DU state is modified."""
        self.connected_rus.remove(ru.name) # Remove RU from DU's tracking
        self.recalculate_number_of_dus()   # Recalculate DUs based on new state
        self.update_du()                   # Update DU state

    # % =================================================================
    # % MOVING INTO CU FUNCTIONS NOW
    # % =================================================================

    def allocate_to_cu(self, cu):
        """Connect the DU to the given CU."""
        self.connected_cu = cu.name  # Connect to the CU
        #loggers['DU_logger'].info(f"allocate_to_cu: DU {self.name} connected to CU {cu.name}.")

    def deallocate_from_cu(self):
        """Disconnect the DU from its connected CU."""
        self.connected_cu = None # Disconnect from the CU
        #loggers['DU_logger'].info(f"deallocate_from_cu: DU {self.name} disconnected from CU.")

    # % =================================================================
    # % MOVING INTO STATE SETTING FUNCTIONS NOW
    # % =================================================================

    def set_state(self, is_selected: Optional[bool] = None, num_dus: Optional[int] = None, used_bandwidth: Optional[int] = None, used_ports: Optional[int] = None, fibre_ru_du: Optional[int] = None, fibre_du_cu: Optional[int] = None, connected_rus: Optional[List[str]] = None, connected_cu: Optional[str] = None):
        """Set the state of this DU based on provided attributes in the initial JSON file."""
        is_selected is not None and self.init_set_is_selected(is_selected)
        num_dus is not None and self.init_set_num_dus(num_dus)
        used_bandwidth is not None and self.init_set_used_bandwidth(used_bandwidth)
        used_ports is not None and self.init_set_used_ports(used_ports)
        fibre_ru_du is not None and self.init_set_fibre_ru_du(fibre_ru_du)
        fibre_du_cu is not None and self.init_set_fibre_du_cu(fibre_du_cu)
        connected_rus is not None and self.init_set_connected_rus(connected_rus)
        connected_cu is not None and self.init_set_connected_cu(connected_cu)
        self.update_du()

    #* -INDIVIDUAL SETTER METHODS -*#

    def init_set_is_selected(self, is_selected: bool):
        """Set the selection status of the DU."""
        self.is_selected = is_selected

    def init_set_num_dus(self, num_dus: int):
        """Set the number of DUs."""
        self.num_dus = num_dus

    def init_set_used_bandwidth(self, used_bandwidth: int):
        """Set the used bandwidth of the DU."""
        self.used_bandwidth = used_bandwidth

    def init_set_used_ports(self, used_ports: int):
        """Set the used ports of the DU."""
        self.used_ports = used_ports

    def init_set_fibre_ru_du(self, fibre_ru_du: int):
        """Set the fibre capacity between RU and DU."""
        self.fibre_ru_du = fibre_ru_du

    def init_set_fibre_du_cu(self, fibre_du_cu: int):
        """Set the fibre capacity between DU and CU."""
        self.fibre_du_cu = fibre_du_cu

    def init_set_connected_rus(self, connected_rus: List[str]):
        """Set the connected RUs for the DU."""
        self.connected_rus = connected_rus if connected_rus else []

    def init_set_connected_cu(self, cu):
        """Set the connected CU for the DU."""
        self.connected_cu = cu