"""
%┏━━━┓┏┓━┏┓━━━━┏━┓┏━┓┏━━━┓┏━━━┓┏┓━┏┓┏┓━━━┏━━━┓
?┃┏━┓┃┃┃━┃┃━━━━┃┃┗┛┃┃┃┏━┓┃┗┓┏┓┃┃┃━┃┃┃┃━━━┃┏━━┛
%┃┃━┗┛┃┃━┃┃━━━━┃┏┓┏┓┃┃┃━┃┃━┃┃┃┃┃┃━┃┃┃┃━━━┃┗━━┓
?┃┃━┏┓┃┃━┃┃━━━━┃┃┃┃┃┃┃┃━┃┃━┃┃┃┃┃┃━┃┃┃┃━┏┓┃┏━━┛
%┃┗━┛┃┃┗━┛┃━━━━┃┃┃┃┃┃┃┗━┛┃┏┛┗┛┃┃┗━┛┃┃┗━┛┃┃┗━━┓
?┗━━━┛┗━━━┛━━━━┗┛┗┛┗┛┗━━━┛┗━━━┛┗━━━┛┗━━━┛┗━━━┛
%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
?━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from dataclasses import dataclass, field
from typing import List, Optional

from modules.logger import loggers
from modules import config

@dataclass
class CentralisedUnit:
    #* ---- SET IN JSON ---- *#
    name: str                                                           # Unique identifier for the CU
    latitude: float = 0                                                 # Latitude of the CU
    longitude: float = 0                                                # Longitude of the CU

    #* ---- SET IN INIT ---- *#
    connected_dus: List[str] = field(default_factory=list)              # List of DUs connected to the CU
    used_bandwidth: int = field(default=0, init=False)                  # Current bandwidth usage from connected DUs
    used_ports: int = field(default=0, init=False)                      # Current ports usage from connected DUs
    total_fibres: int = field(default=0, init=False)                    # Tracks total fibres connected to this CU
    total_cost: int = field(default=0, init=False)                      # Total cost of the CU based on connected DUs
    is_selected: bool = field(default=True, init=True)                  # Selection flag

    solution: Optional['Solution'] = field(default=None, init=False)    # Solution the user is part of
    
    #* ---- STATIC VARIABLES ---- *#
    CUs = {}                                                            # Dictionary of CUs

    #* ---- CONSTANTS FROM CONFIG ---- *#
    KB = config.KB_COST                                                 # Cost per fibre connection to CU
    base_capacity_bandwidth = config.CU_BASE_BANDWIDTH                  # Base bandwidth capacity of a CU
    capacity_ports = config.CU_TOTAL_PORTS                              # Total port capacity of a CU

    @classmethod
    def initialise_units(cls, cu_data):
        """Initialise the class-level dictionary for CUs."""
        def create_cu_dict(cu_list):
            return {
                cu['cu_name']: cls(
                    name=cu['cu_name'],
                    latitude=cu['latitude'],
                    longitude=cu['longitude']
                ) for cu in cu_list
            }

        cls.CUs = create_cu_dict(cu_data['centralised_units'])
    
    def __str__(self):
        """String representation of the CentralisedUnit object."""
        return (
            f"Centralised Unit: {self.name}\n"
            f"  Capacity Bandwidth: {self.base_capacity_bandwidth}\n"
            f"  Capacity Ports: {self.capacity_ports}\n"
            f"  Location: ({self.latitude}, {self.longitude})\n"
            f"  Used Bandwidth: {self.used_bandwidth}\n"
            f"  Used Ports: {self.used_ports}\n"
            f"  Total Fibres: {self.total_fibres}\n"
            f"  Total Cost: {self.total_cost}\n"
            f"  Is Selected: {self.is_selected}"
        )

    def set_solution(self, solution):
        """Set the solution the user is part of."""
        self.solution = solution

    def make_copy(self):
        """Create a deepcopy of the CU, excluding the solution attribute."""
        copy_cu = CentralisedUnit(
            name=self.name,
            latitude=self.latitude,
            longitude=self.longitude,
            is_selected=self.is_selected
        )
        copy_cu.base_capacity_bandwidth = self.base_capacity_bandwidth
        copy_cu.capacity_ports = self.capacity_ports    
        copy_cu.used_bandwidth = self.used_bandwidth
        copy_cu.used_ports = self.used_ports
        copy_cu.total_fibres = self.total_fibres
        copy_cu.total_cost = self.total_cost
        copy_cu.connected_dus = self.connected_dus.copy()
        return copy_cu

    def update_cost(self):
        """Update the cost of the CU based on total fibres and cost per fibre (KB)."""
        self.total_cost = self.total_fibres * self.KB

    def calculate_used_bandwidth(self):
        """Calculate the total bandwidth used by connected DUs."""
        self.used_bandwidth = sum(self.solution.DUs[du].used_bandwidth for du in self.connected_dus or [])

    def calculate_used_ports(self):
        """Calculate the total ports used by connected DUs."""
        self.used_ports = self.total_fibres

    def calculate_total_fibres(self):
        """Calculate the total fibres connected to the CU."""
        self.total_fibres = sum(du.fibre_du_cu for du in (self.solution.DUs[du] for du in self.connected_dus or []) if du.fibre_du_cu is not None)

    def check_selection(self):
        """Check if the CU is selected based on connected DUs."""
        self.is_selected = bool(self.connected_dus)

    def update_cu(self):
        """Update the CU. Calculate used bandwidth, ports, total fibres, and cost."""
        self.calculate_used_bandwidth()
        self.calculate_used_ports()
        self.calculate_total_fibres()
        self.check_selection()
        self.update_cost()        

    def reset_CentralisedUnit(self):
        """Reset the CU. Remove all connected DUs and reset all attributes."""
        self.used_bandwidth = 0     # Reset used bandwidth
        self.used_ports = 0         # Reset used ports
        self.connected_dus.clear()  # Clear connected DUs
        self.total_fibres = 0       # Reset total fibres
        self.total_cost = 0         # Reset total cost
        self.is_selected = True     # Reset selection status(always True)

    # ! =================================================================
    # ! DU FUNCTIONS
    # ! =================================================================

    def add_du(self, du):
        """Add a DU to the CU, delegate the DU to update its own state."""
        self.connected_dus.append(du.name) # Add the DU to the connected list
        self.update_cu()                   # Update the CU's capacity and costs
        #loggers['CU_logger'].info(f"add_du: DU {du.name} added to CU {self.name}. Connected DUs: {len(self.connected_dus)}.")

    def remove_du(self, du):
        """Remove a DU from the CU, delegate the DU to update its own state."""
        du_to_remove = du.name
        self.connected_dus.remove(du_to_remove) # Remove the DU from the connected list
        self.update_cu()                        # Update the CU's capacity and costs
        #loggers['CU_logger'].info(f"remove_du: DU {du.name} removed from CU {self.name}. Remaining DUs: {len(self.connected_dus)}.")

    # ! =================================================================
    # ! STATE SETTING FUNCTIONS
    # ! =================================================================    

    def set_state(self, connected_dus: Optional[List[str]] = None, used_bandwidth: Optional[int] = None, used_ports: Optional[int] = None, total_fibres: Optional[int] = None, is_selected: Optional[bool] = None):
        """Set the state of this CU based on provided attributes in the initial JSON file."""
        used_bandwidth is not None and self.init_set_used_bandwidth(used_bandwidth)
        used_ports is not None and self.init_set_used_ports(used_ports)
        total_fibres is not None and self.init_set_total_fibres(total_fibres)
        is_selected is not None and self.init_set_is_selected(is_selected)
        connected_dus is not None and self.init_set_connected_dus(connected_dus)
        self.update_cu()

    #* ---- INDIVIDUAL SETTER METHODS ---- *#

    def init_set_used_bandwidth(self, used_bandwidth: int):
        """Set the used bandwidth of the CU."""
        self.used_bandwidth = used_bandwidth

    def init_set_used_ports(self, used_ports: int):
        """Set the used ports of the CU."""
        self.used_ports = used_ports

    def init_set_total_fibres(self, total_fibres: int):
        """Set the total fibre capacity of the CU."""
        self.total_fibres = total_fibres

    def init_set_is_selected(self, is_selected: bool):
        """Set the selection status of the CU."""
        self.is_selected = is_selected

    def init_set_connected_dus(self, connected_dus: List[str]):
        """Set the connected DUs for the CU."""
        self.connected_dus = connected_dus if connected_dus else []
