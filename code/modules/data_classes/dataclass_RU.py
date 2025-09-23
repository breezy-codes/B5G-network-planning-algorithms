"""
%┏━━━┓┏┓━┏┓━━━━┏━┓┏━┓┏━━━┓┏━━━┓┏┓━┏┓┏┓━━━┏━━━┓
?┃┏━┓┃┃┃━┃┃━━━━┃┃┗┛┃┃┃┏━┓┃┗┓┏┓┃┃┃━┃┃┃┃━━━┃┏━━┛
%┃┗━┛┃┃┃━┃┃━━━━┃┏┓┏┓┃┃┃━┃┃━┃┃┃┃┃┃━┃┃┃┃━━━┃┗━━┓
?┃┏┓┏┛┃┃━┃┃━━━━┃┃┃┃┃┃┃┃━┃┃━┃┃┃┃┃┃━┃┃┃┃━┏┓┃┏━━┛
%┃┃┃┗┓┃┗━┛┃━━━━┃┃┃┃┃┃┃┗━┛┃┏┛┗┛┃┃┗━┛┃┃┗━┛┃┃┗━━┓
?┗┛┗━┛┗━━━┛━━━━┗┛┗┛┗┛┗━━━┛┗━━━┛┗━━━┛┗━━━┛┗━━━┛
%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
?━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from dataclasses import dataclass, field
from typing import List, Optional

from modules.logger import loggers
from modules import config

@dataclass
class RadioUnit:
    #* ---- SET IN JSON ---- *#
    name: str              # Unique identifier for the RU
    latitude: float = 0    # Location of the RU
    longitude: float = 0   # Location of the RU
    num_rus: int = 0       # Number of RUs at this location

    #* ---- SET IN INIT ---- *#
    connected_users: List[str] = field(default_factory=list)         # List of users connected to this RU
    connected_du: str = field(default=None, init=False)              # DU connected to this RU
    path_segments: List[str] = field(default_factory=list)           # Store segment names instead of RoadSegment objects
    total_cost: int = field(default=0, init=False)                   # Total cost of the RU
    is_selected: bool = field(default=False, init=False)             # Selection flag
    capacity_bandwidth: int = field(default=0, init=False)           # Capacity based on base * num_rus
    used_capacity: int = field(default=0, init=False)                # Tracks currently used capacity
    
    solution: Optional['Solution'] = field(default=None, init=False) # Solution the user is part of

    #* ---- STATIC VARIABLES ---- *#
    RUS_new = {}       # Dictionary of new RUs
    RUS_existing = {}  # Dictionary of existing RUs

    #* ---- CONSTANTS FROM CONFIG ---- *#
    KV = config.RU_COST + config.RU_INSTALL_COST        # Cost per RU including installation
    MR = config.MR_VALUE                                # Maximum number of RUs per location
    UM = config.UM_VALUE                                # Bandwidth each user requires
    base_capacity_bandwidth = config.RU_BASE_BANDWIDTH  # Bandwidth capacity per RU
    
    @classmethod
    def initialise_units(cls, ru_data_new, ru_data_existing):
        """Initialise the class-level dictionaries for new and existing RUs."""
        def create_ru_dict(ru_list):
            return {
                ru['ru_name']: cls(
                    name=ru['ru_name'],
                    latitude=ru['latitude'],
                    longitude=ru['longitude']
                ) for ru in ru_list
            }

        cls.RUs_new = create_ru_dict(ru_data_new['radio_units_new'])
        cls.RUs_existing = create_ru_dict(ru_data_existing['radio_units_existing'])
    
    def __str__(self):
        """String representation of the RadioUnit object."""
        return (
            f"Radio Unit: {self.name}\n"
            f"  Location: ({self.latitude}, {self.longitude})\n"
            f"  Number of RUs: {self.num_rus}\n"
            f"  Number of Connected Users: {len(self.connected_users)}\n"
            f"  Connected Users: {self.connected_users}\n"
            f"  Connected DU: {self.connected_du if self.connected_du else None}\n"
            f"  Total Cost: {self.total_cost}\n"
            f"  Is Selected: {self.is_selected}\n"
            f"  Capacity Bandwidth: {self.capacity_bandwidth}\n"
            f"  Used Capacity: {self.used_capacity}\n"
            f"  Path Segments: {self.path_segments}"
        )

    def set_solution(self, solution):
        """Set the solution the user is part of."""
        self.solution = solution
    
    def make_copy(self):
        """Create a deep copy of the RU, excluding the solution attribute."""
        copy_ru = RadioUnit(
            name=self.name,
            latitude=self.latitude,
            longitude=self.longitude,
            num_rus=self.num_rus,
            connected_users=self.connected_users.copy(),
            path_segments=self.path_segments.copy()
        )
        copy_ru.base_capacity_bandwidth = self.base_capacity_bandwidth
        copy_ru.connected_du = self.connected_du
        copy_ru.total_cost = self.total_cost
        copy_ru.is_selected = self.is_selected
        copy_ru.capacity_bandwidth = self.capacity_bandwidth
        copy_ru.used_capacity = self.used_capacity
        return copy_ru

    def update_cost(self):
        """Update the cost of the RU based on total number of RUs and cost per RU (KV)."""
        self.total_cost = self.num_rus * self.KV if self.num_rus else 0; return self.total_cost

    def calculate_capacity_bandwidth(self):
        """Calculate the total bandwidth capacity of the RU based on base capacity and number of RUs."""
        self.capacity_bandwidth = self.base_capacity_bandwidth * self.num_rus if self.base_capacity_bandwidth and self.num_rus else 0; return self.capacity_bandwidth

    def calculate_used_capacity(self):
        """Calculate the total used capacity of the RU based on connected users."""
        self.used_capacity = 0 if self.UM is None else len(self.connected_users or []) * self.UM; return self.used_capacity

    def calculate_num_users(self):
        """Calculate the number of connected users."""
        return len(self.connected_users or [])

    def calculate_num_rus(self):
        """Calculate the number of RUs based on used capacity."""
        return self.num_rus

    def check_selection(self):
        """Check if the RU is selected based on connected users and DU."""
        self.is_selected = bool(self.connected_users or self.connected_du); return self.is_selected

    def update_ru(self):
        """Update the RU's state based on connected users and DU."""
        self.recalculate_number_of_rus()    # Recalculate the number of RUs based on used capacity
        self.update_cost()                  # Update the RU's total cost

    def activate_ru(self, new_du):
        """Activate the RU by setting num_rus to 1 and to selected, also connects to the DU."""
        self.is_selected = True         # Mark as selected
        self.num_rus = 1                # Start with one RU
        self.used_capacity = 0          # Reset used capacity
        self.connected_du = new_du.name # Connect to the DU
        self.update_ru()                # Update the RU state

    def deactivate_ru(self):
        """Deactivate the RU by resetting all attributes"""
        self.is_selected = False # Mark as not selected
        self.reset_RadioUnit()   # Reset the RU to its initial state

    def recalculate_number_of_rus(self):
        """Recalculate the number of RUs based on used capacity. Increase RUs if over capacity, decrease RUs if underutilised. Never exceed max value in MR. Ensure at least one RU remains."""

        # Recalculate capacity and used capacity
        self.calculate_capacity_bandwidth()
        self.calculate_used_capacity()

        # Check if we are over capacity
        if self.used_capacity > self.capacity_bandwidth:
            # Calculate required RUs directly, avoid loop
            required_rus = min(self.MR, -(-self.used_capacity // self.base_capacity_bandwidth))  # Ceiling division
            if required_rus > self.num_rus:
                self.num_rus = required_rus
                self.calculate_capacity_bandwidth()
                if self.connected_du is not None:
                    du = self.solution.DUs[self.connected_du]
                    self.calculate_num_rus()
                    du.update_du()

        # Check if we are underutilising resources
        elif self.num_rus > 1:
            # Calculate minimum RUs needed
            min_rus = max(1, (self.used_capacity + self.base_capacity_bandwidth - 1) // self.base_capacity_bandwidth)
            if min_rus < self.num_rus:
                self.num_rus = min_rus
                self.calculate_capacity_bandwidth()
                if self.connected_du is not None:
                    du = self.solution.DUs[self.connected_du]
                    self.calculate_num_rus()
                    du.update_du()

        # If RU has no users and no DU, reset it
        if not self.connected_users and self.connected_du is None:
            self.reset_RadioUnit()

    def reset_RadioUnit(self):
        """Reset the RU to its initial state. Disconnect from the DU and all users, clear all paths, and reset internal attributes."""
        if not self.is_selected and not self.connected_users and self.connected_du is None:
            return                    # Already reset, no action needed
        self.connected_du = None      # Disconnect from the DU
        self.connected_users.clear()  # Clear the connected users list
        self.path_segments.clear()    # Ensure no segments remain
        self.num_rus = 0              # Reset the number of RUs
        self.used_capacity = 0        # Set used capacity to 0
        self.is_selected = False      # Reset the selection status
        self.update_ru()              # Update the RU state

    # % =================================================================
    # % USER FUNCTIONS
    # % =================================================================

    def is_user_connected(self, user) -> bool:
        """Check if the user is connected to the RU."""
        return user.user_id in self.connected_users

    def can_add_user(self, user) -> tuple[bool, bool]:
        """Check if a user can be added to the RU, with and without expansion."""

        if self.num_rus == 0:
            return False, False

        # Check if can add without expansion
        current_capacity = self.num_rus * self.base_capacity_bandwidth
        remaining_capacity = current_capacity - self.used_capacity

        can_add_without_expansion = remaining_capacity >= self.UM

        # Check if can add with expansion
        if self.num_rus < self.MR:
            max_possible_capacity = self.MR * self.base_capacity_bandwidth
            max_additional_capacity = max_possible_capacity - self.used_capacity
            can_add_with_expansion = max_additional_capacity >= self.UM
            return can_add_without_expansion, can_add_with_expansion

        return can_add_without_expansion, False

    def true_false_can_add_user(self, user) -> bool:
        """Return True if the user can be added to the RU (with or without expansion), else False."""
        if self.num_rus == 0:
            return False

        # Check if can add without expansion
        current_capacity = self.num_rus * self.base_capacity_bandwidth
        if current_capacity - self.used_capacity >= self.UM:
            return True

        # Check if can add with expansion
        if self.num_rus < self.MR:
            max_possible_capacity = self.MR * self.base_capacity_bandwidth
            if max_possible_capacity - self.used_capacity >= self.UM:
                return True

        return False

    def add_user(self, user):
        """Add a user to the RU. If already assigned to another RU, remove from that RU. If already assigned to this RU, do nothing. If expansion is required and possible, expand the RU before adding the user. Otherwise, raise an error if at max capacity.
        """
        can_without, can_with = self.can_add_user(user)

        if can_without:
            pass
        elif can_with:
            self.num_rus = min(self.MR, self.num_rus + 1)
        else:
            loggers['RU_logger'].error(f"add_user: RU {self.name} is at max capacity and cannot expand. User {user.user_id} cannot be added.")

        self.connected_users.append(user.user_id) # Add the user to the RU
        self.recalculate_number_of_rus()          # Adjust RU count based on new capacity
        self.update_ru()                          # Update the RU state

        #loggers['RU_logger'].info(f"add_user: User {user.user_id} added to RU {self.name}.")

    def remove_user(self, user):
        """Remove a user from the RU. If the user is not connected to the RU, raise an error. Otherwise, remove the user from the RU, update the RU's state, and reset the RU if empty."""

        if not self.is_user_connected(user):
            loggers['RU_logger'].error(f"remove_user: User {user.user_id} is not assigned to RU {self.name}.")

        self.connected_users.remove(user.user_id)  # Remove the user from RU's list
        self.recalculate_number_of_rus()           # Adjust RU count if needed
        self.update_ru()                           # Update the RU state
        #loggers['RU_logger'].info(f"remove_user: User {user.user_id} removed from RU {self.name}.")

    # % =================================================================
    # % DU FUNCTIONS
    # % =================================================================

    def allocate_to_du(self, du: str):
        """Assign the RU to the given DU ID, disconnect from the old DU if needed."""
        self.connected_du = du.name
        #loggers['RU_logger'].info(f"allocate_to_du: RU {self.name} allocated to DU {du.name}.")

    def deallocate_from_du(self):
        """Disconnect the RU from the connected DU."""
        self.connected_du = None
        #loggers['RU_logger'].info(f"deallocate_from_du: RU {self.name} deallocated from DU.")

    # % =================================================================
    # % STATE SETTING FUNCTIONS
    # % =================================================================

    def set_state(self, num_rus: Optional[int] = None, connected_users: Optional[dict[str]] = None, connected_du: Optional[str] = None, is_selected: Optional[bool] = None, used_capacity: Optional[int] = None):
        """Set the state of this RU based on provided attributes in the initial JSON file."""
        num_rus is not None and self.init_set_num_rus(num_rus)
        connected_users is not None and self.init_set_connected_users(connected_users)
        connected_du is not None and self.init_set_connected_du(connected_du)
        is_selected is not None and self.init_set_is_selected(is_selected)
        used_capacity is not None and self.init_set_used_capacity(used_capacity)
        self.update_ru()

    #* ---- INDIVIDUAL SETTER METHODS ---- *#

    def init_set_num_rus(self, num_rus: int):
        """Set the number of RUs."""
        self.num_rus = num_rus

    def init_set_connected_users(self, users: List[str]):
        """Set the connected users for the RU."""
        self.connected_users = []  # Clear existing users
        for user in users:
            self.add_init_user(user)

    def init_set_connected_du(self, du_id: str):
        """Set the connected DU for the RU."""
        self.connected_du = du_id

    def init_set_is_selected(self, is_selected: bool):
        """Set the selection status of the RU."""
        self.is_selected = is_selected

    def init_set_used_capacity(self, used_capacity: int):
        """Set the used capacity of the RU."""
        self.used_capacity = used_capacity

    def add_init_user(self, user):
        """Add a user to the RU. Used for the initial setup, doesn't modify the capacity."""
        self.connected_users.append(user.user_id)
        user.allocate_to_ru(self)