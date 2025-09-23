"""
%┏┓━┏┓┏━━━┓┏━━━┓┏━━━┓━━━━┏━┓┏━┓┏━━━┓┏━━━┓┏┓━┏┓┏┓━━━┏━━━┓
?┃┃━┃┃┃┏━┓┃┃┏━━┛┃┏━┓┃━━━━┃┃┗┛┃┃┃┏━┓┃┗┓┏┓┃┃┃━┃┃┃┃━━━┃┏━━┛
%┃┃━┃┃┃┗━━┓┃┗━━┓┃┗━┛┃━━━━┃┏┓┏┓┃┃┃━┃┃━┃┃┃┃┃┃━┃┃┃┃━━━┃┗━━┓
?┃┃━┃┃┗━━┓┃┃┏━━┛┃┏┓┏┛━━━━┃┃┃┃┃┃┃┃━┃┃━┃┃┃┃┃┃━┃┃┃┃━┏┓┃┏━━┛
%┃┗━┛┃┃┗━┛┃┃┗━━┓┃┃┃┗┓━━━━┃┃┃┃┃┃┃┗━┛┃┏┛┗┛┃┃┗━┛┃┃┗━┛┃┃┗━━┓
?┗━━━┛┗━━━┛┗━━━┛┗┛┗━┛━━━━┗┛┗┛┗┛┗━━━┛┗━━━┛┗━━━┛┗━━━┛┗━━━┛
%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
?━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from dataclasses import dataclass, field
from typing import List, Optional

from modules.logger import loggers
from modules import config

@dataclass
class User:
    user_id: str                                                     # Unique user identifier
    potential_rus: List[str] = field(default_factory=list)           # List of RUs user could connect to
    assigned_ru: str = field(default=None, init=False)               # The single RU the user is connected to
    is_assigned: bool = field(default=False, init=False)             # Selection flag
    solution: Optional['Solution'] = field(default=None, init=False) # Solution the user is part of
        
    #* --- CONSTANTS FROM CONFIG --- *#
    UM = config.UM_VALUE # Bandwidth required by a user

    @classmethod
    def initialise_users(cls, user_ru_mapping, RUs):
        """Initialise the class-level dictionary for Users"""
        users = {}
        for user in user_ru_mapping:
            user_id = user['user_id']
            potential_rus = [RUs[ru_name] for ru_name in user.get('assigned_ru', []) if ru_name in RUs]
            user_instance = cls(user_id=user_id, potential_rus=potential_rus)
            users[user_id] = user_instance
        return users
    
    def __init__(self, user_id, potential_rus=None, assigned_ru=None, is_assigned=False):
        """Initialise a User object."""
        self.user_id = user_id
        self.potential_rus = potential_rus if potential_rus else []
        self.assigned_ru = assigned_ru
        self.is_assigned = is_assigned

    def __str__(self):
        """String representation of the User object."""
        return (
            f"User ID: {self.user_id}\n"
            f"  Assigned RU: {self.assigned_ru}\n"
            f"  Is Assigned: {self.is_assigned}\n"
            f"  Potential RUs: {[ru.name for ru in self.potential_rus]}"
        )
    
    def set_solution(self, solution):
        """Set the solution the user is part of."""
        self.solution = solution

    def make_copy(self):
        """Make a deep copy of the user, excluding the solution attribute."""
        copy_user = User(
            user_id=self.user_id,
            potential_rus=self.potential_rus.copy()
        )
        copy_user.assigned_ru = self.assigned_ru
        copy_user.is_assigned = self.is_assigned
        return copy_user

    def reset_user(self):
        """Reset the user. If the user is assigned to an RU, disconnect the user from the RU."""
        if self.assigned_ru:
            ru = self.solution[self.assigned_ru]
            ru.remove_user(self)  # Disconnect from the RU if assigned
        self.is_assigned = False

    def is_potential_ru(self, ru):
        """Check if the given RU is in the user's potential RUs."""
        return ru.name in [r.name for r in self.potential_rus]

    def update_user(self):
        """Update the user. Update the user assigned RU and assignment flag."""
        self.is_assigned = bool(self.assigned_ru)

    # % =================================================================
    # % RU FUNCTIONS
    # % =================================================================

    def is_connected_to_ru(self, ru=None) -> bool:
        """Check if the user is assigned to an RU. If an RU is provided, check if the user is assigned to that RU."""
        return self.is_assigned if ru is None else self.assigned_ru == ru.name
    
    def allocate_to_ru(self, ru):
        """Assign the user to the given RU."""
        self.assigned_ru = ru.name
        self.is_assigned = True

    def deallocate_from_ru(self):
        """Update the user's assignment to None."""
        self.is_assigned = False
        self.assigned_ru = None