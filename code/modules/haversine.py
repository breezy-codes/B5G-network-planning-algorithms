"""
%┏┓━┏┓┏━━━┓┏┓━━┏┓┏━━━┓┏━━━┓┏━━━┓┏━━┓┏━┓━┏┓┏━━━┓
%┃┃━┃┃┃┏━┓┃┃┗┓┏┛┃┃┏━━┛┃┏━┓┃┃┏━┓┃┗┫┣┛┃┃┗┓┃┃┃┏━━┛
%┃┗━┛┃┃┃━┃┃┗┓┃┃┏┛┃┗━━┓┃┗━┛┃┃┗━━┓━┃┃━┃┏┓┗┛┃┃┗━━┓
%┃┏━┓┃┃┗━┛┃━┃┗┛┃━┃┏━━┛┃┏┓┏┛┗━━┓┃━┃┃━┃┃┗┓┃┃┃┏━━┛
%┃┃━┃┃┃┏━┓┃━┗┓┏┛━┃┗━━┓┃┃┃┗┓┃┗━┛┃┏┫┣┓┃┃━┃┃┃┃┗━━┓
%┗┛━┗┛┗┛━┗┛━━┗┛━━┗━━━┛┗┛┗━┛┗━━━┛┗━━┛┗┛━┗━┛┗━━━┛
%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This file contains the Haversine function to calculate the distance between two geographical coordinates.

It is used in various parts of the project to compute distances between radio units, distributed units, and centralised units based on their latitude and longitude.

"""

import math

def haversine(coord1, coord2):
    """Calculate the Haversine distance between two coordinates."""
    R = 6371
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c