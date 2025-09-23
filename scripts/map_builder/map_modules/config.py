"""
%┏━━━┓┏━━━┓┏━┓━┏┓━┏━━━┓┏━━┓┏━━━┓━━━━┏━┓┏━┓┏━━━┓┏━━━┓┏┓━┏┓┏┓━━━┏━━━┓
%┃┏━┓┃┃┏━┓┃┃┃┗┓┃┃━┃┏━━┛┗┫┣┛┃┏━┓┃━━━━┃┃┗┛┃┃┃┏━┓┃┗┓┏┓┃┃┃━┃┃┃┃━━━┃┏━━┛
%┃┃━┗┛┃┃━┃┃┃┏┓┗┛┃━┃┗━━┓━┃┃━┃┃━┗┛━━━━┃┏┓┏┓┃┃┃━┃┃━┃┃┃┃┃┃━┃┃┃┃━━━┃┗━━┓
%┃┃━┏┓┃┃━┃┃┃┃┗┓┃┃━┃┏━━┛━┃┃━┃┃┏━┓━━━━┃┃┃┃┃┃┃┃━┃┃━┃┃┃┃┃┃━┃┃┃┃━┏┓┃┏━━┛
%┃┗━┛┃┃┗━┛┃┃┃━┃┃┃┏┛┗┓━━┏┫┣┓┃┗┻━┃━━━━┃┃┃┃┃┃┃┗━┛┃┏┛┗┛┃┃┗━┛┃┃┗━┛┃┃┗━━┓
%┗━━━┛┗━━━┛┗┛━┗━┛┗━━┛━━┗━━┛┗━━━┛━━━━┗┛┗┛┗┛┗━━━┛┗━━━┛┗━━━┛┗━━━┛┗━━━┛
%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

def get_color_by_demand(users, dataset):
    """Returns a color based on the number of users in a demand point."""
    if dataset == "small-dataset":
        if users >= 4:
            return 'red'
        elif users >= 3:
            return 'orange'
        elif users <= 2:
            return 'green'
        else:
            return 'blue'
    elif dataset == "full-dataset":
        if users >= 6:
            return 'red'
        elif users >= 4:
            return 'orange'
        elif users >= 3:
            return 'green'
        else:
            return 'blue'
    else:
        return 'blue'
    
def get_radio_unit_color(attribute_value):
    return 'red' if attribute_value == 'selected' else '#710E9D'

def get_distributed_unit_color(attribute_value):
    return 'blue' if attribute_value == 'selected' else 'green'

def get_centralised_unit_color(attribute_value):
    return 'black'
