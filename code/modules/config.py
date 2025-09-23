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
from pathlib import Path

# Parameters that are used in the model and varied in the experiments
UM_VALUE = 2000             # Bandwidth each user requires
FC_VALUE = 40_000           # Bandwidth each fibre can carry
MR_VALUE = 3                # Maximum number of RUs per location
MD_VALUE = 3                # Maximum number of DUs per location
CR_VALUE = 10               # Maximum number of fibre connections between a DU and CU
COVERAGE_THRESHOLD = 0.95   # Coverage threshold of users required

# Device capacity parameters
RU_BASE_BANDWIDTH = 10_000    # Base bandwidth capacity of a Radio Unit (RU)
DU_BASE_BANDWIDTH = 400_000   # Base bandwidth capacity of a Distributed Unit
DU_TOTAL_PORTS = 100           # Total port capacity of a Distributed Unit
CU_BASE_BANDWIDTH = 5_000_000 # Base bandwidth capacity of a Centralised Unit
CU_TOTAL_PORTS = 500          # Total port capacity of a Centralised Unit

# Cost values for the Distributed Unit
DU_COST = 10_500                        # Cost per DU unit
DU_INSTALL_COST = 400                   # Installation cost for the DU
KW_COST = DU_COST + DU_INSTALL_COST     # Total DU Cost for the first one placed
KX_COST = DU_COST                       # Total DU Cost for the rest of the DU placed at a location
KL_COST = 400                           # Cost per fibre connection

# Cost values for the Radio Unit
RU_COST = 6750                          # Cost per RU unit
RU_INSTALL_COST = 400                   # Installation cost for the RU
KV_COST = RU_COST + RU_INSTALL_COST     # Total RU Cost

# Cost values for the Centralised Unit
KB_COST = 800                           # Cost per fibre connection at CU

# Pathing costs
TRENCHING_COST = 80                                     # Cost per metre of trenching
FIBRE_COST = 17.5                                       # Cost per metre of fibre
NUM_FIBRES = 100                                        # Number of fibres to be laid per metre
KY_COST = TRENCHING_COST + (FIBRE_COST * NUM_FIBRES)    # Total cost per metre
KM_COST = 100                                           # Cost per metre of fibre for the last mile

# Run ID for heuristic algorithms
RUN_ID = f"{UM_VALUE}_{COVERAGE_THRESHOLD}_{MR_VALUE}_{MD_VALUE}_{CR_VALUE}"

BASE_LOG_DIR = Path(__file__).parent.parent / "logs"