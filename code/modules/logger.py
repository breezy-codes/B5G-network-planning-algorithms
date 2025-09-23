"""
%┏┓━━━┏━━━┓┏━━━┓┏━━━┓┏━━━┓┏━━━┓━━━━┏━┓┏━┓┏━━━┓┏━━━┓┏┓━┏┓┏┓━━━┏━━━┓
%┃┃━━━┃┏━┓┃┃┏━┓┃┃┏━┓┃┃┏━━┛┃┏━┓┃━━━━┃┃┗┛┃┃┃┏━┓┃┗┓┏┓┃┃┃━┃┃┃┃━━━┃┏━━┛
%┃┃━━━┃┃━┃┃┃┃━┗┛┃┃━┗┛┃┗━━┓┃┗━┛┃━━━━┃┏┓┏┓┃┃┃━┃┃━┃┃┃┃┃┃━┃┃┃┃━━━┃┗━━┓
%┃┃━┏┓┃┃━┃┃┃┃┏━┓┃┃┏━┓┃┏━━┛┃┏┓┏┛━━━━┃┃┃┃┃┃┃┃━┃┃━┃┃┃┃┃┃━┃┃┃┃━┏┓┃┏━━┛
%┃┗━┛┃┃┗━┛┃┃┗┻━┃┃┗┻━┃┃┗━━┓┃┃┃┗┓━━━━┃┃┃┃┃┃┃┗━┛┃┏┛┗┛┃┃┗━┛┃┃┗━┛┃┃┗━━┓
%┗━━━┛┗━━━┛┗━━━┛┗━━━┛┗━━━┛┗┛┗━┛━━━━┗┛┗┛┗┛┗━━━┛┗━━━┛┗━━━┛┗━━━┛┗━━━┛
%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import logging
from pathlib import Path
from typing import Dict, List

from modules import config

# Base log directory
BASE_LOG_DIR = config.BASE_LOG_DIR

def ensure_log_dir(log_dir: Path):
    """Ensure the base log directory and the target log directory exist."""
    BASE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

def setup_logger(name: str, log_dir: Path, log_file: str, level=logging.DEBUG, format='%(levelname)s - %(message)s', propagate=False, overwrite=False):
    """Set up and configure a logger."""
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(level)
        logger.propagate = propagate

        # File handler
        mode = 'w' if overwrite else 'a'
        handler = logging.FileHandler(log_dir / log_file, mode=mode)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(format))
        logger.addHandler(handler)

    return logger

def initialise_loggers(base_dir: Path, config_dict: Dict[str, List[Dict]], aggregate_log_file: str = None, specific_aggregate_loggers: List[str] = None, overwrite=False):
    """Initialise loggers for multiple configurations."""
    loggers = {}
    aggregate_handler = None

    # Create aggregate handler if specified
    if aggregate_log_file and specific_aggregate_loggers:
        aggregate_handler = logging.FileHandler(base_dir / aggregate_log_file, mode='a' if not overwrite else 'w')
        aggregate_handler.setLevel(logging.DEBUG)
        aggregate_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))

    for log_dir_name, logger_configs in config_dict.items():
        log_dir = base_dir / log_dir_name
        ensure_log_dir(log_dir)

        for logger_config in logger_configs:
            logger = setup_logger(
                name=logger_config['name'],
                log_dir=log_dir,
                log_file=logger_config['log_file'],
                level=logger_config.get('level', logging.DEBUG),
                format=logger_config.get('format', '%(levelname)s - %(message)s'),
                propagate=logger_config.get('propagate', False),
                overwrite=overwrite
            )
            loggers[logger_config['name']] = logger

            # Add to specific aggregate handler if applicable
            if aggregate_handler and logger_config['name'] in specific_aggregate_loggers:
                logger.addHandler(aggregate_handler)

            # Inject run header for heuristic logs only
            if log_dir_name == "heuristic":
                run_header = (
                    "\n"
                    + "=" * 80 + "\n"
                    + "=" * 80 + "\n"
                    + f"STARTING NEW RUN - RUN_ID: {config.RUN_ID}\n"
                    + "=" * 80 + "\n"
                    + "=" * 80 + "\n"
                )
                logger.info(run_header)

    return loggers

# Logger configurations for different directories
LOGGER_CONFIGS = {
    "general": [
        {"name": "RU_logger", "log_file": "RU_log.log"},
        {"name": "DU_logger", "log_file": "DU_log.log"},
        {"name": "CU_logger", "log_file": "CU_log.log"},
        {"name": "user_logger", "log_file": "user_log.log"},
        {"name": "Path_logger", "log_file": "path_log.log"},
        {"name": "logger", "log_file": "log.log"},
        {"name": "feasible_logger", "log_file": "feasible.log"},
        {"name": "mutate_logger", "log_file": "mutate.log"},
        {"name": "crossover_logger", "log_file": "crossover.log"},
        {"name": "device_logger", "log_file": "device_log.log"},
    ],
    "heuristic": [ 
        {"name": "genetic_algo", "log_file": f'genetic_{config.RUN_ID}.log', "format": "%(asctime)s - %(message)s"},
        {"name": "greedy_algo", "log_file": f'greedy_{config.RUN_ID}.log', "format": "%(asctime)s - %(message)s"},
        {"name": "local_search_algo", "log_file": f'local_search_{config.RUN_ID}.log', "format": "%(asctime)s - %(message)s"},
    ]
}

# List of loggers to include in the specific aggregate log file
SPECIFIC_AGGREGATE_LOGGERS = [
    "Path_logger",
    "logger",
    "feasible_logger",
    "mutate_logger",
    "crossover_logger",
    "device_logger",
    "RU_logger",
    "DU_logger",
    "user_logger",
]

# Initialise loggers
loggers = initialise_loggers(BASE_LOG_DIR, LOGGER_CONFIGS, aggregate_log_file="specific_aggregate.log", specific_aggregate_loggers=SPECIFIC_AGGREGATE_LOGGERS)

def clear_general_logs(base_dir: Path, clear_aggregate: bool = True):
    """Clear all 'general' log files and optionally the aggregate log."""
    general_dir = base_dir / "general"

    # Clear individual log files from SPECIFIC_AGGREGATE_LOGGERS
    for logger_conf in LOGGER_CONFIGS.get("general", []):
        name = logger_conf["name"]
        if name in SPECIFIC_AGGREGATE_LOGGERS:
            log_path = general_dir / logger_conf["log_file"]
            if log_path.exists():
                log_path.write_text("")

    # Clear the specific aggregate log if it exists in base_dir
    if clear_aggregate:
        aggregate_log = base_dir / "specific_aggregate.log"
        if aggregate_log.exists():
            aggregate_log.write_text("")

def log_begin(logger, func, msg=""): 
    logger.info(f"{func}: BEGIN -------- " + (f"{msg}" if msg else ""))

def log_end(logger, func, msg=""): 
    logger.info(f"{func}: END -------- " + (f"{msg}" if msg else ""))

def log_fail(logger, func, e, msg=""):
    logger.error(f"{func}: FAIL ({e}) -------- " + (f"{msg}" if msg else ""))

def log_fail_no_exception(logger, func, msg=""):
    logger.error(f"{func}: FAIL -------- " + (f"{msg}" if msg else ""))