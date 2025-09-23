# Modules

This folder contains supporting modules for the heuristic algorithm framework. These provide the underlying data structures, configuration, initialisation logic, and utilities that are used by the **genetic algorithm**, **greedy solution generator**, **local search**, and **CPLEX-based solver**.

- **Genetic algorithm** and **greedy solution generator** rely on all core modules for data modeling, configuration, and logging.
- **Local search** modules use the same data classes and configuration utilities for iterative solution improvement.
- **CPLEX files** interface with the same data structures and configuration to solve the problem using mathematical programming.

This modular setup ensures consistency and reusability across all implemented solution approaches.

> 📂 **[modules](.)/** — *Shared modules & utilities*
>
> - 📁 [algorithms](algorithms)/ — *Algorithm-specific helpers*
>   - 🐍 [genetic_algorithm.py](algorithms/genetic_algorithm.py) — *Core implementation of the genetic algorithm*
>   - 🐍 [genetic_algorithm_time.py](algorithms/genetic_algorithm_time.py) — *Genetic algorithm variant with timing instrumentation*
>   - 🐍 [helper_functions.py](algorithms/helper_functions.py) — *Supporting utilities for algorithm runs*
>   - 🐍 [local_search.py](algorithms/local_search.py) — *Local search algorithm implementation*
>   - 🐍 [\_\_init__.py](algorithms/__init__.py) — *Allows importing this folder as a module*
>
> - 📁 [data_classes](data_classes)/ — *Dataclasses for network model components*
>   - 🐍 [dataclass_CU.py](data_classes/dataclass_CU.py) — *Centralised Unit (CU) representation & related functions*
>   - 🐍 [dataclass_DU.py](data_classes/dataclass_DU.py) — *Distributed Unit (DU) logic & operations*
>   - 🐍 [dataclass_RU.py](data_classes/dataclass_RU.py) — *Radio Unit (RU) representation & functionality*
>   - 🐍 [dataclass_roads.py](data_classes/dataclass_roads.py) — *Encapsulation of road/segment data & processing*
>   - 🐍 [dataclass_user.py](data_classes/dataclass_user.py) — *User demand, connectivity requirements & operations*
>   - 🐍 [dataclass_solution.py](data_classes/dataclass_solution.py) — *Solution representation (assignments, costs, etc.)*
>   - 🐍 [\_\_init__.py](data_classes/__init__.py) — *Collects dataclasses for simpler imports*
>
> - 🐍 [config.py](config.py) — *Global experiment parameters*
> - 🐍 [haversine.py](haversine.py) — *Geographic distance calculations*
> - 🐍 [initialise_populations.py](initialise_populations.py) — *Population setup functions for GA and LS*
> - 🐍 [logger.py](logger.py) — *Logging utilities for experiments*
> - 🐍 [results_to_file.py](results_to_file.py) — *Functions to export solution results*
> - 🐍 [save_cplex_results.py](save_cplex_results.py) — *Functions for handling & saving CPLEX outputs*
> - 📄 [README.md](README.md) — *Guide to the modules and structure*

## **Configuration: [`config.py`](config.py)**

The [`config.py`](config.py) file centralises all adjustable parameters for experiments and algorithm runs.  
It defines values such as:

- Cost settings (e.g., trenching, fibre, node costs)
- Coverage radius options
- Bandwidth requirements and limits
- Other experiment-specific variables

To change how the algorithms behave or to tune experiments, edit the relevant values in [`config.py`](config.py).  
This ensures consistent configuration across all modules and solution approaches.

---

## Usage by Algorithms

- The **greedy solution generator** relies on [`config.py`](config.py) for parameters and [`logger.py`](logger.py) for logging.
- The **genetic algorithm** uses all modules in this folder: data classes, configuration ([`config.py`](config.py)), population initialisation, logging ([`logger.py`](logger.py)), and results saving.
- The **local search** algorithm also uses the shared data classes, configuration ([`config.py`](config.py)), and logging utilities ([`logger.py`](logger.py)) for iterative solution improvement.
- The **CPLEX-based solver** uses only the shared configuration ([`config.py`](config.py)) and the CPLEX results handling module ([`save_cplex_results.py`](save_cplex_results.py)) to solve the problem via mathematical programming.

---

## Logging

Detailed logging statements are included in all heuristic algorithm files (genetic algorithm, greedy solution generator, and local search).  

- These log lines are **commented out by default** to keep runs clean.
- To enable step-by-step tracing of heuristic behaviour, simply uncomment the relevant lines.
- This provides fine-grained visibility into intermediate values, execution flow, and decision-making within the heuristics.

This approach allows you to easily switch between standard runs and verbose, exploratory debugging for heuristic development and analysis.

---

## Workflow

1. Configure experiment parameters in [`config.py`](config.py).
2. Choose and run an algorithm:
    - **Greedy solution generator** (uses configuration and logging modules)
    - **Genetic algorithm** (uses all modules)
    - **Local search** (uses shared modules)
    - **CPLEX-based solver** (uses configuration and CPLEX results handler)
3. View generated results in the `results/` directory.
4. For detailed runtime insights, uncomment logging statements in relevant files.

---

This modular structure ensures a clear separation of **configuration**, **data modelling**, and **algorithm logic**, while still making the entire process transparent through optional logs.
