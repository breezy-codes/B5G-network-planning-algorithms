# Heuristic Algorithm – Code

This directory contains the core code for running heuristic-based experiments, including the **Greedy Solution Generator**, **Genetic Algorithm**, and **Local Search**, as well as exact solution models using **CPLEX**. Together, these components create, evolve, and evaluate network deployment solutions.

> 📂 **[code](.)/** — *Core source code*
>
> - 📁 [logs](logs)/ — *Experiment logs & solver outputs*
> - 📁 [modules](modules)/ — *Reusable components, dataclasses & helpers*
> - 📁 [results](results)/ — *Generated results from runs*
>
> - 🐍 [CPLEX_graph_model.py](CPLEX_graph_model.py) — *CPLEX full graph model*
> - 🐍 [CPLEX_shortest_path.py](CPLEX_shortest_path.py) — *CPLEX shortest-path model*
> - 🐍 [genetic_algorithm.py](genetic_algorithm.py) — *Genetic algorithm implementation*
> - 📓 [greedy_solution_generator.ipynb](greedy_solution_generator.ipynb) — *Greedy algorithm notebook*
> - 🐍 [local_search.py](local_search.py) — *Local search implementation*
>
> - 📄 [README.md](README.md) — *Guide to this directory*

---

## File Overview

- [**`greedy-solution-generator.ipynb`**](greedy-solution-generator.ipynb)
  Generates the initial pool of feasible solutions used to seed the genetic and local search algorithms.

  - Produces valid but not necessarily optimal solutions.
  - Run **more iterations than the number of required solutions** to ensure a diverse pool.

    - Example: For 50 starting solutions, run 60–70 greedy iterations.

- [**`genetic_algorithm.py`**](genetic_algorithm.py)
  Evolves greedy-generated solutions toward higher-quality results using crossover, mutation, and selection.

  - Reads input data from the chosen dataset (see [Dataset README](../dataset/README.md)).
  - Outputs solutions, logs, and metrics to the [`results/`](results/) directory.
  - Uses shared components from the [`modules/`](modules/) folder.

- [**`local_search.py`**](local_search.py)
  Evolves greedy-generated solutions toward higher-quality results using crossover, mutation, and selection.

  - Reads input data from the chosen dataset (see [Dataset README](../dataset/README.md)).
  - Outputs solutions, logs, and metrics to the [`results/`](results/) directory.
  - Uses shared components from the [`modules/`](modules/) folder.

- [**`cplex_shortest_path.py`**](cplex_shortest_path.py)
  Builds an exact shortest-path model using CPLEX.

  - Dataset is chosen via the dataset variable.
  - Outputs results to [`results/`](results/).

- [**`cplex_graph.py`**](cplex_graph.py)
  Implements a full graph-based CPLEX model for exact deployment solutions.

  - Dataset is chosen via the dataset variable.
  - Saves results to [`results/`](results/).

- [**`modules/`**](modules/)
  Shared configuration files, dataclasses, and utilities. See the [Modules README](modules/README.md) for details.

- [**`logs/`**](logs/)
  Runtime log files. Auto-generated and excluded via `.gitignore`.

---

## Dataset Selection

All algorithms being the **Greedy**, **Genetic**, **Local Search**, and **CPLEX models** require data from the [`dataset/`](../dataset/) directory.

To select a dataset, edit the `dataset` variable in the code:

```python
# Choose between: "small-dataset" or "full-dataset"
dataset = "small-dataset"
```

**Available datasets:**

- [**`small-dataset`**](../dataset/small-dataset/) – Lightweight runs for debugging and fast testing.
- [**`full-dataset`**](../dataset/full-dataset/) – Full-scale runs for performance evaluation.

**Notes:**

- Use the same dataset consistently across scripts for reproducibility.
- Datasets can be customised or extended by following the structure and instructions in the [Dataset README](../dataset/README.md).

---

## Configuration

All experiment parameters are defined in [**`modules/config.py`**](modules/config.py), including:

- Cost values (trenching, fibre, node costs)
- Coverage radii
- Bandwidth requirements
- Tunable constraints

Modify this file to configure experiments globally across all algorithms and models.

---

## Results

Outputs are saved in the [**`results/`**](results/) directory:

- **Greedy Algorithm:** Final solutions only.
- **Genetic Algorithm:** Solutions, logs per iteration, and used-segment JSONs.
- **Local Search:** Solutions, logs per iteration, and used-segment JSONs.
- **CPLEX Models:**

  - Results and solver logs saved in [`results/`](results/).
  - Solver-generated files (`.lp`, `.sol`) stored in [`results/cplex-generated-files/`](results/cplex-generated-files/).

---

## Usage Guide

1. Run [**`greedy-solution-generator.ipynb`**](greedy-solution-generator.ipynb) to generate an initial solution pool.

   - Ensure iterations > required solutions (e.g., 60–70 runs for 50 needed solutions).

2. Choose a refinement method:

   - [**`genetic_algorithm.py`**](genetic_algorithm.py) → Evolves solutions over generations.
   - [**`local_search.py`**](local_search.py) → Iteratively improves solutions.

3. (Optional) For exact optimisation, run:

   - [**`cplex_shortest_path.py`**](cplex_shortest_path.py) or
   - [**`cplex_graph.py`**](cplex_graph.py).
   - Adjust the `dataset` variable and [**`modules/config.py`**](modules/config.py) as needed.

4. Inspect output files in [**`results/`**](results/).

---

## Workflow Summary

```text
dataset → greedy → genetic → local search → results
dataset → CPLEX (shortest path or graph) → results
```

This workflow combines heuristic (greedy, genetic, local search) and exact (CPLEX) approaches to network design, ensuring all experiments remain **reproducible, configurable, and extensible**.
