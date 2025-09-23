# Map Builder

This directory contains scripts and modules for **visualising the results** as interactive maps.

The maps are generated using the [Folium](https://python-visualization.github.io/folium/) library and can display the chosen devices along with their paths and connections.

---

## 📂 Folder Structure

> 📂 **[map_builder](map_builder)/** — *Scripts and modules for generating network maps*
>
> * 📁 [map_modules](map_builder/map_modules)/ — *Core modules for map building*
>
>   * 🐍 [`config.py`](map_builder/map_modules/config.py) — *Configuration (colours, thresholds, dataset toggle)*
>   * 🐍 [`helper_functions.py`](map_builder/map_modules/helper_functions.py) — *Utility functions shared by scripts*
>   * 🐍 [`build_map_legend.py`](map_builder/map_modules/build_map_legend.py) — *Generates a map legend with RU, DU, CU markers*
>   * 🐍 [`plot_results_graph.py`](map_builder/map_modules/plot_results_graph.py) — *Plots results from graph-based models (Genetic, Local Search, CPLEX)*
>   * 🐍 [`plot_results_short.py`](map_builder/map_modules/plot_results_short.py) — *Plots results from shortest path models (Greedy, CPLEX)*
>   * 🐍 [`__init__.py`](map_builder/map_modules/__init__.py) — *Makes `map_modules` importable as a package*
> * 📄 [`README.md`](map_builder/README.md) — *Documentation of scripts, usage, and customisation*

---

## Scripts

### 1. **Shortest Path Results Map**

**File:** [`plot_results_short.py`](plot_results_short.py)
This script visualises the results of **Greedy** or **CPLEX shortest path models**. It highlights selected/unselected RUs and DUs and overlays the chosen shortest-path connections.

#### How to Use

1. Populate the following variables at the top of the script with your results:

   * `selected_rus`, `not_selected_rus`
   * `selected_dus`, `not_selected_dus`
2. Set `dataset` to either `"small-dataset"` or `"full-dataset"`.
3. Run the script. The output will be saved in the `maps/` directory as:

   ```
   short_{dataset}.html
   ```

#### Example

```bash
python plot_results_short.py
```

#### Customisation

* Update colour thresholds in `get_color_by_demand` (inside `config.py`).
* Switch dataset via the `dataset` variable.
* Adjust map centre and zoom in `create_map_with_layers` if working with a new region.

---

### 2. **Graph Model Results Map**

**File:** [`plot_results_graph.py`](plot_results_graph.py)
This script visualises results from **Genetic Algorithm**, **Local Search**, or **CPLEX graph models**. It draws full graph-based solutions with their selected network segments.

#### How to Use

1. Set `selected_rus`, `not_selected_rus`, `selected_dus`, `not_selected_dus`.
2. Set `dataset` to `"small-dataset"` or `"full-dataset"`.
3. Provide the **segments file** (e.g., `{run_id}_segments.json`) generated during optimisation.
4. Run the script. The HTML map will be saved in the `maps/` directory.

#### Example

```bash
python plot_results_graph.py
```

#### Customisation

* Adjust colour thresholds in `get_color_by_demand`.
* Switch dataset with the `dataset` variable.
* Update map centre/zoom if visualising a new region.

---

### 3. **Map Legend**

**File:** [`build_map_legend.py`](build_map_legend.py)
This script builds a **map legend** to visually distinguish between different network components, including **RUs**, **DUs**, and **CUs**. It produces a stand-alone legend map for clarity.

#### How to Use

1. Set `dataset` to `"small-dataset"` or `"full-dataset"`.
2. Run the script. It generates a legend map HTML file in the `maps/` directory.

#### Example

```bash
python build_map_legend.py
```

#### Customisation

* Update `get_color_by_demand` thresholds in [`config.py`](map_builder/map_modules/config.py).
* Switch dataset using the `dataset` variable.
* Adjust map centre and zoom if using a different geographic region.

---

## Supporting Modules

* [`config.py`](map_builder/map_modules/config.py) – Configuration file for colour schemes, demand thresholds, and dataset selection.
* [`helper_functions.py`](map_builder/map_modules/helper_functions.py) – Shared utilities for processing JSON data, creating feature groups, and styling maps.

---

## General Notes

* All maps are generated as **HTML files** and saved in the [`maps/`](map_builder/maps/) directory. You can open them in any web browser.
* The scripts assume a dataset structure consistent with `small-dataset` and `full-dataset`. If you build your own dataset, replicate this structure.
* Always check the **centre coordinates and zoom level** for correct geographic alignment.
* For larger datasets, you may want to tune demand thresholds and colour mappings for clarity.
