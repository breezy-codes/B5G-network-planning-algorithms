# Scripts Overview

This folder contains supporting scripts for data preparation, validation, and visualisation. These utilities help you build maps, generate user-to-RU mappings, check penalty factors, and create plots. They are not part of the main optimisation logic but are important for data integrity and understanding results.

---

## Structure

> 📂 **[scripts](.)/** — *Supporting utilities and preprocessing tools*  
>
> - 📂 **[map_builder](map_builder)/** — *Map construction & visualisation*  
>   - 📂 [map_modules](map_builder/map_modules)/ — *Plotting & legend utilities*  
>     - 🐍 [build_map_legend.py](map_builder/map_modules/build_map_legend.py) — *Makes map legends for visualisations.*  
>     - 🐍 [plot_results_graph.py](map_builder/map_modules/plot_results_graph.py) — *Shows network results as graphs.*  
>     - 🐍 [plot_results_short.py](map_builder/map_modules/plot_results_short.py) — *Quick visual checks for debugging.*  
>     - 📄 [README.md](map_builder/map_modules/README.md) — *Docs for map modules.*  
>  
>   - 🐍 [build_all_paths.py](map_builder/build_all_paths.py) — *Generates RU ↔ DU ↔ CU paths for models.*  
>   - 🐍 [check_penalty_factors.py](map_builder/check_penalty_factors.py) — *Checks penalty factor growth rates.*  
>   - 🐍 [make_user_mapping.py](map_builder/make_user_mapping.py) — *Creates user-to-RU mappings.*  
>   - 📄 [README.md](map_builder/README.md) — *Guide to map builder scripts.*  
>  
> - 📄 [README.md](README.md) — *Overview of all scripts (this file)*  

---

## File Breakdown

### Preprocessing

Preprocessing scripts are only needed when you modify the dataset.

- [`build_all_paths.py`](map_builder/build_all_paths.py): Run this script if you add, remove, or change the status of devices (e.g., moving devices between "existing" and "new"). This updates all possible connectivity paths between RUs, DUs, and CUs, ensuring the algorithms use the correct network structure.
- [`make_user_mapping.py`](map_builder/make_user_mapping.py): Use this script when you make changes to RUs, such as adding, removing, or adjusting their coverage radii. It regenerates user-to-RU mappings to reflect the updated coverage and assignments.

### Validation

Validation scripts are optional tools for exploring model parameters.

- [`check_penalty_factors.py`](map_builder/check_penalty_factors.py): Use this script if you want to visualise how different penalty factors grow as model parameters change. It helps you compare penalty functions and understand their behaviour, but is not required for standard workflows.

### Visualisation

Visualisation scripts help you interpret results and create figures for reports.

- [`map_builder/map_modules`](map_builder/map_modules): Contains plotting tools and legend generators.
  - [`plot_results_graph.py`](map_builder/map_modules/plot_results_graph.py): Produces detailed network diagrams showing connections and performance metrics.
  - [`plot_results_short.py`](map_builder/map_modules/plot_results_short.py): Offers quick, simple visualisations for debugging and rapid iteration.
  - [`build_map_legend.py`](map_builder/map_modules/build_map_legend.py): Automatically generates legends for maps, making them easier to read and present.
- These tools are useful for understanding results, debugging issues, and preparing publication-quality figures.

---

## Usage Tips

- Re-run preprocessing scripts ([`make_user_mapping.py`](map_builder/make_user_mapping.py), [`build_all_paths.py`](map_builder/build_all_paths.py)) after any dataset changes.
- Use visualisation tools to interpret results and prepare figures.
