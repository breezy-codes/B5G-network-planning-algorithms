"""
%┏┓━━━┏━━━┓┏━━━┓┏━━━┓┏┓━━━━━━━┏━━━┓┏━━━┓┏━━━┓┏━━━┓┏━━━┓┏┓━┏┓
?┃┃━━━┃┏━┓┃┃┏━┓┃┃┏━┓┃┃┃━━━━━━━┃┏━┓┃┃┏━━┛┃┏━┓┃┃┏━┓┃┃┏━┓┃┃┃━┃┃
%┃┃━━━┃┃━┃┃┃┃━┗┛┃┃━┃┃┃┃━━━━━━━┃┗━━┓┃┗━━┓┃┃━┃┃┃┗━┛┃┃┃━┗┛┃┗━┛┃
?┃┃━┏┓┃┃━┃┃┃┃━┏┓┃┗━┛┃┃┃━┏┓━━━━┗━━┓┃┃┏━━┛┃┗━┛┃┃┏┓┏┛┃┃━┏┓┃┏━┓┃
%┃┗━┛┃┃┗━┛┃┃┗━┛┃┃┏━┓┃┃┗━┛┃━━━━┃┗━┛┃┃┗━━┓┃┏━┓┃┃┃┃┗┓┃┗━┛┃┃┃━┃┃
?┗━━━┛┗━━━┛┗━━━┛┗┛━┗┛┗━━━┛━━━━┗━━━┛┗━━━┛┗┛━┗┛┗┛┗━┛┗━━━┛┗┛━┗┛
%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
?━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This module implements a local search algorithm for optimising individual solutions in a network model.

"""

import os
from collections import Counter
from datetime import datetime

from modules.results_to_file import save_results_to_file
from modules.initialise_populations import initialise_multiple_individuals
from modules.algorithms.helper_functions import log_iteration, log_best_individuals_json, log_final_summary_solutions
from modules.logger import loggers

def local_search(individuals, max_iterations=100, num_neighbours=10, num_mutations=5, log_file=None, best_individual_per_iteration_path=None):
    """Run local search over multiple individual solutions. Each individual explores its own neighbourhood per iteration."""
    
    cover_penalty_base = 500  # Initial cover penalty base value

    # Build the initial population of individuals
    current_individuals, best_individuals, best_fitness_scores = [], [], []
    for base_individual in individuals:
        current = base_individual.__deepcopy__()
        current.update_model()
        fitness = current.evaluate_fitness(cover_penalty_base=cover_penalty_base)
        current_individuals.append(current)
        best_individuals.append(current.__deepcopy__())
        best_fitness_scores.append(fitness)

    current_individuals = list(current_individuals)
    best_individuals = list(best_individuals)
    best_fitness_scores = list(best_fitness_scores)

    # Setup the logging which will track our best iteration information
    all_logs = [[] for _ in current_individuals]
    mutation_counters = [Counter() for _ in current_individuals]

    # Log our first iteration details
    for individual_iteration, (sol, fit) in enumerate(zip(best_individuals, best_fitness_scores)):
        msg = f"[Individual {individual_iteration}] Initial cost: {sol.total_cost:.2f}, fitness: {fit:.2f}"
        loggers['feasible_logger'].debug(msg)
        all_logs[individual_iteration].append(msg)

    # Log file for the best individuals per iteration
    for iteration in range(max_iterations):
        start_time = datetime.now()
        cover_penalty_base = min(cover_penalty_base * 1.15, 9e6)

        results = []

        for idx, current_sol in enumerate(current_individuals):
            current_sol.update_model()

            neighbours = []
            for _ in range(num_neighbours):
                copy = current_sol.__deepcopy__()
                copy.run_mutation(num_mutations=num_mutations)
                neighbours.append(copy)

            feasible_costs = []
            best_neighbour = None
            best_neighbour_fit = float('inf')
            best_neighbour_index = None
            best_mutations = []

            # Evaluate the fitness of each neighbour solution
            eval_results = [(neighbour_index, neighbour, neighbour.evaluate_fitness(cover_penalty_base)) for neighbour_index, neighbour in enumerate(neighbours)]

            for neighbour_index, neighbour_solution, fitness in eval_results:
                if neighbour_solution.is_feasible():
                    feasible_costs.append((neighbour_index + 1, neighbour_solution.total_cost, fitness))
                    if fitness < best_neighbour_fit:
                        best_neighbour, best_neighbour_fit = neighbour_solution, fitness
                        best_neighbour_index = neighbour_index + 1
                        best_mutations = neighbour_solution.applied_mutations
                        best_neighbour.update_model()

            results.append((idx, best_neighbour, best_neighbour_fit, best_neighbour_index, best_mutations, feasible_costs))

        # Process results
        for idx, best_neighbour, best_neighbour_fit, best_neighbour_index, mutations, feasible_costs in results:
            prefix = f"[Individual {idx}][Iter {iteration + 1}]"

            current_fit = current_individuals[idx].evaluate_fitness(cover_penalty_base)

            # If a better neighbour is found, update the current solution
            # Here we check each solution's best neighbour and update it if it's better than the current solution
            if best_neighbour and best_neighbour_fit < current_fit:
                current_individuals[idx] = best_neighbour
                if best_neighbour_fit < best_fitness_scores[idx]:
                    best_individuals[idx] = best_neighbour
                    best_fitness_scores[idx] = best_individuals[idx].evaluate_fitness(cover_penalty_base)
                    msg = f"{prefix} New best with fitness {best_neighbour_fit:.2f} and cost {best_neighbour.total_cost:.2f} from N{best_neighbour_index} (mutations: {', '.join(mutations)})"
                    loggers['feasible_logger'].debug(msg)
                    all_logs[idx].append(msg)
                    mutation_counters[idx].update(mutations)
            else:
                # No better neighbour found, but re-evaluate current best individual with new penalty which might have changed the fitness
                best_individuals[idx].update_model()
                best_fitness_scores[idx] = best_individuals[idx].evaluate_fitness(cover_penalty_base)
        
        # Ensure all best individuals are re-evaluated with updated penalty
        for idx in range(len(best_individuals)):
            best_individuals[idx].update_model()
            best_fitness_scores[idx] = best_individuals[idx].evaluate_fitness(cover_penalty_base)

        # Log global best for this iteration
        best_global_idx = best_fitness_scores.index(min(best_fitness_scores))
        best_global_fitness = best_fitness_scores[best_global_idx]
        best_global_individual = best_individuals[best_global_idx]

        # Also log the best individual per iteration to a file for checking later.
        log_best_individuals_json(iteration, best_individuals, best_fitness_scores, best_global_idx, best_global_fitness, best_global_individual, best_individual_per_iteration_path)
        
        # Log the iteration details
        algo = "LS"
        log_iteration(iteration, best_global_fitness, best_global_fitness, best_global_individual, log_file, algo=algo, start_time=start_time)

    return best_individuals, all_logs, mutation_counters

def save_local_search(best_solution_path, RUs, DUs, CUs, users, output_dir, graph, scenario, run_id, max_iterations=100, num_neighbours=10, num_mutations=5, num_solutions=5):
    # Initialise multiple individual solutions from JSON
    solution_dict = initialise_multiple_individuals(best_solution_path, RUs, DUs, CUs, users, graph, scenario)
    individuals = list(solution_dict.values())[:num_solutions]
    solution_names = list(solution_dict.keys())[:num_solutions]

    if not individuals:
        raise ValueError("No feasible individual solutions were initialised.")

    # Paths for saving
    best_output_path = os.path.join(output_dir, f"LS_{run_id}.txt")
    best_json_path = os.path.join(output_dir, f"LS_{run_id}_segments.json")
    log_path = os.path.join(output_dir, f"LS_{run_id}.log")
    best_individual_per_iteration_path = os.path.join(output_dir, f"LS_{run_id}_best_individuals_per_iter.log")

    # Clear log files at the start of the run if they exist
    for path in [best_individual_per_iteration_path]: open(path, "w").close()
    
    # Run local search across multiple individual solutions
    with open(log_path, "w") as log_file:
        final_solutions, best_log_lists, mutation_counters = local_search(individuals, max_iterations=max_iterations, num_neighbours=num_neighbours, num_mutations=num_mutations, log_file=log_file, best_individual_per_iteration_path=best_individual_per_iteration_path)

    # Find overall best final solution
    fitnesses = [s.evaluate_fitness(cover_penalty_base=50) for s in final_solutions]
    best_index = fitnesses.index(min(fitnesses))
    best_solution = final_solutions[best_index]
    best_logs = best_log_lists[best_index]
    best_mutations = mutation_counters[best_index]
    best_name = solution_names[best_index]

    # Save the best solution
    save_results_to_file(best_solution, best_output_path, best_json_path, algorithm=f"local_search")

    # Append logs and mutation summary
    log_final_summary_solutions(best_output_path, best_name, best_logs, best_mutations)