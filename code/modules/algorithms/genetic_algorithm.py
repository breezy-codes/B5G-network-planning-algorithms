"""
%┏━━━┓┏━━━┓┏━┓━┏┓┏━━━┓┏━━━━┓┏━━┓┏━━━┓━━━━┏━━━┓┏┓━━━┏━━━┓┏━━━┓
?┃┏━┓┃┃┏━━┛┃┃┗┓┃┃┃┏━━┛┃┏┓┏┓┃┗┫┣┛┃┏━┓┃━━━━┃┏━┓┃┃┃━━━┃┏━┓┃┃┏━┓┃
%┃┃━┗┛┃┗━━┓┃┏┓┗┛┃┃┗━━┓┗┛┃┃┗┛━┃┃━┃┃━┗┛━━━━┃┃━┃┃┃┃━━━┃┃━┗┛┃┃━┃┃
?┃┃┏━┓┃┏━━┛┃┃┗┓┃┃┃┏━━┛━━┃┃━━━┃┃━┃┃━┏┓━━━━┃┗━┛┃┃┃━┏┓┃┃┏━┓┃┃━┃┃
%┃┗┻━┃┃┗━━┓┃┃━┃┃┃┃┗━━┓━┏┛┗┓━┏┫┣┓┃┗━┛┃━━━━┃┏━┓┃┃┗━┛┃┃┗┻━┃┃┗━┛┃
?┗━━━┛┗━━━┛┗┛━┗━┛┗━━━┛━┗━━┛━┗━━┛┗━━━┛━━━━┗┛━┗┛┗━━━┛┗━━━┛┗━━━┛
%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
?━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""

import os, random, time
from datetime import datetime
from collections import Counter
from itertools import combinations

from modules.results_to_file import save_results_to_file
from modules.initialise_populations import initialise_multiple_individuals, crossover_population
from modules.logger import loggers
from modules.algorithms.helper_functions import log_iteration, log_best_individuals_json, log_final_summary_solutions

def genetic_algorithm(individuals, run_id, output_dir, solution_names=None, RUs=None, DUs=None, CUs=None, users=None, graph=None, max_iterations=100, time_limit_minutes=None, num_neighbours=10, num_mutations=5, crossover_interval=5, elite_pop=5, num_clones = 5, log_file=None, best_individual_per_iteration_path=None, save_interval=None):
    """Run GA over multiple individual solutions. Each individual explores its own neighbourhood per iteration, and all individuals are processed in parallel each iteration."""
    
    cover_penalty_base = 500  # Initial cover penalty base value
    cover_penalty_initial = cover_penalty_base  # Store the initial value for resetting after crossover

    stagnant_counter = 0
    best_seen_fitness = float("inf")
    mutation_boost_counter = 0

    # Configure the time limit and iteration control
    if time_limit_minutes is not None:
        time_limit_seconds = time_limit_minutes * 60
        start_time = time.time()
        end_time = start_time + time_limit_seconds
        use_time_limit = True
    else:
        use_time_limit = False

    # Build the initial population of individuals
    current_individuals, best_individuals, best_fitness_scores = [], [], []
    for base_individual in individuals:
        current = base_individual.__deepcopy__()
        current.update_model()
        fitness = current.evaluate_fitness(cover_penalty_base=cover_penalty_base)
        current_individuals.append(current)
        best_individuals.append(current)
        best_fitness_scores.append(fitness)

    if solution_names is None:
        # If no solution names provided, generate default names
        solution_names = [f"sol_{i}" for i in range(len(individuals))]

    current_individuals = list(current_individuals)
    best_individuals = list(best_individuals)
    best_fitness_scores = list(best_fitness_scores)

    # Setup the logging which will track our best iteration information
    all_logs = [[] for _ in current_individuals]
    mutation_counters = [Counter() for _ in current_individuals]

    # Create a flag to track if an individual is an offspring, then keep a list of penalties for each individual
    is_offspring = [False] * len(individuals)
    individual_penalties = [cover_penalty_base for _ in range(len(individuals))]

    # Log file for the best individuals per iteration
    iteration = 0
    while True:
        # Stopping condition: either time limit or max_iterations
        if use_time_limit:
            if time.time() >= end_time:
                break
        else:
            if iteration >= max_iterations:
                break

        start_time_dt = datetime.now()
        results = []

        #! Mutation Neighbourhood Search Step
        # Only do neighbour search if it's not a crossover or Steiner Tree rebuild iteration
        is_crossover_iter = (iteration + 1) % crossover_interval == 0
        is_steiner_rebuild_iter = (iteration + 1) % crossover_interval == crossover_interval // 2

        if not (is_crossover_iter or is_steiner_rebuild_iter):
            # Loop through each individual, apply mutations, and evaluate neighbours
            for idx, current_sol in enumerate(current_individuals):
                # Decide how many mutations to use this iteration
                effective_num_mutations = num_mutations
                if mutation_boost_counter > 0:
                    if random.random() < 0.5:
                        effective_num_mutations = num_mutations + random.randint(2, 5)
                    else:
                        effective_num_mutations = max(3, num_mutations - random.randint(2, 5))

                neighbours = []
                for _ in range(num_neighbours):
                    copy = current_sol.__deepcopy__()
                    mutation_variation = random.randint(-3, 3)  
                    num_mut = max(1, effective_num_mutations + mutation_variation)
                    copy.run_mutation(num_mutations=num_mut)
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
        else:
            # If skipping neighbour search, append placeholder results for all individuals
            results = [(idx, None, float('inf'), None, [], []) for idx in range(len(current_individuals))]

        # Check if we need to decrement the counter for mutation boost
        if mutation_boost_counter > 0:
            mutation_boost_counter -= 1

        #! Crossover Step, if it's the right iteration
        if (iteration + 1) % crossover_interval == 0:
            # Adaptive crossover count
            if stagnant_counter >= 75:
                # If stagnant, use up to half the population for crossovers, but never exceed available non-elites
                num_crossovers = min(max(1, len(individuals) // 2), len(individuals) - elite_pop)
            else:
                # Default is one third of the population
                num_crossovers = max(1, len(individuals) // 3)
            
            offspring = []
            offspring_names = []
            new_is_offspring = []
            new_individual_penalties = []

            # Step 1: Zip individuals with fitness and sort
            ranked_best = sorted(zip(best_individuals, best_fitness_scores, solution_names), key=lambda x: x[1])

            # Step 2: Generate all unique parent pairs
            ranked_pairs = list(combinations(ranked_best, 2))

            # Step 3: Sort parent pairs by combined fitness (lower total fitness = better)
            ranked_pairs.sort(key=lambda pair: pair[0][1] + pair[1][1])

            # Step 4: Take top num_crossovers as unique pairs
            selected_pairs = []
            for (p1, f1, n1), (p2, f2, n2) in ranked_pairs:
                EPS = 1e-6
                if abs(f1 - f2) > EPS:
                    selected_pairs.append(((p1, f1, n1), (p2, f2, n2)))
                if len(selected_pairs) >= num_crossovers:
                    break

            # Step 5: Perform crossover for each pair
            for i, ((parent1, _, name1), (parent2, _, name2)) in enumerate(selected_pairs):
                child = crossover_population(parent1=parent1, parent2=parent2, RUs=RUs, DUs=DUs, CUs=CUs, users=users, graph=graph)
                child_fitness = child.evaluate_fitness(cover_penalty_initial)
                offspring.append((child, child_fitness))
                offspring_names.append(f"offspring_{iteration + 1}_{i}")
                new_is_offspring.append(True)
                new_individual_penalties.append(cover_penalty_initial)

            # Combine current adults (sorted by fitness) with offspring
            named_current = list(zip(current_individuals, solution_names, best_fitness_scores, is_offspring, individual_penalties))
            named_offspring = [(child, name, fit, True, pen) for (child, fit), name, pen in zip(offspring, offspring_names, new_individual_penalties)]
            named_current.sort(key=lambda x: x[2])  # sort adults by fitness

            # Keep all offspring, fill remaining slots with best adults
            num_to_keep = len(individuals) - len(named_offspring)
            retained_adults = named_current[:num_to_keep]

            # Clone top individuals
            top_individuals_to_clone = named_current[:num_clones]
            clones = [(ind.__deepcopy__(), f"{name}_clone", fit, False, pen) for ind, name, fit, _, pen in top_individuals_to_clone]

            # Replace the worst retained adults with clones
            if num_clones <= len(retained_adults):
                retained_adults[-num_clones:] = clones
            else:
                retained_adults = clones[:len(retained_adults)]

            # Protect top N adults, reset penalty for the rest (after cloning)
            for i in range(elite_pop, len(retained_adults)):
                retained_adults[i] = (*retained_adults[i][:4], cover_penalty_initial)

            combined = named_offspring + retained_adults

            # Unpack updated population info
            current_individuals = [ind for ind, _, _, _, _ in combined]
            solution_names = [name for _, name, _, _, _ in combined]
            best_fitness_scores = [fit for _, _, fit, _, _ in combined]
            is_offspring = [flag for _, _, _, flag, _ in combined]
            individual_penalties = [pen for _, _, _, _, pen in combined]

            # Deepcopy best individuals for the next iteration
            best_individuals = [ind.__deepcopy__() for ind in current_individuals]

            for best_ind in best_individuals:
                best_ind.update_model()

        # Set the penalties for each individual, ensuring they are capped at 5 million
        individual_penalties = [min(p * 1.2, 5e6) for p in individual_penalties]

        #! Exact Steiner Tree Rebuild Step, if it's the right iteration
        if (iteration + 1) % crossover_interval == crossover_interval // 2:
            loggers['feasible_logger'].debug(f"[Iter {iteration + 1}] Rebuilding all individuals using exact Steiner Tree algorithm.")
            for idx, individual in enumerate(current_individuals):
                individual.traversing_tree_algorithm()
                individual.update_model()
                # Re-evaluate fitness after rebuilding the tree
                best_fitness_scores[idx] = individual.evaluate_fitness(individual_penalties[idx])

        #! Update all individuals with results from this iteration
        for idx, best_neighbour, best_neighbour_fit, best_neighbour_index, mutations, feasible_costs in results:
            prefix = f"[{solution_names[idx]}][Iter {iteration + 1}]"
            current_fit = current_individuals[idx].evaluate_fitness(individual_penalties[idx])

            # Detect if this was a crossover or Steiner iteration (no neighbours/mutations)
            is_crossover_iter = (iteration + 1) % crossover_interval == 0
            is_steiner_rebuild_iter = (iteration + 1) % crossover_interval == crossover_interval // 2

            if best_neighbour and best_neighbour_fit < current_fit:
                current_individuals[idx] = best_neighbour
                if best_neighbour_fit < best_fitness_scores[idx]:
                    best_individuals[idx] = best_neighbour
                    best_fitness_scores[idx] = best_individuals[idx].evaluate_fitness(individual_penalties[idx])
                    if is_crossover_iter:
                        msg = (f"{prefix} New best with fitness {best_neighbour_fit:.2f} and cost {best_neighbour.total_cost:.2f} from crossover.")
                    elif is_steiner_rebuild_iter:
                        msg = (f"{prefix} New best with fitness {best_neighbour_fit:.2f} and cost {best_neighbour.total_cost:.2f} from Steiner rebuild.")
                    else:
                        msg = (f"{prefix} New best with fitness {best_neighbour_fit:.2f} and cost {best_neighbour.total_cost:.2f} from N{best_neighbour_index} (total mutations: {len(mutations)}; mutations: {', '.join(mutations)})")
                    mutation_counters[idx].update(mutations)
                    all_logs[idx].append(msg)
                else:
                    best_individuals[idx].update_model()
                    best_fitness_scores[idx] = best_individuals[idx].evaluate_fitness(individual_penalties[idx])
        
        #! Penalty Management Step, ensure all best individuals are re-evaluated with updated penalty
        penalties_this_iter = []

        for idx in range(len(best_individuals)):
            best_fitness_scores[idx] = best_individuals[idx].evaluate_fitness(individual_penalties[idx])
            penalties_this_iter.append(individual_penalties[idx])

        # Log global best for this iteration
        best_global_idx = best_fitness_scores.index(min(best_fitness_scores))
        best_global_fitness = best_fitness_scores[best_global_idx]
        best_global_individual = best_individuals[best_global_idx]

        #! Stagnation Detection and Mutation Boosting Step
        # Check for stagnation
        if best_global_fitness < best_seen_fitness:
            best_seen_fitness = best_global_fitness
            stagnant_counter = 0
            mutation_boost_counter = 0  # reset any ongoing boost
        else:
            stagnant_counter += 1

        if stagnant_counter >= 50 and mutation_boost_counter == 0:
            mutation_boost_counter = 5  # activate boost for 5 iterations
            stagnant_counter = 0

        # Also log the best individual per iteration to a file for checking later.
        log_best_individuals_json(iteration, best_individuals, best_fitness_scores, best_global_idx, best_global_fitness, best_global_individual, best_individual_per_iteration_path, solution_names=solution_names, penalties=penalties_this_iter)

        # Log the iteration details
        algo = "GA"
        log_iteration(iteration, best_global_fitness, best_seen_fitness, best_global_individual, log_file, algo=algo, start_time=start_time_dt)

        # If at the specified save interval, save the best individual and its details
        if save_interval and (iteration + 1) % save_interval == 0 and (iteration + 1) != max_iterations and output_dir and run_id:
            basename = f"GA_{run_id}_iter_{iteration + 1}"
            intermediate_output_path = os.path.join(output_dir, f"{basename}.txt")
            intermediate_json_path = os.path.join(output_dir, f"{basename}_segments.json")
            save_results_to_file(best_global_individual, intermediate_output_path, intermediate_json_path, algorithm="genetic_algorithm")
    
        iteration += 1

    # Determine final best at exit time
    final_best_idx = best_fitness_scores.index(min(best_fitness_scores))
    final_best_individual = best_individuals[final_best_idx]
    final_best_log = all_logs[final_best_idx]
    final_best_mutations = mutation_counters[final_best_idx]
    final_best_name = solution_names[final_best_idx]

    return best_individuals, all_logs, mutation_counters, solution_names, final_best_individual, final_best_log, final_best_mutations, final_best_name

def save_genetic_algorithm(best_solution_path, RUs, DUs, CUs, users, output_dir, graph, scenario, run_id, max_iterations=100, num_neighbours=10, num_mutations=5, num_solutions=5, crossover_interval=5, save_interval=5, elite_pop=5, num_clones=5, time_limit_minutes=None):
    # Initialise multiple individual solutions from JSON

    solution_dict = initialise_multiple_individuals(best_solution_path, RUs, DUs, CUs, users, graph, scenario)
    individuals = list(solution_dict.values())[:num_solutions]
    solution_names = list(solution_dict.keys())[:num_solutions]

    if not individuals:
        raise ValueError("No feasible individual solutions were initialised.")

    # Paths for saving
    basename = f"GA_{run_id}"
    best_output_path = os.path.join(output_dir, f"{basename}.txt")
    best_json_path = os.path.join(output_dir, f"{basename}_segments.json")
    log_path = os.path.join(output_dir, f"{basename}.log")
    best_individual_per_iteration_path = os.path.join(output_dir, f"GA_OTHER_{run_id}_best_ind_per_iter.log")

    # Clear log files at the start of the run if they exist
    for path in [best_individual_per_iteration_path]: open(path, "w").close()
    
    # Run GA across multiple individual solutions
    with open(log_path, "a") as log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        (final_solutions, best_log_lists, mutation_counters, final_solution_names, best_solution, best_logs, best_mutations, best_name) = genetic_algorithm(individuals, run_id, output_dir, solution_names=solution_names, RUs=RUs, DUs=DUs, CUs=CUs, users=users, graph=graph, max_iterations=max_iterations, time_limit_minutes=time_limit_minutes, num_neighbours=num_neighbours, num_mutations=num_mutations, crossover_interval=crossover_interval, elite_pop=elite_pop, num_clones=num_clones, log_file=log_file, best_individual_per_iteration_path=best_individual_per_iteration_path, save_interval=save_interval)

    # Save the best solution
    save_results_to_file(best_solution, best_output_path, best_json_path, algorithm=f"local_search")

    # Append logs and mutation summary
    log_final_summary_solutions(best_output_path, best_name, best_logs, best_mutations)