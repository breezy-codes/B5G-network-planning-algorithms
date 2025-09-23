"""
%┏┓━┏┓┏━━━┓┏┓━━━┏━━━┓┏━━━┓┏━━━┓━━━━━┏━━━┓┏┓━┏┓┏━┓━┏┓┏━━━┓
?┃┃━┃┃┃┏━━┛┃┃━━━┃┏━┓┃┃┏━━┛┃┏━┓┃━━━━━┃┏━━┛┃┃━┃┃┃┃┗┓┃┃┃┏━┓┃
%┃┗━┛┃┃┗━━┓┃┃━━━┃┗━┛┃┃┗━━┓┃┗━┛┃━━━━━┃┗━━┓┃┃━┃┃┃┏┓┗┛┃┃┃━┗┛
?┃┏━┓┃┃┏━━┛┃┃━┏┓┃┏━━┛┃┏━━┛┃┏┓┏┛━━━━━┃┏━━┛┃┃━┃┃┃┃┗┓┃┃┃┃━┏┓
%┃┃━┃┃┃┗━━┓┃┗━┛┃┃┃━━━┃┗━━┓┃┃┃┗┓━━━━┏┛┗┓━━┃┗━┛┃┃┃━┃┃┃┃┗━┛┃
?┗┛━┗┛┗━━━┛┗━━━┛┗┛━━━┗━━━┛┗┛┗━┛━━━━┗━━┛━━┗━━━┛┗┛━┗━┛┗━━━┛
%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
?━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


This module contains helper functions for logging iteration details, best individuals, and final summaries in genetic algorithms and local search algorithms.

"""

from modules.logger import loggers
import json
from datetime import datetime

def log_final_summary_solutions(best_output_path, best_name, best_logs, best_mutations):
    """Append the final summary of the best solution and mutation frequency to the output file."""
    with open(best_output_path, "a") as f:
        f.write(f"\n\n===== Best Solution: {best_name} =====\n")
        f.write("\n===== New Best Solutions Log =====\n")
        for line in best_logs:
            f.write(line + "\n")
        if best_mutations:
            f.write("\n===== Mutation Frequency Summary =====\n")
            for mutation, count in best_mutations.most_common():
                f.write(f"{mutation}: {count}\n")
        else:
            f.write("\n(No mutations applied to this individual that led to improvements.)\n")

def log_iteration(iteration, current_fitness, best_fitness, solution, log_file=None, algo=None, start_time=None):
    """Log the details of the current iteration, including runtime if start_time is provided."""
    num_selected_rus = len(solution.get_selected_RUs())
    num_selected_dus = sum(du.is_selected for du in solution.DUs.values())

    # Log the current iteration details to the console
    if algo is None:
        algo = "NA"
    
    # Calculate elapsed time if start_time is provided
    elapsed_sec = None
    if start_time:
        elapsed_sec = (datetime.now() - start_time).total_seconds()

    # Log the iteration details to the console
    print(f"[{algo}] Iteration {iteration + 1}: Fitness={current_fitness:.2f}, "
          f"Coverage={solution.coverage_percentage:.2f}, Total Cost={round(solution.total_cost)}, "
          f"Distance Used={round(solution.total_distance_used)}, Num RUs={num_selected_rus}, "
          f"Num DUs={num_selected_dus}, Num Users={len(solution.assigned_users)}" +
          (f", Time={elapsed_sec:.2f}s" if elapsed_sec is not None else ""))

    # Prepare the log data
    log_data = {
        "iteration": iteration + 1,
        "current_fitness": current_fitness,
        "best_fitness": best_fitness,
        "coverage_percentage": round(solution.coverage_percentage, 2),
        "total_cost": round(solution.total_cost),
        "total_distance_used": round(solution.total_distance_used),
        "num_RUs": num_selected_rus,
        "num_DUs": num_selected_dus,
        "num_users": len(solution.assigned_users),
        "total_CU_cost": solution.cost_of_CUs,
        "total_DU_cost": solution.cost_of_DUs,
        "total_RU_cost": solution.cost_of_RUs
    }

    if elapsed_sec is not None:
        log_data["elapsed_time_sec"] = round(elapsed_sec, 2)

    if log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        log_file.write(f"{timestamp} - {json.dumps(log_data)}\n")
        log_file.flush()

def log_best_individuals_json(iteration, best_individuals, best_fitness_scores, best_global_idx, best_global_fitness, best_global_individual, output_path, solution_names=None, penalties=None):
    """Log the best individuals and their fitness scores to a JSON file."""
    # Note: the cover_penaly_base here only applies to the genetic algorithm, so it is not used in the local search.
    
    if solution_names is None:
        # If no solution names provided, generate default names
        solution_names = [f"sol_{i}" for i in range(len(best_individuals))]

    # Uncomment the following lines if you want to log the best individuals in a debug log
    #loggers['feasible_logger'].debug(f"[Iter {iteration + 1}] Current best for each individual:")
    #for idx, (sol, fit) in enumerate(zip(best_individuals, best_fitness_scores)):
    #    penalty = penalties[idx] if penalties else None
    #    loggers['feasible_logger'].debug(
    #        f"  → {solution_names[idx]}: fitness={fit:.2f}, cost={sol.total_cost:.2f}, penalty={penalty:.2f}" if penalty is not None else
    #        f"  → {solution_names[idx]}: fitness={fit:.2f}, cost={sol.total_cost:.2f}"
    #    )

    #loggers['feasible_logger'].debug(f"[Iter {iteration + 1}] Global best solution: {solution_names[best_global_idx]} with fitness={best_global_fitness:.2f}, cost={best_global_individual.total_cost:.2f}")

    data = {
        "iteration": iteration + 1,
        "global_best_index": best_global_idx,
        "global_best_name": solution_names[best_global_idx],
        "individuals": {
            solution_names[idx]: {
                "fitness": round(fit, 2),
                "cost": round(ind.total_cost, 2),
                "coverage_percentage": round(ind.coverage_percentage, 2),
                "cover_penalty_base": round(penalties[idx], 2) if penalties else None
            } for idx, (ind, fit) in enumerate(zip(best_individuals, best_fitness_scores))
        }
    }

    # Write the data to the output file
    with open(output_path, "a") as f:
        f.write(json.dumps(data) + "\n\n")