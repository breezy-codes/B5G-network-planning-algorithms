"""
%┏━━━┓┏━━━┓┏━━━┓┏┓━┏┓┏┓━━━┏━━━━┓┏━━━┓━━━━┏━┓┏━┓┏━━━┓┏━━━┓┏┓━┏┓┏┓━━━┏━━━┓
%┃┏━┓┃┃┏━━┛┃┏━┓┃┃┃━┃┃┃┃━━━┃┏┓┏┓┃┃┏━┓┃━━━━┃┃┗┛┃┃┃┏━┓┃┗┓┏┓┃┃┃━┃┃┃┃━━━┃┏━━┛
%┃┗━┛┃┃┗━━┓┃┗━━┓┃┃━┃┃┃┃━━━┗┛┃┃┗┛┃┗━━┓━━━━┃┏┓┏┓┃┃┃━┃┃━┃┃┃┃┃┃━┃┃┃┃━━━┃┗━━┓
%┃┏┓┏┛┃┏━━┛┗━━┓┃┃┃━┃┃┃┃━┏┓━━┃┃━━┗━━┓┃━━━━┃┃┃┃┃┃┃┃━┃┃━┃┃┃┃┃┃━┃┃┃┃━┏┓┃┏━━┛
%┃┃┃┗┓┃┗━━┓┃┗━┛┃┃┗━┛┃┃┗━┛┃━┏┛┗┓━┃┗━┛┃━━━━┃┃┃┃┃┃┃┗━┛┃┏┛┗┛┃┃┗━┛┃┃┗━┛┃┃┗━━┓
%┗┛┗━┛┗━━━┛┗━━━┛┗━━━┛┗━━━┛━┗━━┛━┗━━━┛━━━━┗┛┗┛┗┛┗━━━┛┗━━━┛┗━━━┛┗━━━┛┗━━━┛
%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os, glob, shutil, json
from modules import config
from pulp import value

def save_ilp_results_to_file(results, file_path, log_path=None, num_rus_with_n_cells=None, num_dus_with_n_cells=None, total_users=None, RU_names=None, CU_names=None, K=None, N=None, value=None, L=None, DUs_new=None, J=None, covered_users=None, UC_low=None, total_fibres_per_cu=None):
    with open(file_path, 'w') as file:
        total_cost = (
            results["RU_installation_cost_value"] +
            results["DU_cost_value"] +
            results["segment_cost_value"] +
            results["CU_cost_value"]
        )
        
        ru_cell_counts = ",".join(str(num_rus_with_n_cells.get(n, 0)) for n in range(1, config.MR_VALUE + 1))
        du_cell_counts = ",".join(str(num_dus_with_n_cells.get(n, 0)) for n in range(1, config.MD_VALUE + 1))

        # Construct the summary line
        summary_line = (
            f'{results["user_coverage_percent"]:.0f},'
            f'{len(results["selected_rus"])},'
            f'{len(results["selected_dus"])},'
            f'{results["RU_installation_cost_value"]},'
            f'{results["DU_cost_value"]},'
            f'{results["CU_cost_value"]},'
            f'{results["segment_cost_value"]},'
            f'{total_cost},'
            f'{results["total_segment_distance"]},'
            f'{results["total_users"]},'
            f'{sum(results["fibre_connections_ru_du"].values())},'
            f'{sum(results["fibre_connections_du_cu"].values())},'
            f'{ru_cell_counts},'
            f'{du_cell_counts},'
            f'{config.UM_VALUE},'
            f'{config.MR_VALUE},'
            f'{config.CR_VALUE},'
            f'{config.FC_VALUE},'
            f'short'
        )

        # Write the header
        file.write(
            'Coverage_Percentage,Num_RUs,Num_DUs,Cost_RUs,Cost_DUs,Cost_CUs,Cost_Paths,Total_Cost,Total_Distance,Total_Users,'
            'FibreRU_DU,FibreDU_CU,' +
            ','.join(f'RU_Num{n}cell' for n in range(1, config.MR_VALUE + 1)) + ',' +
            ','.join(f'DU_Num{n}cell' for n in range(1, config.MD_VALUE + 1)) + ',' +
            'UM,MR,CR,FC,Type\n'
        )
        file.write(summary_line + '\n')

        file.write(f'\n\n\n')
        file.write('-' * 80 + '\n\n')
        file.write(f'Results when coverage low: {config.COVERAGE_THRESHOLD}, UM: {config.UM_VALUE}, FC: {config.FC_VALUE}, MR: {config.MR_VALUE}, CR: {config.CR_VALUE}\n\n')
        file.write(f"With the costs of RU(KV): {config.KV_COST}, DU(KW): {config.KW_COST}, Fibre RU-DU(KL): {config.KL_COST}, Fibre DU-CU(KB): {config.KB_COST}\n\n")
        file.write(f'RU Cost: {config.RU_COST}, RU Install: {config.RU_INSTALL_COST}, DU Cost: {config.DU_COST}, DU Install: {config.DU_INSTALL_COST}\n\n')
        file.write('-' * 80 + '\n\n')
        file.write('\nMODEL RESULTS\n\n')
        file.write(f'Cost of road paths: ${results["segment_cost_value"]:,}\n')
        file.write(f'Cost of RUs: ${results["RU_installation_cost_value"]:,}\n')
        file.write(f'Cost of DUs: ${results["DU_cost_value"]:,}\n')
        file.write(f'CU Cost: ${results["CU_cost_value"]:,}\n\n')
        
        file.write(f'TOTAL COST: ${total_cost:,.0f}\n')
        file.write(f'{results["segment_cost_value"]} + {results["RU_installation_cost_value"]} + {results["DU_cost_value"]} + {results["CU_cost_value"]}  = ${total_cost:.0f}\n\n\n')
        
        file.write(f'Number of RUs: {results["num_selected_RUs"]:,}\n')
        file.write("RU Cell Distribution:\n")
        for n, count in num_rus_with_n_cells.items():
            file.write(f'Number of RUs with {n} cell{"s" if n > 1 else ""}: {count:,}\n')
        file.write("\n\n")

        file.write(f'Number of DUs: {results["num_selected_DUs"]:,}\n')
        file.write("DU Cell Distribution:\n")
        for n, count in num_dus_with_n_cells.items():
            file.write(f'Number of DUs with {n} cell{"s" if n > 1 else ""}: {count:,}\n')
        file.write("\n\n")

        file.write(f'Number of fibre connections from RU to DU: {sum(results["fibre_connections_ru_du"].values())}\n')
        file.write(f'Number of fibre connections from DU to CU: {sum(results["fibre_connections_du_cu"].values())}\n\n\n')

        file.write('COVERAGE\n')
        file.write(f'User Coverage: {results["user_coverage_percent"]:.2f}%\n')
        file.write(f'Total Users: {total_users}\n\n\n')
        
        file.write('DISTANCE\n')
        file.write(f'Total distance in m: {results["total_segment_distance"]}m\n')
        total_distance_km = results["total_segment_distance"] / 1000
        file.write(f'Total distance in km: {total_distance_km:.2f} km\n\n\n')
        file.write('-' * 80 + '\n\n')
        
        file.write('\nDEVICES SELECTED AND PATHS\n\n')
        file.write(f'selected_rus = {results["selected_rus"]}\n')
        file.write(f'not_selected_rus = {results["not_selected_rus"]}\n')
        file.write(f'selected_dus = {results["selected_dus"]}\n')
        file.write(f'not_selected_dus = {results["not_selected_dus"]}\n\n')
        file.write(f'ru_to_du_connections = {results["ru_to_du_connections"]}\n')
        file.write(f'du_to_cu_connections = {results["du_to_cu_connections"]}\n\n\n')
        
        file.write('-' * 80 + '\n\n\n')
        
        # Fibre Connections Overview
        file.write('FIBRE CONNECTIONS OVERVIEW\n')
        file.write('-' * 80 + '\n')
        file.write(f'{"DU":<8}{"Total RUs":<11}{"Connected RUs":<61}\n')
        file.write('-' * 80 + '\n')
        max_ru_width = 61
        indent = ' ' * 19
        total_rus_count = 0
        for d in results["selected_dus"]:
            connected_rus = [r for r, du in results["ru_to_du_connections"] if du == d]
            total_rus = len(connected_rus)
            total_rus_count += total_rus
            connected_rus_str = ', '.join(connected_rus)
            lines = []
            while len(connected_rus_str) > max_ru_width:
                cut_index = connected_rus_str.rfind(',', 0, max_ru_width)
                lines.append(connected_rus_str[:cut_index])
                connected_rus_str = connected_rus_str[cut_index + 2:]
            lines.append(connected_rus_str)
            file.write(f'{d:<8}{total_rus:<11}{lines[0]:<61}\n')
            for line in lines[1:]:
                file.write(f'{indent}{line:<61}\n')
        file.write('-' * 80 + '\n')
        file.write(f'{"TOTALS:":<8}{total_rus_count:<11}\n')
        file.write('-' * 80 + '\n\n\n')

        # DU Usage and Capacity Overview - Fibre Connections and Ports
        file.write('DU USAGE AND CAPACITY OVERVIEW (FIBRE CONNECTIONS AND PORTS):\n')
        file.write('-' * 80 + '\n')
        file.write(f'{"DU":<10}{"Used Ports":<18}{"Fibre RU-DU":<18}{"Fibre DU-CU":<18}{"Unused Ports":<18}\n')
        file.write('-' * 80 + '\n')
        total_du_ports = 0
        total_fibre_to_rus = 0
        total_fibre_to_cus = 0
        total_unused_ports = 0
        total_total_ports = 0
        for d in results["selected_dus"]:
            capacity = results["used_du_capacity"][d]
            ports = capacity["ports"]
            fibre_to_rus = sum(round(value(K[(r, d)])) for r in RU_names)
            fibre_to_cus_du = sum(round(value(N[(d, c)])) for c in CU_names)
            total_fibre_to_cus_du = sum(round(value(N[(d, cu)])) for cu in set(cu for (du, cu) in results["du_to_cu_connections"] if du == d))
            total_ports = round(results["total_du_capacity"][d]["ports"] * value(L[d]))
            unused_ports = total_ports - ports
            total_du_ports += ports
            total_fibre_to_rus += fibre_to_rus
            total_fibre_to_cus += total_fibre_to_cus_du
            total_unused_ports += unused_ports
            total_total_ports += total_ports
            file.write(f'{d:<10}{fibre_to_rus:<18}{fibre_to_cus_du:<18}{total_ports:<18}{unused_ports:<18}\n')
        file.write('-' * 80 + '\n')
        file.write(f'{"TOTALS:":<10}{total_fibre_to_rus:<18}{total_fibre_to_cus:<18}{total_total_ports:<18}{total_unused_ports:<18}\n')
        file.write('-' * 80 + '\n\n\n')

        # DU Usage and Capacity Overview - Bandwidth
        file.write('DU USAGE AND CAPACITY OVERVIEW (BANDWIDTH):\n')
        file.write('-' * 80 + '\n')
        file.write(f'{"DU":<10}{"Used BW":<18}{"Available BW":<18}{"Unused BW":<18}{"Num DUs":<18}\n')
        file.write('-' * 80 + '\n')
        total_du_capacity = 0
        total_unused_bandwidth = 0
        total_bandwidth = 0
        total_dus = 0
        for d in results["selected_dus"]:
            capacity = results["used_du_capacity"][d]
            bandwidth = capacity["bandwidth"]
            total_bandwidth_for_du = round(sum(value(N[(d, c)]) * config.FC_VALUE for c in CU_names))
            unused_bandwidth = total_bandwidth_for_du - bandwidth
            num_dus_per_cell = round(value(L[d]))
            total_dus += num_dus_per_cell
            total_du_capacity += bandwidth
            total_bandwidth += total_bandwidth_for_du
            total_unused_bandwidth += unused_bandwidth
            file.write(f'{d:<10}{bandwidth:<18}{total_bandwidth_for_du:<18}{unused_bandwidth:<18}{num_dus_per_cell:<18}\n')
        file.write('-' * 80 + '\n')
        file.write(f'{"TOTALS:":<10}{total_du_capacity:<18}{total_bandwidth:<18}{total_unused_bandwidth:<18}{total_dus:<18}\n')
        file.write('-' * 80 + '\n\n\n')

        # Overview of RU Usage and Capacity
        file.write('RU USAGE AND CAPACITY\n')
        file.write('-' * 80 + '\n')
        file.write(f'{"RU":<10}{"Num RUs":<10}{"Required":<15}{"Available":<15}{"Unused":<14}{"Num Users":<15}\n')
        file.write('-' * 80 + '\n')
        total_ru_capacity = 0
        total_ru_unused_capacity = 0
        total_num_users = 0
        total_num_rus = 0
        total_all_capacity = 0
        for r, capacity in results["used_ru_capacity"].items():
            num_rus = round(value(J[r]))
            total_capacity = round(results["total_ru_capacity"][r] * num_rus)
            unused_capacity = total_capacity - capacity
            num_users = capacity // config.UM_VALUE
            total_ru_capacity += capacity
            total_all_capacity += total_capacity
            total_ru_unused_capacity += unused_capacity
            total_num_users += num_users
            total_num_rus += num_rus
            file.write(f'{r:<10}{num_rus:<10}{capacity:<15}{total_capacity:<15}{unused_capacity:<14}{num_users:<15}\n')
        file.write('-' * 80 + '\n')
        file.write(f'{"TOTALS:":<10}{total_num_rus:<10}{total_ru_capacity:<15}{total_all_capacity:<15}{total_ru_unused_capacity:<14}{total_num_users:<15}\n')
        file.write('-' * 80 + '\n\n')
        
        total_dus_valid = sum(round(value(L[d])) for d in DUs_new)
        file.write('\nValidation Checks\n')
        file.write('-' * 80 + '\n')
        file.write(f'Total number of RUs == Used ports: {total_num_rus == total_du_ports}\n')
        file.write(f'Fibre connections DU to CU * FC == Total DU capacity: {total_fibre_to_cus * config.FC_VALUE >= total_du_capacity}\n')
        file.write(f'Number of users covered equals coverage percentage: {covered_users == UC_low}\n')
        file.write(f'Total RU capacity is less than or equal to total DU capacity: {total_ru_capacity <= total_du_capacity}\n')
        file.write(f'Total used ports + total unused ports == total ports: {total_du_ports + total_unused_ports == total_total_ports}\n')
        file.write(f'Total used bandwidth + total unused bandwidth == total bandwidth: {total_du_capacity + total_unused_bandwidth == total_bandwidth}\n')
        total_fibres = sum(total_fibres_per_cu.values())
        file.write(f'Total cost of CU is equal to num_fibre_du_cu * KB: {results["fibre_cost_du_cu"] == round(total_fibres) * config.KB_COST}\n')
        file.write(f'Total cost of DU is equal to num_dus * KV + num_fibre_ru_du * KL: {results["DU_cost_value"] == total_dus_valid * config.KW_COST + total_fibre_to_rus * config.KL_COST}\n')
        file.write(f'Total cost of RU is equal to num_rus * KV: {results["RU_installation_cost_value"] == total_num_rus * config.KV_COST}\n')
        file.write('-' * 80 + '\n\n\n')

        # Write the CPLEX solver log into the end of the file
        file.write('\n\n\n\n\n\n\n\n\n\nCPLEX Solver Log:\n')
        file.write('-' * 80 + '\n')
        with open(log_path, 'r') as log_file:
            log_content = log_file.read()
            file.write(log_content)

    print(f"Results have been saved to {file_path}")

def save_used_segments(KS, I, results_dir, run_id):
    """Extract used segments from KS/I and save them to results_dir/run_id_segments.json."""
    used_segments, seen_segments = [], set()

    for s in KS.keys():
        if value(I[s]) == 1:
            from_id, to_id = sorted((str(s[0]), str(s[1])))
            segment_tuple = (from_id, to_id)

            if segment_tuple not in seen_segments:
                used_segments.append({"from": from_id, "to": to_id})
                seen_segments.add(segment_tuple)

    json_path = os.path.join(results_dir, f"{run_id}_segments.json")
    with open(json_path, "w") as file:
        json.dump(used_segments, file, indent=4)

    return json_path

def tidy_working_directory(log_path, current_dir, run_id, cplex_files_dir):
    """Clean up logs and move solver-generated files into the correct directory."""
    
    # remove main log
    if os.path.exists(log_path):
        os.remove(log_path)

    # remove scratch logs
    for p in (
        *glob.glob(os.path.join(current_dir, "..", "clone*.log")),
        os.path.join(current_dir, "..", "cplex.log"),
        os.path.join(current_dir, "..", "mipopt"),
    ):
        if os.path.exists(p):
            os.remove(p)

    # move solver-generated files
    for name in (f"{run_id}-pulp.lp", f"{run_id}-pulp.sol"):
        src = os.path.join(current_dir, "..", name)
        if os.path.exists(src):
            shutil.move(src, os.path.join(cplex_files_dir, os.path.basename(src)))