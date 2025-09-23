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

import json
from collections import Counter

from modules import config
from modules.data_classes.dataclass_solution import Solution

def save_results_to_file(results: Solution, file_path: str, json_path: str, algorithm: str):
    with open(file_path, 'w') as file:
        RUs = results.RUs
        DUs = results.DUs
        CUs = results.CUs
        users = results.users
        road_graph = results.graph
        
        # Directly use config values for costs
        du_cost = config.DU_COST
        du_install = config.DU_INSTALL_COST
        KW = config.KW_COST
        KX = config.KX_COST
        KL = config.KL_COST
        ru_cost = config.RU_COST
        ru_install = config.RU_INSTALL_COST
        KV = config.KV_COST
        KB = config.KB_COST
        
        UM = config.UM_VALUE
        MD = config.MD_VALUE
        MR = config.MR_VALUE
        CR = config.CR_VALUE
        FC = config.FC_VALUE
        coverage_threshold = config.COVERAGE_THRESHOLD

        RU_BASE_BANDWIDTH = config.RU_BASE_BANDWIDTH
        DU_BASE_BANDWIDTH = config.DU_BASE_BANDWIDTH
        DU_TOTAL_PORTS = config.DU_TOTAL_PORTS
        CU_BASE_BANDWIDTH = config.CU_BASE_BANDWIDTH
        CU_TOTAL_PORTS = config.CU_TOTAL_PORTS

        # Calculate fibre connections and user mapping
        fibre_connections_ru_du = sum(du.fibre_ru_du for du in DUs.values())
        fibre_connections_du_cu = sum(du.fibre_du_cu for du in DUs.values())
        
        # User coverage
        covered_users = [user.user_id for user in users.values() if user.assigned_ru]
        unassigned_users = [user.user_id for user in users.values() if not user.assigned_ru]
        total_users = len(users)

        # Count RUs and DUs based on the number of cells (handles any number)
        ru_cell_counts = Counter(ru.num_rus for ru in RUs.values() if ru.is_selected)
        du_cell_counts = Counter(du.num_dus for du in DUs.values() if du.is_selected)

        # Dynamically handle RU cell counts up to MR and DU cell counts up to MD
        ru_cell_summary = [ru_cell_counts.get(i, 0) for i in range(1, MR + 1)]
        du_cell_summary = [du_cell_counts.get(i, 0) for i in range(1, MD + 1)]

        selected_RUs = [ru for ru in results.RUs.values() if ru.is_selected]
        selected_DUs = [du for du in results.DUs.values() if du.is_selected]

        selected_RU_names = [ru.name for ru in RUs.values() if ru.is_selected]
        selected_DU_names = [du.name for du in DUs.values() if du.is_selected]
        not_selected_RU_names = [ru.name for ru in RUs.values() if not ru.is_selected]
        not_selected_DU_names = [du.name for du in DUs.values() if not du.is_selected]
        ru_to_du_connections = [(ru.name, ru.connected_du) for ru in RUs.values() if ru.is_selected and ru.connected_du]
        du_to_cu_connections = [(du.name, du.connected_cu if du.connected_cu else "None") for du in DUs.values() if du.is_selected]

        ru_headers = [f'RU_Num{i}cell' for i in range(1, MR + 1)]
        du_headers = [f'DU_Num{i}cell' for i in range(1, MD + 1)]
        header = (
            'Coverage_Percentage,Num_RUs,Num_DUs,Cost_RUs,Cost_DUs,Cost_CUs,Cost_Paths,Total_Cost,Total_Distance,Total_Users,'
            'FibreRU_DU,FibreDU_CU,' +
            ','.join(ru_headers) + ',' +
            ','.join(du_headers) + ',' +
            'UM,MR,MD,CR,FC,Type\n'
        )
        file.write(header)

        summary_line = (
            f'{results.coverage_percentage:.0f},'
            f'{len(selected_RU_names)},'
            f'{len(selected_DU_names)},'
            f'{results.cost_of_RUs},'
            f'{results.cost_of_DUs},'
            f'{results.cost_of_CUs},'
            f'{results.total_segment_cost},'
            f'{results.total_cost},'
            f'{results.total_distance_used},'
            f'{total_users},'
            f'{fibre_connections_ru_du},'
            f'{fibre_connections_du_cu},' +
            ','.join(str(x) for x in ru_cell_summary) + ',' +
            ','.join(str(x) for x in du_cell_summary) + ',' +
            f'{UM},{MR},{MD},{CR},{FC},{algorithm}\n'
        )
        file.write(summary_line)

        file.write('\n\n\n\n' + ('-' * 80) + '\n\n')
        file.write(f'Results when coverage low: {coverage_threshold}, UM: {UM}, FC: {FC}, MR: {MR}, MD: {MD}, CR: {CR}\n\n')
        file.write(f"With the costs of RU(KV): {KV}, DU(KW): {KW}, DU(KX): {KX}, Fibre RU-DU(KL): {KL}, Fibre DU-CU(KB): {KB}, Trenching: {config.TRENCHING_COST}, fibre cost = {config.FIBRE_COST}, number of fibres = {config.NUM_FIBRES}, cost per metre for new path: {config.KY_COST} and existing: {config.KM_COST} \n\n")
        file.write(f'RU Cost: {ru_cost}, RU Install: {ru_install}, DU Cost: {du_cost}, DU Install: {du_install}\n\n')
        file.write(f'RU Base Bandwidth: {RU_BASE_BANDWIDTH}, DU Base Bandwidth: {DU_BASE_BANDWIDTH}, DU Total Ports: {DU_TOTAL_PORTS}, CU Base Bandwidth: {CU_BASE_BANDWIDTH}, CU Total Ports: {CU_TOTAL_PORTS}\n\n')
        file.write('-' * 80 + '\n\n')
        
        file.write('\nMODEL RESULTS\n\n')
        file.write(f'Cost of road paths: ${results.total_segment_cost:,}\n')
        file.write(f'Cost of RUs: ${results.cost_of_RUs:,}\n')
        file.write(f'Cost of DUs: ${results.cost_of_DUs:,}\n')
        file.write(f'CU Cost: ${results.cost_of_CUs:,}\n\n')

        file.write(f'TOTAL COST: ${results.total_cost:,.0f}\n')
        file.write(f'{results.total_segment_cost} + {results.cost_of_RUs} + {results.cost_of_DUs} + {results.cost_of_CUs} = ${results.total_cost:.0f}\n\n\n')
        
        file.write(f'Number of RUs: {len(selected_RUs)}\n')
        file.write('RU Cell Distribution:\n')
        for i, count in enumerate(ru_cell_summary, start=1):
            file.write(f'Number of RUs with {i} cell{"s" if i > 1 else ""}: {count}\n')
        file.write('\n')

        file.write(f'Number of DUs: {len(selected_DUs)}\n\n')
        file.write('DU Cell Distribution:\n')
        for i, count in enumerate(du_cell_summary, start=1):
            file.write(f'Number of DUs with {i} cell{"s" if i > 1 else ""}: {count}\n')
        file.write('\n')
        
        file.write(f'Total fibre connections RU to DU: {fibre_connections_ru_du}\n')
        file.write(f'Total fibre connections DU to CU: {fibre_connections_du_cu}\n\n')

        # Calculate the percentage of covered users
        if total_users > 0:
            coverage_percentage = (len(covered_users) / total_users) * 100
        else:
            coverage_percentage = 0.0  # Handle edge case of zero users

        file.write(f'\nCOVERAGE\n')
        file.write(f'User Coverage: {coverage_percentage:.2f}%\n')
        file.write(f'Count of covered users: {len(covered_users)}\n')
        file.write(f'Count of unassigned users: {len(unassigned_users)}\n')
        file.write(f'Total Users: {total_users}\n\n')
        
        file.write('DISTANCE\n')
        file.write(f'Total distance in m: {results.total_distance_used}m\n')
        total_distance_km = results.total_distance_used / 1000
        file.write(f'Total distance in km: {total_distance_km:.2f} km\n\n\n')

        #% =======================================================================================
        #% RU-DU Connections and DU-CU Connections
        #% =======================================================================================
        file.write('-' * 80 + '\n\n\nDEVICES SELECTED AND PATHS\n\n')

        file.write(f'selected_rus = {sorted(selected_RU_names)}\n')
        file.write(f'not_selected_rus = {sorted(not_selected_RU_names)}\n')
        file.write(f'selected_dus = {sorted(selected_DU_names)}\n')
        file.write(f'not_selected_dus = {sorted(not_selected_DU_names)}\n\n')

        file.write(f'ru_to_du_connections = {sorted(ru_to_du_connections)}\n')
        file.write(f'du_to_cu_connections = {sorted(du_to_cu_connections)}\n\n\n')
        file.write('-' * 80 + '\n\n\n')

        #% =======================================================================================
        #% RU to DU Fibre Connections Overview
        #% =======================================================================================
        file.write('FIBRE CONNECTIONS OVERVIEW\n')
        file.write('-' * 80 + '\n')
        file.write(f'{"DU":<8}{"Total RUs":<11}{"Connected RUs":<61}\n')
        file.write('-' * 80 + '\n')

        max_ru_width = 61
        indent = ' ' * 19
        total_rus_count = 0

        du_to_rus = {}
        for ru_name, du_name in ru_to_du_connections:
            if du_name not in du_to_rus:
                du_to_rus[du_name] = []
            du_to_rus[du_name].append(ru_name)

        for du in selected_DU_names:
            connected_rus = du_to_rus.get(du, [])
            total_rus = len(connected_rus)
            total_rus_count += total_rus
            connected_rus_str = ', '.join(connected_rus)
            lines = []

            while len(connected_rus_str) > max_ru_width:
                cut_index = connected_rus_str.rfind(',', 0, max_ru_width)
                lines.append(connected_rus_str[:cut_index])
                connected_rus_str = connected_rus_str[cut_index + 2:]
            lines.append(connected_rus_str)

            file.write(f'{du:<8}{total_rus:<11}{lines[0]:<61}\n')
            for line in lines[1:]:
                file.write(f'{indent}{line:<61}\n')

        file.write('-' * 80 + '\n')
        file.write(f'{"TOTALS:":<8}{total_rus_count:<11}\n')
        file.write('-' * 80 + '\n\n\n')

        #% =======================================================================================
        #% DU Ports and Fibre Connections Overview
        #% =======================================================================================
        file.write('DU USAGE AND CAPACITY OVERVIEW (FIBRE CONNECTIONS AND PORTS):\n')
        file.write('-' * 80 + '\n')
        file.write(f'{"DU":<10}{"Used Ports":<18}{"Fibre RU-DU":<18}{"Fibre DU-CU":<18}{"Unused Ports":<18}\n')
        file.write('-' * 80 + '\n')

        total_du_capacity = 0
        total_du_ports = 0
        total_fibre_connections_ru_du = 0
        total_fibre_connections_du_cu = 0
        total_unused_ports = 0

        for du in sorted(selected_DUs, key=lambda du: int(''.join(filter(str.isdigit, du.name)))):
            bandwidth = du.used_bandwidth
            used_ports = du.used_ports
            fibre_ru_du = du.fibre_ru_du
            fibre_du_cu = du.fibre_du_cu
            unused_ports = max(0, du.capacity_ports - used_ports)

            total_du_ports += used_ports
            total_fibre_connections_ru_du += fibre_ru_du
            total_fibre_connections_du_cu += fibre_du_cu
            total_unused_ports += unused_ports

            file.write(f'{du.name:<10}{used_ports:<18}{fibre_ru_du:<18}{fibre_du_cu:<18}{unused_ports:<18}\n')

        file.write('-' * 80 + '\n')
        file.write(f'{"TOTALS:":<10}{total_du_ports:<18}{total_fibre_connections_ru_du:<18}{total_fibre_connections_du_cu:<18}{total_unused_ports:<18}\n')
        file.write('-' * 80 + '\n\n\n')

        #% =======================================================================================
        #% DU Bandwidth Overview
        #% =======================================================================================
        file.write('DU USAGE AND CAPACITY OVERVIEW (BANDWIDTH):\n')
        file.write('-' * 80 + '\n')
        file.write(f'{"DU":<10}{"Used BW":<18}{"Available BW":<18}{"Unused BW":<18}{"Num DUs":<18}\n')
        file.write('-' * 80 + '\n')

        total_used_bandwidth = 0
        total_available_bandwidth = 0
        total_unused_bandwidth = 0
        total_dus = 0

        for du in sorted(selected_DUs, key=lambda du: int(''.join(filter(str.isdigit, du.name)))):
            used_bandwidth = du.used_bandwidth
            available_bandwidth = du.fibre_du_cu * FC
            unused_bandwidth = max(0, available_bandwidth - used_bandwidth)
            num_dus_per_cell = du.num_dus

            total_used_bandwidth += used_bandwidth
            total_available_bandwidth += available_bandwidth
            total_unused_bandwidth += unused_bandwidth
            total_dus += num_dus_per_cell

            file.write(f'{du.name:<10}{used_bandwidth:<18}{available_bandwidth:<18}{unused_bandwidth:<18}{num_dus_per_cell:<18}\n')

        file.write('-' * 80 + '\n')
        file.write(f'{"TOTALS:":<10}{total_used_bandwidth:<18}{total_available_bandwidth:<18}{total_unused_bandwidth:<18}{total_dus:<18}\n')
        file.write('-' * 80 + '\n\n\n')

        #% =======================================================================================
        #% RU Bandwidth, number of RUs, and user overview
        #% =======================================================================================
        file.write('RU USAGE AND CAPACITY\n')
        file.write('-' * 80 + '\n')
        file.write(f'{"RU":<10}{"Num RUs":<10}{"Required":<15}{"Available":<15}{"Unused":<14}{"Num Users":<15}\n')
        file.write('-' * 80 + '\n')

        total_required_capacity = 0
        total_available_capacity = 0
        total_unused_capacity = 0
        total_num_rus = 0
        total_num_users = 0

        for ru in sorted(selected_RUs, key=lambda ru: int(''.join(filter(str.isdigit, ru.name)))):
            required = ru.used_capacity
            available = ru.capacity_bandwidth
            unused = available - required
            num_users = len(ru.connected_users)
            
            total_required_capacity += required
            total_available_capacity += available
            total_unused_capacity += unused
            total_num_rus += ru.num_rus
            total_num_users += num_users
            
            file.write(f'{ru.name:<10}{ru.num_rus:<10}{required:<15}{available:<15}{unused:<14}{num_users:<15}\n')

        file.write('-' * 80 + '\n')
        file.write(f'{len(selected_RU_names):<10}{total_num_rus:<10}{total_required_capacity:<15}{total_available_capacity:<15}{total_unused_capacity:<14}{total_num_users:<15}\n')
        file.write('-' * 80 + '\n\n')
        
        #% =======================================================================================
        #% User Assignment Details
        #% =======================================================================================
        file.write(f'\n\n\nUser counts \n' + '-' * 60 + '\n')
        file.write(f'Number of users: {len(users)}, number of assigned users: {len(covered_users)}, number of unassigned users: {len(unassigned_users)}.')
        file.write(f'\nCoverage percentage calculation is: (calculated {len(covered_users) / len(users) * 100:.2f}%) vs (actual {coverage_percentage:.2f}%)\n\n')
        file.write(f'Assigned Users ({len(covered_users)}) =>\n')
        for user_id in covered_users: file.write(f' {user_id},')
        file.write(f'\n\nUnassigned Users ({len(unassigned_users)}) =>\n')
        for user_id in unassigned_users: file.write(f' {user_id},')

        #% =======================================================================================
        #% Road Segment Usage
        #% =======================================================================================
        file.write('\n\n\n' + '-' * 80 + "\n\n\nRoad Segment Usage Counts:\n")
        seen_segments = set()
        json_segments = []
        json_all_segments = []

        for (from_node, to_node), segment in road_graph.segments.items():
            segment_key = tuple(sorted([str(from_node), str(to_node)]))
            if segment_key in seen_segments:
                continue

            seen_segments.add(segment_key)

            associated_rus = ', '.join(segment.associated_rus) if segment.associated_rus else "None"
            associated_dus = ', '.join(segment.associated_dus) if segment.associated_dus else "None"

            file.write(f"{from_node} -> {to_node}: Usage Count = {segment.usage_count}, Associated RUs: [{associated_rus}], Associated DUs: [{associated_dus}]\n")

            segment_entry = {"from": from_node, "to": to_node}
            json_all_segments.append(segment_entry)

            if segment.usage_count > 0:
                json_segments.append(segment_entry)

    # Save all the road segments used in the solution to a JSON file
    with open(json_path, 'w') as f_json:
        json.dump(json_segments, f_json, indent=4)

    #! Output the results in decision variable format for analysis
    # This section is commented out but can be enabled if needed
    '''state_output_path = file_path.replace('.txt', '_state.txt')
    with open(state_output_path, 'w') as f:
        # A: User to RU connections
        for user in users.values():
            for ru_name in RUs:
                is_connected = int(user.assigned_ru == ru_name)
                f.write(f"A,{user.user_id},{ru_name}={is_connected}\n")

        # B: RU to DU connections
        for ru in RUs.values():
            for du_name in DUs:
                is_connected = int(ru.connected_du == du_name)
                f.write(f"B,{ru.name},{du_name}={is_connected}\n")

        # C: DU to CU connections
        for du in DUs.values():
            for cu_name in CUs:
                is_connected = int(du.connected_cu == cu_name)
                f.write(f"C,{du.name},{cu_name}={is_connected}\n")

        # D: RU activation
        for ru in RUs.values():
            f.write(f"D,{ru.name}={int(ru.is_selected)}\n")

        # E: DU activation
        for du in DUs.values():
            f.write(f"E,{du.name}={int(du.is_selected)}\n")

        # J/K: Number of RUs per location
        for ru in RUs.values():
            f.write(f"J,{ru.name}={ru.num_rus}\n")
            f.write(f"K,{ru.name},{ru.connected_du}={ru.num_rus}\n")

        # L/M: Number of DUs per location
        for du in DUs.values():
            f.write(f"L,{du.name}={du.num_dus}\n")
            f.write(f"M,{du.name},{du.connected_cu}={du.num_dus}\n")

        # N: Fibre connections from DU to CU
        for du in DUs.values():
            if du.connected_cu:
                f.write(f"N,{du.name},{du.connected_cu}={du.fibre_du_cu}\n")

        # I: All segments used (bidirectional, both directions, always output)
        for (from_id, to_id), segment in road_graph.segments.items():
            is_used = int(segment.usage_count > 0)
            f.write(f"I,{from_id},{to_id}={is_used}\n")
            f.write(f"I,{to_id},{from_id}={is_used}\n")

        # F: Segment used by RU → DU path (bidirectional, always output)
        for (from_id, to_id), segment in road_graph.segments.items():
            associated_rus = segment.associated_rus or []
            for ru_name in RUs:
                ru = RUs[ru_name]
                du_name = ru.connected_du if ru else None
                if not du_name: continue
                is_used = int(ru_name in associated_rus and segment.usage_count > 0 and du_name in DUs)
                f.write(f"F,{ru_name},{du_name},{from_id},{to_id}={is_used}\n")
                f.write(f"F,{ru_name},{du_name},{to_id},{from_id}={is_used}\n")

        # G: Segment used by DU → CU path (bidirectional, always output)
        for (from_id, to_id), segment in road_graph.segments.items():
            associated_dus = segment.associated_dus or []
            for du_name in DUs:
                du = DUs[du_name]
                cu_name = du.connected_cu if du else None
                if not cu_name: continue
                is_used = int(du_name in associated_dus and segment.usage_count > 0 and cu_name in CUs)
                f.write(f"G,{du_name},{cu_name},{from_id},{to_id}={is_used}\n")
                f.write(f"G,{du_name},{cu_name},{to_id},{from_id}={is_used}\n")'''
