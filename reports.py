"""Reports module (auto-extracted)."""
# Auto-generated modular refactor from phase_2_v2.py
# NOTE: This is an initial split. Cross-module imports assume:
#   from config import *
#   from data_structures import *
# Review and adjust as needed for your project.

import csv
from datetime import datetime

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    # Gurobi might not be available in all environments while reading files
    gp = None
    class GRB:
        pass

from config import *
from data_structures import *
import vrp_utils.logging as logging
logger = logging.getLogger(__name__)

def gather_shipment_rows(scenario_id, wave_id, cluster_id, source_loc_id, phase_label, scenario_descriptions):
    rows = []

    vehicles = {
        key[3]
        for key in Y_Param
        if len(key) == 4 and key[0] == wave_id and key[1] == cluster_id and key[2] == phase_label and Y_Param[key] > 0.5
    }

    for k in vehicles:
        rate = next(r for r in FC if VC.get((k, r), 0) == 1)

        dist = sum(
            Dist.get((i, j), 0) * X_Param.get((wave_id, cluster_id, phase_label, i, j, k), 0)
            for i in LOCS for j in LOCS if i != j
        )

        stops = sum(
            X_Param.get((wave_id, cluster_id, phase_label, i, j, k), 0)
            for i in LOCS for j in LOCS if i != j
        )

        orders = sum(
            Z_Param.get((wave_id, cluster_id, phase_label, m, k), 0)
            for m in ORDERS
        )

        fixed = FC[rate]
        mileage = MC[rate] * dist
        stop_cost = SC[rate] * max(0, stops - Stops_Included_In_Rate[rate])

        shipped_pallets = 0
        shipped_weight = 0
        shipped_cases = 0
        original_pallets = 0
        original_cases = 0

        # --- PATCH BLOCK: phase-agnostic Q_Param reader ---
        for key, q_val in Q_Param.items():
            if len(key) == 6:
                # Phase 0/1 format
                if key[:3] == (wave_id, cluster_id, phase_label):
                    _, _, _, m, p, veh = key
                # Phase 2 format
                elif key[0] == phase_label and key[1:3] == (wave_id, cluster_id):
                    _, _, _, m, p, veh = key
                else:
                    continue

                if veh != k:
                    continue

                shipped_pallets += q_val
                shipped_weight += q_val * WeightPerPallet.get(p, 0.0)
                shipped_cases += q_val * CasesPerPallet.get(p, 0)

        # --- PATCH BLOCK: original denominator using Z_Param only ---
        for (m, o, d, p), qty in InputOriginalD.items():
            if Z_Param.get((wave_id, cluster_id, phase_label, m, k), 0) > 0.5:
                original_pallets += qty
                original_cases += qty * CasesPerPallet.get(p, 0)

        var_pallets = shipped_pallets - original_pallets
        var_cases = shipped_cases - original_cases
        vehicle_cap = WCap.get(rate, 0.0)

        # --- PATCH BLOCK: keep as-is ---
        percent_util = shipped_weight / vehicle_cap if vehicle_cap else 0
        cost_per_case = (fixed + mileage + stop_cost) / original_cases if original_cases else 0

        # Rounding to match Access exports
        timestamp = datetime.now().strftime('%a %b %d %H:%M:%S %Y')
        scenario_description = scenario_descriptions.get(scenario_id, "TEMP SCEN DESC")

        rows.append([
            scenario_id,
            timestamp,
            scenario_id,
            scenario_description,
            source_loc_id,
            wave_id,
            cluster_id,
            k,
            rate,
            round(cost_per_case, 4),
            round(fixed + mileage + stop_cost, 2),
            round(fixed, 2),
            round(mileage, 2),
            round(stop_cost, 2),
            round(dist, 2),
            int(stops),
            int(orders),
            round(shipped_weight, 2),
            round(percent_util, 4),  # as fraction
            int(shipped_pallets),
            int(original_pallets),
            int(var_pallets),
            int(shipped_cases),
            int(original_cases),
            int(var_cases),
            "XXNBL_DTS",
            timestamp,
            "XXNBL_DTS",
            timestamp
        ])

    return rows
def gather_scenario_stats_rows(scenario_id, phase_label, scenario_descriptions):
    """
    Generates a single scenario statistics row for the given scenario_id.
    Aggregates across all clusters, waves, and vehicles.
    """
    rows = []
    timestamp = datetime.now().strftime("%a %b %d %H:%M:%S.%f %Y")
    scenario_description = scenario_descriptions.get(scenario_id, "TEMP SCEN DESC")

    # Identify all relevant wave/cluster pairs for this scenario
    wave_clusters = set((w, c) for (_, _, w, c, _, _, _) in SLOC_RATE_WAVE_CLUST_SUBCLUST_DLOC_ORD)

    total_cost = 0
    total_distance = 0
    total_shipped_weight = 0
    total_original_weight = 0
    total_shipped_cases = 0
    total_original_cases = 0
    total_shipped_pallets = 0
    total_original_pallets = 0
    total_orders = 0
    total_vehicles = set()
    total_capacity = 0

    for wave_id, cluster_id in wave_clusters:
        # Get all vehicles used in this cluster
        vehicles = {
            key[3] for key in Y_Param
            if len(key) == 4 and key[0] == wave_id and key[1] == cluster_id and key[2] == phase_label and Y_Param[key] > 0.5
        }

        total_vehicles.update(vehicles)

        for k in vehicles:
            rate = next((r for r in FC if VC.get((k, r), 0) == 1), None)
            if not rate:
                continue

            dist = sum(
                Dist.get((i, j), 0) * X_Param.get((wave_id, cluster_id, phase_label, i, j, k), 0)
                for i in LOCS for j in LOCS if i != j
            )
            stops = sum(
                X_Param.get((wave_id, cluster_id, phase_label, i, j, k), 0)
                for i in LOCS for j in LOCS if i != j
            )
            orders = sum(
                Z_Param.get((wave_id, cluster_id, phase_label, m, k), 0)
                for m in ORDERS
            )
            fixed = FC.get(rate, 0.0)
            mileage = MC.get(rate, 0.0) * dist
            stop_cost = SC.get(rate, 0.0) * max(0, stops - Stops_Included_In_Rate.get(rate, 0))
            cost = fixed + mileage + stop_cost

            total_cost += cost
            total_distance += dist
            total_orders += orders
            total_capacity += WCap.get(rate, 0.0)

            for key, q_val in Q_Param.items():
                if len(key) == 6:
                    if key[:3] == (wave_id, cluster_id, phase_label):
                        _, _, _, m, p, veh = key
                    elif key[0] == phase_label and key[1:3] == (wave_id, cluster_id):
                        _, _, _, m, p, veh = key
                    else:
                        continue
                    if veh != k:
                        continue
                    total_shipped_pallets += q_val
                    total_shipped_weight += q_val * WeightPerPallet.get(p, 0.0)
                    total_shipped_cases += q_val * CasesPerPallet.get(p, 0)

            for (m, o, d, p), qty in InputOriginalD.items():
                if Z_Param.get((wave_id, cluster_id, phase_label, m, k), 0) > 0.5:
                    total_original_pallets += qty
                    total_original_cases += qty * CasesPerPallet.get(p, 0)

    cost_per_case = total_cost / total_shipped_cases if total_shipped_cases > 0 else 0
    percent_util = total_shipped_weight / total_capacity if total_capacity else 0
    avg_route_distance = total_distance / len(total_vehicles) if total_vehicles else 0
    var_pallets = total_shipped_pallets - total_original_pallets
    var_cases = total_shipped_cases - total_original_cases
    percent_var_pallets = var_pallets / total_original_pallets if total_original_pallets else 0

    rows.append([
        scenario_id,
        timestamp,
        scenario_description,
        None,              # START_TIME
        None,              # END_TIME
        "0H:0M",           # RUN_TIME (placeholder)
        round(cost_per_case, 5),
        round(total_cost, 2),
        int(total_orders),
        len(total_vehicles),
        round(avg_route_distance, 2),
        round(total_distance, 2),
        round(total_shipped_weight, 2),
        round(percent_util, 4),
        int(total_shipped_pallets),
        int(total_original_pallets),
        int(var_pallets),
        round(percent_var_pallets, 4),
        int(total_shipped_cases),
        int(total_original_cases),
        int(var_cases)
    ])

    return rows
def gather_order_rows(scenario_id, wave_id, cluster_id, source_loc_id, phase_label, scenario_descriptions):
    """
    Generates order report rows aligned with the defined ORDER_REPORT structure.
    """
    rows = []
    timestamp = datetime.now().strftime("%a %b %d %H:%M:%S %Y")
    scenario_description = scenario_descriptions.get(scenario_id, "TEMP SCENARIO DESCRIPTION")

    vehicles = {
        key[3] for key in Y_Param
        if len(key) == 4
        and key[0] == wave_id
        and key[1] == cluster_id
        and key[2] == phase_label
        and Y_Param[key] > 0.5
    }

    for k in vehicles:
        for z_key, z_val in Z_Param.items():
            if (
                len(z_key) == 5 and
                z_key[0] == wave_id and
                z_key[1] == cluster_id and
                z_key[2] == phase_label and
                z_key[4] == k and
                z_val > 0.5
            ):
                m = z_key[3]  # order_id

                cust_po = ""
                for (mm, cpo, pp) in ORD_CUSTPO_PRD:
                    if mm == m:
                        cust_po = cpo
                        break

                dest_loc = ""
                dest_name = ""
                for (mm, o, d, p) in ORD_SLOC_DLOC_PROD:
                    if mm == m:
                        dest_loc = d
                        dest_name = LocName.get(d, "")
                        break

                for p in PRD:
                    q_key = (wave_id, cluster_id, phase_label, m, p, k)
                    q_ship = Q_Param.get(q_key, 0.0)

                    if q_ship > 0:
                        original_sum = sum(
                            InputOriginalD.get((m, o, d, p), 0.0)
                            for (mm, o, d, pp) in ORD_SLOC_DLOC_PROD
                            if mm == m and pp == p
                        )

                        var_pallets = q_ship - original_sum
                        shipped_weight = q_ship * WeightPerPallet.get(p, 0)
                        shipped_cases = q_ship * CasesPerPallet.get(p, 0)
                        original_cases = original_sum * CasesPerPallet.get(p, 0)
                        var_cases = shipped_cases - original_cases

                        rows.append([
                            None,  # ORDER_REPORT_ID
                            timestamp,
                            scenario_id,
                            scenario_description,
                            source_loc_id,
                            LocName.get(source_loc_id, ""),
                            wave_id,
                            cluster_id,
                            m,
                            cust_po,
                            p,
                            ProductDescription.get(p, ""),
                            round(shipped_weight, 2),
                            round(q_ship, 2),
                            round(original_sum, 2),
                            round(var_pallets, 2),
                            round(shipped_cases, 2),
                            round(original_cases, 2),
                            round(var_cases, 2),
                            k,
                            dest_loc,
                            dest_name,
                            LocCity.get(source_loc_id, ""),
                            LocState.get(source_loc_id, ""),
                            LocZipCode.get(source_loc_id, ""),
                            LocCity.get(dest_loc, ""),
                            LocState.get(dest_loc, ""),
                            LocZipCode.get(dest_loc, ""),
                            "XXNBL_DTS",  # INSERT_USER
                            datetime.now().strftime("%d-%b-%y"),  # INSERT_DATE
                            "XXNBL_DTS",  # UPDATE_USER
                            datetime.now().strftime("%d-%b-%y")  # UPDATE_DATE
                        ])

    return rows
def gather_route_rows(scenario_id, wave_id, cluster_id, source_loc_id, phase_label, scenario_descriptions):
    """
    Generates route report rows aligned with the defined ROUTE_REPORT structure,
    creating one row per (vehicle, stop, order), as in the original AMPL report.
    """
    from datetime import datetime
    rows = []
    global StartDepotByVeh
    timestamp = datetime.now().strftime("%a %b %d %H:%M:%S %Y")
    scenario_description = scenario_descriptions.get(scenario_id, "TEMP SCENARIO DESCRIPTION")

    # vehicles = {
    #     key[3] for key in Y_Param
    #     if len(key) == 4
    #     and key[0] == wave_id
    #     and key[1] == cluster_id
    #     and key[2] == phase_label
    #     and Y_Param[key] > 0.5
    #     and StartDepotByVeh.get(key[3]) == source_loc_id
    # }
    # --- build raw vehicle set from Y_Param ---
    # --- build vehicle set for this source, wave, cluster, phase ---
    vehicles = {
        InputVehicles[(o, r, w, c, sc, i, m)]
        for (o, r, w, c, sc, i, m) in SLOC_RATE_WAVE_CLUST_SUBCLUST_DLOC_ORD
        if o == source_loc_id and w == wave_id and c == cluster_id
    }
    # Keep only those actually active in this phase
    vehicles = {
        k for k in vehicles
        if Y_Param.get((wave_id, cluster_id, phase_label, k), 0) > 0.5
    }



    for k in vehicles:
        # Get route segments for this vehicle in this wave/cluster/phase
        route_segments = []
        for key in X_Param:
            if (
                len(key) == 6 and
                key[0] == wave_id and
                key[1] == cluster_id and
                key[2] == phase_label and
                key[5] == k and
                X_Param[key] > 0.5
            ):
                i, j = key[3], key[4]
                route_segments.append({
                    'from_loc': i,
                    'to_loc': j,
                    'transit_time': TT.get((i, j), 0),
                    'distance': Dist.get((i, j), 0.0),
                    'service_time': ST.get(j, 0)
                })

        # To track service times per stop (optional, as before)
        arrival_times = []
        service_end_times = []

        prev_end_time = 0
        for seg_idx, segment in enumerate(route_segments):
            from_loc = segment['from_loc']
            to_loc = segment['to_loc']
            from_name = LocName.get(from_loc, "")
            to_name = LocName.get(to_loc, "")
            from_type = LocType.get(from_loc, "")
            to_type = LocType.get(to_loc, "")

            # Calculate arrival and service end times
            if seg_idx == 0:
                arrival_time = 0
            else:
                arrival_time = prev_end_time + segment['transit_time']
            service_end_time = arrival_time + segment['service_time']
            arrival_times.append(arrival_time)
            service_end_times.append(service_end_time)
            prev_end_time = service_end_time

            # For delivery locations: list all orders delivered at this location by this vehicle
            order_rows_added = 0
            if to_type == "DELIVERY_LOCATION":
                delivered_orders = set()
                for (m, o, d, p) in ORD_SLOC_DLOC_PROD:
                    # Only orders delivered to this dest and assigned to this vehicle
                    # (Z_Param signals assignment)
                    z_key = (wave_id, cluster_id, phase_label, m, k)
                    if d == to_loc and Z_Param.get(z_key, 0) > 0.5:
                        delivered_orders.add(m)
                # Emit a row for each delivered order at this stop
                for order_id in delivered_orders:
                    # Get cust_po
                    cust_po = ""
                    for (mm, cpo, pp) in ORD_CUSTPO_PRD:
                        if mm == order_id:
                            cust_po = cpo
                            break
                    rows.append([
                        None,  # ROUTE_REPORT_ID
                        timestamp,
                        scenario_id,
                        scenario_description,
                        source_loc_id,
                        LocName.get(source_loc_id, ""),
                        wave_id,
                        cluster_id,
                        k,
                        to_loc,
                        to_name,
                        to_type,
                        order_id,
                        cust_po,
                        seg_idx + 1,  # STOP_NUM
                        arrival_time,
                        service_end_time,
                        segment['transit_time'],
                        0,  # IDLE_TIME_TO_NEXT_STOP (set if modeled)
                        segment['distance'],
                        LocCity.get(source_loc_id, ""),
                        LocState.get(source_loc_id, ""),
                        LocZipCode.get(source_loc_id, ""),
                        LocCity.get(to_loc, ""),
                        LocState.get(to_loc, ""),
                        LocZipCode.get(to_loc, ""),
                        arrival_time,
                        service_end_time,
                        0,  # DATE_OFFSET
                        "XXNBL_DTS",  # INSERT_USER
                        datetime.now().strftime("%d-%b-%y"),
                        "XXNBL_DTS",
                        datetime.now().strftime("%d-%b-%y")
                    ])
                    order_rows_added += 1
            # For non-delivery stops, just write a single row with blank order fields
            if order_rows_added == 0:
                rows.append([
                    None,  # ROUTE_REPORT_ID
                    timestamp,
                    scenario_id,
                    scenario_description,
                    source_loc_id,
                    LocName.get(source_loc_id, ""),
                    wave_id,
                    cluster_id,
                    k,
                    to_loc,
                    to_name,
                    to_type,
                    "",     # ORDER_ID
                    "",     # CUST_PO
                    seg_idx + 1,  # STOP_NUM
                    arrival_time,
                    service_end_time,
                    segment['transit_time'],
                    0,  # IDLE_TIME_TO_NEXT_STOP
                    segment['distance'],
                    LocCity.get(source_loc_id, ""),
                    LocState.get(source_loc_id, ""),
                    LocZipCode.get(source_loc_id, ""),
                    LocCity.get(to_loc, ""),
                    LocState.get(to_loc, ""),
                    LocZipCode.get(to_loc, ""),
                    arrival_time,
                    service_end_time,
                    0,  # DATE_OFFSET
                    "XXNBL_DTS",  # INSERT_USER
                    datetime.now().strftime("%d-%b-%y"),
                    "XXNBL_DTS",
                    datetime.now().strftime("%d-%b-%y")
                ])

    return rows
def write_scenario_stats_csv(final_filename, all_scenario_stats_rows):
    """
    Writes scenario statistics aligned with XXNBL_SCENARIO_STATS structure.
    """
    header = [
        "SCENARIO_ID", "TIME_STAMP", "SCENARIO_DESCRIPTION",
        "START_TIME", "END_TIME", "RUN_TIME",
        "COST_PER_CASE", "TOTAL_COST", "NUM_ORDERS", "NUM_VEHICLES_USED",
        "AVG_ROUTE_DISTANCE", "DISTANCE_TRAVELED",
        "SHIPPED_WEIGHT", "PERCENT_UTIL_WEIGHT",
        "SHIPPED_PALLETS", "ORIGINAL_PALLETS", "VAR_PALLETS", "PERCENT_VAR_PALLETS",
        "SHIPPED_CASES", "ORIGINAL_CASES", "VAR_CASES"
    ]

    with open(final_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_scenario_stats_rows)

    logger.info(f"[OK] Wrote Scenario Stats CSV: {final_filename} with {len(all_scenario_stats_rows)} rows.")
def write_shipment_csv(final_filename, all_shipment_rows):
    """
    Takes a list of shipment rows (from all scenarios),
    writes them to final_filename with a single header row.
    """

    header = [
        "SCENARIO_ID", "TIME_STAMP", "SCENARIO_ID", "SCENARIO_DESCRIPTION",
        "SOURCE_LOCATION_ID", "WAVE_ID", "CLUSTER_ID", "VEHICLE_ID", "RATE_ID",
        "COST_PER_CASE", "TOTAL_COST", "FIXED_COST", "MILEAGE_COST", "STOPPAGE_COST",
        "DISTANCE_TRAVELED", "NUM_STOPS", "NUM_ORDERS",
        "SHIPPED_WEIGHT", "PERCENT_UTIL_WEIGHT",
        "SHIPPED_PALLETS", "ORIGINAL_PALLETS", "VAR_PALLETS",
        "SHIPPED_CASES", "ORIGINAL_CASES", "VAR_CASES",
        "INSERT_USER", "INSERT_DATE", "UPDATE_USER", "UPDATE_DATE"
    ]

    with open(final_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_shipment_rows)

    logger.info(f"[OK] Wrote Shipment CSV: {final_filename} with {len(all_shipment_rows)} rows.")
def write_order_csv(final_filename, all_order_rows):
    header = [
        "ORDER_REPORT_ID",
        "TIME_STAMP",
        "SCENARIO_ID",
        "SCENARIO_DESCRIPTION",
        "SOURCE_LOCATION_ID",
        "SOURCE_LOCATION_NAME",
        "WAVE_ID",
        "CLUSTER_ID",
        "ORDER_ID",
        "CUST_PO",
        "PRODUCT_ID",
        "PRODUCT_DESCRIPTION",
        "SHIPPED_WEIGHT",
        "SHIPPED_PALLETS",
        "ORIGINAL_PALLETS",
        "VAR_PALLETS",
        "SHIPPED_CASES",
        "ORIGINAL_CASES",
        "VAR_CASES",
        "VEHICLE_ID",
        "DEST_LOCATION_ID",
        "DEST_LOCATION_NAME",
        "SOURCE_CITY",
        "SOURCE_STATE",
        "SOURCE_ZIP_CODE",
        "DEST_CITY",
        "DEST_STATE",
        "DEST_ZIP_CODE",
        "INSERT_USER",
        "INSERT_DATE",
        "UPDATE_USER",
        "UPDATE_DATE"
    ]
    with open(final_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_order_rows)

    logger.info(f"[OK] Wrote Order CSV: {final_filename} with {len(all_order_rows)} rows.")
def write_route_csv(final_filename, all_route_rows):

    header = [
    "ROUTE_REPORT_ID", "TIME_STAMP", "SCENARIO_ID", "SCENARIO_DESCRIPTION",
    "SOURCE_LOCATION_ID", "SOURCE_LOCATION_NAME", "WAVE_ID", "CLUSTER_ID",
    "VEHICLE_ID", "DEST_LOCATION_ID", "DEST_LOCATION_NAME", "DEST_LOCATION_TYPE",
    "ORDER_ID", "CUST_PO", "STOP_NUM", "ARRIVAL_TIME", "SERVICE_END_TIME",
    "TRANSIT_TIME_TO_NEXT_STOP", "IDLE_TIME_TO_NEXT_STOP", "TRANSIT_DIST_TO_NEXT_STOP",
    "SOURCE_CITY", "SOURCE_STATE", "SOURCE_ZIP_CODE",
    "DEST_CITY", "DEST_STATE", "DEST_ZIP_CODE",
    "A_TIME", "SE_TIME", "DATE_OFFSET",
    "INSERT_USER", "INSERT_DATE", "UPDATE_USER", "UPDATE_DATE"
    ]       
    with open(final_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_route_rows)

    logger.info(f"[OK] Wrote Route CSV: {final_filename} with {len(all_route_rows)} rows.")