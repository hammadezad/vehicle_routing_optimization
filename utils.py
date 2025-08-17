from config import *
from data_structures import *
from data_loader import load_data_for_scenario
from rate_builder import build_global_rate_dicts
from reports import gather_order_rows
from reports import gather_route_rows
from reports import gather_shipment_rows
from reports import gather_scenario_stats_rows
import vrp_utils.logging as logging
logger = logging.getLogger(__name__)


try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    # Gurobi might not be available in all environments while reading files
    gp = None
    class GRB:
        pass

# Import functions directly to avoid circular imports
import phase0_solver
import phase1_solver
import phase2_solver

def post_process_final_q(phase_label, wave_id, cluster_id):
    """
    Post-process the final solution for the given (wave_id, cluster_id) at phase_label
    to replicate the AMPL 'snap' logic:
     1) clamp each q[m,p,k] up to a minimum threshold (origD - flexDown)
     2) attempt to flex up in +1 increments, reverting if we exceed capacity or location/product flex
    """
    if UseFlexQty != 1:
        return  # If no flex usage, no post-processing needed

    # Identify which orders, vehicles, and products belong to this wave/cluster.
    # For example, you might do something like this if your code organizes them similarly:
    # (Adjust as needed to match your naming or data structures.)
    sub_orders = set()
    sub_vehicles = set()
    sub_products = set()

    # Typically you'd replicate how you identified sub_orders, sub_vehicles, sub_products
    # e.g. the same logic you used in solve_phase2_for_cluster to gather them
    for (o, r, w, c, s_c, i, m) in SLOC_RATE_WAVE_CLUST_SUBCLUST_DLOC_ORD:
        if w == wave_id and c == cluster_id:
            sub_orders.add(m)
            v = InputVehicles.get((o, r, w, c, s_c, i, m), None)
            if v is not None:
                sub_vehicles.add(v)
            # also gather products
            # you can loop over ORD_SLOC_DLOC_PROD to find p if m in ...
            # or do it like you do in solve_phase2_for_cluster
    for (m, o, d, p) in ORD_SLOC_DLOC_PROD:
        if m in sub_orders:
            sub_products.add(p)

    # ============== STEP 1: Clamp each q[m,p,k] to min (origD - flexDown) ==============
    for m in sub_orders:
        for p in sub_products:
            # compute total origD. If you do single-loc, you might do something else.
            # For multi-loc, you typically pick the i that belongs to the order m
            # If each m only has one i, you can do:
            # i = the location for that order
            # or sum up multiple i's if needed
            # We'll do a simple approach:
            origD = 0
            for (mm, oo, dd, pp) in ORD_SLOC_DLOC_PROD:
                if mm == m and pp == p:
                    origD += InputOriginalD.get((mm, oo, dd, pp), 0)

            # pick a location i for flexDown, or sum it. For simplicity:
            # you might do location i from that order
            # We'll just pick the min flexDown across all possible i. 
            # The AMPL code logic is more intricate for multi-loc orders.
            possible_i = [
                dd for (mm, oo, dd, pp) in ORD_SLOC_DLOC_PROD
                if mm == m and pp == p
            ]
            # pick the first or do a min. We'll just do one:
            if possible_i:
                i_loc = possible_i[0]
            else:
                i_loc = None

            if i_loc is not None:
                loc_flexDown = LocFlexDownPerOrder.get(i_loc, 0)
            else:
                loc_flexDown = 0

            min_q_allowed = max(0, origD - ProductFlexDown.get(p,0) - loc_flexDown)  # simplified

            for k in sub_vehicles:
                key = (phase_label, wave_id, cluster_id, m, p, k)
                old_val = Q_Param.get(key, 0.0)
                if old_val < min_q_allowed:
                    Q_Param[key] = min_q_allowed

    # ============== STEP 2: Attempt to flex up in +1 increments ==============
    # we do multiple passes, for example 10:
    for iteration in range(10):
        # loop through vehicles, orders, products
        for k in sub_vehicles:
            for m in sub_orders:
                for p in sub_products:
                    key = (phase_label, wave_id, cluster_id, m, p, k)
                    old_val = Q_Param.get(key, 0.0)

                    # see if we can add +1 pallet
                    new_val = old_val + 1
                    # check if new_val <= (origD + productFlexUp + locFlexUp)
                    # for simplicity, do:
                    # if new_val <= origD + ProductFlexUp[p] + ...
                    # and also check capacity with new_val

                    # compute a maximum q. E.g.:
                    # max_q = ...
                    # For short example:
                    max_q = old_val + ProductFlexUp.get(p, 0) + LocFlexUpPerOrder.get(i_loc, 0)
                    # You might do a more accurate approach summing demands across locations

                    if new_val <= max_q:
                        # tentatively store it
                        Q_Param[key] = new_val
                        if not check_feasible_capacity(k, wave_id, cluster_id, phase_label, sub_orders, sub_products):
                            # revert
                            Q_Param[key] = old_val

    # done post-processing
    logger.info(f"[post_process_final_q] Completed clamping for {phase_label}, wave={wave_id}, cluster={cluster_id}.")
def check_feasible_capacity(k, wave_id, cluster_id, phase_label, sub_orders, sub_products):
    """
    Example function that sums up Q_Param for the given vehicle k,
    checks if we exceed WCap for that k's rate.
    Return True if it's still feasible, else False.
    """
    # find the rate r that has VC[k,r] == 1
    # or handle multi-rates if your code allows that
    chosen_rate = None
    for r in FC.keys():
        if VC.get((k,r),0) == 1:
            chosen_rate = r
            break
    if chosen_rate is None:
        # no rate found, so let's not fail
        return True

    total_wt = 0.0
    for m in sub_orders:
        for p in sub_products:
            q_val = Q_Param.get((phase_label, wave_id, cluster_id, m, p, k), 0.0)
            wt = WeightPerPallet.get(p,0.0)
            total_wt += (q_val * wt)

    # compare
    cap = WCap.get(chosen_rate, 9999999)
    return (total_wt <= cap)
def parse_scenario(scenario_id, dataframes, shipment_rows_acc, order_rows_acc, route_rows_acc, scenario_descriptions, scenario_stats_acc):
    logger.info(f"=== parse_scenario(scenario_id={scenario_id}) start ===")
    load_data_for_scenario(dataframes, scenario_id)
    build_global_rate_dicts()

    # Track all orders and vehicles across phases
    all_sub_orders = set()
    all_sub_vehicles = set()
    used_vehicles_dict = {}

    wave_ids = set()
    cluster_ids = set()
    for (o, r, w, c, s_c, i, m) in SLOC_RATE_WAVE_CLUST_SUBCLUST_DLOC_ORD:
        wave_ids.add(w)
        cluster_ids.add(c)

    phase2_executed = False

    # First pass: Solve Phase 0 and Phase 1 for all clusters
    for wave_id in wave_ids:
        for cluster_id in cluster_ids:
            relevant_keys = [
                (o, r, w, c, sc, i, m)
                for (o, r, w, c, sc, i, m) in SLOC_RATE_WAVE_CLUST_SUBCLUST_DLOC_ORD
                if w == wave_id and c == cluster_id
            ]
            sub_orders = set(k[6] for k in relevant_keys)
            sub_vehicles = set(InputVehicles[k] for k in relevant_keys if InputVehicles[k] is not None)
            
            all_sub_orders.update(sub_orders)
            all_sub_vehicles.update(sub_vehicles)

            possible_sourceIDs = {o for (o, r, w, c, sc, i, m) in relevant_keys}
            for source_loc_id in possible_sourceIDs:
                subclusters = set(sc for (o, r, w, c, sc, i, m) in relevant_keys if o == source_loc_id)
                for sc in subclusters:
                    # solve_phase0_subcluster(sc, wave_id, cluster_id, source_loc_id)
                    used_veh_sub = phase0_solver.solve_phase0_subcluster(sc, wave_id, cluster_id, source_loc_id)
                    used_vehicles_dict[sc] = used_veh_sub                    
                phase1_solver.solve_phase1_for_cluster(wave_id, cluster_id, source_loc_id, used_vehicles_dict)

    # Now calculate global shipped sum using the collected orders and vehicles
    global_original_sum = sum(InputOriginalD.get((m, o, d, p), 0) 
                             for (m, o, d, p) in ORD_SLOC_DLOC_PROD 
                             if m in all_sub_orders)
    
    global_shipped_sum = sum(Q_Param.get(("PHASE1", m, p, k), 0) 
                            for m in all_sub_orders 
                            for p in PRD 
                            for k in all_sub_vehicles)

    if all_sub_orders:  # Avoid division by zero
        total_flex_frac = (global_shipped_sum - global_original_sum) / global_original_sum * 100
        if total_flex_frac > GlobalMaxFlexUpPercent:
            phase2_executed = True
            # Second pass: Solve Phase 2 for clusters needing flex allocation
            for wave_id in wave_ids:
                for cluster_id in cluster_ids:
                    relevant_keys = [
                        (o, r, w, c, sc, i, m)
                        for (o, r, w, c, sc, i, m) in SLOC_RATE_WAVE_CLUST_SUBCLUST_DLOC_ORD
                        if w == wave_id and c == cluster_id
                    ]
                    possible_sourceIDs = {o for (o, r, w, c, sc, i, m) in relevant_keys}
                    for source_loc_id in possible_sourceIDs:
                        phase2_solver.flex_allocation_and_phase2(source_loc_id, wave_id)

    final_label = "PHASE2" if phase2_executed else "PHASE1"

    # Generate reports using the final phase data
    for wave_id in wave_ids:
        for cluster_id in cluster_ids:
            relevant_keys = [
                (o, r, w, c, sc, i, m)
                for (o, r, w, c, sc, i, m) in SLOC_RATE_WAVE_CLUST_SUBCLUST_DLOC_ORD
                if w == wave_id and c == cluster_id
            ]
            possible_sources = {o for (o, r, w, c, sc, i, m) in relevant_keys}
            for source_loc_id in possible_sources:
                ship_rows = gather_shipment_rows(scenario_id, wave_id, cluster_id, source_loc_id, final_label, scenario_descriptions)
                order_rows = gather_order_rows(scenario_id, wave_id, cluster_id, source_loc_id, final_label, scenario_descriptions)
                route_rows = gather_route_rows(scenario_id, wave_id, cluster_id, source_loc_id, final_label, scenario_descriptions)
                                
                shipment_rows_acc.extend(ship_rows)
                order_rows_acc.extend(order_rows)
                route_rows_acc.extend(route_rows)

    stats_rows = gather_scenario_stats_rows(scenario_id, final_label, scenario_descriptions)
    scenario_stats_acc.extend(stats_rows)

    logger.info(f"=== parse_scenario(scenario_id={scenario_id}) end ===")
    logger.info(f"   → Added {len(stats_rows)} scenario stats rows")