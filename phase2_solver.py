

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    # Gurobi might not be available in all environments while reading files
    gp = None
    class GRB:
        pass

from cgi import print_environ
from config import *
from data_structures import *
import vrp_utils.logging as logging
logger = logging.getLogger(__name__)

def flex_allocation_and_phase2(source_loc_id, wave_id):
    """
    1) Check total flex across all clusters vs. global limit
    2) If exceeded, do 'flex allocation' to assign each cluster a new max up-limit
    3) Re-solve each cluster in 'Phase 2' with UseFlexUpPercentLimit=1
       and the new per-cluster limit.
    4) Store final solutions in X_Param, Y_Param, Z_Param, Q_Param with keys=("PHASE2", ...).
    """

    # ------------------------------------------------------------------------
    # A) Identify all CLUSTERS that belong to the scenario (if needed).
    #    If you have multiple clusters in SLOC_RATE_WAVE_CLUST_SUBCLUST_DLOC_ORD,
    #    gather them. If you only have 1 cluster (ClusterID), you can skip this.
    # ------------------------------------------------------------------------
    all_clusters = set()
    for (o, r, w, c, s_c, i, m) in SLOC_RATE_WAVE_CLUST_SUBCLUST_DLOC_ORD:
        if o == source_loc_id and w == wave_id:
            all_clusters.add(c)

    if not all_clusters:
        # If for some reason there's no cluster, do nothing
        logger.info("No clusters found. Skipping flex allocation.")
        return

    # ------------------------------------------------------------------------
    # B) For each cluster c, compute total shipped after Phase 1
    #    i.e. sum of Q_Param[("PHASE1", m, p, k)]
    #    Also sum the total original demand for that cluster.
    #
    #    We'll also define a dictionary 'ClustDiscount' (if you want to replicate
    #    the AMPL approach of discount per cluster). For example, just do zero or some value.
    # ------------------------------------------------------------------------
    ClustTotQ = {}
    ClustOrigD = {}
    ClustDiscount = {}
    for c in all_clusters:
        ClustTotQ[c] = 0.0
        ClustOrigD[c] = 0.0
        ClustDiscount[c] = 0  # or set to something if you want

    # We'll need a way to see if an order 'm' belongs to cluster c
    # Based on SLOC_RATE_WAVE_CLUST_SUBCLUST_DLOC_ORD structure:
    # Each (o, r, w, c, s_c, i, m) is a specific cluster c for that order m.
    # We'll build a dictionary: order_to_cluster[m] = c
    # NOTE: If an order can appear in multiple clusters, you'd adapt the logic.
    order_to_cluster = {}
    for (o, r, w, c, s_c, i, m) in SLOC_RATE_WAVE_CLUST_SUBCLUST_DLOC_ORD:
        if o == source_loc_id and w == wave_id:
            order_to_cluster[m] = c

    # Now sum up Q from Phase1 solutions:
    for (key, q_val) in Q_Param.items():
        # key might be ("PHASE1", m, p, k) or ("PHASE0",...) etc.
        if len(key) == 4:
            ph_label, m, p, k = key
            if ph_label == "PHASE1":
                # find cluster c for this order m
                c = order_to_cluster.get(m, None)
                if c in all_clusters:
                    ClustTotQ[c] += q_val

    # Sum up original demands
    for (m, o, d, p) in ORD_SLOC_DLOC_PROD:
        # if that order m belongs to a cluster c
        c = order_to_cluster.get(m, None)
        if c in all_clusters:
            ClustOrigD[c] += InputOriginalD.get((m, o, d, p), 0.0)

    # ------------------------------------------------------------------------
    # C) Check total (shipped - original) vs. the scenario's global flex up limit
    # ------------------------------------------------------------------------
    global_original_sum = sum(ClustOrigD[c] for c in all_clusters)
    global_shipped_sum = sum(ClustTotQ[c] for c in all_clusters)
    # The fraction of flex up used:
    total_flex_frac = (global_shipped_sum - global_original_sum) / max(1e-9, global_original_sum)
    logger.info(f"Global flex usage fraction = {100*total_flex_frac:.2f}%  vs. allowed {GlobalMaxFlexUpPercent}%")

    if total_flex_frac*100 <= GlobalMaxFlexUpPercent:
        # We are within limit, so no re-allocation needed
        logger.info("Global Flex Up usage is within limit. No reallocation needed.")
        # We still might do a Phase 2 solve if you want, but by default the AMPL code
        # only triggers Phase 2 if we exceed the global limit. We replicate that logic:
        return
    else:
        logger.info("We exceeded the global limit -> Running Flex Allocation...")

    # ------------------------------------------------------------------------
    # D) "Flex Allocation Model": Reassign how much each cluster is allowed
    #    to ship above original demand. The AMPL code sets 'ClustAllocQty[c]'
    #    initially to min(OriginalD, TotQ), then increments for clusters
    #    with the biggest discount, etc. We'll replicate:
    # ------------------------------------------------------------------------
    ClustAllocQty = {}
    for c in all_clusters:
        # Start with "lowest" of (original) or (already shipped)
        # (the AMPL code: "min(ClustOriginalD[c], ClustTotQ[c])")
        # But typically if we have shipped more than original, we keep it at original
        ClustAllocQty[c] = min(ClustOrigD[c], ClustTotQ[c])

    # total global flex allowance:
    # (1 + GlobalMaxFlexUpPercent/100)*global_original_sum - sum(ClustAllocQty[c])
    # i.e. how many "pallets" we can add across all clusters
    flex_allowance = (1 + GlobalMaxFlexUpPercent/100) * global_original_sum - sum(ClustAllocQty.values())

    # We'll also shuffle clusters by discount, from largest discount to smallest.
    # If you don't actually use discounts, they are all zero -> the loop won't do much.
    # but let's replicate the logic anyway.
    unprocessed = set(all_clusters)
    while unprocessed and flex_allowance > 0.0001:
        # find cluster with the maximum discount
        c_with_max = max(unprocessed, key=lambda c: ClustDiscount[c])
        unprocessed.remove(c_with_max)

        # how much "extra" that cluster shipped beyond original
        over_ship = max(ClustTotQ[c_with_max] - ClustOrigD[c_with_max], 0)
        # see how much we can allow from the flex allowance
        temp_flex_qty = min(over_ship, flex_allowance)
        ClustAllocQty[c_with_max] += temp_flex_qty
        flex_allowance -= temp_flex_qty

    # Now each cluster c has ClustAllocQty[c], which is effectively "the allocated total quantity" for c
    # We'll convert that into a final "max flex up percent" for cluster c:
    # The AMPL code: MaxFlexUpPercent_AllClusters[c] = 100*(ClustAllocQty[c]-ClustOrigD[c])/ClustOrigD[c]
    # But watch out for zero denominators
    MaxFlexUpPercent_Cluster = {}
    for c in all_clusters:
        if ClustOrigD[c] < 1e-9:
            MaxFlexUpPercent_Cluster[c] = 0
        else:
            # i.e. the fraction above originalD for that cluster
            MaxFlexUpPercent_Cluster[c] = 100.0 * (ClustAllocQty[c] - ClustOrigD[c]) / ClustOrigD[c]

    # ------------------------------------------------------------------------
    # E) Now we do PHASE 2 solves for each cluster, with:
    #    - UseFlexUpPercentLimit = 1
    #    - MaxFlexUpPercent_CurrentCluster = the new per-cluster limit
    #    - Possibly a bigger discount, e.g. "CurrClustDiscount = 2 * ClustDiscount[c]"
    #    - Warm start from the Phase 1 solution
    #    - Then store final solutions in X_Param[("PHASE2",...), ...] etc.
    # ------------------------------------------------------------------------
    for c in all_clusters:
        _solve_phase2_for_cluster(
            cluster_id=c,
            new_flex_up_percent=MaxFlexUpPercent_Cluster[c],
            new_discount=2*ClustDiscount[c]  # replicate the AMPL doubling
        )
def _solve_phase2_for_cluster(cluster_id, new_flex_up_percent, new_discount):

    if model.Status == GRB.OPTIMAL:
        # After storing Phase 2 results, remove previous phase entries for these vehicles
        for k in sub_vehicles:
            # Remove Phase 0 entries
            phase0_keys = [key for key in X_Param if key[0] != "PHASE2" and key[3] == k]
            for key in phase0_keys:
                del X_Param[key]
            # Similarly for Y_Param, Z_Param, Q_Param
            phase0_y_keys = [key for key in Y_Param if key[0] != "PHASE2" and key[1] == k]
            for key in phase0_y_keys:
                del Y_Param[key]
            phase0_z_keys = [key for key in Z_Param if key[0] != "PHASE2" and key[2] == k]
            for key in phase0_z_keys:
                del Z_Param[key]
    """
    This is basically the same as solve_phase1_cluster, but specialized to
    - use UseFlexUpPercentLimit=1
    - set MaxFlexUpPercent_CurrentCluster = new_flex_up_percent
    - set CurrClustDiscount = new_discount
    - warm-start from Phase1 solutions
    - store final results as X_Param[("PHASE2", i, j, k)], etc.
    """

    logger.info(f"\n=== PHASE 2: cluster-level solve start for cluster={cluster_id}, newFlex={new_flex_up_percent:.2f}% ===")

    # 1) Identify all orders/vehicles for this cluster
    relevant_keys = [
        (o, r, w, c, s_c, i, m)
        for (o, r, w, c, s_c, i, m) in SLOC_RATE_WAVE_CLUST_SUBCLUST_DLOC_ORD
        if o == source_loc_id and w == wave_id and c == cluster_id
    ]
    sub_orders = set(k[6] for k in relevant_keys)
    sub_vehicles_list = [InputVehicles[k] for k in relevant_keys if InputVehicles[k] is not None]
    sub_vehicles = set(sub_vehicles_list)
    sub_rates = set(k[1] for k in relevant_keys)
    sub_dests = set(k[5] for k in relevant_keys)

    start_depot = {l for l in LOCS if LocType.get(l, "") == "START_DEPOT" and l == source_loc_id}
    end_depot = {l for l in LOCS if LocType.get(l, "") == "END_DEPOT"}
    delivery_locs = sub_dests - start_depot - end_depot
    locs_cluster = start_depot.union(delivery_locs).union(end_depot)

    # 2) Build the same Gurobi model as in Phase 1,
    #    but with UseFlexUpPercentLimit=1, MaxFlexUpPercent_CurrentCluster=new_flex_up_percent,
    #    CurrClustDiscount=new_discount

    # (Model building code is nearly identical to solve_phase1_cluster,
    #  so we won't re-paste all constraints. The only differences:
    #  - We define a discount term = new_discount * sum(q).
    #  - We set "UseFlexUpPercentLimit = 1" and enforce the sum of q up to (1 + new_flex_up_percent/100)*OriginalD at the cluster level.)

    # ...
    # CREATE Gurobi model
    model = gp.Model("Phase2_Cluster")
    model.Params.OutputFlag = 1
    model.Params.TimeLimit = SolveTimeForCluster

    # Variables x, y, z, q, etc. (same as before)
    # [Not rewriting them here in detail...]

    # 3) Warm start from Phase1 solutions
    # E.g. if we had X_Param[("PHASE1", i, j, k)] = ...
    for k in sub_vehicles:
        used_any = 0
        for i in locs_cluster:
            for j in locs_cluster:
                if (("PHASE1", i, j, k) in X_Param) and X_Param[("PHASE1", i, j, k)] > 0.5:
                    used_any = 1
                    break


    # ...
    ############################################################################
    # 1) Build the Phase 2 variables exactly like in Phase 1:
    ############################################################################
    x = {}
    y = {}
    z = {}
    q_var = {}

    # Also the additional ones for time, stops, etc. if you want the same constraints:
    t = {}
    ns = {}
    it_var = {}
    u = {}

    # Create them:
    for i in locs_cluster:
        for j in locs_cluster:
            if i != j:
                for k in sub_vehicles:
                    x[(i,j,k)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")

    for k in sub_vehicles:
        y[k] = model.addVar(vtype=GRB.BINARY, name=f"y_{k}")
        ns[k] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"ns_{k}")

    for m in sub_orders:
        for k in sub_vehicles:
            z[(m,k)] = model.addVar(vtype=GRB.BINARY, name=f"z_{m}_{k}")

    for m in sub_orders:
        for p in cluster_products:
            for k in sub_vehicles:
                q_var[(m,p,k)] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"q_{m}_{p}_{k}")

    for i in locs_cluster:
        for k in sub_vehicles:
            t[(i,k)] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"t_{i}_{k}")

    for i in locs_cluster:
        for j in locs_cluster:
            if i != j:
                for k in sub_vehicles:
                    it_var[(i,j,k)] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"it_{i}_{j}_{k}")

    # For the MTZ sub-tour constraints
    Q_loc = {}
    for i in locs_cluster:
        if i in start_depot:
            Q_loc[i] = 0
        else:
            Q_loc[i] = 1
    Q_Total_local = sum(Q_loc[i] for i in locs_cluster)

    for i in locs_cluster:
        for k in sub_vehicles:
            u[(i,k)] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"u_{i}_{k}")

    model.update()


    ############################################################################
    # 2) WARM-START from Phase 1 solution
    ############################################################################
    for k in sub_vehicles:
        # Check if vehicle was used at all in Phase 1
        used_any = 0
        for i in locs_cluster:
            for j in locs_cluster:
                if i != j:
                    # If X_Param[("PHASE1", i, j, k)] = 1, means used
                    if X_Param.get(("PHASE1", i, j, k), 0.0) > 0.5:
                        used_any = 1
                        break
        y[k].Start = used_any

    # x, z, q as well
    for i in locs_cluster:
        for j in locs_cluster:
            if i != j:
                for k in sub_vehicles:
                    val = X_Param.get(("PHASE1", i, j, k), 0.0)
                    x[(i,j,k)].Start = val

    for m in sub_orders:
        for k in sub_vehicles:
            valz = Z_Param.get(("PHASE1", m, k), 0.0)
            z[(m,k)].Start = valz

    for m in sub_orders:
        for p in cluster_products:
            for k in sub_vehicles:
                valq = Q_Param.get(("PHASE1", m, p, k), 0.0)
                q_var[(m,p,k)].Start = valq


    ############################################################################
    # 3) Build constraints (like Phase 1), plus the 5.b.7 cluster-level flex limit
    ############################################################################

    # For the cluster-level flex up limit:
    # sum_{m,p,k} q_var[m,p,k] <= (1 + new_flex_up_percent/100) * sum_of_original_demand
    original_cluster_demand = 0.0
    for (mm, oo, dd, pp) in ORD_SLOC_DLOC_PROD:
        if mm in sub_orders:
            original_cluster_demand += InputOriginalD.get((mm, oo, dd, pp), 0.0)

    lhs_total_q = gp.quicksum(q_var[(m,p,k)]
                            for m in sub_orders
                            for p in cluster_products
                            for k in sub_vehicles)
    rhs_flex_cap = (1.0 + new_flex_up_percent/100.0) * original_cluster_demand
    model.addConstr(lhs_total_q <= rhs_flex_cap, name="ClusterFlexUpLimit")


    for k in cluster_vehicles:
        lhs = ns[k]
        rhs = gp.quicksum(x[(i, j, k)] for i in locs_cl for j in locs_cl if i != j) \
            - gp.quicksum(y[k] * VC.get((k, r), 0) * Stops_Included_In_Rate[r] for r in cluster_rates)
        model.addConstr(lhs >= rhs, name=f"nsBound_{k}")


    #
    # 5.a.1 => sum{k} z[m,k] = 1 for each order m
    #
    for m in cluster_orders:
        model.addConstr(
            gp.quicksum(z[(m, k)] for k in cluster_vehicles) == 1,
            name=f"OrderAssign_{m}"
        )


    #
    # 5.a.2 => y[k] >= z[m,k]
    #
    for m in cluster_orders:
        for k in cluster_vehicles:
            model.addConstr(y[k] >= z[(m, k)], name=f"VehUsedIfOrder_{m}_{k}")


    #
    # 5.a.3 => y[k] <= sum{m} z[m,k]
    #
    for k in cluster_vehicles:
        model.addConstr(
            y[k] <= gp.quicksum(z[(m, k)] for m in cluster_orders),
            name=f"VehDeact_{k}"
        )


    #
    # 5.b.1 => q[m,p,k] >= z[m,k] * 1 if PO_cl[m,p] == 1
    #
    for m in cluster_orders:
        for p in cluster_products:
            if PO_cl.get((m, p), 0) == 1:
                for k in cluster_vehicles:
                    model.addConstr(
                        q_var[(m, p, k)] >= z[(m, k)],
                        name=f"q_lb_{m}_{p}_{k}"
                    )


    def get_originalD_cluster(m, p, i):
        total = 0
        for (mm, oo, dd, pp) in ORD_SLOC_DLOC_PROD:
            if mm == m and pp == p and dd == i:
                total += InputOriginalD.get((mm, oo, dd, pp), 0.0)
        return total


    #
    # 5.b.2 => q[m,p,k] >= z[m,k]*(OriginalD - possible flex down)
    #
    for i in delivery_locs_cl:
        for m in cluster_orders:
            if LO_cl[(i, m)] == 1:
                for p in cluster_products:
                    if PO_cl.get((m, p), 0) == 1:
                        for k in cluster_vehicles:
                            origD_mp = get_originalD_cluster(m, p, i) # sum up the OriginalD for that (m, p, i)
                            val_down = origD_mp - PO_cl[(m, p)] * min(ProductFlexDown[p], LocFlexDownPerOrder[i]) * UseFlexQty
                            if val_down < 0:
                                val_down = 0
                            model.addConstr(
                                q_var[(m, p, k)] >= z[(m, k)] * val_down,
                                name=f"q_lb_flex_{m}_{p}_{k}_{i}"
                            )


    #
    # 5.b.3 => q[m,p,k] <= z[m,k]*(OriginalD + possible flex up)
    #
    for i in delivery_locs_cl:
        for m in cluster_orders:
            if LO_cl[(i, m)] == 1:
                for p in cluster_products:
                    if PO_cl.get((m, p), 0) == 1:
                        for k in cluster_vehicles:
                            origD_mp = get_originalD_cluster(m, p, i) # sum up the OriginalD for that (m, p, i)
                            val_up = origD_mp + PO_cl[(m, p)] * min(ProductFlexUp[p], LocFlexUpPerOrder[i]) * UseFlexQty
                            model.addConstr(
                                q_var[(m, p, k)] <= z[(m, k)] * val_up,
                                name=f"q_ub_flex_{m}_{p}_{k}_{i}"
                            )


    #
    # 5.b.4 => sum_{m,p} ( q[m,p,k]*Weight[p] ) <= sum_r ( y[k]*VC[k,r]*WCap[r] )
    #
    for k in cluster_vehicles:
        lhs = gp.quicksum(q_var[(m, p, k)] * WeightPerPallet[p]
                        for m in cluster_orders for p in cluster_products)
        rhs = gp.quicksum(y[k] * VC.get((k, r), 0) * WCap[r] for r in cluster_rates)
        model.addConstr(lhs <= rhs, name=f"capacity_{k}")


    #
    # 5.b.5 => location flex down limit
    #
    for i in delivery_locs_cl:
        lhs = gp.quicksum(q_var[(m, p, k)]
                        for m in cluster_orders
                        for p in cluster_products
                        for k in cluster_vehicles
                        if LO_cl[(i, m)] == 1)
        # sum of original demand for all orders that deliver to i
        total_orig_i = 0
        for (mm, oo, dd, pp) in ORD_SLOC_DLOC_PROD:
            if mm in cluster_orders and dd==i:
                total_orig_i += InputOriginalD.get((mm, oo, dd, pp), 0.0)
        model.addConstr(lhs >= total_orig_i - LocFlexDownPerOrder[i]*UseFlexQty, name=f"loc_down_{i}")


    # 5.b.6 => location flex up limit
    for i in delivery_locs_cl:
        # Sum of q_var for that location i
        lhs = gp.quicksum(
            q_var[(m, p, k)]
            for m in cluster_orders
            for p in cluster_products
            for k in cluster_vehicles
            if LO_cl[(i, m)] == 1
        )
        # Sum of original demand for location i in cluster_orders
        total_orig_i = 0
        for (mm, oo, dd, pp) in ORD_SLOC_DLOC_PROD:
            if mm in cluster_orders and dd == i:
                total_orig_i += InputOriginalD.get((mm, oo, dd, pp), 0.0)

        model.addConstr(
            lhs <= total_orig_i + LocFlexUpPerOrder[i] * UseFlexQty,
            name=f"loc_up_{i}"
        )

    #
    # 5.b.7 => cluster-level flex up => OFF (UseFlexUpPercentLimit=0), so skip
    #


    #
    # 5.c.1 => t[j,k] >= EDT_cl[m] * z[m,k] if LO_cl[j,m]=1
    #
    for m in cluster_orders:
        e_m = EDT_cl[m]
        for j in delivery_locs_cl:
            if LO_cl[(j, m)] == 1:
                for k in cluster_vehicles:
                    model.addConstr(t[(j, k)] >= e_m * z[(m, k)], name=f"EarlyDel_{j}_{m}_{k}")


    #
    # 5.c.2 => t[j,k] + ST[j] <= LDT_cl[m] + (1 - z[m,k])*MaxTime_cl
    #
    for m in cluster_orders:
        l_m = LDT_cl[m]
        for j in delivery_locs_cl:
            if LO_cl[(j, m)] == 1:
                for k in cluster_vehicles:
                    model.addConstr(
                        t[(j, k)] + ST[j] <= l_m + (1 - z[(m, k)])*MaxTime_cl,
                        name=f"LateDel_{j}_{m}_{k}"
                    )


    #
    # 5.c.3 => t[i,k] + ST[i] + TT[i,j] + it[i,j,k] = t[j,k] + (1 - x[i,j,k])*MaxTime_cl
    #
    for i in locs_cl:
        for j in delivery_locs_cl:
            if i != j:
                for k in cluster_vehicles:
                    tt_ij = TT.get((i, j), 0)
                    model.addConstr(
                        t[(i, k)] + ST[i] + tt_ij + it_var[(i, j, k)]
                        == t[(j, k)] + (1 - x[(i, j, k)])*MaxTime_cl,
                        name=f"ArrivalSeq_{i}_{j}_{k}"
                    )


    #
    # 5.c.4 => it[i,j,k] <= MaxIdleTimeBtwStops*x[i,j,k] + (1-x[i,j,k])*(2*MaxTime_cl)
    #
    if UseMaxIdleTimeConstr == 1:
        for i in locs_cl:
            for j in delivery_locs_cl:
                if i != j:
                    for k in cluster_vehicles:
                        model.addConstr(
                            it_var[(i, j, k)]
                            <= MaxIdleTimeBtwStops*x[(i, j, k)]
                            + (1 - x[(i, j, k)])*(2*MaxTime_cl),
                            name=f"MaxIdle_{i}_{j}_{k}"
                        )


    #
    # 5.c.5 => t[j,k] <= sum_{m} (LO_cl[j,m]*z[m,k]) * MaxTime_cl
    #
    for j in delivery_locs_cl:
        for k in cluster_vehicles:
            lhs = t[(j, k)]
            rhs = gp.quicksum(LO_cl[(j, m)]*z[(m, k)] for m in cluster_orders)*MaxTime_cl
            model.addConstr(lhs <= rhs, name=f"tZero_{j}_{k}")


    #
    # 5.c.6 => t[o,k] >= MinStartTime_cl*y[k]
    #
    for k in cluster_vehicles:
        for o in start_depot_cl:
            model.addConstr(
                t[(o, k)] >= MinStartTime_cl*y[k],
                name=f"MinRouteStart_{o}_{k}"
            )


    #
    # 5.c.7 => t[o,k] <= MaxStartTime_cl*y[k]
    #
    for k in cluster_vehicles:
        for o in start_depot_cl:
            model.addConstr(
                t[(o, k)] <= MaxStartTime_cl*y[k],
                name=f"MaxRouteStart_{o}_{k}"
            )


    #
    # 5.d.1 => sum{x[i,j,k]} <= sum_r(y[k]*VC[k,r]*MaxStops[r])
    #
    for k in cluster_vehicles:
        lhs = gp.quicksum(x[(i, j, k)] for i in locs_cl for j in locs_cl if i != j)
        rhs = gp.quicksum(y[k]*VC.get((k, r), 0)*MaxStops[r] for r in cluster_rates)
        model.addConstr(lhs <= rhs, name=f"MaxStops_{k}")


    #
    # 5.d.2 => t[i,k] - t[o,k] <= y[k]*( sum_r( VC[k,r]*MaxRouteDuration[r] ) - ST[i] )
    #
    for k in cluster_vehicles:
        for o in start_depot_cl:
            for i in delivery_locs_cl:
                dur = gp.quicksum(VC.get((k, r), 0)*MaxRouteDuration[r] for r in cluster_rates)
                model.addConstr(
                    t[(i, k)] - t[(o, k)] <= y[k]*(dur - ST[i]),
                    name=f"MaxRouteDur_{o}_{i}_{k}"
                )


    #
    # 5.d.3 => Dist[i,j]*x[i,j,k] <= sum_r( y[k]*VC[k,r]*MaxDistBetweenStops_val[r] )
    #
    for i in delivery_locs_cl:
        for j in delivery_locs_cl:
            if i != j:
                for k in cluster_vehicles:
                    lhs = Dist.get((i, j), 0)*x[(i, j, k)]
                    rhs = gp.quicksum(y[k]*VC.get((k, r), 0)*MaxDistBetweenStops_val[r] for r in cluster_rates)
                    model.addConstr(lhs <= rhs, name=f"MaxDistBtwStops_{i}_{j}_{k}")


    #
    # 5.e.1 => sum{i} x[i,o,k] = 0 if o in start_depot
    #
    for k in cluster_vehicles:
        for o in start_depot_cl:
            lhs = gp.quicksum(x.get((i, o, k), 0) for i in locs_cl if i != o)
            model.addConstr(lhs == 0, name=f"NoInboundStart_{o}_{k}")


    #
    # 5.e.2 => sum{i} x[i,e,k] = y[k] if e in end_depot
    #
    for k in cluster_vehicles:
        for e in end_depot_cl:
            lhs = gp.quicksum(x.get((i, e, k), 0) for i in locs_cl if i != e)
            model.addConstr(lhs == y[k], name=f"EndDepotRoute_{e}_{k}")


    #
    # 5.e.3 => sum{i} x[i,j,k] = z[m,k] if LO_cl[j,m] = 1
    #
    for j in delivery_locs_cl:
        for k2 in cluster_vehicles:
            for m in cluster_orders:
                if LO_cl[(j, m)] == 1:
                    lhs = gp.quicksum(x.get((i, j, k2), 0) for i in locs_cl if i != j)
                    model.addConstr(lhs == z[(m, k2)], name=f"Inbound_{j}_{m}_{k2}")


    #
    # 5.f.1 => sum{j} x[o,j,k] = y[k] if o in start_depot
    #
    for k in cluster_vehicles:
        for o in start_depot_cl:
            lhs = gp.quicksum(x.get((o, j, k), 0) for j in locs_cl if j != o)
            model.addConstr(lhs == y[k], name=f"Outbound_{o}_{k}")


    #
    # 5.f.2 => sum{j} x[i,j,k] = 0 if i in end_depot
    #
    for k in cluster_vehicles:
        for i in end_depot_cl:
            lhs = gp.quicksum(x.get((i, j, k), 0) for j in locs_cl if j != i)
            model.addConstr(lhs == 0, name=f"NoOutboundEnd_{i}_{k}")


    #
    # 5.f.3 => sum{j} x[i,j,k] = z[m,k] if LO_cl[i,m] = 1
    #
    for i in delivery_locs_cl:
        for k2 in cluster_vehicles:
            for m in cluster_orders:
                if LO_cl[(i, m)] == 1:
                    lhs = gp.quicksum(x.get((i, j, k2), 0) for j in locs_cl if j != i)
                    model.addConstr(lhs == z[(m, k2)], name=f"Outbound_{i}_{m}_{k2}")


    #
    # 5.g.2 => x[i,j,k] + x[j,i,k] <= 1
    #
    for i in delivery_locs_cl:
        for j in delivery_locs_cl:
            if i != j:
                for k2 in cluster_vehicles:
                    model.addConstr(
                        x.get((i, j, k2), 0) + x.get((j, i, k2), 0) <= 1,
                        name=f"NoRoundTrip_{i}_{j}_{k2}"
                    )


    #
    # 5.g.3 => x[o,e,k] = 0 for o in start_depot, e in end_depot
    #
    for k2 in cluster_vehicles:
        for o in start_depot_cl:
            for e in end_depot_cl:
                if (o, e, k2) in x:
                    model.addConstr(x[(o, e, k2)] == 0, name=f"NoDirect_{o}_{e}_{k2}")


    #
    # 5.h.1 => u[i,k] - u[j,k] + Q_Total_local*x[i,j,k] + (Q_Total_local - Q_loc[i] - Q_loc[j])*x[j,i,k] <= Q_Total_local - Q_loc[j]
    #
    for i in locs_cl:
        for j in locs_cl:
            if i != j:
                for k2 in cluster_vehicles:
                    lhs = u[(i, k2)] - u[(j, k2)] \
                        + Q_Total_local*x[(i, j, k2)] \
                        + (Q_Total_local - Q_loc[i] - Q_loc[j])*x.get((j, i, k2), 0)
                    rhs = Q_Total_local - Q_loc[j]
                    model.addConstr(lhs <= rhs, name=f"MTZ_{i}_{j}_{k2}")


    #
    # 5.h.2 => u[j,k] - sum{i}( Q_loc[i]*x[i,j,k] ) >= Q_loc[j]
    #
    for j in locs_cl:
        for k2 in cluster_vehicles:
            lhs = u[(j, k2)] - gp.quicksum(Q_loc[i]*x.get((i, j, k2), 0) for i in locs_cl if i != j)
            rhs = Q_loc[j]
            model.addConstr(lhs >= rhs, name=f"MTZ2_{j}_{k2}")


    #
    # 5.h.3 => u[i,k] + sum{j}( Q_loc[j]*x[i,j,k] ) <= Q_Total_local
    #
    for i in locs_cl:
        for k2 in cluster_vehicles:
            lhs = u[(i, k2)] + gp.quicksum(Q_loc[j]*x.get((i, j, k2), 0) for j in locs_cl if j != i)
            model.addConstr(lhs <= Q_Total_local, name=f"MTZ3_{i}_{k2}")


    #
    # 5.h.4 => u[j,k] <= Q_loc[j]*x[o,j,k] + Q_Total_local*(1 - x[o,j,k]) for the first route stop
    #
    for k2 in cluster_vehicles:
        for o in start_depot_cl:
            for j in locs_cl:
                if o != j:
                    lhs = u[(j, k2)]
                    rhs = Q_loc[j]*x.get((o, j, k2), 0) + Q_Total_local*(1 - x.get((o, j, k2), 0))
                    model.addConstr(lhs <= rhs, name=f"MTZ4_{o}_{j}_{k2}")


    #
    # 5.i.1 => sum{j} x[i,j,k] = sum{j} x[j,i,k]
    # for i in delivery_locs_cl
    #
    for i in delivery_locs_cl:
        for k2 in cluster_vehicles:
            lhs = gp.quicksum(x.get((i, j, k2), 0) for j in locs_cl if j != i)
            rhs = gp.quicksum(x.get((j, i, k2), 0) for j in locs_cl if j != i)
            model.addConstr(lhs == rhs, name=f"FlowBalance_{i}_{k2}")



    ############################################################################
    # 4) Objective = sum fixed + mileage + stop cost - UseFlexQty * new_discount * sum(q)
    ############################################################################
    fixed_part = gp.quicksum(FC[r]*VC.get((k,r),0)*y[k] 
                            for k in sub_vehicles for r in sub_rates)
    mileage_part = gp.quicksum(MC[r]*VC.get((k,r),0)*Dist.get((i,j),0)*x[(i,j,k)]
                            for i in locs_cluster for j in locs_cluster if i!=j
                            for k in sub_vehicles for r in sub_rates)
    stop_part = gp.quicksum(SC[r]*VC.get((k,r),0)*ns[k]
                            for k in sub_vehicles for r in sub_rates)
    discount_part = new_discount * gp.quicksum(q_var[(m,p,k)]
                        for m in sub_orders for p in cluster_products for k in sub_vehicles)

    model.setObjective(fixed_part + mileage_part + stop_part
                    - UseFlexQty * discount_part,
                    GRB.MINIMIZE)
    # after subclusters:
    used_vehicles = set()
    for s_c in subclusters:
        for k in sub_vehicles_union:
            if Y_Param.get((s_c, k), 0) > 0.5:
                used_vehicles.add(k)

    for k in sub_vehicles_union:
        if k not in used_vehicles:
            model.addConstr(y[k] == 0, name=f"FixOut_{k}")


    model.optimize()

   
    if model.Status == GRB.OPTIMAL:
        # Store X_Param with cluster/wave/phase keys
        for i in locs_cluster:
            for j in locs_cluster:
                if i != j:
                    for k in sub_vehicles:
                        key = (wave_id, cluster_id, "PHASE2", i, j, k)
                        X_Param[key] = x[(i, j, k)].X

        # Store Y_Param
        for k in sub_vehicles:
            key = (wave_id, cluster_id, "PHASE2", k)
            Y_Param[key] = y[k].X

        # Store Z_Param
        for m in sub_orders:
            for k in sub_vehicles:
                key = (wave_id, cluster_id, "PHASE2", m, k)
                Z_Param[key] = z[(m, k)].X


        for m in sub_orders:          # define or gather sub_orders as in your code
            for p in sub_products:    # define or gather sub_products as in your code
                for k in sub_vehicles:
                    Q_Param[("PHASE2", wave_id, cluster_id, m, p, k)] = q_var[(m,p,k)].X

        # -- 2) NOW call the post-processing clamp function
        post_process_final_q("PHASE2", wave_id, cluster_id)
        # Clean up old entries (optional)
        for key in list(X_Param.keys()):
            if key[2] == "PHASE1" and key[0] == wave_id and key[1] == cluster_id:
                del X_Param[key]
        ############################################################################
        # >>> POST-PROCESS Q SNAP (3C) <<<
        # Suppose we want to clamp q to [origD - flexDown, origD + flexUp].
        # We'll do a direct override in Q_Param based on the model solution.
        if UseFlexQty == 1:
            for m in sub_orders:
                for p in cluster_products:
                    for k in sub_vehicles:
                        val_q = q_var[(m,p,k)].X
                        # find location(s) i for that (m). We'll assume a single i for brevity:
                        # or you can sum it across i. 
                        # Example approach:
                        # get the original demand for that (m,p):
                        sum_orig = 0.0
                        for (mm,oo,dd,pp) in ORD_SLOC_DLOC_PROD:
                            if mm==m and pp==p:
                                sum_orig += InputOriginalD.get((mm,oo,dd,pp),0.0)

                        # clamp to minQ..maxQ:
                        min_q = sum_orig - ProductFlexDown.get(p,0.0)
                        max_q = sum_orig + ProductFlexUp.get(p,0.0)
                        if min_q<0: min_q=0

                        # snap:
                        if val_q < min_q:
                            val_q = min_q
                        elif val_q > max_q:
                            val_q = max_q

                        # store back:
                        # Q_Param[("PHASE2", m, p, k)] = val_q
                        Q_Param[(wave_id, cluster_id, "PHASE2", m, p, k)] = q_var[(m,p,k)].X
                        # key = (wave_id, cluster_id, "PHASE2", m, p, k)
                        # Q_Param[key] = val_q
        ############################################################################
    else:
        logger.info(f"PHASE 2 => solver ended with status={model.Status}")

    model.dispose()
    
    logger.info(f"=== PHASE 2: cluster-level solve done for cluster={cluster_id} ===")