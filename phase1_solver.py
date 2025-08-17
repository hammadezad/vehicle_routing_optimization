import vrp_utils.logging as logging
logger = logging.getLogger(__name__)

import data_structures as ds
import config as cfg

from vrp_utils.decorators import log_and_time
from exceptions import Phase1Error

# Gurobi import (keep this pattern in every solver)
try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    class GRB:  # fallback stub so the module still imports
        pass


@log_and_time("solve_phase1_for_cluster", error_cls=Phase1Error)
def solve_phase1_for_cluster(wave_id, cluster_id, source_loc_id, used_vehicles_dict):
    """
    Gathers union of subclusters for the cluster, builds one cluster-level model,
    warm-starts from Phase 0 solution artifacts, and writes Phase 1 results into
    ds.X_Param, ds.Y_Param, ds.Z_Param, ds.Q_Param keyed by
    (wave_id, cluster_id, "PHASE1", ...).
    """
    logger.info("PHASE 1: wave=%s, cluster=%s, source_loc_id=%s", wave_id, cluster_id, source_loc_id)
    logger.info("=== PHASE 1: cluster-level solve start ===")

    # Which subclusters belong to this (wave, cluster)
    subclusters = set()
    for (o, r, w, c, s_c, i, m) in ds.SLOC_RATE_WAVE_CLUST_SUBCLUST_DLOC_ORD:
        if w == wave_id and c == cluster_id:
            subclusters.add(s_c)

    # Union across subclusters
    sub_orders_union = set()
    sub_vehicles_union = set()
    sub_rates_union = set()
    sub_locs_union = set()

    # Vehicles that Phase 0 said were used in each subcluster
    used_vehicles_union = set()
    for sc in subclusters:
        if sc in used_vehicles_dict:
            used_vehicles_union |= set(used_vehicles_dict[sc])

    sub_vehicles_union = used_vehicles_union

    for s_c in subclusters:
        relevant_keys = [
            (o, r, w, c, sc, i, m)
            for (o, r, w, c, sc, i, m) in ds.SLOC_RATE_WAVE_CLUST_SUBCLUST_DLOC_ORD
            if w == wave_id and c == cluster_id and sc == s_c
        ]
        for (o, r, w, c, sc, i, m) in relevant_keys:
            sub_orders_union.add(m)
            these_veh = [ds.InputVehicles[k] for k in relevant_keys if ds.InputVehicles.get(k) is not None]
            sub_vehicles_union |= set(these_veh)
            sub_rates_union |= {k[1] for k in relevant_keys}
            sub_locs_union |= {k[5] for k in relevant_keys}

    # Start/end depots and delivery locations
    start_depot = {l for l in ds.LOCS if ds.LocType.get(l, "") == "START_DEPOT" and l == source_loc_id}
    end_depot = {l for l in ds.LOCS if ds.LocType.get(l, "") == "END_DEPOT"}
    delivery_locs = sub_locs_union - start_depot - end_depot
    locs_cluster = start_depot | delivery_locs | end_depot

    # Vehicles used in any subcluster according to Phase 0
    veh_used_any = set()
    for s_c in subclusters:
        for k in sub_vehicles_union:
            if ds.Y_Param.get((s_c, k), 0.0) > 0.5:
                veh_used_any.add(k)

    # Products for the cluster’s orders
    cluster_products = set()
    for (m, o, d, p) in ds.ORD_SLOC_DLOC_PROD:
        if m in sub_orders_union:
            cluster_products.add(p)

    # Helpers to aggregate original demand
    def get_originalD_cluster(m, p, i):
        total = 0.0
        for (mm, oo, dd, pp) in ds.ORD_SLOC_DLOC_PROD:
            if mm == m and pp == p and dd == i:
                total += ds.InputOriginalD.get((mm, oo, dd, pp), 0.0)
        return total

    # LO_cl[(i,m)] and PO_cl[(m,p)]
    LO_cl = {}
    for i in locs_cluster:
        for m in sub_orders_union:
            val = 1 if i in start_depot else 0
            if not val:
                for (mm, oo, dd, pp) in ds.ORD_SLOC_DLOC_PROD:
                    if mm == m and dd == i and ds.InputOriginalD.get((mm, oo, dd, pp), 0.0) > 0:
                        val = 1
                        break
            LO_cl[(i, m)] = val

    PO_cl = {}
    for m in sub_orders_union:
        for p in cluster_products:
            tot = 0.0
            for (mm, oo, dd, pp) in ds.ORD_SLOC_DLOC_PROD:
                if mm == m and pp == p:
                    tot += ds.InputOriginalD.get((mm, oo, dd, pp), 0.0)
            PO_cl[(m, p)] = 1 if tot > 0 else 0

    # EDT_cl and LDT_cl
    EDT_cl, LDT_cl = {}, {}
    for m in sub_orders_union:
        e_candidates, l_candidates = [], []
        for (mm, oo, dd, pp) in ds.ORD_SLOC_DLOC_PROD:
            if mm == m:
                e_candidates.append(ds.InputEDT.get((mm, oo, dd, pp), 0.0))
                l_candidates.append(ds.InputLDT.get((mm, oo, dd, pp), 0.0))
        EDT_cl[m] = max(e_candidates) if e_candidates else 0.0
        LDT_cl[m] = max(l_candidates) if l_candidates else 999999.0

    # MaxTime_cl, MinStartTime_cl, MaxStartTime_cl
    max_LDT = max(LDT_cl.values()) if LDT_cl else 0.0
    max_ST = max((ds.ST.get(loc, 0.0) for loc in locs_cluster), default=0.0)
    max_TT = 0.0
    for i in locs_cluster:
        for j in locs_cluster:
            max_TT = max(max_TT, ds.TT.get((i, j), 0.0))
    MaxTime_cl = 2 * (max_LDT + max_ST + max_TT)

    min_edt = min(EDT_cl.values()) if EDT_cl else 0.0
    max_tt_od = 0.0
    for o in start_depot:
        for d in delivery_locs:
            max_tt_od = max(max_tt_od, ds.TT.get((o, d), 0.0))
    max_st_o = max((ds.ST.get(o, 0.0) for o in start_depot), default=0.0)
    MinStartTime_cl = max(0.0, min_edt - max_tt_od - max_st_o)

    max_ldt_ = max(LDT_cl.values()) if LDT_cl else 0.0
    min_tt_od = min((ds.TT.get((o, d), 999999.0) for o in start_depot for d in delivery_locs), default=999999.0)
    if min_tt_od == 999999.0:
        min_tt_od = 0.0
    min_st_o = min((ds.ST.get(o, 0.0) for o in start_depot), default=0.0)
    MaxStartTime_cl = max_ldt_ - min_tt_od - min_st_o
    if MaxStartTime_cl < 0.0:
        MaxStartTime_cl = MaxTime_cl

    logger.info("CHECKPOINT: Built LO_cl, PO_cl, EDT_cl, LDT_cl, and timing bounds for Phase 1")

    # Model
    model = gp.Model("Phase1_Cluster")
    model.Params.OutputFlag = 1
    # model.Params.TimeLimit = cfg.SolveTimeForCluster

    # Decision variables with warm-starts from Phase 0
    x = {}
    for i in locs_cluster:
        for j in locs_cluster:
            if i != j:
                for k in sub_vehicles_union:
                    var = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")
                    key = (wave_id, cluster_id, "PHASE0", i, j, k)
                    if key in ds.X_Param:
                        var.Start = ds.X_Param[key]
                    x[(i, j, k)] = var

    y = {}
    for k in sub_vehicles_union:
        var = model.addVar(vtype=GRB.BINARY, name=f"y_{k}")
        key = (wave_id, cluster_id, "PHASE0", k)
        if key in ds.Y_Param:
            var.Start = ds.Y_Param[key]
        y[k] = var

    z = {}
    for m in sub_orders_union:
        for k in sub_vehicles_union:
            var = model.addVar(vtype=GRB.BINARY, name=f"z_{m}_{k}")
            key = (wave_id, cluster_id, "PHASE0", m, k)
            if key in ds.Z_Param:
                var.Start = ds.Z_Param[key]
            z[(m, k)] = var

    q_var = {}
    for m in sub_orders_union:
        for p in cluster_products:
            for k in sub_vehicles_union:
                var = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"q_{m}_{p}_{k}")
                key = (wave_id, cluster_id, "PHASE0", m, p, k)
                if key in ds.Q_Param:
                    var.Start = ds.Q_Param[key]
                q_var[(m, p, k)] = var

    t = {(i, k): model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"t_{i}_{k}")
         for i in locs_cluster for k in sub_vehicles_union}
    ns = {k: model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"ns_{k}")
          for k in sub_vehicles_union}
    it_var = {(i, j, k): model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"it_{i}_{j}_{k}")
              for i in locs_cluster for j in locs_cluster if i != j for k in sub_vehicles_union}
    Q_loc = {i: (0 if i in start_depot else 1) for i in locs_cluster}
    Q_Total_local = sum(Q_loc.values())
    u = {(i, k): model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"u_{i}_{k}")
         for i in locs_cluster for k in sub_vehicles_union}

    model.update()
    model.Params.StartNodeLimit = 1

    # Objective
    def fixed_cost(k):
        return gp.quicksum(ds.FC[r] * ds.VC.get((k, r), 0) * y[k] for r in sub_rates_union)

    def mileage_cost(k):
        return gp.quicksum(
            ds.MC[r] * ds.VC.get((k, r), 0) * ds.Dist.get((i, j), 0.0) * x[(i, j, k)]
            for i in locs_cluster for j in locs_cluster if i != j for r in sub_rates_union
        )

    def stoppage_cost(k):
        return gp.quicksum(ds.SC[r] * ds.VC.get((k, r), 0) * ns[k] for r in sub_rates_union)

    def discount_total(k):
        return cfg.CurrClustDiscount * gp.quicksum(q_var[(m, p, k)] for m in sub_orders_union for p in cluster_products)

    obj = gp.quicksum(fixed_cost(k) + mileage_cost(k) + stoppage_cost(k) for k in sub_vehicles_union) \
          - cfg.UseFlexQty * gp.quicksum(discount_total(k) for k in sub_vehicles_union)
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints
    cluster_orders = sub_orders_union
    cluster_vehicles = sub_vehicles_union
    cluster_rates = sub_rates_union
    start_depot_cl = start_depot
    end_depot_cl = end_depot
    delivery_locs_cl = delivery_locs
    locs_cl = start_depot_cl | delivery_locs_cl | end_depot_cl

    # ns[k] bound
    for k in cluster_vehicles:
        lhs = ns[k]
        rhs = gp.quicksum(x[(i, j, k)] for i in locs_cl for j in locs_cl if i != j) \
              - gp.quicksum(y[k] * ds.VC.get((k, r), 0) * ds.Stops_Included_In_Rate[r] for r in cluster_rates)
        model.addConstr(lhs >= rhs, name=f"nsBound_{k}")

    # Order assignment and vehicle activation
    for m in cluster_orders:
        model.addConstr(gp.quicksum(z[(m, k)] for k in cluster_vehicles) == 1, name=f"OrderAssign_{m}")
        for k in cluster_vehicles:
            model.addConstr(y[k] >= z[(m, k)], name=f"VehUsedIfOrder_{m}_{k}")
    for k in cluster_vehicles:
        model.addConstr(y[k] <= gp.quicksum(z[(m, k)] for m in cluster_orders), name=f"VehDeact_{k}")

    # q lower bound when product present
    for m in cluster_orders:
        for p in cluster_products:
            if PO_cl.get((m, p), 0) == 1:
                for k in cluster_vehicles:
                    model.addConstr(q_var[(m, p, k)] >= z[(m, k)], name=f"q_lb_{m}_{p}_{k}")

    # Flex down/up at location level and capacity
    for i in delivery_locs_cl:
        lhs = gp.quicksum(
            q_var[(m, p, k)]
            for m in cluster_orders for p in cluster_products for k in cluster_vehicles
            if LO_cl[(i, m)] == 1
        )
        total_orig_i = 0.0
        for (mm, oo, dd, pp) in ds.ORD_SLOC_DLOC_PROD:
            if mm in cluster_orders and dd == i:
                total_orig_i += ds.InputOriginalD.get((mm, oo, dd, pp), 0.0)
        model.addConstr(lhs >= total_orig_i - ds.LocFlexDownPerOrder.get(i, 0.0) * cfg.UseFlexQty, name=f"loc_down_{i}")

    for i in delivery_locs_cl:
        lhs = gp.quicksum(
            q_var[(m, p, k)]
            for m in cluster_orders for p in cluster_products for k in cluster_vehicles
            if LO_cl[(i, m)] == 1
        )
        total_orig_i = 0.0
        for (mm, oo, dd, pp) in ds.ORD_SLOC_DLOC_PROD:
            if mm in cluster_orders and dd == i:
                total_orig_i += ds.InputOriginalD.get((mm, oo, dd, pp), 0.0)
        model.addConstr(lhs <= total_orig_i + ds.LocFlexUpPerOrder.get(i, 0.0) * cfg.UseFlexQty, name=f"loc_up_{i}")

    for k in cluster_vehicles:
        lhs = gp.quicksum(q_var[(m, p, k)] * ds.WeightPerPallet.get(p, 0.0) for m in cluster_orders for p in cluster_products)
        rhs = gp.quicksum(y[k] * ds.VC.get((k, r), 0) * ds.WCap[r] for r in cluster_rates)
        model.addConstr(lhs <= rhs, name=f"capacity_{k}")

    # Optional global flex-up percent limit
    if cfg.UseFlexUpPercentLimit == 1:
        total_ship = gp.quicksum(q_var[(m, p, k)] for m in cluster_orders for p in cluster_products for k in cluster_vehicles)
        total_orig = 0.0
        for m in cluster_orders:
            for p in cluster_products:
                for (mm, oo, dd, pp) in ds.ORD_SLOC_DLOC_PROD:
                    if mm == m and pp == p:
                        total_orig += ds.InputOriginalD.get((mm, oo, dd, pp), 0.0)
        rhs = (1 + cfg.MaxFlexUpPercent_AllClusters.get(cluster_id, 100) / 100.0) * total_orig
        model.addConstr(total_ship <= rhs, name="ClusterFlexUp")

    # Time window and sequencing
    for m in cluster_orders:
        e_m, l_m = EDT_cl[m], LDT_cl[m]
        for j in delivery_locs_cl:
            if LO_cl[(j, m)] == 1:
                for k in cluster_vehicles:
                    model.addConstr(t[(j, k)] >= e_m * z[(m, k)], name=f"EarlyDel_{j}_{m}_{k}")
                    model.addConstr(t[(j, k)] + ds.ST.get(j, 0.0) <= l_m + (1 - z[(m, k)]) * MaxTime_cl,
                                    name=f"LateDel_{j}_{m}_{k}")

    for i in locs_cl:
        for j in delivery_locs_cl:
            if i != j:
                for k in cluster_vehicles:
                    tt_ij = ds.TT.get((i, j), 0.0)
                    model.addConstr(
                        t[(i, k)] + ds.ST.get(i, 0.0) + tt_ij + it_var[(i, j, k)]
                        == t[(j, k)] + (1 - x[(i, j, k)]) * MaxTime_cl,
                        name=f"ArrivalSeq_{i}_{j}_{k}"
                    )

    if cfg.UseMaxIdleTimeConstr == 1:
        for i in locs_cl:
            for j in delivery_locs_cl:
                if i != j:
                    for k in cluster_vehicles:
                        model.addConstr(
                            it_var[(i, j, k)]
                            <= cfg.MaxIdleTimeBtwStops * x[(i, j, k)]
                            + (1 - x[(i, j, k)]) * (2 * MaxTime_cl),
                            name=f"MaxIdle_{i}_{j}_{k}"
                        )

    for j in delivery_locs_cl:
        for k in cluster_vehicles:
            lhs = t[(j, k)]
            rhs = gp.quicksum(LO_cl[(j, m)] * z[(m, k)] for m in cluster_orders) * MaxTime_cl
            model.addConstr(lhs <= rhs, name=f"tZero_{j}_{k}")

    for k in cluster_vehicles:
        for o in start_depot_cl:
            model.addConstr(t[(o, k)] >= MinStartTime_cl * y[k], name=f"MinRouteStart_{o}_{k}")
            model.addConstr(t[(o, k)] <= MaxStartTime_cl * y[k], name=f"MaxRouteStart_{o}_{k}")

    # Rate-based operational limits
    for k in cluster_vehicles:
        lhs = gp.quicksum(x[(i, j, k)] for i in locs_cl for j in locs_cl if i != j)
        rhs = gp.quicksum(y[k] * ds.VC.get((k, r), 0) * ds.MaxStops[r] for r in cluster_rates)
        model.addConstr(lhs <= rhs, name=f"MaxStops_{k}")

    for k in cluster_vehicles:
        for o in start_depot_cl:
            for i in delivery_locs_cl:
                dur = gp.quicksum(ds.VC.get((k, r), 0) * ds.MaxRouteDuration[r] for r in cluster_rates)
                model.addConstr(t[(i, k)] - t[(o, k)] <= y[k] * (dur - ds.ST.get(i, 0.0)),
                                name=f"MaxRouteDur_{o}_{i}_{k}")

    for i in delivery_locs_cl:
        for j in delivery_locs_cl:
            if i != j:
                for k in cluster_vehicles:
                    lhs = ds.Dist.get((i, j), 0.0) * x[(i, j, k)]
                    rhs = gp.quicksum(y[k] * ds.VC.get((k, r), 0) * ds.MaxDistBetweenStops_val[r] for r in cluster_rates)
                    model.addConstr(lhs <= rhs, name=f"MaxDistBtwStops_{i}_{j}_{k}")

    # Flow constraints and degree constraints
    for k in cluster_vehicles:
        for o in start_depot_cl:
            model.addConstr(gp.quicksum(x.get((i, o, k), 0) for i in locs_cl if i != o) == 0, name=f"NoInboundStart_{o}_{k}")
        for e in end_depot_cl:
            model.addConstr(gp.quicksum(x.get((i, e, k), 0) for i in locs_cl if i != e) == y[k], name=f"EndDepotRoute_{e}_{k}")

    for j in delivery_locs_cl:
        for k2 in cluster_vehicles:
            for m in cluster_orders:
                if LO_cl[(j, m)] == 1:
                    lhs = gp.quicksum(x.get((i, j, k2), 0) for i in locs_cl if i != j)
                    model.addConstr(lhs == z[(m, k2)], name=f"Inbound_{j}_{m}_{k2}")

    for k in cluster_vehicles:
        for o in start_depot_cl:
            model.addConstr(gp.quicksum(x.get((o, j, k), 0) for j in locs_cl if j != o) == y[k], name=f"Outbound_{o}_{k}")

    for k in cluster_vehicles:
        for i in end_depot_cl:
            model.addConstr(gp.quicksum(x.get((i, j, k), 0) for j in locs_cl if j != i) == 0, name=f"NoOutboundEnd_{i}_{k}")

    for i in delivery_locs_cl:
        for k2 in cluster_vehicles:
            for m in cluster_orders:
                if LO_cl[(i, m)] == 1:
                    lhs = gp.quicksum(x.get((i, j, k2), 0) for j in locs_cl if j != i)
                    model.addConstr(lhs == z[(m, k2)], name=f"Outbound_{i}_{m}_{k2}")

    for i in delivery_locs_cl:
        for k2 in cluster_vehicles:
            if (i, i, k2) in x:
                model.addConstr(x[(i, i, k2)] == 0, name=f"NoSelfLoop_{i}_{k2}")

    for i in delivery_locs_cl:
        for j in delivery_locs_cl:
            if i != j:
                for k2 in cluster_vehicles:
                    model.addConstr(x.get((i, j, k2), 0) + x.get((j, i, k2), 0) <= 1, name=f"NoRoundTrip_{i}_{j}_{k2}")

    for k2 in cluster_vehicles:
        for o in start_depot_cl:
            for e in end_depot_cl:
                if (o, e, k2) in x:
                    model.addConstr(x[(o, e, k2)] == 0, name=f"NoDirect_{o}_{e}_{k2}")

    # MTZ family
    for i in locs_cl:
        for j in locs_cl:
            if i != j:
                for k2 in cluster_vehicles:
                    lhs = u[(i, k2)] - u[(j, k2)] + Q_Total_local * x[(i, j, k2)] \
                          + (Q_Total_local - Q_loc[i] - Q_loc[j]) * x.get((j, i, k2), 0)
                    model.addConstr(lhs <= Q_Total_local - Q_loc[j], name=f"MTZ_{i}_{j}_{k2}")

    for j in locs_cl:
        for k2 in cluster_vehicles:
            lhs = u[(j, k2)] - gp.quicksum(Q_loc[i] * x.get((i, j, k2), 0) for i in locs_cl if i != j)
            model.addConstr(lhs >= Q_loc[j], name=f"MTZ2_{j}_{k2}")

    for i in locs_cl:
        for k2 in cluster_vehicles:
            lhs = u[(i, k2)] + gp.quicksum(Q_loc[j] * x.get((i, j, k2), 0) for j in locs_cl if j != i)
            model.addConstr(lhs <= Q_Total_local, name=f"MTZ3_{i}_{k2}")

    for k2 in cluster_vehicles:
        for o in start_depot_cl:
            for j in locs_cl:
                if o != j:
                    lhs = u[(j, k2)]
                    rhs = Q_loc[j] * x.get((o, j, k2), 0) + Q_Total_local * (1 - x.get((o, j, k2), 0))
                    model.addConstr(lhs <= rhs, name=f"MTZ4_{o}_{j}_{k2}")

    # Balance
    for i in delivery_locs_cl:
        for k2 in cluster_vehicles:
            lhs = gp.quicksum(x.get((i, j, k2), 0) for j in locs_cl if j != i)
            rhs = gp.quicksum(x.get((j, i, k2), 0) for j in locs_cl if j != i)
            model.addConstr(lhs == rhs, name=f"FlowBalance_{i}_{k2}")

    # Never-used vehicles forced off
    for k in sub_vehicles_union:
        if k not in veh_used_any:
            model.addConstr(y[k] == 0, name=f"VehNeverUsed_{k}")

    # Warm starts from subclusters
    for k in sub_vehicles_union:
        val_sum = sum(ds.Y_Param.get((s_c, k), 0.0) for s_c in subclusters)
        y[k].Start = 1 if val_sum > 0.5 else 0

    for s_c in subclusters:
        for (key, val) in ds.X_Param.items():
            if len(key) == 4 and key[0] == s_c:
                _, i, j, v_k = key
                if i in locs_cluster and j in locs_cluster and v_k in sub_vehicles_union and (i, j, v_k) in x:
                    x[(i, j, v_k)].Start = val
        for (key_z, valz) in ds.Z_Param.items():
            if len(key_z) == 3 and key_z[0] == s_c:
                _, m, v_k = key_z
                if m in sub_orders_union and v_k in sub_vehicles_union and (m, v_k) in z:
                    z[(m, v_k)].Start = valz
        for (key_q, valq) in ds.Q_Param.items():
            if len(key_q) == 4 and key_q[0] == s_c:
                _, m, p, v_k = key_q
                if m in sub_orders_union and p in cluster_products and v_k in sub_vehicles_union and (m, p, v_k) in q_var:
                    q_var[(m, p, v_k)].Start = valq

    # Solve
    model.optimize()

    # Write back results
    if model.Status == GRB.OPTIMAL:
        logger.info("Cluster-level (PHASE 1) => OPTIMAL, ObjVal=%s", getattr(model, "ObjVal", None))
        for i in locs_cluster:
            for j in locs_cluster:
                if i != j:
                    for k in sub_vehicles_union:
                        ds.X_Param[(wave_id, cluster_id, "PHASE1", i, j, k)] = x[(i, j, k)].X
        for k in sub_vehicles_union:
            ds.Y_Param[(wave_id, cluster_id, "PHASE1", k)] = int(round(y[k].X))
        for m in sub_orders_union:
            for k in sub_vehicles_union:
                ds.Z_Param[(wave_id, cluster_id, "PHASE1", m, k)] = int(round(z[(m, k)].X))
        for m in sub_orders_union:
            for p in cluster_products:
                for k in sub_vehicles_union:
                    ds.Q_Param[(wave_id, cluster_id, "PHASE1", m, p, k)] = q_var[(m, p, k)].X
    else:
        logger.info("Cluster-level (PHASE 1) => solver ended with status=%s", model.Status)

    logger.info("PHASE 1 done: wave=%s, cluster=%s", wave_id, cluster_id)
