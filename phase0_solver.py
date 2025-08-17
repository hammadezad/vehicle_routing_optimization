

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

def solve_phase0_subcluster(subcluster_id, wave_id, cluster_id, source_loc_id):
    """
    Builds a subcluster-level model with constraints (5.a..5.i, etc.).
    Solves, storing solution in X_Param, Y_Param, Z_Param, Q_Param with keys=(subcluster_id,...).
    (Identical to your existing approach.)
    """
    relevant_keys = [
        (o, r, w, c, sc, i, m)
        for (o, r, w, c, sc, i, m) in SLOC_RATE_WAVE_CLUST_SUBCLUST_DLOC_ORD
        if o == source_loc_id and w == wave_id and c == cluster_id and sc == subcluster_id
    ]
    logger.info(f"PHASE 0: source_loc_id={source_loc_id}, wave={wave_id}, cluster={cluster_id}, subcluster={subcluster_id}")



    # 1) Identify relevant sets
    logger.info(SLOC_RATE_WAVE_CLUST_SUBCLUST_DLOC_ORD)
    relevant_keys = [
        (o, r, w, c, sc, i, m)
        for (o, r, w, c, sc, i, m) in SLOC_RATE_WAVE_CLUST_SUBCLUST_DLOC_ORD
        if o == source_loc_id and w == wave_id and c == cluster_id and sc == subcluster_id
    ]
    sub_orders = set(k[6] for k in relevant_keys)
    sub_vehicles_list = [InputVehicles[k] for k in relevant_keys if InputVehicles[k] is not None]
    sub_vehicles = set(sub_vehicles_list)
    sub_rates = set(k[1] for k in relevant_keys)
    sub_dests = set(k[5] for k in relevant_keys)

    start_depot_sub = {l for l in LOCS if LocType.get(l, "") == "START_DEPOT" and l == source_loc_id}
    end_depot_sub = {l for l in LOCS if LocType.get(l, "") == "END_DEPOT"}
    delivery_locs_sub = sub_dests - start_depot_sub - end_depot_sub
    locs_sub = start_depot_sub.union(delivery_locs_sub).union(end_depot_sub)

    logger.info(f"  subcluster={subcluster_id} => #Orders={len(sub_orders)}, #Vehicles={len(sub_vehicles)}, #DeliveryLocs={len(delivery_locs_sub)}")
    # 2) Build partial sets of products for these orders
    sub_products = set()
    for (m,o,d,p) in ORD_SLOC_DLOC_PROD:
        if m in sub_orders:
            sub_products.add(p)

    # Build LO_sub, PO_sub, EDT_sub, LDT_sub for sub sets
    LO_sub = {}
    def get_originalD(m,p,i):
        # sum of InputOriginalD for that (m,?,i,p)
        s=0
        for (mm,oo,dd,pp) in ORD_SLOC_DLOC_PROD:
            if mm==m and pp==p and dd==i:
                s += InputOriginalD.get((mm,oo,dd,pp),0)
        return s

    for i in locs_sub:
        for m in sub_orders:
            val=0
            if i in start_depot_sub:
                val=1
            else:
                # check if i is a dest for that order with >0 qty
                for (mm,oo,dd,pp) in ORD_SLOC_DLOC_PROD:
                    if mm==m and dd==i:
                        if InputOriginalD.get((mm,oo,dd,pp),0)>0:
                            val=1
                            break
            LO_sub[(i,m)] = val

    PO_sub = {}
    for m in sub_orders:
        for p in sub_products:
            tot=0
            for (mm,oo,dd,pp) in ORD_SLOC_DLOC_PROD:
                if mm==m and pp==p:
                    tot += InputOriginalD.get((mm,oo,dd,pp),0)
            PO_sub[(m,p)] = 1 if tot>0 else 0

    EDT_sub = {}
    LDT_sub = {}
    for m in sub_orders:
        e_list=[]
        l_list=[]
        for (mm,oo,dd,pp) in ORD_SLOC_DLOC_PROD:
            if mm==m:
                e_list.append(InputEDT.get((mm,oo,dd,pp),0))
                l_list.append(InputLDT.get((mm,oo,dd,pp),0))
        EDT_sub[m] = max(e_list) if e_list else 0
        LDT_sub[m] = max(l_list) if l_list else 999999

    # build MaxTime local
    max_LDT = max(LDT_sub.values()) if LDT_sub else 0
    max_ST = max(ST.get(i,0) for i in locs_sub) if locs_sub else 0
    max_TT = 0
    for i in locs_sub:
        for j in locs_sub:
            if (i,j) in TT and TT[(i,j)]>max_TT:
                max_TT=TT[(i,j)]
    MaxTime_local = 2*(max_LDT + max_ST + max_TT)

    # quick min start time
    min_edt = min(EDT_sub.values()) if EDT_sub else 0
    max_tt_od=0
    for o in start_depot_sub:
        for d in delivery_locs_sub:
            if (o,d) in TT and TT[(o,d)]>max_tt_od:
                max_tt_od=TT[(o,d)]
    if start_depot_sub:
        min_st_o = max(ST[o] for o in start_depot_sub)
    else:
        min_st_o=0
    MinStartTime_local = min_edt - max_tt_od - min_st_o
    if MinStartTime_local<0:
        MinStartTime_local=0

    # max start time
    max_ldt_ = max(LDT_sub.values()) if LDT_sub else 0
    min_tt_od=999999
    for o in start_depot_sub:
        for d in delivery_locs_sub:
            if (o,d) in TT and TT[(o,d)]<min_tt_od:
                min_tt_od=TT[(o,d)]
    if min_tt_od==999999:
        min_tt_od=0
    if start_depot_sub:
        sm = min(ST[o] for o in start_depot_sub)
    else:
        sm=0
    MaxStartTime_local = max_ldt_ - min_tt_od - sm
    if MaxStartTime_local<0:
        MaxStartTime_local=MaxTime_local

    # 3) Build Gurobi model
    model_name = f"Phase0_{subcluster_id}"
    model = gp.Model(model_name)
    model.Params.OutputFlag=1
    # model.Params.TimeLimit = SolveTimePerSubCluster

    # Decision variables
    x = {}
    for i in locs_sub:
        for j in locs_sub:
            if i!=j:
                for k in sub_vehicles:
                    x[(i,j,k)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")

    y = {}
    for k in sub_vehicles:
        y[k] = model.addVar(vtype=GRB.BINARY, name=f"y_{k}")

    z = {}
    for m in sub_orders:
        for k in sub_vehicles:
            z[(m,k)] = model.addVar(vtype=GRB.BINARY, name=f"z_{m}_{k}")

    q_var = {}
    for m in sub_orders:
        for p in sub_products:
            for k in sub_vehicles:
                q_var[(m,p,k)] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"q_{m}_{p}_{k}")

    t = {}
    for i in locs_sub:
        for k in sub_vehicles:
            t[(i,k)] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"t_{i}_{k}")

    ns = {}
    for k in sub_vehicles:
        ns[k] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"ns_{k}")

    it_var = {}
    for i in locs_sub:
        for j in locs_sub:
            if i!=j:
                for k in sub_vehicles:
                    it_var[(i,j,k)] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"it_{i}_{j}_{k}")

    u = {}
    # Q[i] = 0 if start depot else 1
    Q_loc = {}
    for i in locs_sub:
        if i in start_depot_sub:
            Q_loc[i]=0
        else:
            Q_loc[i]=1
    Q_Total_local = sum(Q_loc[i] for i in locs_sub)
    for i in locs_sub:
        for k in sub_vehicles:
            u[(i,k)] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"u_{i}_{k}")

    model.update()

    # 4) Objective
    def fixed_cost(k):
        return gp.quicksum(FC[r]*VC.get((k,r),0)*y[k] for r in sub_rates)
    def mileage_cost(k):
        return gp.quicksum(MC[r]*VC.get((k,r),0)*Dist.get((i,j),0)*x[(i,j,k)]
                           for i in locs_sub for j in locs_sub if i!=j for r in sub_rates)
    def stoppage_cost(k):
        return gp.quicksum(SC[r]*VC.get((k,r),0)*ns[k] for r in sub_rates) # Check stppage cost - would it be for each vehicle once or would eb based on the number of stops 

    def discount_qty(k):
        # Phase 0 => CurrClustDiscount=0 => always 0
        return 0.0

    obj = gp.quicksum((fixed_cost(k)+mileage_cost(k)+stoppage_cost(k)) for k in sub_vehicles) \
         - UseFlexQty*gp.quicksum(discount_qty(k) for k in sub_vehicles)
    model.setObjective(obj, GRB.MINIMIZE)
    #minimize Total_Cost_Discounted: sum{k in VEHICLES}(fixed_cost[k] + mileage_cost[k] + stoppage_cost[k]) - UseFlexQty *(sum{k in VEHICLES}(discount_total_quantity[k]) );

    # 5) Constraints
    # 4.2 => ns[k] >= sum x[i,j,k] - sum( y[k]* VC[k,r]*Stops_Included[r] )
    for k in sub_vehicles:
        lhs = ns[k]
        rhs = gp.quicksum(x[(i,j,k)] for i in locs_sub for j in locs_sub if i!=j) \
              - gp.quicksum(y[k]*VC.get((k,r),0)*Stops_Included_In_Rate[r] for r in sub_rates)
        model.addConstr(lhs>=rhs, name=f"nsBound_{k}")

    # 5.a.1 => sum_k z[m,k] =1
    for m in sub_orders:
        model.addConstr(gp.quicksum(z[(m,k)] for k in sub_vehicles)==1, name=f"OrderAssign_{m}")

    # 5.a.2 => y[k]>=z[m,k]
    for m in sub_orders:
        for k in sub_vehicles:
            model.addConstr(y[k]>=z[(m,k)], name=f"VehUsed_{m}_{k}")

    # 5.a.3 => y[k]<= sum_m z[m,k]
    for k in sub_vehicles:
        lhs = y[k]
        rhs = gp.quicksum(z[(m,k)] for m in sub_orders)
        model.addConstr(lhs<=rhs, name=f"VehDeact_{k}")
    #########################################################
    ##########  Order quantity and Vehicle Capacity Constraints
    #########################################################
    # 5.b.1 => q[m,p,k]>= z[m,k] if PO[m,p]==1
    for m in sub_orders:
        for p in sub_products:
            if PO_sub.get((m,p),0)==1:
                for k in sub_vehicles:
                    model.addConstr(q_var[(m,p,k)]>= z[(m,k)]*1, name=f"LBQ_{m}_{p}_{k}")




    # 5.b.2 => q[m,p,k]>= z[m,k]*(origD - flexDown)
    for i in delivery_locs_sub:
        for m in sub_orders:
            if LO_sub[(i,m)]==1:
                for p in sub_products:
                    if PO_sub.get((m,p),0)==1:
                        for k in sub_vehicles:
                            origD_mp = get_originalD(m,p,i)
                            val_down = origD_mp - PO_sub[(m,p)]*min(ProductFlexDown.get(p,0),LocFlexDownPerOrder.get(i,0))*UseFlexQty
                            if val_down<0:
                                val_down=0
                            model.addConstr(q_var[(m,p,k)]>= z[(m,k)]*val_down, name=f"FlexDown_{m}_{p}_{k}_{i}")

    # 5.b.3 => q[m,p,k]<= z[m,k]*(origD + flexUp)
    for i in delivery_locs_sub:
        for m in sub_orders:
            if LO_sub[(i,m)]==1:
                for p in sub_products:
                    if PO_sub.get((m,p),0)==1:
                        for k in sub_vehicles:
                            origD_mp = get_originalD(m,p,i)
                            val_up = origD_mp + PO_sub[(m,p)]*min(ProductFlexUp.get(p,0), LocFlexUpPerOrder.get(i,0))*UseFlexQty
                            model.addConstr(q_var[(m,p,k)]<= z[(m,k)]*val_up, name=f"FlexUp_{m}_{p}_{k}_{i}")

    # 5.b.4 => sum_{m,p} q_var*(weight) <= sum_r(y[k]*VC[k,r]*WCap[r])
    for k in sub_vehicles:
        lhs = gp.quicksum(q_var[(m,p,k)]*WeightPerPallet.get(p,0) for m in sub_orders for p in sub_products)
        rhs = gp.quicksum(y[k]*VC.get((k,r),0)*WCap[r] for r in sub_rates)
        model.addConstr(lhs<=rhs, name=f"Cap_{k}")

    # 5.b.5 => location flex down
    for i in delivery_locs_sub:
        lhs = gp.quicksum(q_var[(m,p,k)] for m in sub_orders for p in sub_products for k in sub_vehicles if LO_sub[(i,m)]==1)
        total_orig_i=0
        for (mm,oo,dd,pp) in ORD_SLOC_DLOC_PROD:
            if mm in sub_orders and dd==i:
                total_orig_i += InputOriginalD.get((mm,oo,dd,pp),0)
        rhs = total_orig_i - LocFlexDownPerOrder.get(i,0)*UseFlexQty
        model.addConstr(lhs>=rhs, name=f"LocDown_{i}")

    # 5.b.6 => location flex up
    for i in delivery_locs_sub:
        lhs = gp.quicksum(q_var[(m,p,k)] for m in sub_orders for p in sub_products for k in sub_vehicles if LO_sub[(i,m)]==1)
        total_orig_i=0
        for (mm,oo,dd,pp) in ORD_SLOC_DLOC_PROD:
            if mm in sub_orders and dd==i:
                total_orig_i += InputOriginalD.get((mm,oo,dd,pp),0)
        rhs = total_orig_i + LocFlexUpPerOrder.get(i,0)*UseFlexQty
        model.addConstr(lhs<=rhs, name=f"LocUp_{i}")

    # 5.b.7 => cluster-level flex => OFF in Phase 0 => skipped because we dont need it for the sub-clusters

    #########################################################
    ###########    Order Delivery Time/Date Constraints
    #########################################################
    # 5.c.1 => t[j,k]>= EDT[m]*z[m,k] if LO_sub[j,m]=1
    for m in sub_orders:
        e_m = EDT_sub[m]
        for j in delivery_locs_sub:
            if LO_sub[(j,m)]==1:
                for k in sub_vehicles:
                    model.addConstr(t[(j,k)]>= e_m*z[(m,k)], name=f"EarlyDel_{j}_{m}_{k}")

    # 5.c.2 => t[j,k] + ST[j] <= LDT[m] + (1-z[m,k])*MaxTime_local
    for m in sub_orders:
        l_m = LDT_sub[m]
        for j in delivery_locs_sub:
            if LO_sub[(j,m)]==1:
                for k in sub_vehicles:
                    model.addConstr(t[(j,k)] + ST[j] <= l_m + (1-z[(m,k)])*MaxTime_local,
                                    name=f"LateDel_{j}_{m}_{k}")

    # 5.c.3 => t[i,k] + ST[i] + TT[i,j]+ it[i,j,k] = t[j,k] + (1-x[i,j,k])*MaxTime_local
    for i in locs_sub:
        for j in delivery_locs_sub:
            if i!=j:
                for k in sub_vehicles:
                    tt_ij = TT.get((i,j),0)
                    model.addConstr(t[(i,k)] + ST[i] + tt_ij + it_var[(i,j,k)] ==
                                    t[(j,k)] + (1-x[(i,j,k)])*MaxTime_local,
                                    name=f"ArrTime_{i}_{j}_{k}")

    # 5.c.4 => it[i,j,k] <= MaxIdleTimeBtwStops*x[i,j,k] + (1-x[i,j,k])*(2*MaxTime_local)
    if UseMaxIdleTimeConstr==1:
        for i in locs_sub:
            for j in delivery_locs_sub:
                if i!=j:
                    for k in sub_vehicles:
                        model.addConstr(
                            it_var[(i,j,k)] <= MaxIdleTimeBtwStops*x[(i,j,k)] + (1-x[(i,j,k)])*(2*MaxTime_local),
                            name=f"MaxIdle_{i}_{j}_{k}"
                        )

    # 5.c.5 => t[j,k]<= sum_m( LO_sub[j,m]*z[m,k] ) * MaxTime_local
    for j in delivery_locs_sub:
        for k in sub_vehicles:
            lhs = t[(j,k)]
            rhs = gp.quicksum(LO_sub[(j,m)]*z[(m,k)] for m in sub_orders)*MaxTime_local
            model.addConstr(lhs<=rhs, name=f"TimeZero_{j}_{k}")

    # 5.c.6 => t[o,k]>= MinStartTime_local*y[k]
    for k in sub_vehicles:
        for o in start_depot_sub:
            model.addConstr(t[(o,k)]>= MinStartTime_local*y[k],
                            name=f"MinRt_{o}_{k}")

    # 5.c.7 => t[o,k]<= MaxStartTime_local*y[k]
    for k in sub_vehicles:
        for o in start_depot_sub:
            model.addConstr(t[(o,k)]<= MaxStartTime_local*y[k],
                            name=f"MaxRt_{o}_{k}")
    #########################################################
    #######            Carrier Constraints
    #########################################################
    # 5.d.1 => sum{x[i,j,k]} <= sum_r( y[k]* VC[k,r]* MaxStops[r] )
    for k in sub_vehicles:
        lhs = gp.quicksum(x[(i,j,k)] for i in locs_sub for j in locs_sub if i!=j)
        rhs = gp.quicksum(y[k]*VC.get((k,r),0)*MaxStops[r] for r in sub_rates)
        model.addConstr(lhs<=rhs, name=f"MaxStops_{k}")

    # 5.d.2 => t[i,k] - t[o,k] <= y[k]*( sum_r(VC[k,r]*MaxRouteDuration[r]) - ST[i] )
    for k in sub_vehicles:
        for o in start_depot_sub:
            for i in delivery_locs_sub:
                dur = gp.quicksum(VC.get((k,r),0)*MaxRouteDuration[r] for r in sub_rates)
                model.addConstr(t[(i,k)] - t[(o,k)] <= y[k]*(dur - ST[i]),
                                name=f"MaxDur_{o}_{i}_{k}")

    # 5.d.3 => Dist[i,j]*x[i,j,k] <= sum_r( y[k]*VC[k,r]* MaxDistBetweenStops_val[r] )
    for i in delivery_locs_sub:
        for j in delivery_locs_sub:
            if i!=j:
                for k in sub_vehicles:
                    lhs = Dist.get((i,j),0)*x[(i,j,k)]
                    rhs = gp.quicksum(y[k]*VC.get((k,r),0)*MaxDistBetweenStops_val[r] for r in sub_rates)
                    model.addConstr(lhs<=rhs, name=f"MaxDistStop_{i}_{j}_{k}")
    #########################################################
    #######            Inbound Degree Constraints
    #########################################################
    # 5.e.1 => sum{i} x[i,o,k]=0 if o in start depot
    for k in sub_vehicles:
        for o in start_depot_sub:
            lhs = gp.quicksum(x.get((i,o,k),0) for i in locs_sub if i!=o)
            model.addConstr(lhs==0, name=f"InboundStart_{o}_{k}")

    # 5.e.2 => sum{i} x[i,e,k] = y[k] if e in end depot
    for k in sub_vehicles:
        for e in end_depot_sub:
            lhs = gp.quicksum(x.get((i,e,k),0) for i in locs_sub if i!=e)
            model.addConstr(lhs==y[k], name=f"InboundEnd_{e}_{k}")

    # 5.e.3 => sum{i} x[i,j,k]= z[m,k] if LO_sub[j,m]==1
    for j in delivery_locs_sub:
        for k2 in sub_vehicles:
            for m in sub_orders:
                if LO_sub[(j,m)]==1:
                    lhs = gp.quicksum(x.get((i,j,k2),0) for i in locs_sub if i!=j)
                    model.addConstr(lhs==z[(m,k2)], name=f"InboundDel_{j}_{m}_{k2}")

    # 5.f.1 => sum{j} x[o,j,k] = y[k] if o in start depot
    for k in sub_vehicles:
        for o in start_depot_sub:
            lhs = gp.quicksum(x.get((o,j,k),0) for j in locs_sub if j!=o)
            model.addConstr(lhs==y[k], name=f"OutboundStart_{o}_{k}")

    # 5.f.2 => sum{j} x[i,j,k] =0 if i in end depot
    for k in sub_vehicles:
        for i in end_depot_sub:
            lhs = gp.quicksum(x.get((i,j,k),0) for j in locs_sub if j!=i)
            model.addConstr(lhs==0, name=f"OutboundEnd_{i}_{k}")

    # 5.f.3 => sum{j} x[i,j,k] = z[m,k] if LO_sub[i,m]==1
    for i in delivery_locs_sub:
        for k2 in sub_vehicles:
            for m in sub_orders:
                if LO_sub[(i,m)]==1:
                    lhs = gp.quicksum(x.get((i,j,k2),0) for j in locs_sub if j!=i)
                    model.addConstr(lhs== z[(m,k2)], name=f"OutboundDel_{i}_{m}_{k2}")

    # 5.g.1 => x[i,i,k] = 0
    for i in delivery_locs_sub:
        for k2 in sub_vehicles:
            if (i,i,k2) in x:
                model.addConstr(x[(i,i,k2)] == 0, name=f"NoSelfLoop_{i}_{k2}")

    # 5.g.2 => x[i,j,k]+ x[j,i,k]<=1
    for i in delivery_locs_sub:
        for j in delivery_locs_sub:
            if i!=j:
                for k2 in sub_vehicles:
                    model.addConstr(x.get((i,j,k2),0)+x.get((j,i,k2),0)<=1,
                                    name=f"No2StepLoop_{i}_{j}_{k2}")

    # 5.g.3 => x[o,e,k]=0 for o in start depot, e in end depot
    for k2 in sub_vehicles:
        for o in start_depot_sub:
            for e in end_depot_sub:
                if (o,e,k2) in x:
                    model.addConstr(x[(o,e,k2)]==0, name=f"NoDirect_{o}_{e}_{k2}")

    # 5.h.1 => u[i,k] - u[j,k] + Q_Total*x[i,j,k] + (Q_Total-Q_loc[i]-Q_loc[j])*x[j,i,k] <= Q_Total - Q_loc[j]
    for i in locs_sub:
        for j in locs_sub:
            if i!=j:
                for k2 in sub_vehicles:
                    lhs = u[(i,k2)] - u[(j,k2)] + Q_Total_local*x[(i,j,k2)] \
                          + (Q_Total_local - Q_loc[i]-Q_loc[j])*x.get((j,i,k2),0)
                    rhs = Q_Total_local - Q_loc[j]
                    model.addConstr(lhs<=rhs, name=f"MTZ_{i}_{j}_{k2}")

    # 5.h.2 => u[j,k] - sum{i}(Q_loc[i]*x[i,j,k]) >= Q_loc[j]
    for j in locs_sub:
        for k2 in sub_vehicles:
            lhs = u[(j,k2)] - gp.quicksum(Q_loc[i]*x.get((i,j,k2),0) for i in locs_sub if i!=j)
            rhs = Q_loc[j]
            model.addConstr(lhs>=rhs, name=f"MTZ2_{j}_{k2}")

    # 5.h.3 => u[i,k] + sum{j}(Q_loc[j]*x[i,j,k]) <= Q_Total_local
    for i in locs_sub:
        for k2 in sub_vehicles:
            lhs = u[(i,k2)] + gp.quicksum(Q_loc[j]*x.get((i,j,k2),0) for j in locs_sub if j!=i)
            model.addConstr(lhs<=Q_Total_local, name=f"MTZ3_{i}_{k2}")

    # 5.h.4 => u[j,k] <= Q_loc[j]*x[o,j,k] + Q_Total_local*(1-x[o,j,k])
    for k2 in sub_vehicles:
        for o in start_depot_sub:
            for j in locs_sub:
                if o!=j:
                    lhs = u[(j,k2)]
                    rhs = Q_loc[j]*x.get((o,j,k2),0) + Q_Total_local*(1-x.get((o,j,k2),0))
                    model.addConstr(lhs<=rhs, name=f"MTZ4_{o}_{j}_{k2}")

    # 5.i.1 => sum{j} x[i,j,k] = sum{j} x[j,i,k] for i in delivery locs
    for i in delivery_locs_sub:
        for k2 in sub_vehicles:
            lhs = gp.quicksum(x.get((i,j,k2),0) for j in locs_sub if j!=i)
            rhs = gp.quicksum(x.get((j,i,k2),0) for j in locs_sub if j!=i)
            model.addConstr(lhs==rhs, name=f"InOutBal_{i}_{k2}")


    # add_arc_removal_constraints(model, sub_vehicles, start_depot_sub, delivery_locs_sub, x)
    # Solve
    logger.info(f"CHECKPOINT: solving subcluster model {model_name} ...")

    model.optimize()

    if model.Status == GRB.OPTIMAL:
        logger.info(f"  subcluster={subcluster_id} => OPTIMAL, ObjVal={model.ObjVal}")

        # --------------------- 1) ROUND THE SOLUTION -----------------------
        # Round x[i,j,k], y[k], z[m,k], q[m,p,k] so they are strictly 0 or 1 (or integral).
        for i in locs_sub:
            for j in locs_sub:
                if i != j:
                    for k in sub_vehicles:
                        X_Param[(wave_id, cluster_id, "PHASE0", i, j, k)] = x[(i, j, k)].X
                        # X_Param[(subcluster_id, i, j, k)] = x[(i, j, k)].X
                        

        for k in sub_vehicles:
            Y_Param[(wave_id, cluster_id, "PHASE0", k)] = y[k].X
            # Y_Param[(subcluster_id, k)] = y[k].X

        for m in sub_orders:
            for k in sub_vehicles:
                Z_Param[(wave_id, cluster_id, "PHASE0", m, k)] = z[(m, k)].X
                # Z_Param[(subcluster_id, m, k)] = z[(m, k)].X
            for p in sub_products:
                for k in sub_vehicles:
                    Q_Param[(subcluster_id, m, p, k)] = q_var[(m, p, k)].X

        # --------------------- 2) RECORD WHICH VEHICLES ARE USED -----------------------
        used_vehicles_sub = []
        for k in sub_vehicles:
            if y[k].X >= 1:     # means y[k] = 1 after rounding
                used_vehicles_sub.append(k)
            else:
                # y[k] ended up 0, so this vehicle is NOT used in this subcluster
                pass

        # *** NOW store final solutions in X_Param, Y_Param, etc. ***
        logger.info(f"CHECKPOINT: storing solutions for subcluster={subcluster_id} ...")
        for i in locs_sub:
            for j in locs_sub:
                if i != j:
                    for k in sub_vehicles:
                        X_Param[(subcluster_id, i, j, k)] = x[(i, j, k)].X

        for k in sub_vehicles:
            Y_Param[(subcluster_id, k)] = y[k].X
        for m in sub_orders:
            for k in sub_vehicles:
                Z_Param[(subcluster_id, m, k)] = z[(m, k)].X
            for p in sub_products:
                for k in sub_vehicles:
                    Q_Param[(wave_id, cluster_id, "PHASE0", m, p, k)] = q_var[(m, p, k)].X
                    # Q_Param[(subcluster_id, m, p, k)] = q_var[(m, p, k)].X

        # We also want to RETURN that used_vehicles_sub list so Phase1 can see it
        return used_vehicles_sub

    else:
        logger.info(f"  subcluster={subcluster_id} => solver ended with status={model.Status}")
        return []
    




    logger.info(f"PHASE 0: Subcluster {subcluster_id}, wave={wave_id}, cluster={cluster_id}")