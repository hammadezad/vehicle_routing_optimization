import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import csv
import math
from datetime import datetime
import os
###############################################################################
# SCENARIO / CONTROL PARAMETERS
###############################################################################

# For Phase 0, we do no cluster-level flex limit or discount:
UseFlexQty = 1
UseFlexUpPercentLimit = 0
MaxFlexUpPercent_CurrentCluster = 100
CurrClustDiscount = 0
MaxFlexUpPercent_AllClusters = {} 

# If you want the idle time constraint turned on:
UseMaxIdleTimeConstr = 0
MaxIdleTimeBtwStops = 1  # e.g. 120 minutes

SolveTimePerSubCluster = 30
SolveTimeForCluster = 60

GlobalMaxFlexUpPercent = 100  # (E.g. from the scenario?)

###############################################################################
# GLOBAL DATA STRUCTURES
###############################################################################
PRD = set()
ORDERS = set()
LOCS = set()
ORD_SLOC_DLOC_PROD = set()
ORIGIN_DEST = set()
SLOC_RATE = set()
ORD_CUSTPO_PRD = set()
SLOC_RATE_WAVE_CLUST_SUBCLUST_DLOC_ORD = set()

# FILTERED_SUB_CLUSTERS = set()

InputVehicles = {}
InputOriginalD = {}
InputEDT = {}
InputLDT = {}

InputFC = {}
InputMC = {}
InputSC = {}
InputWCap = {}
InputStops_Included_In_Rate = {}
InputMaxStops = {}
InputMaxRouteDuration = {}
InputMaxDistBetweenStops = {}

LocFlexDownPerOrder = {}
LocFlexUpPerOrder = {}
LocType = {}
ST = {}
Dist = {}
TT = {}

WeightPerPallet = {}
CasesPerPallet = {}
ProductFlexDown = {}
ProductFlexUp = {}
ProductDescription = {}
LocName = {}
LocCity = {}
LocState = {}
LocZipCode = {}

# Derived dictionaries for building constraints
FC = {}
MC = {}
SC = {}
WCap = {}
Stops_Included_In_Rate = {}
MaxStops = {}
MaxRouteDuration = {}
MaxDistBetweenStops_val = {}
VC = {}   # (vehicle, rate) -> 0/1

# We store solutions from phases here
# Phase 0 => (subcluster_id, i, j, k) etc.
# Phase 1 => ("PHASE1", i, j, k) etc.
# Phase 2 => ("PHASE2", i, j, k) etc.
X_Param = {}
Y_Param = {}
Z_Param = {}
Q_Param = {}

###############################################################################
# 1) LOAD DATA
###############################################################################
def load_data_for_scenario(dataframes, scenario_id):
    print("CHECKPOINT: Starting load_data_from_dataframes...")
    global PRD, LOCS, ORD_SLOC_DLOC_PROD, ORIGIN_DEST, SLOC_RATE, ORDERS
    global ORD_CUSTPO_PRD, SLOC_RATE_WAVE_CLUST_SUBCLUST_DLOC_ORD
    global InputVehicles, InputOriginalD, InputEDT, InputLDT,LocFlexDownPerOrder, LocFlexUpPerOrder
    global LocType, ST, Dist, TT
    global InputFC, InputMC, InputSC, InputWCap
    global InputStops_Included_In_Rate, InputMaxStops, InputMaxRouteDuration
    global WeightPerPallet, CasesPerPallet, ProductFlexDown, ProductFlexUp
    global InputMaxDistBetweenStops
    global LocName, LocCity, LocState, LocZipCode

    PRD.clear()
    ORDERS.clear()
    LOCS.clear()
    ORD_SLOC_DLOC_PROD.clear()
    ORIGIN_DEST.clear()
    SLOC_RATE.clear()
    ORD_CUSTPO_PRD.clear()
    SLOC_RATE_WAVE_CLUST_SUBCLUST_DLOC_ORD.clear()

    InputVehicles.clear()
    InputOriginalD.clear()
    InputEDT.clear()
    InputLDT.clear()
    LocFlexDownPerOrder.clear()
    LocFlexUpPerOrder.clear()
    LocType.clear()
    ST.clear()
    Dist.clear()
    TT.clear()
    InputFC.clear()
    InputMC.clear()
    InputSC.clear()
    InputWCap.clear()
    InputStops_Included_In_Rate.clear()
    InputMaxStops.clear()
    InputMaxRouteDuration.clear()
    InputMaxDistBetweenStops.clear()

    WeightPerPallet.clear()
    CasesPerPallet.clear()
    ProductFlexDown.clear()
    ProductFlexUp.clear()
    ProductDescription.clear()
    LocName.clear()
    LocCity.clear()
    LocState.clear() 
    LocZipCode.clear()


    if "XXNBL_PRODUCT_PARAMS" in dataframes:
        df = dataframes["XXNBL_PRODUCT_PARAMS"]
        df = df[df["SCENARIO_ID"] == scenario_id]
        for row in df.itertuples(index=False):
            p = getattr(row, "PRODUCT_ID")
            PRD.add(p)
            ProductDescription[p] = getattr(row, "PRODUCT_DESCRIPTION", "")
            WeightPerPallet[p] = getattr(row, "WEIGHT_PER_PALLET", 0.0)
            CasesPerPallet[p] = getattr(row, "CASES_PER_PALLET", 0)
            ProductFlexDown[p] = getattr(row, "FLEX_DOWN_PALLETS", 0)
            ProductFlexUp[p] = getattr(row, "FLEX_UP_PALLETS", 0)

    if "XXNBL_LOCATION_PARAMS" in dataframes:
        df = dataframes["XXNBL_LOCATION_PARAMS"]
        df = df[df["SCENARIO_ID"] == scenario_id]
        for row in df.itertuples(index=False):
            loc = getattr(row, "LOCATION_ID")
            LOCS.add(loc)
            LocName[loc] = getattr(row, "LOCATION_NAME", "")
            LocType[loc] = getattr(row, "LOCATION_TYPE", "")
            ST[loc] = getattr(row, "SERVICE_TIME", 0.0)
            LocFlexDownPerOrder[loc] = getattr(row, "FLEX_DOWN_PALLETS_PER_ORDER", 0)
            LocFlexUpPerOrder[loc] = getattr(row, "FLEX_UP_PALLETS_PER_ORDER", 0)
            LocCity[loc] = getattr(row, "CITY", "")
            LocState[loc] = getattr(row, "STATE", "")
            LocZipCode[loc] = getattr(row, "ZIP_CODE", "")

    if "XXNBL_ORDER_PARAMS" in dataframes:
        df = dataframes["XXNBL_ORDER_PARAMS"]
        df = df[df["SCENARIO_ID"] == scenario_id]
        for row in df.itertuples(index=False):
            m = getattr(row, "ORDER_ID")
            ORDERS.add(m)
            m = getattr(row, "ORDER_ID")
            o = getattr(row, "SOURCE_LOCATION_ID")
            d = getattr(row, "DESTINATION_LOCATION_ID")
            p = getattr(row, "PRODUCT_ID", "NO_PRODUCT")
            ORD_SLOC_DLOC_PROD.add((m, o, d, p))

            qty = getattr(row, "ORIGINAL_QTY_PALLETS", 0.0)
            InputOriginalD[(m, o, d, p)] = qty

            e = getattr(row, "EDT", 0)
            l = getattr(row, "LDT", 0)
            InputEDT[(m, o, d, p)] = e
            InputLDT[(m, o, d, p)] = l

            if hasattr(row, "CUST_PO"):
                cpo = getattr(row, "CUST_PO")
                ORD_CUSTPO_PRD.add((m, cpo, p))

    if "XXNBL_TIME_DIST_PARAMS" in dataframes:
        df = dataframes["XXNBL_TIME_DIST_PARAMS"]
        df = df[df["SCENARIO_ID"] == scenario_id]
        for row in df.itertuples(index=False):
            orig = getattr(row, "ORIGIN")
            dest = getattr(row, "DESTINATION")
            ORIGIN_DEST.add((orig, dest))
            Dist[(orig, dest)] = getattr(row, "DISTANCE", 0.0)
            TT[(orig, dest)] = getattr(row, "TRANSIT_TIME", 0.0)

    if "XXNBL_RATE_PARAMS" in dataframes:
        df = dataframes["XXNBL_RATE_PARAMS"]
        df = df[df["SCENARIO_ID"] == scenario_id]
        for row in df.itertuples(index=False):
            sloc = getattr(row, "SOURCE_LOCATION_ID")
            rr = getattr(row, "RATE_ID")
            SLOC_RATE.add((sloc, rr))
            InputFC[(sloc, rr)] = getattr(row, "FIXED_COST", 0.0)
            InputMC[(sloc, rr)] = getattr(row, "PER_MILE_COST", 0.0)
            InputSC[(sloc, rr)] = getattr(row, "PER_STOP_COST", 0.0)
            InputWCap[(sloc, rr)] = getattr(row, "VEHICLE_CAPACITY", 0.0)
            InputStops_Included_In_Rate[(sloc, rr)] = getattr(row, "STOPS_INCLUDED_IN_RATE", 0)
            InputMaxStops[(sloc, rr)] = getattr(row, "MAX_STOPS", 0)
            InputMaxRouteDuration[(sloc, rr)] = getattr(row, "MAX_ROUTE_DURATION", 0.0)
            InputMaxDistBetweenStops[(sloc, rr)] = getattr(row, "MAX_DISTANCE_BTW_STOPS", 0.0)

    if "XXNBL_CLUSTER_PARAMS" in dataframes:
        df = dataframes["XXNBL_CLUSTER_PARAMS"]
        df = df[df["SCENARIO_ID"] == scenario_id]
        for row in df.itertuples(index=False):
            o = getattr(row, "SOURCE_LOCATION_ID")
            r = getattr(row, "RATE_ID")
            w = getattr(row, "WAVE_ID")
            c = getattr(row, "CLUSTER_ID")
            s_c = getattr(row, "SUB_CLUSTER_ID")
            i = getattr(row, "DESTINATION_LOCATION_ID")
            m = getattr(row, "ORDER_ID")
            SLOC_RATE_WAVE_CLUST_SUBCLUST_DLOC_ORD.add((o, r, w, c, s_c, i, m))
            InputVehicles[(o, r, w, c, s_c, i, m)] = getattr(row, "VEHICLE_ID")

    print("CHECKPOINT: load_data_from_dataframes done.")



###############################################################################
# 2) BUILD RATE DICTS
###############################################################################
def build_global_rate_dicts():
    print("CHECKPOINT: build_global_rate_dicts start...")
    global FC, MC, SC, WCap, Stops_Included_In_Rate
    global MaxStops, MaxRouteDuration, MaxDistBetweenStops_val, VC

    FC.clear()
    MC.clear()
    SC.clear()
    WCap.clear()
    Stops_Included_In_Rate.clear()
    MaxStops.clear()
    MaxRouteDuration.clear()
    MaxDistBetweenStops_val.clear()
    VC.clear()
    start_depots = {loc for loc in LOCS if LocType.get(loc, "") == "START_DEPOT"}
 
    all_rate_ids = set(rr for (_, rr) in SLOC_RATE)
    for r in all_rate_ids:
        # (2) Replace the old line with this one:
        matching_slocs = [s for (s, rr) in SLOC_RATE if rr == r and s in start_depots]

        fc_val = max(InputFC.get((sl, r), 0) for sl in matching_slocs) if matching_slocs else 0
        mc_val = max(InputMC.get((sl, r), 0) for sl in matching_slocs) if matching_slocs else 0
        sc_val = max(InputSC.get((sl, r), 0) for sl in matching_slocs) if matching_slocs else 0
        wcap_val = max(InputWCap.get((sl, r), 0) for sl in matching_slocs) if matching_slocs else 999999
        stops_incl = max(InputStops_Included_In_Rate.get((sl, r), 0) for sl in matching_slocs) if matching_slocs else 0
        mx_stops = max(InputMaxStops.get((sl, r), 0) for sl in matching_slocs) if matching_slocs else 999
        mx_dur = max(InputMaxRouteDuration.get((sl, r), 0) for sl in matching_slocs) if matching_slocs else 999999
        mx_dist = max(InputMaxDistBetweenStops.get((sl, r), 0) for sl in matching_slocs) if matching_slocs else 999999

        FC[r] = fc_val
        MC[r] = mc_val
        SC[r] = sc_val
        WCap[r] = wcap_val
        Stops_Included_In_Rate[r] = stops_incl
        MaxStops[r] = mx_stops
        MaxRouteDuration[r] = mx_dur
        MaxDistBetweenStops_val[r] = mx_dist

    VC = {}
    # Track vehicles to ensure each is assigned to only one rate
    processed_vehicles = set()
    for (o, r, w, c, s_c, i, m), veh in InputVehicles.items():
        if veh is not None:
            if veh not in processed_vehicles:
                # Assign the vehicle to this rate
                VC[(veh, r)] = 1
                processed_vehicles.add(veh)
            else:
                # Check if the vehicle is already assigned to a different rate
                existing_rate = next((rr for (v, rr) in VC if v == veh), None)
                if existing_rate != r:
                    raise ValueError(f"Vehicle {veh} is assigned to multiple rates: {existing_rate} and {r}")

    # Initialize all other (veh, r) combinations to 0
    for veh in processed_vehicles:
        for r in all_rate_ids:
            if (veh, r) not in VC:
                VC[(veh, r)] = 0

    print("CHECKPOINT: build_global_rate_dicts done.")


###############################################################################
# 4) PHASE 0: Solve Sub-Clusters
###############################################################################
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
    print(f"PHASE 0: source_loc_id={source_loc_id}, wave={wave_id}, cluster={cluster_id}, subcluster={subcluster_id}")



    # 1) Identify relevant sets
    print(SLOC_RATE_WAVE_CLUST_SUBCLUST_DLOC_ORD)
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

    print(f"  subcluster={subcluster_id} => #Orders={len(sub_orders)}, #Vehicles={len(sub_vehicles)}, #DeliveryLocs={len(delivery_locs_sub)}")
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
    print(f"CHECKPOINT: solving subcluster model {model_name} ...")

    model.optimize()

    if model.Status == GRB.OPTIMAL:
        print(f"  subcluster={subcluster_id} => OPTIMAL, ObjVal={model.ObjVal}")

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
        print(f"CHECKPOINT: storing solutions for subcluster={subcluster_id} ...")
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
        print(f"  subcluster={subcluster_id} => solver ended with status={model.Status}")
        return []
    




    print(f"PHASE 0: Subcluster {subcluster_id}, wave={wave_id}, cluster={cluster_id}")


###############################################################################
# 5) PHASE 1: Solve Single Cluster
###############################################################################
def solve_phase1_for_cluster(wave_id, cluster_id, source_loc_id, used_vehicles_dict):
    """
    Gathers union of subclusters for the cluster, does one big cluster-level model,
    warm-start from Phase 0 solutions, then solves storing final cluster solution in
    X_Param, Y_Param, Z_Param, Q_Param with keys=("PHASE1", i, j, k), etc.
    (Same as your existing code.)
    """
    print(f"PHASE 1: wave={wave_id}, cluster={cluster_id}, source_loc_id={source_loc_id}")
    print("\n=== PHASE 1: cluster-level solve start ===")
    subclusters = set()
    for (o, r, w, c, s_c, i, m) in SLOC_RATE_WAVE_CLUST_SUBCLUST_DLOC_ORD:
        if w == wave_id and c == cluster_id:
            subclusters.add(s_c)    

    # 1) Union all subcl data
    sub_orders_union = set()
    sub_vehicles_union = set()
    sub_rates_union = set()
    sub_locs_union = set()
    used_vehicles_union = set()
    for sc in subclusters:
        if sc in used_vehicles_dict:
            used_vehicles_union |= set(used_vehicles_dict[sc])  # union them
        else:
            # if we didn't solve or it was empty
            pass
            
    sub_vehicles_union = used_vehicles_union




    for s_c in subclusters:
        relevant_keys = [
            (o, r, w, c, sc, i, m)
            for (o, r, w, c, sc, i, m) in SLOC_RATE_WAVE_CLUST_SUBCLUST_DLOC_ORD
            if w == wave_id and c == cluster_id and sc == s_c
        ]
        # union them into your sub_orders_union, etc.
        for (o, r, w, c, sc, i, m) in relevant_keys:
            sub_orders_union.add(m)
            these_veh = [InputVehicles[k] for k in relevant_keys if InputVehicles[k] is not None]
            sub_vehicles_union |= set(these_veh)
            sub_rates_union |= set(k[1] for k in relevant_keys)
            sub_locs_union |= set(k[5] for k in relevant_keys)

    # plus start depot, end depot
    start_depot = {l for l in LOCS if LocType.get(l,"")=="START_DEPOT" and l==source_loc_id}
    end_depot = {l for l in LOCS if LocType.get(l,"")=="END_DEPOT"}
    delivery_locs = sub_locs_union - start_depot - end_depot
    locs_cluster = start_depot.union(delivery_locs).union(end_depot)

    # 2) Figure out which vehicles were used in ANY subcluster
    veh_used_any = set()
    for s_c in subclusters:
        for k in sub_vehicles_union:
            val = Y_Param.get((s_c, k), 0.0)
            if val>0.5:
                veh_used_any.add(k)

    # 3) Gather product set for these orders
    cluster_products = set()
    for (m,o,d,p) in ORD_SLOC_DLOC_PROD:
        if m in sub_orders_union:
            cluster_products.add(p)
    ###############################################################################
    # A) BUILD CLUSTER-LEVEL LO_cl, PO_cl, EDT_cl, LDT_cl, ETC.
    ###############################################################################

    # We'll define:
    #   - LO_cl[(i,m)] => 1 if i is start depot or if i has >0 demand for order m
    #   - PO_cl[(m,p)] => 1 if order m has product p with >0 total quantity
    #   - EDT_cl[m], LDT_cl[m] => earliest & latest times for order m
    #   - MaxTime_cl, MinStartTime_cl, MaxStartTime_cl, etc.

    # We'll also define a helper to sum InputOriginalD for (m,p,i)
    def get_originalD_cluster(m, p, i):
        total = 0
        for (mm, oo, dd, pp) in ORD_SLOC_DLOC_PROD:
            if mm == m and pp == p and dd == i:
                total += InputOriginalD.get((mm, oo, dd, pp), 0.0)
        return total

    # 1) LO_cl[(i,m)]
    LO_cl = {}
    for i in locs_cluster:
        for m in sub_orders_union:
            val = 0
            if i in start_depot:
                val = 1
            else:
                # check if i is a destination for that order m with >0 original qty
                for (mm, oo, dd, pp) in ORD_SLOC_DLOC_PROD:
                    if mm == m and dd == i:
                        if InputOriginalD.get((mm, oo, dd, pp), 0) > 0:
                            val = 1
                            break
            LO_cl[(i, m)] = val

    # 2) PO_cl[(m,p)]
    PO_cl = {}
    for m in sub_orders_union:
        for p in cluster_products:
            tot = 0
            for (mm, oo, dd, pp) in ORD_SLOC_DLOC_PROD:
                if mm == m and pp == p:
                    tot += InputOriginalD.get((mm, oo, dd, pp), 0.0)
            PO_cl[(m, p)] = 1 if tot > 0 else 0

    # 3) EDT_cl[m], LDT_cl[m]
    EDT_cl = {}
    LDT_cl = {}
    for m in sub_orders_union:
        e_candidates = []
        l_candidates = []
        for (mm, oo, dd, pp) in ORD_SLOC_DLOC_PROD:
            if mm == m:
                e_candidates.append(InputEDT.get((mm, oo, dd, pp), 0.0))
                l_candidates.append(InputLDT.get((mm, oo, dd, pp), 0.0))
        EDT_cl[m] = max(e_candidates) if e_candidates else 0
        LDT_cl[m] = max(l_candidates) if l_candidates else 999999

    # 4) Compute MaxTime_cl
    max_LDT = max(LDT_cl.values()) if LDT_cl else 0
    max_ST = 0
    if locs_cluster:
        max_ST = max(ST.get(loc, 0) for loc in locs_cluster)
    max_TT = 0
    for i in locs_cluster:
        for j in locs_cluster:
            if (i,j) in TT:
                if TT[(i,j)] > max_TT:
                    max_TT = TT[(i,j)]
    MaxTime_cl = 2*(max_LDT + max_ST + max_TT)

    # 5) Compute MinStartTime_cl
    min_edt = min(EDT_cl.values()) if EDT_cl else 0
    max_tt_od = 0
    for o in start_depot:
        for d in delivery_locs:
            if (o, d) in TT:
                if TT[(o,d)] > max_tt_od:
                    max_tt_od = TT[(o,d)]
    if start_depot:
        max_st_o = max(ST[o] for o in start_depot)
    else:
        max_st_o = 0
    MinStartTime_cl = min_edt - max_tt_od - max_st_o
    if MinStartTime_cl < 0:
        MinStartTime_cl = 0

    # 6) Compute MaxStartTime_cl
    max_ldt_ = max(LDT_cl.values()) if LDT_cl else 0
    min_tt_od = 999999
    for o in start_depot:
        for d in delivery_locs:
            if (o, d) in TT and TT[(o,d)] < min_tt_od:
                min_tt_od = TT[(o,d)]
    if min_tt_od == 999999:
        min_tt_od = 0
    if start_depot:
        min_st_o = min(ST[o] for o in start_depot)
    else:
        min_st_o = 0
    MaxStartTime_cl = max_ldt_ - min_tt_od - min_st_o
    if MaxStartTime_cl < 0:
        MaxStartTime_cl = MaxTime_cl

    # 7) Q_loc[i] and Q_Total_local
    # Q_loc = {}
    # for i in locs_cluster:
    #     if i in start_depot:
    #         Q_loc[i] = 0
    #     else:
    #         Q_loc[i] = 1
    # Q_Total_local = sum(Q_loc[i] for i in locs_cluster)

    print("CHECKPOINT: Built LO_cl, PO_cl, EDT_cl, LDT_cl, etc. for Phase 1.")


    # 4) Build model
    model_name = "Phase1_Cluster"
    model = gp.Model(model_name)
    model.Params.OutputFlag=1
    # model.Params.TimeLimit=SolveTimeForCluster

    # x = {}
    # for i in locs_cluster:
    #     for j in locs_cluster:
    #         if i!=j:
    #             for k in sub_vehicles_union:
    #                 x[(i,j,k)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")

    x = {}
    for i in locs_cluster:
        for j in locs_cluster:
            if i!=j:
                for k in sub_vehicles_union:
                    var = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")
                    key = (wave_id, cluster_id, "PHASE0", i, j, k)
                    if key in X_Param:
                        var.Start = X_Param[key]
                    x[(i, j, k)] = var

    # y = {}
    # for k in sub_vehicles_union:
    #     y[k] = model.addVar(vtype=GRB.BINARY, name=f"y_{k}")

    y = {}
    for k in sub_vehicles_union:
        var = model.addVar(vtype=GRB.BINARY, name=f"y_{k}")
        key = (wave_id, cluster_id, "PHASE0", k)
        if key in Y_Param:
            var.Start = Y_Param[key]
        y[k] = var


    # z = {}
    # for m in sub_orders_union:
    #     for k in sub_vehicles_union:
    #         z[(m,k)] = model.addVar(vtype=GRB.BINARY, name=f"z_{m}_{k}")

    z = {}
    for m in sub_orders_union:
        for k in sub_vehicles_union:
            var = model.addVar(vtype=GRB.BINARY, name=f"z_{m}_{k}")
            key = (wave_id, cluster_id, "PHASE0", m, k)
            if key in Z_Param:
                var.Start = Z_Param[key]
            z[(m, k)] = var

    # q_var = {}
    # for m in sub_orders_union:
    #     for p in cluster_products:
    #         for k in sub_vehicles_union:
    #             q_var[(m,p,k)] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"q_{m}_{p}_{k}")
    q_var = {}
    for m in sub_orders_union:
        for p in cluster_products:
            for k in sub_vehicles_union:
                var = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"q_{m}_{p}_{k}")
                key = (wave_id, cluster_id, "PHASE0", m, p, k)
                if key in Q_Param:
                    var.Start = Q_Param[key]
                q_var[(m, p, k)] = var


    t = {}
    for i in locs_cluster:
        for k in sub_vehicles_union:
            t[(i,k)] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"t_{i}_{k}")
    ns = {}
    for k in sub_vehicles_union:
        ns[k] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"ns_{k}")
    it_var = {}
    for i in locs_cluster:
        for j in locs_cluster:
            if i!=j:
                for k in sub_vehicles_union:
                    it_var[(i,j,k)] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"it_{i}_{j}_{k}")

    u = {}
    Q_loc = {}
    for i in locs_cluster:
        if i in start_depot:
            Q_loc[i]=0
        else:
            Q_loc[i]=1
    Q_Total_local = sum(Q_loc[i] for i in locs_cluster)
    for i in locs_cluster:
        for k in sub_vehicles_union:
            u[(i,k)] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"u_{i}_{k}")

    model.update()
    model.Params.StartNodeLimit = 1

    # 5) Objective
    def fixed_cost(k):
        return gp.quicksum(FC[r]*VC.get((k,r),0)*y[k] for r in sub_rates_union)
    def mileage_cost(k):
        return gp.quicksum(MC[r]*VC.get((k,r),0)*Dist.get((i,j),0)*x[(i,j,k)]
                           for i in locs_cluster for j in locs_cluster if i!=j for r in sub_rates_union)
    def stoppage_cost(k):
        return gp.quicksum(SC[r]*VC.get((k,r),0)*ns[k] for r in sub_rates_union)
    # def discount_total(k):
    #     # still no cluster-level discount in phase 1
    #     return 0.0

    


    # 6) Constraints (same approach as Phase 0, but for union sets)
    # Example definitions for Phase 1:
    cluster_orders = sub_orders_union
    cluster_vehicles = sub_vehicles_union
    cluster_rates = sub_rates_union
    start_depot_cl = start_depot
    end_depot_cl = end_depot
    delivery_locs_cl = delivery_locs
    locs_cl = start_depot_cl.union(delivery_locs_cl).union(end_depot_cl)

    def discount_total(k):
        return CurrClustDiscount * gp.quicksum(
            q_var[(m, p, k)] for m in cluster_orders for p in cluster_products
        )

    obj = gp.quicksum((fixed_cost(k)+mileage_cost(k)+stoppage_cost(k)) for k in sub_vehicles_union) \
         - UseFlexQty*gp.quicksum(discount_total(k) for k in sub_vehicles_union)
    model.setObjective(obj, GRB.MINIMIZE)

    ###############################################################################
    # 6) Constraints (same approach as Phase 0, but for the cluster-level sets)
    ###############################################################################

    #
    # 4.2 => ns[k] >= sum{x[i,j,k]} - sum{ y[k]*VC[k,r]*Stops_Included_In_Rate[r] }
    #
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
    # ------------------------------------------------------------------
    # 5.b.7  cluster-aggregate flex-up limit (OFF for Phase-1)
    # ------------------------------------------------------------------
    if UseFlexUpPercentLimit == 1:
        total_ship = gp.quicksum(q_var[(m, p, k)]
                                 for m in cluster_orders
                                 for p in cluster_products
                                 for k in cluster_vehicles)

        total_orig = 0
        for m in cluster_orders:
            for p in cluster_products:
                # sum of original D across *ALL* delivery locations
                for (mm, oo, dd, pp) in ORD_SLOC_DLOC_PROD:
                    if mm == m and pp == p:
                        total_orig += InputOriginalD[(mm, oo, dd, pp)]

        rhs = (1 + MaxFlexUpPercent_AllClusters.get(cluster_id, 100) / 100) * total_orig
        model.addConstr(total_ship <= rhs, name="ClusterFlexUp")


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

    # 5.g.1 => x[i,i,k] = 0
    for i in delivery_locs_cl:
        for k2 in cluster_vehicles:
            if (i,i,k2) in x:
                model.addConstr(x[(i,i,k2)] == 0, name=f"NoSelfLoop_{i}_{k2}")

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


    #
    # 5.i.2 => "ArcRemovalOnLocationPairs" is typically optional; skip if not needed.
    ############################################################################
    # >>> ARC REMOVAL SNIPPET (3B) FOR CLUSTER-LEVEL <<<
    # Same idea as sub-cluster. We'll do an example:
    # for i in delivery_locs_cl:
    #     for j in delivery_locs_cl:
    #         if i != j:
    #             for k2 in cluster_vehicles:
    #                 # EXAMPLE condition for removing arc:
    #                 direct_oj = Dist.get((source_loc_id, j), 999999)
    #                 dist_ij   = Dist.get((i,j), 999999)
    #                 if direct_oj < 0.7 * dist_ij:
    #                     condA = True

    #                 if condA:
    #                     model.addConstr(
    #                         x[(i,j,k2)] == 0,
    #                         name=f"ArcRemoval_{i}_{j}_{k2}"
    #                     )

    ###############################################################################
    # END PHASE 1 CONSTRAINTS
    ###############################################################################
        

    # For demonstration, let's fix y[k] = 0 if never used
    for k in sub_vehicles_union:
        if k not in veh_used_any:
            model.addConstr(y[k]==0, name=f"VehNeverUsed_{k}")

    # 7) Provide warm start from Phase 0 subcluster solutions
    # We'll do .Start values for x, y, z, q, not fully fix
    for k in sub_vehicles_union:
        # if used in any subcluster, let's start y[k].Start=1
        val_sum=0
        for s_c in subclusters:
            val_sum += Y_Param.get((s_c, k), 0.0)
        if val_sum>0.5:
            y[k].Start = 1
        else:
            y[k].Start = 0
    # For x, z, q, we gather them from subcluster solutions
    for s_c in subclusters:
        # x => X_Param[(s_c, i, j, k)]
        for (key,val) in X_Param.items():
            # key might be (subcluster_id, i, j, v_k)
            if len(key)==4 and key[0]==s_c:
                (_, i, j, v_k) = key
                # if i,j in locs_cluster, v_k in sub_vehicles_union
                if i in locs_cluster and j in locs_cluster and v_k in sub_vehicles_union:
                    if (i,j,v_k) in x:
                        x[(i,j,v_k)].Start = val
        for (key_z,valz) in Z_Param.items():
            # key_z might be (s_c, m, v_k)
            if len(key_z)==3 and key_z[0]==s_c:
                (_, m, v_k) = key_z
                if m in sub_orders_union and v_k in sub_vehicles_union:
                    if (m,v_k) in z:
                        z[(m,v_k)].Start = valz
        for (key_q, valq) in Q_Param.items():
            # key_q might be (s_c, m, p, v_k)
            if len(key_q)==4 and key_q[0]==s_c:
                (_, m, p, v_k) = key_q
                if m in sub_orders_union and p in cluster_products and v_k in sub_vehicles_union:
                    if (m,p,v_k) in q_var:
                        q_var[(m,p,v_k)].Start = valq
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
        print(f"  Cluster-level (PHASE 1) => OPTIMAL, ObjVal={model.ObjVal}")
        # Store solutions in X_Param, Y_Param, Z_Param, Q_Param
        for i in locs_cluster:
            for j in locs_cluster:
                if i != j:
                    for k in sub_vehicles_union:
                        key = (wave_id, cluster_id, "PHASE1", i,j,k)
                        X_Param[key] = x[(i,j,k)].X
        for k in sub_vehicles_union:
            key = (wave_id, cluster_id, "PHASE1", k)
            Y_Param[key] = round(y[k].X)
        for m in sub_orders_union:
            for k in sub_vehicles_union:
                key = (wave_id, cluster_id, "PHASE1", m, k)
                Z_Param[key] = round(z[(m, k)].X)
        for m in sub_orders_union:
            for p in cluster_products:
                for k in sub_vehicles_union:
                    key = (wave_id, cluster_id, "PHASE1", m, p, k)
                    Q_Param[key] = q_var[(m, p, k)].X
    else:
        print(f"  Cluster-level (PHASE 1) => solver ended with status={model.Status}")
    


    print(f"PHASE 1: wave={wave_id}, cluster={cluster_id}")


###############################################################################
### (NEW) FLEX ALLOCATION & PHASE 2
###############################################################################
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
        print("No clusters found. Skipping flex allocation.")
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
    print(f"Global flex usage fraction = {100*total_flex_frac:.2f}%  vs. allowed {GlobalMaxFlexUpPercent}%")

    if total_flex_frac*100 <= GlobalMaxFlexUpPercent:
        # We are within limit, so no re-allocation needed
        print("Global Flex Up usage is within limit. No reallocation needed.")
        # We still might do a Phase 2 solve if you want, but by default the AMPL code
        # only triggers Phase 2 if we exceed the global limit. We replicate that logic:
        return
    else:
        print("We exceeded the global limit -> Running Flex Allocation...")

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

################################################################################
# ARC REMOVAL CONSTRAINTS (PHASE 0 or PHASE 1 or PHASE 2)
################################################################################
def add_arc_removal_constraints(model, sub_vehicles, start_depot_sub, delivery_locs_sub, x):
    """
    Add constraints that set x[i,j,k] = 0 if certain conditions are met.
    For example:
      1) If cost(o->j) < cost(i->j),
      2) i_demand + j_demand > capacity,
      3) earliest arrival at i plus TT[i,j] > LDT of j,
      ... etc.

    'model' is your Gurobi model
    'sub_vehicles' is the set of vehicles used in this phase
    'start_depot_sub' are the possible start locations
    'delivery_locs_sub' are the possible delivery locations
    'x' is the dictionary of x[(i,j,k)] variables
    """
    for k in sub_vehicles:
        for o in start_depot_sub:
            for i in delivery_locs_sub:
                for j in delivery_locs_sub:
                    if i != j:
                        # Condition A: compare cost of o->j vs i->j
                        cost_dep_j = 0.0
                        cost_i_j   = 0.0
                        # figure out which rate 'r' is used by vehicle k
                        # so you can pick FC[r], MC[r], etc.
                        for r in FC.keys():
                            if VC.get((k,r),0) == 1:
                                # sum for each rate if needed
                                cost_dep_j += (FC[r] + MC[r]*Dist.get((o,j),0))
                                cost_i_j   += (MC[r]*Dist.get((i,j),0))

                        condA = (cost_dep_j < cost_i_j)

                        # Condition B: i’s demand + j’s demand together exceed capacity
                        # We'll do a simple check for min demands
                        i_min_demand = 0  # fill in your logic
                        j_min_demand = 0  # fill in your logic
                        # e.g. i_min_demand = sum of (origD - possible flexDown) for location i
                        condB = False
                        # if (i_min_demand + j_min_demand) > <some capacity> => condB = True

                        # Condition C: earliest i + ST[i] + TT[i,j] > LDT[j] => x[i,j,k]=0
                        condC = False
                        # you'd fill in logic for earliest i or check if t[i,k] + ST[i] + TT[i,j] > LDT_j

                        # If ANY condition is true => x[i,j,k] = 0
                        if condA or condB or condC:
                            model.addConstr(x[(i,j,k)] == 0,
                                            name=f"ArcRemoval_{i}_{j}_{k}")

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
    print(f"[post_process_final_q] Completed clamping for {phase_label}, wave={wave_id}, cluster={cluster_id}.")


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

    print(f"\n=== PHASE 2: cluster-level solve start for cluster={cluster_id}, newFlex={new_flex_up_percent:.2f}% ===")

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
        print(f"PHASE 2 => solver ended with status={model.Status}")

    model.dispose()
    
    print(f"=== PHASE 2: cluster-level solve done for cluster={cluster_id} ===")




def parse_scenario(scenario_id, dataframes, shipment_rows_acc, order_rows_acc, route_rows_acc, scenario_descriptions, scenario_stats_acc):
    print(f"=== parse_scenario(scenario_id={scenario_id}) start ===")
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
                    used_veh_sub = solve_phase0_subcluster(sc, wave_id, cluster_id, source_loc_id)
                    used_vehicles_dict[sc] = used_veh_sub                    
                solve_phase1_for_cluster(wave_id, cluster_id, source_loc_id, used_vehicles_dict)

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
                        flex_allocation_and_phase2(source_loc_id, wave_id)

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

    print(f"=== parse_scenario(scenario_id={scenario_id}) end ===")
    print(f"   → Added {len(stats_rows)} scenario stats rows")





# def gather_shipment_rows(scenario_id, wave_id, cluster_id, source_loc_id, phase_label, scenario_descriptions):
#     rows = []

#     vehicles = {
#         key[3]
#         for key in Y_Param
#         if len(key) == 4 and key[0] == wave_id and key[1] == cluster_id and key[2] == phase_label and Y_Param[key] > 0.5
#     }

#     for k in vehicles:
#         rate = next(r for r in FC if VC.get((k, r), 0) == 1)

#         dist = sum(
#             Dist.get((i, j), 0) * X_Param.get((wave_id, cluster_id, phase_label, i, j, k), 0)
#             for i in LOCS for j in LOCS if i != j
#         )

#         stops = sum(
#             X_Param.get((wave_id, cluster_id, phase_label, i, j, k), 0)
#             for i in LOCS for j in LOCS if i != j
#         )

#         orders = sum(
#             Z_Param.get((wave_id, cluster_id, phase_label, m, k), 0)
#             for m in ORDERS
#         )

#         fixed = FC[rate]
#         mileage = MC[rate] * dist
#         stop_cost = SC[rate] * max(0, stops - Stops_Included_In_Rate[rate])

#         shipped_pallets = 0
#         shipped_weight = 0
#         shipped_cases = 0
#         original_pallets = 0
#         original_cases = 0
#         assigned_mps = set()

#         # for key, q_val in Q_Param.items():
#         #     if len(key) == 6:
#         #         w, c, ph, m, p, veh = key
#         #         if (w, c, ph, veh) == (wave_id, cluster_id, phase_label, k) and q_val > 0:
#         #             shipped_pallets += q_val
#         #             shipped_weight += q_val * WeightPerPallet.get(p, 0)
#         #             shipped_cases += q_val * CasesPerPallet.get(p, 0)
#         #             assigned_mps.add((m, p))
#         for key, q_val in Q_Param.items():

#         # phase‑1/0 shape
#             if len(key) == 6 and key[:3] == (wave_id, cluster_id, phase_label):
#             _, _, _, m, p, veh = key
#             # phase‑2 shape
#             elif len(key) == 6 and key[0] == phase_label and key[1:3] == (wave_id, cluster_id):
#                 _, _, _, m, p, veh = key
#             else:
#                 continue          # any other key belongs to a different phase / cluster

#             if veh != k: 
#                 continue          # different vehicle – skip

#     # *always* add the quantity – AMPL’s shipped‑pallet metric is Σq even if q = 0
#             shipped_pallets  += q_val
#             shipped_weight   += q_val * WeightPerPallet.get(p, 0.0)
#             shipped_cases    += q_val * CasesPerPallet.get(p, 0)

#         for (m, o, d, p), qty in InputOriginalD.items():
#             if Z_Param.get((wave_id, cluster_id, phase_label, m, k), 0) > 0.5:  # vehicle really serves order m
#                 original_pallets += qty
#                 original_cases += qty * CasesPerPallet.get(p, 0)

#         var_pallets = shipped_pallets - original_pallets
#         var_cases = shipped_cases - original_cases
#         vehicle_cap = WCap.get(rate, 1)

#         percent_util = shipped_weight / vehicle_cap if vehicle_cap else 0
#         # cost_per_case = (fixed + mileage + stop_cost) / shipped_cases if shipped_cases else 0
#         cost_per_case = (fixed + mileage + stop_cost) / original_cases if original_cases else 0
#         timestamp = datetime.now().strftime('%a %b %d %H:%M:%S %Y')
#         scenario_description = scenario_descriptions.get(scenario_id, "TEMP SCEN DESC")

#         # Match original column structure exactly
#         rows.append([
#             scenario_id,
#             timestamp,
#             scenario_id,
#             scenario_description,
#             source_loc_id,
#             wave_id,
#             cluster_id,
#             k,
#             rate,
#             round(cost_per_case, 4),
#             round(fixed + mileage + stop_cost, 2),
#             round(fixed, 2),
#             round(mileage, 2),
#             round(stop_cost, 2),
#             round(dist, 2),
#             int(stops),
#             int(orders),
#             round(shipped_weight, 2),
#             round(percent_util, 4),
#             int(shipped_pallets),
#             int(original_pallets),
#             int(var_pallets),
#             int(shipped_cases),
#             int(original_cases),
#             int(var_cases),
#             "XXNBL_DTS",  # INSERT_USER
#             timestamp,    # INSERT_DATE
#             "XXNBL_DTS",  # UPDATE_USER
#             timestamp     # UPDATE_DATE
#         ])

#     return rows

from datetime import datetime

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


# def gather_route_rows(scenario_id, wave_id, cluster_id, source_loc_id, phase_label, scenario_descriptions):
#     """
#     Generates route report rows aligned with the defined ROUTE_REPORT structure.
#     """
#     rows = []
#     timestamp = datetime.now().strftime("%a %b %d %H:%M:%S %Y")
#     scenario_description = scenario_descriptions.get(scenario_id, "TEMP SCENARIO DESCRIPTION")

#     vehicles = {
#         key[3] for key in Y_Param
#         if len(key) == 4
#         and key[0] == wave_id
#         and key[1] == cluster_id
#         and key[2] == phase_label
#         and Y_Param[key] > 0.5
#     }

#     for k in vehicles:
#         route_segments = []
#         for key in X_Param:
#             if (
#                 len(key) == 6 and
#                 key[0] == wave_id and
#                 key[1] == cluster_id and
#                 key[2] == phase_label and
#                 key[5] == k and
#                 X_Param[key] > 0.5
#             ):
#                 i, j = key[3], key[4]
#                 route_segments.append({
#                     'from_loc': i,
#                     'to_loc': j,
#                     'transit_time': TT.get((i, j), 0),
#                     'distance': Dist.get((i, j), 0.0),
#                     'service_time': ST.get(i, 0)
#                 })

#         for idx, segment in enumerate(route_segments, 1):
#             from_loc = segment['from_loc']
#             to_loc = segment['to_loc']

#             from_name = LocName.get(from_loc, "")
#             to_name = LocName.get(to_loc, "")
#             from_type = LocType.get(from_loc, "")
#             to_type = LocType.get(to_loc, "")

#             order_id = ""
#             cust_po = ""
#             if to_type == "DELIVERY_LOCATION":
#                 for (m, o, d, p) in ORD_SLOC_DLOC_PROD:
#                     if d == to_loc:
#                         order_id = m
#                         for (mm, cpo, pp) in ORD_CUSTPO_PRD:
#                             if mm == m:
#                                 cust_po = cpo
#                                 break
#                         break

#             arrival_time = None
#             service_end_time = None

#             rows.append([
#                 None,  # ROUTE_REPORT_ID
#                 timestamp,
#                 scenario_id,
#                 scenario_description,
#                 source_loc_id,
#                 LocName.get(source_loc_id, ""),
#                 wave_id,
#                 cluster_id,
#                 k,
#                 to_loc,
#                 to_name,
#                 to_type,
#                 order_id,
#                 cust_po,
#                 idx,
#                 arrival_time,
#                 service_end_time,
#                 segment['transit_time'],
#                 0,  # IDLE_TIME_TO_NEXT_STOP
#                 segment['distance'],
#                 LocCity.get(source_loc_id, ""),
#                 LocState.get(source_loc_id, ""),
#                 LocZipCode.get(source_loc_id, ""),
#                 LocCity.get(to_loc, ""),
#                 LocState.get(to_loc, ""),
#                 LocZipCode.get(to_loc, ""),
#                 arrival_time,  # A_TIME
#                 service_end_time,  # SE_TIME
#                 0,  # DATE_OFFSET
#                 "XXNBL_DTS",  # INSERT_USER
#                 datetime.now().strftime("%d-%b-%y"),  # INSERT_DATE
#                 "XXNBL_DTS",  # UPDATE_USER
#                 datetime.now().strftime("%d-%b-%y")  # UPDATE_DATE
#             ])

#         # Time calculations
#         if rows:
#             rows[0][15] = 0  # ARRIVAL_TIME
#             rows[0][16] = rows[0][15] + ST.get(rows[0][9], 0)  # SERVICE_END_TIME
#             rows[0][26] = rows[0][15]
#             rows[0][27] = rows[0][16]

#             for i in range(1, len(rows)):
#                 prev_end = rows[i - 1][16]
#                 prev_to = rows[i - 1][9]
#                 curr_to = rows[i][9]
#                 transit = TT.get((prev_to, curr_to), 0)
#                 arrival = prev_end + transit
#                 end = arrival + ST.get(curr_to, 0)

#                 rows[i][15] = arrival
#                 rows[i][16] = end
#                 rows[i][26] = arrival
#                 rows[i][27] = end

#     return rows

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
###############################################################################
# 2) WRITE FUNCTIONS: each writes a final CSV once we have all rows
###############################################################################


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

    print(f"[OK] Wrote Scenario Stats CSV: {final_filename} with {len(all_scenario_stats_rows)} rows.")




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

    print(f"[OK] Wrote Shipment CSV: {final_filename} with {len(all_shipment_rows)} rows.")


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

    print(f"[OK] Wrote Order CSV: {final_filename} with {len(all_order_rows)} rows.")



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

    print(f"[OK] Wrote Route CSV: {final_filename} with {len(all_route_rows)} rows.")

###############################################################################
# 6) MAIN
###############################################################################
def main():
    #--------------------------------------------------------------------------
    # Paths for all input files:
    #--------------------------------------------------------------------------

    file_paths = {
        "XXNBL_PRODUCT_PARAMS": "C:/Users/HammadEzad/OneDrive - Royal Cyber Inc/Desktop/Niagara/Apex files/new_files/xxnbl_product_params_37836.csv",
        "XXNBL_LOCATION_PARAMS": "C:/Users/HammadEzad/OneDrive - Royal Cyber Inc/Desktop/Niagara/Apex files/new_files/xxnbl_location_params_37836.csv",
        "XXNBL_ORDER_PARAMS": "C:/Users/HammadEzad/OneDrive - Royal Cyber Inc/Desktop/Niagara/Apex files/new_files/xxnbl_order_params_37836.csv",
        "XXNBL_TIME_DIST_PARAMS": "C:/Users/HammadEzad/OneDrive - Royal Cyber Inc/Desktop/Niagara/Apex files/new_files/xxnbl_time_dist_params_37836.csv",
        "XXNBL_RATE_PARAMS": "C:/Users/HammadEzad/OneDrive - Royal Cyber Inc/Desktop/Niagara/Apex files/new_files/xxnbl_rate_params_37836.csv",
        "XXNBL_CLUSTER_PARAMS": "C:/Users/HammadEzad/OneDrive - Royal Cyber Inc/Desktop/Niagara/Apex files/new_files/xxnbl_cluster_params_37836.csv",
        "XXNBL_SCENARIOS": "C:/Users/HammadEzad/OneDrive - Royal Cyber Inc/Desktop/Niagara/Apex files/new_files/xxnbl_scenarios_37836.csv",
    }
    # file_paths = {
    #     "XXNBL_PRODUCT_PARAMS": "C:/Users/HammadEzad/OneDrive - Royal Cyber Inc/Desktop/Niagara/Apex files/38308/product_params_38308.csv",
    #     "XXNBL_LOCATION_PARAMS": "C:/Users/HammadEzad/OneDrive - Royal Cyber Inc/Desktop/Niagara/Apex files/38308/location_params_38308.csv",
    #     "XXNBL_ORDER_PARAMS": "C:/Users/HammadEzad/OneDrive - Royal Cyber Inc/Desktop/Niagara/Apex files/38308/order_params_38308.csv",
    #     "XXNBL_TIME_DIST_PARAMS": "C:/Users/HammadEzad/OneDrive - Royal Cyber Inc/Desktop/Niagara/Apex files/38308/time_dist_params_38308.csv",
    #     "XXNBL_RATE_PARAMS": "C:/Users/HammadEzad/OneDrive - Royal Cyber Inc/Desktop/Niagara/Apex files/38308/rate_params_38308.csv",
    #     "XXNBL_CLUSTER_PARAMS": "C:/Users/HammadEzad/OneDrive - Royal Cyber Inc/Desktop/Niagara/Apex files/38308/cluster_params_38308.csv",
    #     "XXNBL_SCENARIOS": "C:/Users/HammadEzad/OneDrive - Royal Cyber Inc/Desktop/Niagara/Apex files/38308/scenario_38308.csv",
    # }


    #--------------------------------------------------------------------------
    # Load the CSV data into DataFrames
    #--------------------------------------------------------------------------
    dataframes = {}
    for key, path in file_paths.items():
        try:
            df = pd.read_csv(path)
            dataframes[key] = df
        except Exception as e:
            print(f"Error reading {path}: {e}")
            dataframes[key] = pd.DataFrame()

    #--------------------------------------------------------------------------
    # Identify which scenarios have RC_STATUS='SUBMITTED'
    #--------------------------------------------------------------------------
    df_scen = dataframes.get("XXNBL_SCENARIOS", pd.DataFrame())
    df_submitted = df_scen[df_scen["RC_STATUS"] == "SUBMITTED"]
    scenario_ids = df_submitted["SCENARIO_ID"].unique().tolist()
    scenario_descriptions = dict(zip(df_submitted["SCENARIO_ID"], df_submitted["DESCRIPTION"]))
    #--------------------------------------------------------------------------
    # For each submitted scenario => parse_scenario
    #--------------------------------------------------------------------------
    # for idx, row in df_submitted.iterrows():
    #     this_scenario_id = int(row["SCENARIO_ID"])
    #     parse_scenario(this_scenario_id, dataframes)
    if len(scenario_ids) > 0:
        scenario_id = scenario_ids[0]
        scen_row = df_submitted[df_submitted["SCENARIO_ID"] == scenario_id].iloc[0]
        global GlobalMaxFlexUpPercent, UseFlexQty, UseMaxIdleTimeConstr, MaxIdleTimeBtwStops, SolveTimeForCluster, SolveTimePerSubCluster
        GlobalMaxFlexUpPercent = scen_row["MAX_FLEX_UP_PCT"]
        UseFlexQty = scen_row["IND_USE_FLEX_QUANTITY"]
        UseMaxIdleTimeConstr = scen_row["IND_USE_MAX_IDLE_CSTR"]
        MaxIdleTimeBtwStops = scen_row["MAX_IDLE_TIME_BTW_STOPS"]
        SolveTimeForCluster = scen_row["SOLVE_TIME_PER_CLUSTER"]
        SolveTimePerSubCluster = scen_row["SOLVE_TIME_PER_SUB_CLUSTER"]

    # 2) We'll accumulate rows for all scenarios
    all_shipment_rows = []
    all_order_rows = []
    all_route_rows = []
    all_scenario_stats_rows = []

    # 3) For each scenario => parse => gather rows
    for this_scenario_id in scenario_ids:
        parse_scenario(
            this_scenario_id,
            dataframes,
            all_shipment_rows,
            all_order_rows,
            all_route_rows, 
            scenario_descriptions,
            all_scenario_stats_rows
        )

    # 4) At the end => write the combined CSVs
    # write_shipment_csv("shipment_report_all_scenarios_38308.csv", all_shipment_rows)
    # write_order_csv("order_report_all_scenarios_38308.csv", all_order_rows)
    # write_route_csv("route_report_all_scenarios_38308.csv", all_route_rows)
    # write_scenario_stats_csv("scenario_stats_all_scenarios_38308.csv", all_scenario_stats_rows)
    output_folder = f"reports_{scenario_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_folder, exist_ok=True)

    # Now write each report into this folder
    write_shipment_csv(os.path.join(output_folder, f"shipment_report_latest_{scenario_id}_.csv"), all_shipment_rows)
    write_order_csv(os.path.join(output_folder, f"order_report_latest_{scenario_id}.csv"), all_order_rows)
    write_route_csv(os.path.join(output_folder, f"route_report_latest_{scenario_id}.csv"), all_route_rows)
    write_scenario_stats_csv(os.path.join(output_folder, f"scenario_stats_latest_{scenario_id}.csv"), all_scenario_stats_rows)


# Standard Python boilerplate to call main() if run as a script:
if __name__ == "__main__":
    main()

