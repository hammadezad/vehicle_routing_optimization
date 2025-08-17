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

def load_data_for_scenario(dataframes, scenario_id):
    logger.info("CHECKPOINT: Starting load_data_from_dataframes...")
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

    logger.info("CHECKPOINT: load_data_from_dataframes done.")