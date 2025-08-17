import vrp_utils.logging as logging
import data_structures as ds
import config as cfg
from vrp_utils.decorators import log_and_time
from exceptions import RateBuildError

logger = logging.getLogger(__name__)

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    class GRB:
        pass


def build_global_rate_dicts():
    logger.info("CHECKPOINT: build_global_rate_dicts start...")

    # Validate that all required globals exist in data_structures
    required = [
        "FC","MC","SC","WCap","Stops_Included_In_Rate",
        "MaxStops","MaxRouteDuration","MaxDistBetweenStops_val","VC",
        "LOCS","LocType","SLOC_RATE",
        "InputFC","InputMC","InputSC","InputWCap",
        "InputStops_Included_In_Rate","InputMaxStops",
        "InputMaxRouteDuration","InputMaxDistBetweenStops",
        "InputVehicles",
    ]
    missing = [name for name in required if not hasattr(ds, name)]
    if missing:
        raise AttributeError(
            f"Missing globals in data_structures: {missing}. "
            f"Open data_structures.py and define them."
        )

    # Clear in place so references held by other modules remain valid
    ds.FC.clear()
    ds.MC.clear()
    ds.SC.clear()
    ds.WCap.clear()
    ds.Stops_Included_In_Rate.clear()
    ds.MaxStops.clear()
    ds.MaxRouteDuration.clear()
    ds.MaxDistBetweenStops_val.clear()
    ds.VC.clear()

    # Only consider rates from START_DEPOT SLOCs
    start_depots = {loc for loc in ds.LOCS if ds.LocType.get(loc, "") == "START_DEPOT"}
    all_rate_ids = {rr for (_, rr) in ds.SLOC_RATE}

    for r in all_rate_ids:
        matching_slocs = [s for (s, rr) in ds.SLOC_RATE if rr == r and s in start_depots]

        fc_val     = max((ds.InputFC.get((sl, r), 0) for sl in matching_slocs), default=0)
        mc_val     = max((ds.InputMC.get((sl, r), 0) for sl in matching_slocs), default=0)
        sc_val     = max((ds.InputSC.get((sl, r), 0) for sl in matching_slocs), default=0)
        wcap_val   = max((ds.InputWCap.get((sl, r), 0) for sl in matching_slocs), default=999_999)
        stops_incl = max((ds.InputStops_Included_In_Rate.get((sl, r), 0) for sl in matching_slocs), default=0)
        mx_stops   = max((ds.InputMaxStops.get((sl, r), 0) for sl in matching_slocs), default=999)
        mx_dur     = max((ds.InputMaxRouteDuration.get((sl, r), 0) for sl in matching_slocs), default=999_999)
        mx_dist    = max((ds.InputMaxDistBetweenStops.get((sl, r), 0) for sl in matching_slocs), default=999_999)

        ds.FC[r] = fc_val
        ds.MC[r] = mc_val
        ds.SC[r] = sc_val
        ds.WCap[r] = wcap_val
        ds.Stops_Included_In_Rate[r] = stops_incl
        ds.MaxStops[r] = mx_stops
        ds.MaxRouteDuration[r] = mx_dur
        ds.MaxDistBetweenStops_val[r] = mx_dist

    # Build VC without rebinding the dict
    processed_vehicles = set()

    for (o, r, w, c, s_c, i, m), veh in ds.InputVehicles.items():
        if veh is None:
            continue
        if veh not in processed_vehicles:
            ds.VC[(veh, r)] = 1
            processed_vehicles.add(veh)
        else:
            # ensure a vehicle isn't tied to multiple rates
            existing_rate = next((rr for (v, rr) in ds.VC if v == veh and ds.VC[(v, rr)] == 1), None)
            if existing_rate is not None and existing_rate != r:
                raise ValueError(f"Vehicle {veh} is assigned to multiple rates: {existing_rate} and {r}")

    # Initialize remaining (veh, r) pairs to 0
    for veh in processed_vehicles:
        for r in all_rate_ids:
            ds.VC.setdefault((veh, r), 0)

    logger.info("CHECKPOINT: build_global_rate_dicts done.")
