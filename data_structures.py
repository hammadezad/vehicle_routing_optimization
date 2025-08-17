"""Global sets/dicts and solution stores."""
from typing import Any, Tuple, Set, Dict

# Core sets
PRD: Set[Any] = set()
ORDERS: Set[Any] = set()
LOCS: Set[Any] = set()
ORD_SLOC_DLOC_PROD: Set[Tuple[Any, Any, Any, Any]] = set()
ORIGIN_DEST: Set[Tuple[Any, Any]] = set()
SLOC_RATE: Set[Tuple[Any, Any]] = set()
ORD_CUSTPO_PRD: Set[Tuple[Any, Any, Any]] = set()
SLOC_RATE_WAVE_CLUST_SUBCLUST_DLOC_ORD: Set[Tuple[Any, ...]] = set()

# Raw inputs
InputVehicles: Dict[Tuple[Any, ...], Any] = {}
InputOriginalD: Dict[Tuple[Any, Any, Any, Any], float] = {}
InputEDT: Dict[Tuple[Any, Any, Any, Any], float] = {}
InputLDT: Dict[Tuple[Any, Any, Any, Any], float] = {}

InputFC: Dict[Tuple[Any, Any], float] = {}
InputMC: Dict[Tuple[Any, Any], float] = {}
InputSC: Dict[Tuple[Any, Any], float] = {}
InputWCap: Dict[Tuple[Any, Any], float] = {}
InputStops_Included_In_Rate: Dict[Tuple[Any, Any], int] = {}
InputMaxStops: Dict[Tuple[Any, Any], int] = {}
InputMaxRouteDuration: Dict[Tuple[Any, Any], float] = {}
InputMaxDistBetweenStops: Dict[Tuple[Any, Any], float] = {}

# Location and product attributes
LocFlexDownPerOrder: Dict[Any, float] = {}
LocFlexUpPerOrder: Dict[Any, float] = {}
LocType: Dict[Any, str] = {}
ST: Dict[Any, float] = {}
Dist: Dict[Tuple[Any, Any], float] = {}
TT: Dict[Tuple[Any, Any], float] = {}

WeightPerPallet: Dict[Any, float] = {}
CasesPerPallet: Dict[Any, float] = {}
ProductFlexDown: Dict[Any, float] = {}
ProductFlexUp: Dict[Any, float] = {}
ProductDescription: Dict[Any, str] = {}

LocName: Dict[Any, str] = {}
LocCity: Dict[Any, str] = {}
LocState: Dict[Any, str] = {}
LocZipCode: Dict[Any, str] = {}

# Derived rate-level dictionaries (used by rate_builder)
FC: Dict[Any, float] = {}
MC: Dict[Any, float] = {}
SC: Dict[Any, float] = {}
WCap: Dict[Any, float] = {}
Stops_Included_In_Rate: Dict[Any, int] = {}
MaxStops: Dict[Any, int] = {}
MaxRouteDuration: Dict[Any, float] = {}
MaxDistBetweenStops_val: Dict[Any, float] = {}
VC: Dict[Tuple[Any, Any], int] = {}

# Solution stores
X_Param: Dict[Tuple[Any, ...], float] = {}
Y_Param: Dict[Tuple[Any, ...], float] = {}
Z_Param: Dict[Tuple[Any, ...], float] = {}
Q_Param: Dict[Tuple[Any, ...], float] = {}

def reset_all() -> None:
    """Clear all containers before loading a new scenario."""
    PRD.clear(); ORDERS.clear(); LOCS.clear()
    ORD_SLOC_DLOC_PROD.clear(); ORIGIN_DEST.clear(); SLOC_RATE.clear()
    ORD_CUSTPO_PRD.clear(); SLOC_RATE_WAVE_CLUST_SUBCLUST_DLOC_ORD.clear()

    InputVehicles.clear(); InputOriginalD.clear(); InputEDT.clear(); InputLDT.clear()
    InputFC.clear(); InputMC.clear(); InputSC.clear(); InputWCap.clear()
    InputStops_Included_In_Rate.clear(); InputMaxStops.clear()
    InputMaxRouteDuration.clear(); InputMaxDistBetweenStops.clear()

    LocFlexDownPerOrder.clear(); LocFlexUpPerOrder.clear(); LocType.clear()
    ST.clear(); Dist.clear(); TT.clear()

    WeightPerPallet.clear(); CasesPerPallet.clear()
    ProductFlexDown.clear(); ProductFlexUp.clear(); ProductDescription.clear()

    LocName.clear(); LocCity.clear(); LocState.clear(); LocZipCode.clear()

    FC.clear(); MC.clear(); SC.clear(); WCap.clear()
    Stops_Included_In_Rate.clear(); MaxStops.clear()
    MaxRouteDuration.clear(); MaxDistBetweenStops_val.clear(); VC.clear()

    X_Param.clear(); Y_Param.clear(); Z_Param.clear(); Q_Param.clear()
