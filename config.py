# config.py
"""
Centralized scenario and solver configuration with legacy aliases.
Works with main.py's update_from_row(...) + sync_aliases().
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class ScenarioConfig:
    # Phase and flex controls
    UseFlexQty: int = 1
    UseFlexUpPercentLimit: int = 0
    MaxFlexUpPercent_CurrentCluster: int = 100
    CurrClustDiscount: float = 0.0
    MaxFlexUpPercent_AllClusters: Dict[int, int] = None

    # Idle time constraint
    UseMaxIdleTimeConstr: int = 0
    MaxIdleTimeBtwStops: float = 1.0  # minutes

    # Time limits in minutes
    SolveTimePerSubCluster: int = 30
    SolveTimeForCluster: int = 60

    # Global flex cap
    GlobalMaxFlexUpPercent: int = 100

    def __post_init__(self):
        if self.MaxFlexUpPercent_AllClusters is None:
            self.MaxFlexUpPercent_AllClusters = {}

    def update(self, **kwargs) -> None:
        """
        Programmatic override of fields, with safety for unknown keys.
        Example:
            SCENARIO.update(SolveTimePerSubCluster=45)
        """
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise AttributeError(f"Unknown config field: {k}")

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Singleton instance
SCENARIO = ScenarioConfig()

# Legacy module-level aliases so code using `from config import UseFlexQty, ...` keeps working
UseFlexQty = SCENARIO.UseFlexQty
UseFlexUpPercentLimit = SCENARIO.UseFlexUpPercentLimit
MaxFlexUpPercent_CurrentCluster = SCENARIO.MaxFlexUpPercent_CurrentCluster
CurrClustDiscount = SCENARIO.CurrClustDiscount
MaxFlexUpPercent_AllClusters = SCENARIO.MaxFlexUpPercent_AllClusters

UseMaxIdleTimeConstr = SCENARIO.UseMaxIdleTimeConstr
MaxIdleTimeBtwStops = SCENARIO.MaxIdleTimeBtwStops

SolveTimePerSubCluster = SCENARIO.SolveTimePerSubCluster
SolveTimeForCluster = SCENARIO.SolveTimeForCluster

GlobalMaxFlexUpPercent = SCENARIO.GlobalMaxFlexUpPercent


def update_from_row(row) -> None:
    """
    Accepts a pandas Series or dict with these keys:
      MAX_FLEX_UP_PCT
      IND_USE_FLEX_QUANTITY
      IND_USE_MAX_IDLE_CSTR
      MAX_IDLE_TIME_BTW_STOPS
      SOLVE_TIME_PER_CLUSTER
      SOLVE_TIME_PER_SUB_CLUSTER
    Updates SCENARIO but does not change the aliases until sync_aliases() is called.
    """
    # Support both Series and dict
    get = row.get if hasattr(row, "get") else (lambda k: row[k])

    SCENARIO.update(
        GlobalMaxFlexUpPercent=int(get("MAX_FLEX_UP_PCT")),
        UseFlexQty=int(get("IND_USE_FLEX_QUANTITY")),
        UseMaxIdleTimeConstr=int(get("IND_USE_MAX_IDLE_CSTR")),
        MaxIdleTimeBtwStops=float(get("MAX_IDLE_TIME_BTW_STOPS")),
        SolveTimeForCluster=int(get("SOLVE_TIME_PER_CLUSTER")),
        SolveTimePerSubCluster=int(get("SOLVE_TIME_PER_SUB_CLUSTER")),
    )


def sync_aliases() -> None:
    """
    Push SCENARIO values back into the legacy module-level names so solvers that
    import module-level symbols see your latest scenario settings.
    """
    global UseFlexQty, UseFlexUpPercentLimit, MaxFlexUpPercent_CurrentCluster, CurrClustDiscount
    global MaxFlexUpPercent_AllClusters, UseMaxIdleTimeConstr, MaxIdleTimeBtwStops
    global SolveTimePerSubCluster, SolveTimeForCluster, GlobalMaxFlexUpPercent

    UseFlexQty = SCENARIO.UseFlexQty
    UseFlexUpPercentLimit = SCENARIO.UseFlexUpPercentLimit
    MaxFlexUpPercent_CurrentCluster = SCENARIO.MaxFlexUpPercent_CurrentCluster
    CurrClustDiscount = SCENARIO.CurrClustDiscount
    MaxFlexUpPercent_AllClusters = SCENARIO.MaxFlexUpPercent_AllClusters
    UseMaxIdleTimeConstr = SCENARIO.UseMaxIdleTimeConstr
    MaxIdleTimeBtwStops = SCENARIO.MaxIdleTimeBtwStops
    SolveTimePerSubCluster = SCENARIO.SolveTimePerSubCluster
    SolveTimeForCluster = SCENARIO.SolveTimeForCluster
    GlobalMaxFlexUpPercent = SCENARIO.GlobalMaxFlexUpPercent


__all__ = [
    "ScenarioConfig",
    "SCENARIO",
    # legacy aliases
    "UseFlexQty",
    "UseFlexUpPercentLimit",
    "MaxFlexUpPercent_CurrentCluster",
    "CurrClustDiscount",
    "MaxFlexUpPercent_AllClusters",
    "UseMaxIdleTimeConstr",
    "MaxIdleTimeBtwStops",
    "SolveTimePerSubCluster",
    "SolveTimeForCluster",
    "GlobalMaxFlexUpPercent",
    # helpers
    "update_from_row",
    "sync_aliases",
]
