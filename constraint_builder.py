"""Constraint Builder module (auto-extracted)."""
# Auto-generated modular refactor from phase_2_v2.py
# NOTE: This is an initial split. Cross-module imports assume:
#   from config import *
#   from data_structures import *
# Review and adjust as needed for your project.

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