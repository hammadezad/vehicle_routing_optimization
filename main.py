import os
from datetime import datetime
import pandas as pd

import config as cfg
import data_structures as ds

import data_loader
import rate_builder
import phase0_solver
import phase1_solver
import phase2_solver
import flex_allocator
import constraint_builder
import reports
import solution_processor
import utils
import vrp_utils.logging as logging
from vrp_utils.context import set_context, scenario_context


def run_pipeline():
    print("🚀 Starting VRP optimization pipeline...")

    logging.setup(log_dir="logs", level="INFO", capture_print=False)
    logger = logging.getLogger(__name__)

    if hasattr(ds, "reset_all"):
        ds.reset_all()

    file_paths = {
        "XXNBL_PRODUCT_PARAMS": "C:/Users/HammadEzad/OneDrive - Royal Cyber Inc/Desktop/Niagara/Apex files/new_files/xxnbl_product_params_37836.csv",
        "XXNBL_LOCATION_PARAMS": "C:/Users/HammadEzad/OneDrive - Royal Cyber Inc/Desktop/Niagara/Apex files/new_files/xxnbl_location_params_37836.csv",
        "XXNBL_ORDER_PARAMS": "C:/Users/HammadEzad/OneDrive - Royal Cyber Inc/Desktop/Niagara/Apex files/new_files/xxnbl_order_params_37836.csv",
        "XXNBL_TIME_DIST_PARAMS": "C:/Users/HammadEzad/OneDrive - Royal Cyber Inc/Desktop/Niagara/Apex files/new_files/xxnbl_time_dist_params_37836.csv",
        "XXNBL_RATE_PARAMS": "C:/Users/HammadEzad/OneDrive - Royal Cyber Inc/Desktop/Niagara/Apex files/new_files/xxnbl_rate_params_37836.csv",
        "XXNBL_CLUSTER_PARAMS": "C:/Users/HammadEzad/OneDrive - Royal Cyber Inc/Desktop/Niagara/Apex files/new_files/xxnbl_cluster_params_37836.csv",
        "XXNBL_SCENARIOS": "C:/Users/HammadEzad/OneDrive - Royal Cyber Inc/Desktop/Niagara/Apex files/new_files/xxnbl_scenarios_37836.csv",
    }

    dataframes = {}
    for key, path in file_paths.items():
        try:
            df = pd.read_csv(path)
            dataframes[key] = df
            logger.info("Loaded %s with %d rows", key, len(df))
        except Exception:
            logger.exception("Error reading %s", path)
            dataframes[key] = pd.DataFrame()

    df_scen = dataframes.get("XXNBL_SCENARIOS", pd.DataFrame())
    if df_scen.empty or "RC_STATUS" not in df_scen.columns:
        logger.warning("No scenarios DataFrame or RC_STATUS column missing")
        return

    df_submitted = df_scen[df_scen["RC_STATUS"] == "SUBMITTED"].copy()
    if df_submitted.empty:
        logger.info("No scenarios with RC_STATUS='SUBMITTED'")
        return

    scenario_ids = df_submitted["SCENARIO_ID"].dropna().unique().tolist()
    scenario_descriptions = dict(zip(
        df_submitted["SCENARIO_ID"],
        df_submitted.get("DESCRIPTION", pd.Series()).fillna("")
    ))

    scenario_id = scenario_ids[0]
    set_context(scenario_id=scenario_id)

    scen_row = df_submitted[df_submitted["SCENARIO_ID"] == scenario_id].iloc[0]
    cfg.update_from_row(scen_row)
    cfg.sync_aliases()
    logger.info("Scenario %s config applied", scenario_id)

    all_shipment_rows = []
    all_order_rows = []
    all_route_rows = []
    all_scenario_stats_rows = []

    for this_scenario_id in scenario_ids:
        with scenario_context(scenario_id=this_scenario_id):
            try:
                utils.parse_scenario(
                    this_scenario_id,
                    dataframes,
                    all_shipment_rows,
                    all_order_rows,
                    all_route_rows,
                    scenario_descriptions,
                    all_scenario_stats_rows
                )
                logger.info("parse_scenario completed for %s", this_scenario_id)
            except Exception:
                logging.getLogger("utils.parse_scenario").exception("parse_scenario failed for %s", this_scenario_id)
                raise

    output_folder = f"reports_{scenario_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_folder, exist_ok=True)

    try:
        reports.write_shipment_csv(os.path.join(output_folder, f"shipment_report_latest_{scenario_id}.csv"), all_shipment_rows)
        reports.write_order_csv(os.path.join(output_folder, f"order_report_latest_{scenario_id}.csv"), all_order_rows)
        reports.write_route_csv(os.path.join(output_folder, f"route_report_latest_{scenario_id}.csv"), all_route_rows)
        reports.write_scenario_stats_csv(os.path.join(output_folder, f"scenario_stats_latest_{scenario_id}.csv"), all_scenario_stats_rows)
        logger.info("Reports written to %s", output_folder)
    except Exception:
        logging.getLogger("reports").exception("Writing reports failed")
        raise



def main():
    try:
        run_pipeline()
    except Exception:
        logging.getLogger(__name__).exception("Pipeline failed")
        raise

if __name__ == '__main__':
    main()
