"""
Context utilities to inject scenario/cluster/subcluster IDs into log records.
"""
import logging
from contextvars import ContextVar
from contextlib import contextmanager
from typing import Optional

scenario_id_var: ContextVar[Optional[str]] = ContextVar("scenario_id", default=None)
cluster_id_var: ContextVar[Optional[str]] = ContextVar("cluster_id", default=None)
subcluster_id_var: ContextVar[Optional[str]] = ContextVar("subcluster_id", default=None)

def set_context(scenario_id: Optional[str] = None, cluster_id: Optional[str] = None, subcluster_id: Optional[str] = None) -> None:
    if scenario_id is not None:
        scenario_id_var.set(str(scenario_id))
    if cluster_id is not None:
        cluster_id_var.set(str(cluster_id))
    if subcluster_id is not None:
        subcluster_id_var.set(str(subcluster_id))

def clear_context() -> None:
    scenario_id_var.set(None)
    cluster_id_var.set(None)
    subcluster_id_var.set(None)

class LogContextFilter(logging.Filter):
    """
    Adds contextvars to LogRecord so formatters can print them.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        record.scenario_id = scenario_id_var.get() or "-"
        record.cluster_id = cluster_id_var.get() or "-"
        record.subcluster_id = subcluster_id_var.get() or "-"
        return True

@contextmanager
def scenario_context(scenario_id: str, cluster_id: Optional[str] = None, subcluster_id: Optional[str] = None):
    prev_s, prev_c, prev_sc = scenario_id_var.get(), cluster_id_var.get(), subcluster_id_var.get()
    try:
        set_context(scenario_id, cluster_id, subcluster_id)
        yield
    finally:
        scenario_id_var.set(prev_s)
        cluster_id_var.set(prev_c)
        subcluster_id_var.set(prev_sc)
