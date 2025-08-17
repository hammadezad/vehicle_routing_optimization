"""
Custom exception hierarchy for clearer error handling.
Catch and rethrow with these in your modules for better diagnostics.
"""
class VrpError(Exception):
    """Base class for VRP-related errors."""

class ConfigError(VrpError):
    pass

class DataLoadError(VrpError):
    pass

class RateBuildError(VrpError):
    pass

class Phase0Error(VrpError):
    pass

class Phase1Error(VrpError):
    pass

class Phase2Error(VrpError):
    pass

class ReportWriteError(VrpError):
    pass
