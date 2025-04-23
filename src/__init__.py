import time
import warnings

if not hasattr(time, 'clock'):
    # Define clock using perf_counter for compatibility
    time.clock = time.perf_counter
    warnings.warn("time.clock is deprecated. Using time.perf_counter instead.", DeprecationWarning)