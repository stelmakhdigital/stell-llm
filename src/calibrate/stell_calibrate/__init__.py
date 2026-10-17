"""stell_calibrate — калибровщик-арбитр (PLAN.md §4, поток E).

Любое ускорение (драфтер, KV-квант, NVMe-тир) включается только если
измеримо быстрее/не хуже на данном железе. Политика чистая и тестируется
на синтетических замерах; замер-харнесс измеряет A/B через бэкенд движка.
"""

from .config import CalibrationConfig
from .harness import CalibrationHarness, timed_call
from .policy import CalibrationDecision, decide
from .runner import CalibrationRun, decode_run, measure_tok_s, run_calibration

__all__ = ["CalibrationConfig", "CalibrationDecision", "decide",
           "CalibrationHarness", "timed_call", "CalibrationRun",
           "decode_run", "measure_tok_s", "run_calibration"]
