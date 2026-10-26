"""regime-hmm-engine: log-space Gaussian HMM in PyTorch with finance and EEG benchmarks."""

from .core import GaussianHMM
from .metrics import (
    align_states_hungarian,
    calculate_classification_metrics,
    compute_bic,
    confusion_matrix,
)
from .finance_pipeline import prepare_finance_data
from .neuro_pipeline import prepare_sleep_data

__version__ = "0.1.0"

__all__ = [
    "GaussianHMM",
    "compute_bic",
    "align_states_hungarian",
    "calculate_classification_metrics",
    "confusion_matrix",
    "prepare_finance_data",
    "prepare_sleep_data",
    "__version__",
]
