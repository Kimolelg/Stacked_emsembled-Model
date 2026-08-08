"""
Backward-compat shim for joblib pickles saved before MetaModel4 moved into meta_models.py.

Do not add new logic here — import from src.models.meta_models instead.
"""

from src.models.meta_models import (  # noqa: F401
    MetaModel4,
    MetaModel4Config,
    evaluate_binary_model,
    train_meta_model4,
)

__all__ = [
    "MetaModel4",
    "MetaModel4Config",
    "evaluate_binary_model",
    "train_meta_model4",
]
