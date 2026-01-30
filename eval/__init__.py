"""Evaluation harness for ConvergeBench."""

from .run_roundtrip import (
    load_model,
    load_edit_pairs,
    run_single_evaluation,
    run_benchmark
)

from .compute_metrics import (
    AggregateMetrics,
    compute_half_life,
    compute_auc,
    aggregate_results
)

__all__ = [
    "load_model",
    "load_edit_pairs", 
    "run_single_evaluation",
    "run_benchmark",
    "AggregateMetrics",
    "compute_half_life",
    "compute_auc",
    "aggregate_results"
]
