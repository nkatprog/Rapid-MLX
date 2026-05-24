"""Rapid-MLX: High-performance machine learning inference on Apple Silicon using MLX.

This package provides optimized utilities and pipelines for running ML models
with Apple's MLX framework, focusing on speed, memory efficiency, and ease of use.

Personal fork notes:
- Using this for experimenting with local LLM inference on my M2 MacBook Pro
- See README for upstream project: https://github.com/raullenchai/Rapid-MLX
"""

__version__ = "0.1.0"
__author__ = "Rapid-MLX Contributors"
__license__ = "MIT"

from rapid_mlx.core import RapidModel
from rapid_mlx.pipeline import InferencePipeline
from rapid_mlx.utils import benchmark, profile_memory

__all__ = [
    "RapidModel",
    "InferencePipeline",
    "benchmark",
    "profile_memory",
    "__version__",
]
