"""
Stellar Machine Learning Library

"""

__all__ = [
    "data",
    "layer",
    "mapper",
    "utils",
    "StellarDiGraph",
    "StellarGraph",
    "__version__",
]

from ...imports import *

# Version
from .version import __version__

# Import modules
from . import mapper, layer, utils

# Top-level imports
from .core.graph import StellarGraph, StellarDiGraph
from .core.schema import GraphSchema
from .utils.calibration import TemperatureCalibration, IsotonicCalibration
from .utils.calibration import (
    plot_reliability_diagram,
    expected_calibration_error,
)
from .utils.ensemble import Ensemble

# Custom layers for keras deserialization:
custom_keras_layers = {
    "GraphConvolution": layer.GraphConvolution,
    "GraphAttention": layer.GraphAttention,
    "GraphAttentionSparse": layer.GraphAttentionSparse,
    "SqueezedSparseConversion": layer.SqueezedSparseConversion,
    "MeanAggregator": layer.graphsage.MeanAggregator,
    "MaxPoolingAggregator": layer.graphsage.MaxPoolingAggregator,
    "MeanPoolingAggregator": layer.graphsage.MeanPoolingAggregator,
    "AttentionalAggregator": layer.graphsage.AttentionalAggregator,
    "MeanHinAggregator": layer.hinsage.MeanHinAggregator,
}
