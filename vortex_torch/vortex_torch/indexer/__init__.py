r"""
Indexer-side operator API.

This module collects the core ops used on the *indexer path* of vFlow
pipelines. These operators are typically applied to query tensors or
intermediate scoring tensors to construct sparse routing decisions
(e.g., top-k page selection, attention scoring, pagewise normalization).

Included components:

- Matrix operations:
  :class:`GeMM`, :class:`GeMV`
  for page-tiled GEMM/GEMV used in similarity scoring.

- Output routing:
  :class:`topK`
  for selecting sparse page indices based on per-token scores.

- Reductions:
  :class:`Max`, :class:`Mean`, :class:`Min`, :class:`L2Norm`, :class:`Sum`
  for aggregating scores along query or key dimensions.

- Scans / normalization:
  :class:`Softmax`, :class:`Normalize`
  for in-place probability and magnitude normalization.

- Data layout transforms:
  :class:`Transpose`
  for switching between [B, N, D] and [B, D, N] style views.

- Binary/unary elementwise ops:
  :class:`Maximum`, :class:`Minimum`, :class:`Multiply`, :class:`Add`,
  :class:`Relu`, :class:`Sigmoid`, :class:`Silu`, :class:`Add_Mul`,
  :class:`Abs`.

- Utilities:
  :mod:`utils_sglang` for SGLang-related helpers.

- Runtime context:
  :class:`Context`, :func:`get_ctx`
  for accessing per-step dynamic state (page offsets, head count,
  max token budget, etc.).

These operators constitute the standard toolkit for building sparse
attention indexers in vFlow-compatible systems.
"""


from .matmul import GeMM, GeMV
from .output_func import topK
from .reduce import Max, Mean, Min, L2Norm, Sum
from .scan import Softmax, Normalize
from .transpose import Transpose
from .elementwise_binary import Maximum, Minimum, Multiply, Add
from .elementwise import Relu, Sigmoid, Silu, Add_Mul, Abs
from .save_load import Save, Load
from . import utils_sglang, compiler
from .context import Context, get_ctx
__all__ = [ 
    "GeMM", "GeMV",
    "topK",
    "Max", "Mean", "Min", "L2Norm", "Sum",
    "Softmax", "Normalize",
    "Transpose",
    "Maximum", "Minimum", "Multiply", "Add",
    "Relu", "Sigmoid", "Silu", "Add_Mul", "Abs",
    "Save", "Load",
    "utils_sglang",
    "Context",
    "get_ctx",
    "compiler",
]

