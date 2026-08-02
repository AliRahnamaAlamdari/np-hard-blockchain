"""Pluggable NP-hard problem classes for the useful-work layer (reviewer comment 9).

Importing this package registers every built-in class. Select one by name with
:func:`get_problem_class`.
"""

from .base import (
    ProblemClass,
    ProblemInstance,
    Solution,
    available_classes,
    get_problem_class,
    register,
)

# Importing each module runs its register() call.
from . import tsp_class  # noqa: E402,F401
from . import knapsack  # noqa: E402,F401
from . import graph_coloring  # noqa: E402,F401
from . import sat  # noqa: E402,F401

__all__ = [
    "ProblemClass",
    "ProblemInstance",
    "Solution",
    "available_classes",
    "get_problem_class",
    "register",
]
