"""Deprecated: use ``geom_mesh_net.fields.oracle``."""

import importlib
import sys
import warnings

NEW_NAME = "geom_mesh_net.fields.oracle"

warnings.warn(
    f"{__name__} is deprecated; import {NEW_NAME} instead",
    DeprecationWarning,
    stacklevel=2,
)

# Replace this module with the new one, so both names are the same object.
sys.modules[__name__] = importlib.import_module(NEW_NAME)
