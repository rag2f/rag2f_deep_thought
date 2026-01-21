"""Source package for OpenAI Embedder plugin"""

import importlib
import os

_version_module = None
try:
    _version_module = importlib.import_module(f"{__name__}._version")
except ImportError:
    _version_module = None

if _version_module is not None:
    __version__ = _version_module.__version__
    __version_tuple__ = _version_module.__version_tuple__
    __commit__ = _version_module.__commit__
    __distance__ = _version_module.__distance__
    __dirty__ = _version_module.__dirty__
else:
    __version__ = "unknown"
    __version_tuple__ = ()
    __commit__ = "unknown"
    __distance__ = 0
    __dirty__ = False

__all__ = [
    "get_plugin_path",
    "__version__",
    "__version_tuple__",
    "__commit__",
    "__distance__",
    "__dirty__",
]


def get_plugin_path():
    """Return the absolute path to the plugin directory."""
    # This file is in src/, plugin root is parent directory
    return os.path.dirname(os.path.abspath(__file__))
