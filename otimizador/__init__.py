# pylint: disable='missing-module-docstring'
from . import sbert


try:
    import importlib.metadata as _importlib_metadata

except ModuleNotFoundError:
    import importlib_metadata as _importlib_metadata  # type: ignore


try:
    __version__ = _importlib_metadata.version(__name__)

except _importlib_metadata.PackageNotFoundError:
    __version__ = "1.0.0"
