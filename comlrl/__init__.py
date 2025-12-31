from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("comlrl")
except PackageNotFoundError:
    # Package is not installed, use a default version
    __version__ = "0.1.0-dev"

__all__ = ["__version__"]
