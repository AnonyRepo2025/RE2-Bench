import functools
import inspect
import itertools
import warnings
import pkg_resources
from .common import BACKEND_ENTRYPOINTS, BackendEntrypoint

STANDARD_BACKENDS_ORDER = ["netcdf4", "h5netcdf", "scipy"]

def build_engines(pkg_entrypoints):
    backend_entrypoints = {}
    for backend_name, backend in BACKEND_ENTRYPOINTS.items():
        if backend.available:
            backend_entrypoints[backend_name] = backend
    pkg_entrypoints = remove_duplicates(pkg_entrypoints)
    external_backend_entrypoints = backends_dict_from_pkg(pkg_entrypoints)
    backend_entrypoints.update(external_backend_entrypoints)
    backend_entrypoints = sort_backends(backend_entrypoints)
    set_missing_parameters(backend_entrypoints)
    return {name: backend() for name, backend in backend_entrypoints.items()}
