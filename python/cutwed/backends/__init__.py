"""Backend implementations for TWED algorithm."""

import importlib
import warnings
from typing import Optional, Dict, Callable, Any, List, Union

# Dictionary of available backends
_BACKENDS = {}
_CURRENT_BACKEND = None

def register_backend(name: str, module_name: str, priority: int = 0) -> None:
    """Register a backend with the given name and module.
    
    Args:
        name: The name of the backend
        module_name: The module name to import
        priority: Priority of this backend (higher is preferred)
    """
    _BACKENDS[name] = {"module_name": module_name, "module": None, "priority": priority}


def get_available_backends() -> List[str]:
    """Get a list of available backends."""
    available = []
    for name, backend_dict in _BACKENDS.items():
        try:
            backend_module = _load_backend(name)
            if backend_module is not None:
                available.append(name)
                print(f"Successfully loaded backend: {name}")
            else:
                print(f"Failed to load backend: {name}")
        except Exception as e:
            print(f"Error loading backend {name}: {type(e).__name__}: {e}")
    return available


def _load_backend(name: str) -> Any:
    """Load a backend module if it's not already loaded."""
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend: {name}")
        
    backend_dict = _BACKENDS[name]
    if backend_dict["module"] is None:
        try:
            backend_dict["module"] = importlib.import_module(backend_dict["module_name"])
        except ImportError as e:
            raise ImportError(f"Could not import backend {name}: {e}")
    
    return backend_dict["module"]


def set_backend(name: Optional[str] = None) -> str:
    """Set the current backend.
    
    Args:
        name: The name of the backend to use, or None to use the highest priority 
              available backend

    Returns:
        The name of the backend that was selected
    """
    global _CURRENT_BACKEND
    
    if name is not None:
        if name not in _BACKENDS:
            raise ValueError(f"Unknown backend: {name}. Available backends: {list(_BACKENDS.keys())}")
        try:
            _load_backend(name)
            _CURRENT_BACKEND = name
            return name
        except ImportError as e:
            raise ImportError(f"Could not load backend {name}: {e}")
    
    # Find the highest priority available backend
    available_backends = get_available_backends()
    if not available_backends:
        raise ImportError("No backends available. Please install one of: numpy, pytorch, jax, or cupy")
    
    # Sort by priority
    available_backends.sort(key=lambda x: _BACKENDS[x]["priority"], reverse=True)
    _CURRENT_BACKEND = available_backends[0]
    return _CURRENT_BACKEND


def get_backend() -> Any:
    """Get the current backend module."""
    global _CURRENT_BACKEND
    
    if _CURRENT_BACKEND is None:
        set_backend()
    
    return _load_backend(_CURRENT_BACKEND)


def get_backend_name() -> str:
    """Get the name of the current backend."""
    global _CURRENT_BACKEND
    
    if _CURRENT_BACKEND is None:
        set_backend()
    
    return _CURRENT_BACKEND


# Register available backends with optimized priorities
register_backend("numpy", "cutwed.backends.numpy_backend", priority=0)

# On Mac with Metal, JAX tends to be much faster than PyTorch
import platform
if platform.system() == 'Darwin' and platform.processor() == 'arm':
    # Prioritize JAX on Apple Silicon
    register_backend("jax", "cutwed.backends.jax_backend", priority=40)  
    register_backend("pytorch", "cutwed.backends.torch_backend", priority=10)  # Lower priority on Mac
else:
    # Standard priorities for other platforms
    register_backend("pytorch", "cutwed.backends.torch_backend", priority=10)
    register_backend("jax", "cutwed.backends.jax_backend", priority=20)

register_backend("cupy", "cutwed.backends.cupy_backend", priority=30)
register_backend("cuda", "cutwed.backends.cuda_backend", priority=50)  # Highest priority

# Try to set the best available backend
try:
    set_backend()
except ImportError:
    warnings.warn("No backend could be loaded. Please install one of: numpy, pytorch, jax, or cupy")