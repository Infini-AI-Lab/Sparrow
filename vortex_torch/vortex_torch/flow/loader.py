from __future__ import annotations
import os
import uuid
import types
import inspect
import importlib.util
from typing import Any, Dict, Optional
from .flow import vFlow
from .registry import RegistryError, get as reg_get, list_keys

class FlowLoadError(Exception):
    """Errors while loading/exec user modules or resolving classes."""
    ...

class FlowInitError(Exception):
    """Errors while validating or constructing the flow instance."""
    ...

def _load_module_from_file(file_path: str) -> types.ModuleType:
    """
    Optionally execute a user file so that its internal @register(...) calls
    are performed and classes become available in the global registry.

    This does NOT return a class; it only ensures side effects (registration).
    """
    if not os.path.isfile(file_path):
        raise FlowLoadError(f"File does not exist: {file_path}")

    # Use a random module name to avoid name clashes and caching issues.
    mod_name = f"user_flow_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        raise FlowLoadError(f"Failed to create import spec for: {file_path}")

    module = importlib.util.module_from_spec(spec)
    try:
        # Execute the module; user code at top-level will run here.
        # Any @register(...) decorators inside will populate the registry.
        spec.loader.exec_module(module)
    except Exception as e:
        raise FlowLoadError(f"Failed to import user module: {e}") from e

    return module

def _validate_kwargs(cls: type, init_kwargs: Dict[str, Any]) -> None:
    """
    Validate constructor arguments against the class __init__ signature
    before actually instantiating, to produce clearer error messages.
    """
    sig = inspect.signature(cls.__init__)
    try:
        # First positional param is 'self'; bind with a dummy value.
        sig.bind_partial(None, **(init_kwargs or {}))
    except TypeError as e:
        raise FlowInitError(f"Constructor arguments mismatch: {e}") from e

def build_vflow(
    selected: str,
    init_kwargs: Optional[Dict[str, Any]] = None,
    user_file: Optional[str] = None
) -> vFlow:
    """
    Build a vFlow instance from a previously registered class.

    Args:
        selected: Registration key identifying which subclass to instantiate.
        init_kwargs: Dict of constructor kwargs for the subclass.
        user_file: Optional absolute path to a user file to execute first,
                   so that any new registrations inside it take effect.

    Behavior:
        - If user_file is provided, it will be executed (imported) to trigger
          any @register(...) calls inside the file.
        - Then we fetch the class from the global registry by 'selected'.
        - We validate constructor kwargs and instantiate the class.

    Raises:
        FlowLoadError: When file import fails or selected key is missing.
        FlowInitError: When constructor validation/instantiation fails.
    """
    if user_file:
        _load_module_from_file(user_file)

    try:
        FlowCls = reg_get(selected)
    except RegistryError as e:
        # Include available keys in the error message for better UX.
        raise FlowLoadError(f"{e} (available: {list_keys()})")

    _validate_kwargs(FlowCls, init_kwargs or {})
    try:
        return FlowCls(**(init_kwargs or {}))
    except Exception as e:
        raise FlowInitError(f"Failed to instantiate: {e}") from e
