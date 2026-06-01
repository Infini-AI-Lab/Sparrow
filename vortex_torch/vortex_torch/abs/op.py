from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from .context_base import ContextBase
from ..utils import Mode, Schedule


class vOp(ABC):
    """Base class for defining virtual operators that support profiling and execution modes.

    This abstract base class provides a unified interface for defining
    virtual operators that have two main phases:

    - **Profiling phase** – used to pre-compute shapes, allocate buffers, or collect statistics.
    - **Execution phase** – performs the actual operator computation.

    Subclasses must implement :meth:`profile` and :meth:`execute`.
    The :meth:`__call__` method automatically dispatches between these modes
    based on the provided context.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.schedule = Schedule.S  #: Schedule type (e.g. W or S) that may be used by implementations.

    @abstractmethod
    def profile(self, *args: Any, ctx: ContextBase = None, **kwargs: Any) -> Any:
        """Abstract method: profile.

        Called during the profiling or preparation phase.

        Typical use cases:
          - Allocate persistent output buffers.
          - Compute static shapes.
          - Collect performance statistics.

        Subclasses must implement this method.

        Args:
            *args: Positional arguments.
            ctx (ContextBase, optional): The execution context.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The result of the profiling operation.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """

        raise NotImplementedError

    
    @abstractmethod
    def execute(self, *args: Any, ctx: ContextBase = None, **kwargs: Any) -> Any:
        """Abstract method: execute.

        Called during the normal execution phase.

        This method implements the actual operator logic.
        Subclasses must provide their own implementation.

        Args:
            *args: Positional arguments for the operation.
            ctx (ContextBase, optional): The execution context.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The result of the operator execution.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        
        raise NotImplementedError


    def __call__(self, *args: Any, ctx: ContextBase = None, **kwargs: Any) -> Any:
        """Callable entry point.

        Dispatches the call to either :meth:`profile` or :meth:`execute`
        depending on the execution mode specified in ``ctx``.

        Behavior:
          - If ``ctx.mode == "profile"`` → calls :meth:`self.profile(*args, **kwargs)`
          - If ``ctx.mode == "execute"`` or ``ctx.mode is None`` → calls :meth:`self.execute(*args, **kwargs)`
          - Any other mode raises :class:`ValueError`

        Args:
            *args: Positional arguments to pass to the underlying method.
            ctx (ContextBase, optional): The execution context containing the mode.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The result from either :meth:`profile` or :meth:`execute`.

        Raises:
            ValueError: If ``ctx.mode`` is not ``"profile"`` or ``"execute"``.
        """
        
        if ctx.mode is None or ctx.mode == Mode.execute:
            return self.execute(*args, ctx=ctx, **kwargs)
        if ctx.mode == Mode.profile:
            return self.profile(*args, ctx=ctx, **kwargs)
        raise ValueError(f"Unknown mode: {ctx.mode!r}, expected 'profile' or 'execute'")
    
    # ------------------------------ helpers ------------------------------ #
    def _prefix(self) -> str:
        """Prefix for assertion/log messages with class name."""
        return f"{self.__class__.__name__}: "