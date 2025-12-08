from ..backends.base import BaseBackend
from .base_module import BaseModule

class CleanupModule(BaseModule):
    """
    Cleanup module for HyperSpace.

    This module provides cleanup operations for vectors using various methods
    such as resonator, identity, and hopfield network cleanup.
    """
    def __init__(self, backend: BaseBackend):
        """
        Initialize the CleanupModule.

        Arguments:
            backend : BaseBackend
                The backend to use for encoding operations.
        """
        super().__init__()
        self.backend: BaseBackend = backend

        if not isinstance(backend, BaseBackend):
            raise TypeError(f"Expected the argued backend to extend the BaseBackend class; got {type(self.backend)}")

    def __call__(self, tensor, method: str = "resonator"):
        """
        Apply the specified cleanup method to the input tensor.
        """
        return self.backend.cleanup(tensor, method)