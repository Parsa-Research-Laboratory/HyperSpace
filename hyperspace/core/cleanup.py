from .base_module import BaseModule

class CleanupModule(BaseModule):
    """
    Cleanup module for HyperSpace.

    This module provides cleanup operations for vectors using various methods
    such as resonator, identity, and hopfield network cleanup.
    """
    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    def __call__(self, tensor, method: str = "resonator"):
        """
        Apply the specified cleanup method to the input tensor.
        """
        return self.backend.cleanup(tensor, method)