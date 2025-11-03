from .base_module import BaseModule

class MemoryStorageModule(BaseModule):
    """
    Memory Storage module for HyperSpace.

    This module provides memory storage operations for vectors using various methods
    such as key-value storage and associative memory.
    """
    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    def __call__(self, tensor, method: str = "key_value"):
        """
        Apply the specified memory storage method to the input tensor.
        """
        return self.backend.memory_storage(tensor, method)