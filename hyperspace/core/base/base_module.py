
class BaseModule:
    """
    Abstract base class for HyperSpace modules.
    """
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the __call__ method.")
