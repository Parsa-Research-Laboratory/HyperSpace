import torch
import torch.nn as nn
from torch import device, Tensor
from typing import List, Tuple

from ...backends.base import BaseBackend
from ..base.base_module import BaseModule

def _module_device(module: nn.Module) -> torch.device:
    """
    Infer the device on which a PyTorch module resides.

    This utility determines the device of a module by inspecting its
    parameters first, then its registered buffers. If the module has
    neither parameters nor buffers, the device is assumed to be CPU.

    This function is intended for lightweight validation and testing
    utilities, where models are expected to reside on a single device.

    Arguments:
        module : nn.Module
            The PyTorch module whose device should be inferred.

    Returns:
        torch.device
            The device associated with the module's parameters or buffers,
            or CPU if none are present.
    """

    try:
        return next(module.parameters()).device
    except StopIteration:
        try:
            return next(module.buffers()).device
        except StopIteration:
            return torch.device("cpu")

def _create_dummy_input(feature_dim: int, device: device, batched: bool = False) -> Tensor:
    """
    Create a dummy input tensor for model validation.

    This function generates a random input tensor with the appropriate
    shape and moves it to the specified device. It supports both batched
    and non-batched inputs for validating model forward behavior.

    Arguments:
        feature_dim : int
            Dimensionality of the feature axis expected by the model.
        device : torch.device
            Device on which the input tensor should be allocated.
        batched : bool, optional
            If True, returns a batched input tensor of shape
            (batch_size, feature_dim). If False, returns a single
            unbatched input tensor of shape (feature_dim,).
            Defaults to False.

    Returns:
        Tensor
            A randomly initialized input tensor on the specified device.
    """

    i = None

    if batched:
        i = torch.rand((16, feature_dim,))
    else:
        i = torch.rand((feature_dim,))

    i = i.to(device)

    return i

def _validate_model(model: nn.Module, feature_dim: int, value_dim: int):
    """
    Validate a regression model's forward interface and output shapes.

    This function performs a minimal functional validation of a model by:
      1) inferring the model's device,
      2) generating dummy batched and non-batched inputs,
      3) executing forward passes, and
      4) verifying that the output shapes match the expected value dimension.

    The model is expected to accept inputs of shape:
      - (feature_dim,) for non-batched input
      - (batch_size, feature_dim) for batched input

    and return outputs of shape:
      - (value_dim,) for non-batched input
      - (batch_size, value_dim) for batched input

    Arguments:
        model : nn.Module
            The model to validate.
        feature_dim : int
            Expected dimensionality of the input feature vector.
        value_dim : int
            Expected dimensionality of the output value vector.

    Raises:
        ValueError
            If the model's output shape does not match the expected
            shape for either batched or non-batched inputs.
    """

    m_d: torch.device = _module_device(model)
    m_input: Tensor = _create_dummy_input(feature_dim, m_d)
    m_input_batch: Tensor = _create_dummy_input(feature_dim, m_d, batched=True)

    m_output: Tensor = model(m_input)
    m_output_batch: Tensor = model(m_input_batch)

    m_o_shape: Tuple = (value_dim,)
    m_o_b_shape: Tuple = (m_output_batch.shape[0], value_dim)

    if m_output.shape != m_o_shape:
        raise ValueError(f"Expected non-batched output to have shape {m_o_shape}; got {m_output.shape}")
    
    if m_output_batch.shape != m_o_b_shape:
        raise ValueError(f"Expected non-batched output to have shape {m_o_b_shape}; got {m_output_batch.shape}")

class RegressionModule(BaseModule):
    """
    Regression module for HyperSpace.

    This module provides regression operations for vectors using various methods
    such as pseudo-inverse and ridge regression.
    """

    valid_methods: List[str] = ["codebook", "neural"]
    network_needed: bool = False
    network_ready: bool = False

    def __init__(self, backend: BaseBackend, codebook: Tensor, values: Tensor,
                 method: str = "codebook", *, temperature: float = 0.10):
        """
        Initialize the RegressionModule.

        Arguments:
            backend : BaseBackend
                The backend to use for encoding operations.
            codebook : Tensor
                The set of vectors representing discrete values
            values : Tensor
                The discrete values represented by the codebook
            method: str
                The type of decoding method to leverage for the
                regression process
            temperature: float
                TODO
        """
        super().__init__()
        self.backend: BaseBackend = backend
        self.codebook: Tensor = codebook
        self.values: Tensor = values
        self.method: str = method
        self.temperature: float = temperature

        if not isinstance(backend, BaseBackend):
            raise TypeError(f"Expected the argued backend to extend the BaseBackend class; got {type(self.backend)}")
        
        if not isinstance(codebook, Tensor):
            raise TypeError(f"Expected the argued codebook to be a Tensor; got {type(codebook)}")
        
        if not isinstance(values, Tensor):
            raise TypeError(f"Expected the argued values to be Tensor; got {type(values)}")
        
        if not isinstance(method, str):
            raise TypeError(f"Expected the argued method to be string; got {type(method)}")
        
        if self.codebook.ndim != 2:
            raise ValueError(f"Codebook must be 2D with shape (batch_size, vector_dim); got {self.codebook.shape}")
        
        if self.codebook.shape[-1] != self.backend.vector_dim:
            raise ValueError(f"Codebook[-1] must match vector dim; got {self.codebook.shape[-1]} and {self.backend.vector_dim}")

        if self.values.ndim != 2:
            raise ValueError(f"Values must be 2D with shape (batch_size, value_dim); got {self.values.shape}")
        
        if self.values.shape[-1] != self.backend.value_dim:
            raise ValueError(f"Values[-1] must match value dim; got {self.values.shape[-1]} and {self.backend.value_dim}")
        
        if method not in self.valid_methods:
            raise ValueError(f"Expected method to be on of [{self.valid_methods}]; got {method}")
        
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0; got {self.temperature}")
        
        if method in ["neural"]:
            self.network_needed = True

    def __call__(self, v: Tensor) -> Tuple[Tensor, dict]:
        """
        Apply the specified regression method to the input tensor.
        """

        # --------------------
        # argument validation
        # --------------------
        if not isinstance(v, Tensor):
            raise TypeError(f"Expected v to be a Tensor; got {type(v)}")
        
        if v.ndim not in [1, 2]:
            raise ValueError(f"Expected v to be a 1D or 2D Tensor; got shape {v.shape}")
        
        if v.shape[-1] != self.backend.vector_dim:
            raise ValueError(f"Expected dimensionality of v to match the backend; got {v.shape[-1]} and {self.backend.vector_dim}")
        
        # --------------------------------------------------------
        # check if the appropriate modules are loaded for runtime
        # --------------------------------------------------------
        if self.network_needed and not self.network_ready:
            raise AttributeError(f"Unable to perform {self.method} regression without a network; please use self.load_network() method.")
        
        if not self.network_needed and self.network_ready:
            raise ValueError(f"Error: a network has been loaded when not needed.")
        
        if self.values.shape[-1] != 1:
            raise NotImplementedError("The RegressionModule doesn't support decoding with more than 1 value dimension.")
        
        # --------------------
        # Execute
        # --------------------
        if self.method == "neural":
            assert self.model is not None
            return self.model(v), {}

        # method == "codebook"
        return self._codebook_attention_decode(v), {}
    
    # -----------------------------------------------------------------
    # Core decode math: pairwise cosine similarity + attention
    # -----------------------------------------------------------------
    def _codebook_attention_decode(self, v: Tensor) -> Tensor:
        """
        Attention-style decode.

        Input:
          v: (D,) or (B, D)

        Output:
          (value_dim,) or (B, value_dim)
        """
        if v.ndim == 1:
            v2 = v.unsqueeze(0)     # (1,D)
            squeeze_batch = True
        else:
            v2 = v                  # (B,D)
            squeeze_batch = False

        # device alignment policy (strict)
        if self.codebook.device != v2.device or self.values.device != v2.device:
            raise ValueError(
                f"Device mismatch: v={v2.device}, codebook={self.codebook.device}, x_values={self.values.device}"
            )

        B = v2.shape[0]

        sims_rows = []
        for i in range(B):
            # (D,) vs (C,D) -> (C,)  (backend sees this as (D,) vs (B,D))
            s_i, _ = self.backend.similarity(v2[i], self.codebook)   # (C,)
            sims_rows.append(s_i)

        sims = torch.stack(sims_rows, dim=0)  # (B,C)

        weights = torch.softmax(sims / self.temperature, dim=-1)     # (B,C)

        # expected x: (B,) = (B,C) @ (C,)
        x_hat = weights @ self.values

        # return as (B,1) / (1,)
        # out = x_hat.unsqueeze(-1)  # (B,1)
        if squeeze_batch:
            out = x_hat.squeeze(0)   # (1,)
        else:
            out = x_hat

        return out

    def load_neural_network(self, model: nn.Module):
        """
        Load and register a fully constructed neural network module.

        This method attaches an externally defined ``torch.nn.Module`` to the
        current object after validating its structural and interface compatibility.
        Once successfully loaded, the internal ``network_ready`` flag is set,
        indicating that the instance is prepared for forward execution.

        Arguments:
            model : nn.Module
                A fully initialized PyTorch module representing the neural network
                to be used by this object.

        Raises:
            TypeError
                If ``model`` is not an instance of ``torch.nn.Module``.
            ValueError
                If ``model`` fails internal structural validation performed by
                ``_validate_model``.

        Side Effects:
            - Sets ``self.model`` to the provided neural network module.
            - Sets ``self.network_ready`` to ``True``.

        Notes:
            This method does not modify the parameters or buffers of ``model``.
            The caller is responsible for configuring the model's device placement,
            dtype, training/evaluation mode, and optimizer state prior to loading.
        """

        # --------------------
        # argument validation
        # --------------------
        if not isinstance(model, nn.Module):
            raise TypeError(f"Expected model to be an nn.Module; got {type(model)}")
        
        if not self.network_needed:
            raise ValueError("A network is being loaded when not needed.")
        
        _validate_model(
            model=model,
            feature_dim=self.backend.vector_dim,
            value_dim=self.backend.value_dim
        )
        
        self.model: nn.Module = model
        self.network_ready = True
        