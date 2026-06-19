"""Implementation of the reservoir layer of ParalESN."""

__all__ = ["Reservoir", "ReservoirConfig"]

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.utils import (
    associative_scan,
    init_bias,
    init_diag_matrix,
    init_input_matrix,
)


@dataclass
class ReservoirConfig:
    """Hyperparameters' configuration for the reservoir layer.

    Attributes
    ----------
    leaky : float, optional, default=1.0
        Leaky rate (0, 1].
    rho : tuple[float, float], optional, default=(0.0, 1.0)
        A tuple of the form (rho_min, rho_max) specifying the desired range of the 
        radius (magnitude) of the eigenvalues of the recurrent kernel.
    phase : tuple[float, float], optional, default=(0.0, 2 * np.pi)
        A tuple of the form (phase_min, phase_max) specifying the desired range of the 
        phase (angle) of the eigenvalues of the recurrent kernel.
    bias_scaling : float, optional, default=1.0
        Bias scaling for the recurrent kernel bias.
    """
    leaky: float = 1.0
    rho: tuple[float, float] = (0.0, 1.0)
    phase: tuple[float, float] = (0.0, 2 * np.pi)
    bias_scaling: float = 1.0


class Reservoir(nn.Module):
    """
    The (untrained) reservoir layer implements a linear recurrent layer, enabling 
    parallel temporal processing via associative scan. 
    
    The reservoir consists of (i) a complex-valued, diagonal recurrent weight matrix; 
    and (ii) a complex-valued input weight matrix. For the diagonal recurrent matrix, 
    we store only the non-zero entries. The input weight matrix is implemented as a 
    dense, fully-connected matrix if the input dimension is different from the number 
    of hidden units in the reservoir (very first layer), and as a ring topology 
    matrix otherwise (subsequent layers). The ring topology matrix has the following 
    form:

    C = [[0, 0, 0, w1],
        [w2, 0, 0, 0],
        [0, w3, 0, 0],
        [0, 0, w4, 0]]

    Note that, the ring topology matrix simply applies a circulat shift to the input 
    vector plus rescaling. Thus, we can store just the non-zero entries and implement 
    it as `scaling * torch.roll(x, shifts=1, dims=-1)`.

    Parameters
    ----------
    config : ReservoirConfig
        Configuration for the reservoir layer.
    in_size : int, optional, default=1
        Number of expected input features.
    n_units : int, optional, default=128
        Number of hidden units in the reservoir.

    Attributes
    ----------
    in_kernel : Tensor
        The input weight matrix of the reservoir, of shape (`in_size`, `n_units`) if `in_size` != `n_units` and ring topology is not used. Otherwise, it is 
        implemented as a lambda function that applies a circular shift and scaling to 
        the input.
    in_scaling : Tensor
        The scaling vector for the ring topology input weight matrix, of shape 
        (`n_units`,). Only defined if `in_size` == `n_units` and ring topology is used.
    recurrent_kernel : Tensor
        The recurrent weight matrix of the reservoir, of shape (`n_units`,).
    bias : Tensor
        The bias vector of the reservoir, of shape (`n_units`,).
    """

    def __init__(
        self,
        config: ReservoirConfig,
        in_size: int = 1,
        n_units: int = 128,
    ) -> None:
        super().__init__()
        self.config = config
        self.in_size = in_size
        self.n_units = n_units

        recurrent_kernel = init_diag_matrix(
            M=n_units,
            leaky=config.leaky,
            rho=config.rho,
            phase=config.phase,
        )
        self.register_buffer("recurrent_kernel", recurrent_kernel)

        if in_size != n_units: # dense
            in_kernel = init_input_matrix(
                M=in_size, N=n_units, lambdas=recurrent_kernel
            )
            self.register_buffer("in_kernel", in_kernel)
        else: # ring topology
            in_scaling = init_input_matrix(
                M=in_size, N=n_units, lambdas=recurrent_kernel, return_scaling=True
            )
            self.register_buffer("in_scaling", in_scaling)
            self.in_kernel = lambda x: self.in_scaling * torch.roll(
                x, shifts=1, dims=-1
            )

        bias = init_bias(M=n_units, scaling=config.bias_scaling)
        self.register_buffer("bias", bias)

    def _init_hidden_state(self, batch_size: int) -> Tensor:
        """Initialize a complex-valued, zero-valued hidden state for the reservoir.

        Parameters
        ----------
        batch_size : int
            Size of the batch for which to initialize the hidden state.

        Returns
        -------
        Tensor
            A complex-valued tensor of shape (batch_size (B), n_units (H)).
        """
        return torch.zeros(batch_size, self.n_units, dtype=torch.complex64)
    
    def _forward_recurrent(self, x: Tensor, h_prev: Tensor) -> Tensor:
        """Sequential recurrence (slow).

        Parameters
        ----------
        x : Tensor
            Complex-valued input tensor of shape (batch_size (B), seq_len (T), 
            n_features (H)).
        h_prev : Tensor
            Complex-valued hidden state at the previous time step, of shape (B, H).

        Returns
        -------
        Tensor
            A complex-valued tensor of shape (B, T, H) containing the hidden states at 
            each time step.
        """
        B, T, _ = x.shape
        h = torch.zeros(B, T, self.cell.n_units, device=x.device, dtype=torch.complex64)

        for t in range(T):
            h_prev = self.cell(x[:, t], h_prev)
            h[:, t, :] = h_prev

        return h


    @torch.compiler.disable
    def _forward_scan(self, x: Tensor, h_prev: Tensor) -> Tensor:
        """Parallel recurrence via associative scan (fast).

        Parameters
        ----------
        x : Tensor
            Complex-valued input tensor of shape (batch_size (B), seq_len (T), 
            n_features (H)).
        h_prev : Tensor
            Complex-valued hidden state at the previous time step, of shape (B, H).

        Returns
        -------
        Tensor
            A complex-valued tensor of shape (B, T, H) containing the hidden states at 
            each time step.
        """
        # diagonal elements of the recurrent matrix
        lambdas = self.recurrent_kernel

        # repeat lambdas to match shape (T, H)
        lambda_elements = lambdas.tile(x.shape[1], 1)

        # compute input projections
        if self.in_size != self.n_units:  # dense: linear projection with matrix
            Win_elements = self.config.leaky * (x @ self.in_kernel + self.bias)
        else:  # ring topology: circular shift
            Win_elements = self.config.leaky * (self.in_kernel(x) + self.bias)

        if h_prev is not None:
            Win_elements[:, 0, :] = Win_elements[:, 0, :] + (lambdas * h_prev)
        # Vmap the associative scan since `Win_elements` is a batch of B sequences.
        inner_state_fn = lambda Bu_seq: associative_scan(
            elems=(lambda_elements, Bu_seq),
        )[1]

        return torch.vmap(inner_state_fn)(Win_elements)

    @torch.no_grad()
    def forward(
        self,
        x: Tensor,
        mode: str,
        h_prev: Tensor | None = None,
    ) -> Tensor:
        """
        Based on `mode`, recurrence is computed sequentially or in parallel.

        Parameters
        ----------
        x : Tensor
            Complex-valued input tensor of shape (batch_size (B), seq_len (T), 
            n_features (H)).
        mode : str
            Mode for computing the recurrence. If `mode=scan`, the recurrence is 
            computed in parallel via associative scan. If `mode=recurrent`, the 
            recurrence is computed sequentially.
        h_prev : Tensor, optional, default=None
            Complex-valued hidden state at the previous time step, of shape (B, H). If 
            None, it is initialized to a zero vector.

        Returns
        -------
        Tensor
            A complex-valued tensor of shape (B, T, H) containing the hidden states at 
            each time step.
        """
        if h_prev is None:
            h_prev = self._init_hidden_state(x.shape[0]).to(x.device)

        match mode:
            case "recurrent":
                return self._forward_recurrent(x, h_prev)
            case "scan":
                return self._forward_scan(x, h_prev)
