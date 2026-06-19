"""Implementation of the mixing layer of ParalESN."""

__all__ = ["Mixer", "MixerConfig"]

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.utils import (
    init_bias,
    init_conv1d,
)


@dataclass
class MixerConfig:
    """Hyperparameters' configuration for the mixing layer.

    Attributes
    ----------
    in_scaling : float, optional, default=1.0
        Input scaling for the for the convolutional kernel, weights uniformly 
        initialized within (-`in_scaling`, `in_scaling`).
    bias_scaling : float, optional, default=0.0
        Bias scaling for the convolutional kernel bias, uniformly initialized within 
        (-`bias_scaling`, `bias_scaling`).
    kernel_size : int, optional, default=3
        Size of the 1D convolutional kernel.
    """
    in_scaling: float = 1.0
    bias_scaling: float = 0.0
    kernel_size: int = 3


class Mixer(nn.Module):
    """
    The (untrained) mixing layer combines the reservoir states across the hidden 
    dimension, enabling interaction among the components of the hidden state.
    
    A 1D convolutional kernel is applied to the reservoir's output states, sliding 
    across the hidden dimension. The same kernel, of shape `config.kernel_size`, is 
    applied across channels (i.e., the time dimension) of the input tensor. This means 
    we need to store only `config.kernel_size` weights and 1 bias, independently of the 
    sequence length. After the 1D convolution, we take the real part of the output and 
    apply a Tanh non-linearity. All parameters are randomly initialized and then left 
    untrained.

    Parameters
    ----------
    config : MixerConfig
        Configuration for the mixing layer.

    Attributes
    ----------
    activation : Callable[[Tensor], Tensor]
        A Tanh activation function applied to the real part of the output of the 
        convolutional operation.
    weight : Tensor
        The 1D convolutional kernel. Shape is (1, 1, `config.kernel_size`).
    bias : Tensor
        The bias of the convolutional kernel.Sshape is (1,).
    """
    def __init__(self, config: MixerConfig) -> None:
        super().__init__()
        self.config = config
        self.activation = nn.Tanh()

        weight = init_conv1d(
            shape=(1, 1, config.kernel_size),
            scaling=config.mix_scaling,
        )
        self.register_buffer("weight", weight)

        bias = init_bias(M=1, scaling=config.mix_bias_scaling)
        self.register_buffer("bias", bias)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """
        To apply the same 1D kernel across all channels/timesteps efficiently, we 
        change the view of the input channels by having channels stored in the batch 
        dimension. See answer by user `ptrblck`:
        - https://discuss.pytorch.org/t/best-way-to-convolve-on-different-channels-with-a-single-kernel/16501/3

        Parameters
        ----------
        x : Tensor
            Complex-valued input tensor of shape (batch_size (B), seq_len (T),  
            n_features (H)) or (B, H) if all timesteps or only the last timestep are  
            provided, respectively.

        Returns
        -------
        Tensor
            Real-valued output tensor of shape (B, T, H) or (B, H) depending on the 
            input.
        """
        # if the input tensors contains only the last timestep, add a dummy time 
        # dimension to be able to apply the `F.conv1d` operation
        if len(x.shape) == 2:
            x = x.view(x.size(0), 1, x.size(1))
        batch, seqlen, h_size = x.shape

        # `W_{mix}(...) + b_{mix}`
        out = F.conv1d(
            input=x.view(-1, 1, h_size),
            weight=self.weight,
            bias=self.bias,
            padding="same",
        ).view(batch, seqlen, h_size)

        # `Tanh(R(...))`
        return self.activation(out.real)
