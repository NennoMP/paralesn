"""Implementation of Parallel Echo State Networks (ParalESN)."""

__all__ = ["ParalESN"]

import torch
import torch.nn as nn
from torch import Tensor

from src.paralesn.layers import (
    Mixer,
    MixerConfig,
    Reservoir,
    ReservoirConfig,
)


class ParalESNLayer(nn.Module):
    """Implementation of a (`Reservoir` -> `Mixer`) layer of LinearESN.

    Parameters
    ----------
    reservoir : Reservoir
        An instance of the `Reservoir` class.
    mixer : Mixer
        An instance of the `Mixer` class.

    Attributes
    ----------
    reservoir : Reservoir
        The initialized linear reservoir component.
    mixer : Mixer
        The initialized mixing layer component.
    """

    def __init__(self, reservoir: Reservoir, mixer: Mixer) -> None:
        """
        Raises
        ------
        AssertionError
            If `reservoir` is not an instance of `Reservoir`.
        AssertionError
            If `mixer` is not an instance of `Mixer`.
        """
        super().__init__()
        assert isinstance(reservoir, Reservoir), (
            "`reservoir` must be an instance of `Reservoir`."
        )
        assert isinstance(mixer, Mixer), (
            "`mixer` must be an instance of `Mixer`."
        )

        self.reservoir = reservoir
        self.mixer = mixer

    def forward(self, x: Tensor, mode: str) -> Tensor:
        return self.mixer(self.reservoir(x, mode=mode))


class ParalESN(nn.Module):
    """
    ParalESN consists of multiple (`Reservoir` -> `Mixer`) layers. All 
    parameters are complex-valued, randomly initialized, and left untrained.
     
    The linear reservoir is essentially an untrained linear recurrent layer based on a 
    diagonal recurrent matrix. The mixer introduces non-linearity in the model and 
    combines the reservoir states across the hidden dimension, enablign interaction 
    among the components of the hidden state.

    Parameters
    ----------
    concat : bool, optional, default=False
        Whether to concatenate the hidden states of all layers or return only the last
        layer's hidden states.
    n_layers : int, optional, default=1
        Number of (`Reservoir` -> `Mixer`) layers.
    in_size : int, optional, default=1
        Number of expected input features.
    n_units : int, optional, default=128
        Number of hidden units in the reservoir.
    reservoir_config, inter_reservoir_config : ReservoirConfig, optional, default=None
        Configuration for the first reservoir layer (l=1) and subsequent reservoir 
        layers (l > 1). If `reservoir_config` is None, the default configuration is 
        used. If `inter_reservoir_config` is None, the same configuration as 
        `reservoir_config` is used.
    mixer_config, inter_mixer_config : MixerConfig, optional, default=None
        Configuration for the first mixing layer (l=1) and subsequent mixing layers (l 
        > 1). If `mixer_config` is None, the default configuration is used. If 
        `inter_mixer_config` is None, the same configuration as `mixer_config` is used.

    Attributes
    ----------
    layers_units : int
        Number of hidden units in each reservoir layer. If `concat` is True, this may
        be different from `n_units` if an even distribution of units among layers is not
        possible.
    first_layer_units : int
        Number of hidden units in the first reservoir layer. If `concat` is True, this
        may be different from `layers_units` if an even distribution of units among
        layers is not possible.
    layers : nn.Sequential
        A sequential container of (`Reservoir` -> `Mixer`) layers.
    """

    def __init__(
        self,
        concat: bool = False,
        n_layers: int = 1,
        in_size: int = 1,
        n_units: int = 128,
        reservoir_config: ReservoirConfig = None,
        inter_reservoir_config: ReservoirConfig | None = None,
        mixer_config: MixerConfig = None,
        inter_mixer_config: MixerConfig | None = None,
        **kwargs,
    ) -> None:
        """
        Raises
        ------
        AssertionError
            If `n_layers` is not greater than 0.
        AssertionError
            If `in_size` is not greater than 0.
        AssertionError
            If `n_units` is not greater than 0.
        """
        super().__init__()
        assert n_layers > 0, "`n_layers` must be greater than 0."
        assert in_size > 0, "`in_size` must be greater than 0."
        assert n_units > 0, "`n_units` must be greater than 0."

        if not reservoir_config:
            reservoir_config = ReservoirConfig()
        if not mixer_config:
            mixer_config = MixerConfig()

        self.__dict__.update(kwargs)
        self.in_size = in_size
        self.n_units = n_units
        self.n_layers = n_layers
        self.concat = concat

        self.reservoir_config = reservoir_config
        self.inter_reservoir_config = reservoir_config
        if inter_reservoir_config:
            self.inter_reservoir_config = inter_reservoir_config

        self.mixer_config = mixer_config
        self.inter_mixer_config = mixer_config
        if inter_mixer_config:
            self.inter_mixer_config = inter_mixer_config

        # if `concat == True` the number of reservoir units `n_units_ is evenly divided 
        # among the layersm if an even distribution is not possible, the extra units 
        # are allocated to the first layer
        self.layers_units = self.first_layer_units = n_units
        if concat:
            self.layers_units = n_units // n_layers
            self.first_layer_units = self.layers_units + n_units % n_layers

        self.layers = self._make_layers()

    def _make_layers(self) -> nn.Sequential:
        """Initialize the (`Reservoir` -> `Mixer`) layers of ParalESN.

        Returns
        -------
        nn.Sequential
            A sequential container of (`Reservoir` -> `Mixer`) layers.
        """
        layers = [
            ParalESNLayer(
                reservoir=Reservoir(
                    config=self.reservoir_config,
                    in_size=self.in_size,
                    n_units=self.first_layer_units,
                ),
                mixer=Mixer(
                    config=self.mixer_config,
                ),
            )
        ]

        # subsequent layers
        h_dim = self.first_layer_units
        for _ in range(1, self.n_layers):
            layers.append(
                ParalESNLayer(
                    reservoir=Reservoir(
                        config=self.reservoir_config,
                        in_size=h_dim,
                        n_units=self.layers_units,
                    ),
                    mixer=Mixer(
                        config=self.inter_mixer_config,
                    ),
                )
            )
            h_dim = self.layers_units

        return nn.Sequential(*layers)

    @torch.no_grad()
    def forward(self, x: Tensor, mode: str = "scan") -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Complex-valued input tensor of shape (batch_size (B), seq_len (T), 
            n_features (H)).
        mode : str, optional, default="scan"
            Mode for computing the recurrence. If `mode=scan`, the recurrence is 
            computed in parallel via associative scan. If `mode=recurrent`, the 
            recurrence is computed sequentially.

        Returns
        -------
        Tensor
            A complex-valued tensor of shape (B, T, H).
        """
        states = []
        for layer in self.layers:
            x = x.to(torch.complex64)
            x = layer(x, mode=mode)
            states.append(x)
        
        match self.concat:
            case False:
                return states[-1]
            case True:
                return torch.cat(states, dim=2)