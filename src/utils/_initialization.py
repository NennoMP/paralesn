__all__ = [
    "init_bias",
    "init_conv1d",
    "init_diag_matrix",
    "init_input_matrix",
]


import numpy as np
import torch
from torch import Tensor


def init_input_matrix(
    M: int,
    N: int,
    lambdas: Tensor,
    return_scaling: bool = False,
) -> Tensor:
    """Initialize a (`M`, `N`) complex-valued, dense matrix.

    The matrix is initialized uniformly in the range [-1, 1] and then scaled by 
    `lambdas`, a vector containing the eigenvalues of the reservoir diagonal recurrent 
    weight matrix. Specifically, the scalings are computed as `sqrt(1 - |lambda_i|^2)`  
    for each eigenvalue `lambda_i`, which ensures that the input weights are 
    appropriately scaled to maintain stability in the reservoir dynamics.

    Parameters
    ----------
    M : int
        Number of rows.
    N : int
        Number of columns.
    lambdas : Tensor
        Eigenvalues (i.e., diagonal elements) of the reservoir diagonal recurrent 
        weight matrix.
    return_scaling : bool, optional, default=False
        If True, the scaling factor is returned. Useful for ring topology reservoirs.
        
    Returns
    -------
    Tensor
        A (`M`, `N`) complex-valued tensor containing the initialized fully-connected
        matrix.
    """
    W = torch.zeros((M, N), dtype=torch.complex64).uniform_(-1, 1)

    # scale by sqrt(1 - |lambda_i|^2)
    scaling = torch.sqrt(1 - torch.abs(lambdas) ** 2).reshape(1, -1) # (N, 1)
    if return_scaling:
        return scaling
    W *= scaling

    return W


def init_diag_matrix(
    M: int,
    leaky: float,
    rho: tuple[float, float],
    phase: tuple[float, float],
) -> Tensor:
    """Initialize a (`M`, `M`) complex-valued, diagonal matrix.

    The diagonal elements are sampled in a desired subspace of the complex plane, which 
    makes it easy to control the eigenvalues of the matrix. The magnitudes of the 
    eigenvalues are sampled in the range specified by `rho`, and the angles of the 
    eigenvalues are sampled in the range specified by `phase`. Then, to account for 
    leakage, the eigenvalues are shifted towards 1 by a factor of `leaky`. The matrix 
    is stored as a 1D vector of shape (`M`,), as we are only interested in the 
    non-zero, diagonal elements.

    Parameters
    ----------
    M : int
        Number of rows/columns.
    leaky : float
        Leaky rate, used to shift the center of the eigenvalues.
    rho : tuple[float, float]
        A tuple of the form (rho_min, rho_max) specifying the desired range of the 
        radius (magnitude) of the eigenvalues.
    phase : tuple[float, float]
        A tuple of the form (phase_min, phase_max) specifying the desired range of the 
        phase (angle) of the eigenvalues, in radians.

    Returns
    -------
    Tensor
        A (`M`,) complex-valued tensor containing the diagonal elements of the matrix.
    """
    rho_min, rho_max = rho
    phase_min, phase_max = phase
    
    # radii (magnitudes)
    radii = torch.sqrt(torch.zeros(M).uniform_(rho_min**2, rho_max**2))
    # phases (angles)
    thetas = torch.zeros(M).uniform_(phase_min, phase_max)

    return (1 - leaky) + leaky * (radii * torch.exp(1j * thetas))


def init_conv1d(
    shape: tuple[int, ...],
    scaling: float,
) -> Tensor:
    """
    Initialize a complex-valued, 1D convolutional kernel.
    
    The kernel is initialized uniformly in the range [-1, 1] and then scaled by the
    specified `scaling` factor. Then, the kernel is normalized to have a unit
    norm, useful for ensuring that the convolution operation does not 
    disproportionately amplify or attenuate the input signal.

    Parameters
    ----------
    shape: tuple[int, ...],
        Shape of the kernel to be initialized.
    scaling : float,
        Scaling factor for the kernel weights.

    Returns
    -------
    Tensor
        A (`shape`,) complex-valued tensor.
    """
    K = torch.zeros(shape, dtype=torch.complex64).uniform_(-1, 1)
    K *= scaling / torch.linalg.norm(K)
    return K


def init_bias(
    M: int,
    scaling: float,
) -> Tensor:
    """Initialize a (`M`,) complex-valued bias vector.

    The bias vector is initialized uniformly in the range [-`scaling`, `scaling`].
    specified `scaling` factor.

    Parameters
    ----------
    M : int
        Size of the bias vector.
    scaling : float
        Scaling factor for the bias

    Returns
    -------
    Tensor
        A (`M`,) complex-valued tensor containing the initialized bias vector.
    """
    return torch.zeros(M, dtype=torch.complex64).uniform_(-scaling, scaling)