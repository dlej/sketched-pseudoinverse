from abc import ABC, abstractmethod, abstractproperty
from functools import cache, cached_property
from typing import Callable, Iterable, Optional

import numpy as np
from pyrsistent import inc
from scipy.optimize import root_scalar


EPSILON = np.finfo(float).eps
BIGEPSILON = 1e-3


class Spectrum(ABC):
    """Spectrum of a matrix or linear operator

    This class is an abstract class for the spectrum of a matrix.
    """

    @abstractproperty
    def min_nz_eigval(self) -> float:
        """The minimum nonzero eigenvalue of the matrix"""
        pass
    
    @abstractmethod
    def rank(self) -> float:
        """The normalized rank of the matrix

        That is, returns the value of `rank(Sigma) / p` 
        """
        pass

    @cache
    def trace(self) -> float:
        """Convenience function to compute the normalized trace of the matrix

        That is, returns the value of `trace(Sigma) / p`
        """

        return self.trace_resolvent_poly(zeta=-1, num_deg=1, denom_deg=0)

    def stieltjes_transform(self, zeta: float) -> float:
        """Convenience function to compute the Stieltjes transform of the spectrum

        That is, returns the value of `trace((Sigma - zeta I)^-1) / p`
        """

        return self.trace_resolvent_poly(zeta, num_deg=0, denom_deg=1)

    @abstractmethod
    def trace_resolvent_poly(self, zeta: float, num_deg: float = 0, denom_deg: float = 1) -> float:
        """Compute trace functionals of the matrix times its resolvent to power `k`

        That is, returns the value of `trace(Sigma^k1 (Sigma - zeta I)^-k2) / p`
        """
        pass

    @abstractmethod
    def trace_resolvent_poly_other(self, B: 'Spectrum', zeta: float, num_deg: float = 0, denom_deg: float = 0) -> float:
        """Compute trace functionals of the matrix times its resolvent to power `k`,
        times another matrix `B` specified by `other`

        That is, returns the value of `trace(Sigma^k1 B (Sigma - zeta I)^-k2) / p`
        """
        pass


class DiscreteSpectrum(Spectrum):
    """Discrete spectral distribution
    """

    def __init__(self, eigvals: Iterable, weights: Optional[Iterable] = None) -> None:
        """Object constructor
        
        Distribution has `eigvals` with probabilities `weights`, so lenths should
        be equal

        """
        
        self.eigvals = np.asarray(eigvals)

        if weights is None:
            # default to uniform weight
            self.weights = np.ones_like(self.eigvals) / len(self.eigvals)

        elif len(weights) == len(self.eigvals):
            self.weights = np.asarray(weights)
            self.weights /= np.sum(self.weights)

        else:
            raise ValueError(f'Invalid weights with length {len(weights)} vs {len(self.eigvals)} of eigenvalues')
    
    @cached_property
    def min_nz_eigval(self) -> float:
        return np.min(self.eigvals[self.eigvals != 0])
        
    @cache
    def rank(self) -> float:
        return np.sum(self.weights[self.eigvals != 0])

    def trace_resolvent_poly(self, zeta: float, num_deg: float = 0, denom_deg: float = 1) -> float:
        
        if abs(zeta) < EPSILON:
            zeta -= EPSILON

        return self.weights @ (self.eigvals ** num_deg / (self.eigvals - zeta) ** denom_deg)

    def trace_resolvent_poly_other(self, B: Spectrum, zeta: float, num_deg: float = 0, denom_deg: float = 1) -> float:

        if not isinstance(B, DiscreteSpectrum):
            raise ValueError('Can only multiply with another DiscreteSpectrum.')
        if len(self) != len(B) or not np.allclose(self.weights, B.weights):
            raise ValueError(f'Spectrum lengths ({len(self)} vs {len(B)}) and weights must match.')
        
        if abs(zeta) < EPSILON:
            zeta -= EPSILON

        return self.weights @ (self.eigvals ** num_deg * B.eigvals / (self.eigvals - zeta) ** denom_deg)
    
    def __len__(self) -> int:
        return len(self.eigvals)


class BasicEquivalence(object):
    """Basic deterministic equivalence of random matrices

    This class provides functionality for computing the basic deterministic
    asymptotic equivalence for random matrices with deterministic covariance
    matrix. In the function documentation below, the following notation is used:

    `Sigma_spectrum` : a `Spectrum` object defining the spectrum of the covariance matrix `Sigma`

    `X` : a `n x p` matrix whose rows are i.i.d. samples of random vectors with covariance `Sigma`

    `p_n` : the data aspect ratio, equal to `p / n`

    `z` : the argument to the data resolvent `(X^H X / n - z I)^-1`

    `zeta` : the equivalent argument to the covariance resolvent `(Sigma + zeta I)^-1`    

    The asymptotic equivlance refers to the first-order trace functional equivalence
    
        `z (X^H X / n -z I)^-1 ~= zeta (Sigma - zeta I)^-1`

    """

    def __init__(self, Sigma_spectrum: Spectrum, p_n: float) -> None:

        self.Sigma_spectrum = Sigma_spectrum
        self.p_n = p_n
    
    def get_zeta_min(self, z: float = 0) -> float:
        """get the lower bound for solving for zeta"""
        return z - self.p_n * self.Sigma_spectrum.trace()

    @cached_property
    def zeta_0(self) -> float:

        def f(zeta):
            return self.p_n * self.Sigma_spectrum.trace_resolvent_poly(zeta, 2, 2) - 1
        
        bracket = self.get_zeta_min() + EPSILON, self.Sigma_spectrum.min_nz_eigval - EPSILON
        
        return _find_root_scalar(f, bracket=bracket, increasing=True)
    
    @cached_property
    def z_0(self) -> float:
        return self.get_z_from_zeta(self.zeta_0)

    def get_z_from_zeta(self, zeta: float) -> float:

        if zeta > self.zeta_0:
            return None

        return zeta * (1 - self.p_n * self.Sigma_spectrum.trace_resolvent_poly(zeta, 1, 1))
    
    def get_zeta_from_z(self, z: float) -> float:

        if z > self.z_0:
            return None

        def f(zeta):
            return self.get_z_from_zeta(zeta) - z
        
        bracket = self.get_zeta_min(z) + EPSILON, self.zeta_0 - EPSILON

        return _find_root_scalar(f, bracket=bracket, increasing=True)


class SketchedEquivalence(object):

    def __init__(self, A_spectrum: Spectrum, alpha: float) -> None:

        self.A_spectrum = A_spectrum
        self.alpha = alpha
        self.basic_equivalence = BasicEquivalence(self.A_spectrum, 1 / alpha)
    
    @property
    def mu_0(self) -> float:
        return -self.basic_equivalence.zeta_0

    @property
    def lamda_0(self) -> float:
        return -self.basic_equivalence.z_0

    def get_lamda_from_mu(self, mu: float) -> float:

        z = self.basic_equivalence.get_z_from_zeta(-mu)

        if z is None:
            return None
        else:
            return -z
    
    def get_mu_from_lamda(self, lamda: float) -> float:

        zeta = self.basic_equivalence.get_zeta_from_z(-lamda)

        if zeta is None:
            return None
        else:
            return -zeta

    def get_mu_prime(self, Psi: Spectrum, lamda: Optional[float] = None, mu: Optional[float] = None):

        if lamda is None and mu is None:
            raise ValueError('One of `lamda` or `mu` must be specified')
        
        if lamda is not None:
            return self._get_mu_prime(Psi, self.get_mu_from_lamda(lamda))
        elif mu is not None:
            return self._get_mu_prime(Psi, mu)
    
    def _get_mu_prime(self, Psi: Spectrum, mu: Optional[float]) -> float:

        if mu is None:
            return None
        
        # handle division by 0 issues
        if abs(mu) < EPSILON:
            mu_prime_plus = self.get_mu_prime(Psi, mu=EPSILON)
            mu_prime_minus = self.get_mu_prime(Psi, mu=EPSILON)

            if mu_prime_minus is None:
                return mu_prime_plus
            else:
                return (mu_prime_plus + mu_prime_minus) / 2
        
        # num = mu * self.A_spectrum.trace_resolvent_poly_other(Psi, -mu, 0, 2) / self.alpha
        # denom = lamda / mu ** 2 + self.A_spectrum.trace_resolvent_poly(-mu, 1, 2) / self.alpha

        num = mu ** 2 * self.A_spectrum.trace_resolvent_poly_other(Psi, -mu, 0, 2) / self.alpha
        denom = 1 - self.A_spectrum.trace_resolvent_poly(-mu, 2, 2) / self.alpha

        return num / denom


class JointSketchingEquivalence(object):

    def __init__(self, A_spectrum: Spectrum, alpha: float, eta: float, phi: float) -> None:

        self.A_spectrum = A_spectrum
        self.alpha = alpha
        self.eta = eta
        self.phi = phi

        self.T_equiv = SketchedEquivalence(self.A_spectrum, eta)

        pass

    def get_mu_max(self, lamda: float = 0) -> float:
        """get the upper bound for solving for mu"""
        return lamda + (1 / self.phi + 1) / self.eta * self.A_spectrum.trace()

    @cached_property
    def mu_0(self) -> float:

        bracket = self.T_equiv.mu_0 + BIGEPSILON, self.get_mu_max()
        return _find_root_scalar(self._mu_0_f, bracket=bracket, increasing=True)
    
    def _mu_0_f(self, mu):
        mu_prime = self.T_equiv.get_mu_prime(self.A_spectrum, mu=mu)
        RHS = self.A_spectrum.trace_resolvent_poly(-mu, 2, 2)
        RHS += mu_prime * self.A_spectrum.trace_resolvent_poly(-mu, 1, 2)
        RHS /= self.eta * self.phi
        return 1 - RHS

    @cached_property
    def theta_0(self) -> float:
        return self.get_theta_from_mu(self.mu_0)
    
    @cached_property
    def lamda_0(self) -> float:
        return self.get_lamda_from_mu(self.mu_0)
    
    def get_lamda_from_mu(self, mu: float) -> float:

        if mu < self.mu_0:
            return None
        
        # handle division by 0 issues
        if abs(mu) < EPSILON:
            lamda_plus = self.get_lamda_from_mu(EPSILON)
            lamda_minus = self.get_lamda_from_mu(-EPSILON)

            if lamda_minus is None:
                return lamda_plus
            else:
                return (lamda_plus + lamda_minus) / 2
        
        theta = self.get_theta_from_mu(mu)

        return theta * (1 + 1 / self.phi * (theta / mu - 1))
    
    def get_theta_from_mu(self, mu: float) -> float:

        if mu < self.mu_0:
            return None
        
        return self.T_equiv.get_lamda_from_mu(mu)
    
    def get_mu_from_lamda(self, lamda: float):

        if lamda < self.lamda_0:
            return None
        
        def f(mu):
            return self.get_lamda_from_mu(mu) - lamda
        
        bracket = self.mu_0 + EPSILON, self.get_mu_max(lamda)

        return _find_root_scalar(f, bracket=bracket, increasing=True)


def get_orthogonal_equiv(spectrum: Spectrum, alpha: float, lamda: float) -> float:

    assert alpha > 0

    if lamda <= 0:
        raise NotImplementedError('lamda <= 0 not implemented')

    def f(g):
        return spectrum.trace_resolvent_poly(-g, 0, 1) * (g - alpha * lamda) - (1 - alpha)
    
    return _find_root_scalar(f, bracket=(EPSILON, 1 / EPSILON), increasing=True)


def _find_root_scalar(f: Callable[[float], float], bracket: tuple[float, float], increasing: Optional[bool] = None) -> float:
    """Wrapper function for `scipy.optimize.root_scalar` with `brentq` method

    If `increasing` is not `None`, checks for incorrect signs at boundaries and
    returns relevant boundary in that case, since it is the closest to a root.
    """

    if increasing is not None:

        if increasing:
            if f(bracket[0]) > 0:
                return bracket[0]
            if f(bracket[1]) < 0:
                return bracket[1]
        
        if not increasing:
            if f(bracket[0]) < 0:
                return bracket[0]
            if f(bracket[1]) > 0:
                return bracket[1]

    res = root_scalar(f, method='brentq', bracket=bracket)
    return res.root

