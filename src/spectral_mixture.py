from typing import Optional

import numpy as np
import tensorflow as tf
from gpflow import Parameter
from gpflow.kernels import Sum, Kernel

from src.priors import load_log_variance_prior, load_frequency_prior, load_log_beta_prior, \
    load_asymfrequency_prior


def build_2d_spectral_mixture(
                              n_components: int,
                              nyquist_freq: float,
                              fundamental_freq: float = 1.0,
                              ):
    """  Adds together several spectral components, configured with priors to match the data.

    :param n_components: Total number of spectral components
    :param nyquist_freq: Maximum frequency considered, equal to half the sampling frequency
    :param fundamental_freq: Minimum frequency supported by the data, demarcates bass and treble
    """

    kernel_list = []
    n_nonperiodic = n_components
    variance_prior = load_log_variance_prior()

    for q in range(n_nonperiodic):
        component_name = "c" + str(q)
        component = Integrable2DSpectralComponent(
            name=component_name,
            nyquist_freq=nyquist_freq,
            fundamental_freq=fundamental_freq,
            variance_prior=variance_prior,
        )
        kernel_list.append(component)

    return Sum(kernel_list)


class Integrable2DSpectralComponent(Kernel):
    """ An implementation of the spectral mixture kernel designed to permit easier integration
    of its hyperparameters. Hyperparameters are specified such that their priors ought to be uniform. """

    def __init__(self,
                 nyquist_freq: float,
                 fundamental_freq: float,
                 variance: float = 1.0,
                 bandwidth: float = 1.0,
                 bandwidth2: float = 1.0,
                 freq: float = 1.0,
                 freq2: float = 1.0,
                 trainable_freq: bool = True,
                 active_dims=None,
                 variance_prior=None,
                 name: str = 'component'):
        """

        :param variance: Variance associated with the component
        :param lengthscale: Defined as the inverse of the mean frequency
        :param bandwidth: Frequency range spanned by the spectral component
        :param active_dims:
        """
        super().__init__(active_dims=active_dims, name=name)

        if variance_prior is None:
            variance_prior = load_log_variance_prior()

        self.input_dim = 2
        self.fundamental_freq = fundamental_freq
        self.nyquist_freq = nyquist_freq

        variance_name = 'variance'
        freq_name = 'frequency_' + name
        freq_prior = load_frequency_prior(fundamental_freq, nyquist_freq)
        asym_freq_prior = load_asymfrequency_prior(fundamental_freq, nyquist_freq)
        log_beta_prior = load_log_beta_prior()

        self.log_variance = Parameter(np.log(variance), prior=variance_prior, name=variance_name)

        # One frequency must be strictly positive, others have freedom to be negative
        freq0 = Parameter(freq, prior=freq_prior, name=freq_name, trainable=trainable_freq)
        freq1 = Parameter(freq2, prior=asym_freq_prior, name=freq_name, trainable=trainable_freq)

        logb0 = Parameter(np.log(bandwidth), prior=log_beta_prior, name='log bandwidth0')
        logb1 = Parameter(np.log(bandwidth2), prior=log_beta_prior, name='log bandwidth1')

        self.freq = [freq0, freq1]
        self.log_beta = [logb0, logb1]

    @property
    def variance(self):
        return tf.exp(self.log_variance)

    @property
    def bandwidth(self):
        """ Determine bandwidth of component. """
        return tf.exp(self.log_beta[0]) * self.fundamental_freq

    @property
    def bandwidth2(self):
        """ Determine bandwidth of component. """
        return tf.exp(self.log_beta[1]) * self.fundamental_freq

    def K(self, X: tf.Tensor, X2: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Calculates the kernel matrix K(X, X2) (or K(X, X) if X2 is None).
        Handles the slicing as well as scaling and computes k(x, x') = k(r),
        where r = x - x'.
        Would be faster to compute this just once for the full sum over components,
        but computing the form of K is unlikely to be a bottleneck.
        """

        if X2 is None:
            X2 = X

        # Introduce dummy dimension so we can use broadcasting to compute all pairwise separations
        f = tf.expand_dims(X, -2)  # ... x N x 1 x D
        f2 = tf.expand_dims(X2, -3)  # ... x 1 x M x D

        r = f - f2  # N x M x D
        original_shape = r.shape
        reshaped_r = tf.reshape(r, (-1, self.input_dim))

        output = self.K_r(reshaped_r)

        return tf.reshape(output, original_shape[:2])

    def K_diag(self, X: tf.Tensor) -> tf.Tensor:
        """ Evaluate the kernel at the provided X values. """
        return tf.fill(X.shape[:-1], tf.squeeze(self.variance))

    def K_r(self, r: tf.Tensor) -> tf.Tensor:
        """ Evaluate the kernel as a function of separation r.
         Note that this r is not rescaled as it is in native GPflow kernels.
        """

        cos_term = tf.cos(2. * np.pi * (r[:, 0] * self.freq[0] + r[:, 1] * self.freq[1]))

        rb0 = r[:, 0] * self.bandwidth
        rb1 = r[:, 1] * self.bandwidth2

        exp_term = tf.exp(-2 * np.pi ** 2 * (tf.square(rb0) + tf.square(rb1)))

        return self.variance * exp_term * cos_term
