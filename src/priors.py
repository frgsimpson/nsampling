""" Routines relating to the prior distribution of parameters in gpflow models. """
import re

import gpflow as gf
from gpflow import Parameter
import numpy as np
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import LogNormal, Normal, Uniform

from src.split import SplitPrior, MirrorPrior, HalfLogNormal


def set_default_priors_on_hyperparameters(module: gf.base.Module, replace_all: bool = False) -> int:
    """ Assigns default priors to any trainable hyperparameters which lack priors.

    :param module: Any GPflow module, such as a Kernel or GPModel
    :param replace_all: Whether to override existing priors assigned to the parameter
    :return number of parameters which have been assigned default priors
    """

    n_defaults = 0
    param_dict = gf.utilities.leaf_components(module)

    for path, parameter in param_dict.items():
        set_prior = parameter.trainable and (replace_all or parameter.prior is None)

        if set_prior:
            parameter.prior = load_default_prior(parameter, path)
            n_defaults += 1

    return n_defaults


def load_default_prior(parameter: Parameter, path: str = 'default') \
        -> tfp.distributions.Distribution:
    """  Assigns default priors to a parameter based upon their path and choice of transform.
    If parameter is unknown, a unit normal distribution is provided.

    :param parameter: The parameter whose prior distribution we wish to determine
    :param path: The path is helpful in determining the nature of the parameter
    :return: Prior probability distribution
    """

    parameter_name = "".join(re.split("[^a-zA-Z]*", parameter.name))
    transform_name = "null" if parameter.transform is None else parameter.transform.name

    sigma = 3.
    if path.endswith('variance') or parameter_name == 'bias':
        prior = LogNormal(loc=np.float64(-2.), scale=np.float64(sigma))
    elif path.endswith('lengthscale') or transform_name in ['exp', 'softplus']:
        prior = LogNormal(loc=np.float64(0.), scale=np.float64(sigma))
    else:
        prior = Normal(loc=np.float64(0.), scale=np.float64(1.))

    return prior


def uniform_prior(low, high) -> tfp.distributions.Distribution:
    """ Creates a uniform distribution. """
    return Uniform(low=np.float64(low), high=np.float64(high))


def load_log_variance_prior():
    """ Prior on the log of the variance per component. """
    return uniform_prior(-10, 7)


def load_log_beta_prior():
    return uniform_prior(-10, 5)


def load_treble_prior(fundamental_freq, nyquist_freq):
    """ Place a uniform prior on the treble frequencies. """
    return uniform_prior(low=fundamental_freq, high=nyquist_freq)


def load_bass_prior(fundamental_freq, scale=7):
    """ Creates a prior to span the frequencies below the fundamental frequency. """

    location = np.log(fundamental_freq)
    return HalfLogNormal(loc=location, scale=scale)


def load_frequency_prior(fundamental_freq, nyquist_freq):
    """ Frequency prior consists of separate bass and treble components. """

    bass_prior = load_bass_prior(fundamental_freq)
    treble_prior = load_treble_prior(fundamental_freq, nyquist_freq)
    frequency_prior = SplitPrior(bass_prior, treble_prior, 0.3333)

    return frequency_prior


def load_asymfrequency_prior(fundamental_freq, nyquist_freq):
    """ Frequency prior that can span positive and negative values. """

    positive_freq_prior = load_frequency_prior(fundamental_freq, nyquist_freq)
    return MirrorPrior(positive_freq_prior)
