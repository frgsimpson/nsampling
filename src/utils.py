from typing import List, Tuple, Optional

import numpy as np
import gpflow as gf
import tensorflow as tf
from dynesty.utils import Results, resample_equal
from gpflow.models import GPModel


def randomise_hyperparameters(module: gf.base.Module):
    """ Sets hyperparameters to random samples from their prior distribution.
    :param module: Any GPflow module, such as a Kernel or GPModel
    """
    for param in module.trainable_parameters:
        if param.prior is not None:
            if param.prior.batch_shape == param.prior.event_shape == [] and tf.rank(param) == 1:
                sample = param.prior.sample(tf.shape(param))
            else:
                sample = param.prior.sample()
            param.assign(sample)


def assign_parameters_from_vector(model: GPModel, flat_parameters: tf.Tensor) -> GPModel:
    """Configure our model from a vector of desired parameter values."""

    start = 0
    for parameter in model.trainable_parameters:
        if parameter.prior is not None:
            param_length = tf.size(input=parameter)
            end = start + param_length

            target_value = flat_parameters[start:end]
            reshaped_target = tf.reshape(target_value, tf.shape(input=parameter))

            parameter.assign(reshaped_target)
            start += param_length

    tf.debugging.assert_equal(
        start, tf.size(input=flat_parameters), "Insufficient parameters provided for model"
    )

    return model


def evaluate_log_predictive_density_from_hypersamples(
    model: GPModel, samples: List[tf.Tensor], x_predict, f_true
):
    """
    Compute the log predictive density given samples from the posterior distribution of a model's
    hyperparameters.
    """

    n_samples = len(samples)
    n_predict = x_predict.shape[0]
    log_predictive_densities = np.zeros((n_predict, n_samples))

    for (i, sample) in enumerate(samples):
        lpd = evaluate_lpd_from_parameters(model, sample, x_predict, f_true)
        log_predictive_densities[:, i] = tf.squeeze(lpd)

    lpd = tf.math.reduce_logsumexp(input_tensor=log_predictive_densities, axis=1, keepdims=True)

    return lpd - np.log(n_samples)


def evaluate_mean_and_variance_from_hypersamples(
    model: GPModel, samples: List[tf.Tensor], x_predict
):
    """
    Evaluate the mean and variance of the Gaussian mixture distribution associated with the
    superposition of predictions from a list of parameter values.
    """

    n_samples = len(samples)
    n_predict = x_predict.shape[0]
    means = np.zeros((n_predict, n_samples))
    variances = np.zeros((n_predict, n_samples))

    for (i, sample) in enumerate(samples):
        m, v = predict_from_parameters(model, sample, x_predict)
        means[:, i] = tf.squeeze(m)
        variances[:, i] = tf.squeeze(v)

    mixture_mean = tf.reduce_mean(input_tensor=means, axis=-1)
    mixture_variance = (
        tf.reduce_mean(input_tensor=variances, axis=-1)
        + tf.reduce_mean(input_tensor=means**2, axis=-1)
        - mixture_mean**2
    )

    return mixture_mean[:, None], mixture_variance[:, None]


@tf.function
def evaluate_lpd_from_parameters(
    model: GPModel, parameter_values: tf.Tensor, x_predict: tf.Tensor, truth: tf.Tensor
) -> tf.Tensor:
    """Set model parameters, then evaluate predictive density."""

    mean, var = predict_from_parameters(model, parameter_values, x_predict)
    return model.likelihood.predict_log_density(x_predict, mean, var, truth)


def predict_from_parameters(
    model: GPModel, parameters: tf.Tensor, x_predict
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Compute the mean and variance of the predictive density for a particular set of model
    parameters.
    """

    assign_parameters_from_vector(model, parameters)
    return model.predict_f(x_predict)


def generate_equal_samples(result: Results, max_samples: Optional[int] = None):
    """
    Produce equally weighted samples of the posterior distribution from the
    [results](https://dynesty.readthedocs.io/en/latest/quickstart.html#results).
    """

    weights = np.exp(result["logwt"] - result["logz"][-1])
    postsamples = resample_equal(result.samples, weights)
    n_samples = postsamples.shape[0]

    if max_samples is not None and max_samples < n_samples:
        indices = np.random.choice(n_samples, max_samples, replace=False)
        postsamples = postsamples[indices, :]

    return postsamples
