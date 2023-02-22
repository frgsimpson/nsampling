"""
The hypermodel holds a set of different hyperparameter values which are used when making
predictions.
"""

from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.models import GPModel

from utils import (
    assign_parameters_from_vector,
    evaluate_log_predictive_density_from_hypersamples,
    evaluate_mean_and_variance_from_hypersamples,
)


class HyperGPModel(GPModel):
    """This class holds a conventional GPmodel and a list of its hyperparameter values."""

    def __init__(
        self, model: GPModel, hypersamples: List[np.ndarray], max_likelihood: float = np.nan
    ):
        """
        :param model: The GPmodel to use.
        :param hypersamples:  Random draws from the posterior distribution of a model's
            hyperparameters, such as lengthscales and variances. We use the term hypersamples here
            to avoid conflicting notations with GPflow. The number of parameter values in a
            hypersample must match the total size of trainable hyperparameters.
        : param max_likelihood: The highest log likelihood value achieved.
        """

        if model.kernel is not None:
            super().__init__(
                kernel=model.kernel,
                likelihood=model.likelihood,
                num_latent_gps=model.num_latent_gps,
            )

        if hasattr(model, "data"):
            self.data = model.data
        self.model = model

        self.n_hypersamples = len(hypersamples)
        self.hypersamples = tf.stack([tf.constant(sample) for sample in hypersamples])
        self.results = None
        self.max_likelihood = tf.constant(max_likelihood)
        self.noise_generator = tfp.distributions.Normal(loc=np.float64(0.0), scale=np.float64(1.0))

    def predict_log_density(self, data, full_cov: bool = False, full_output_cov: bool = False):
        """
        Evaluate the log probability of the datapoints ``f_true`` based on the samples of
        hyperparameter values.
        """
        x, y = data
        return evaluate_log_predictive_density_from_hypersamples(
            self.model, self.hypersamples, x, y
        )

    def predict_samples(self, Xnew, num_samples, predict_y: bool, full_cov: bool) -> tf.Tensor:
        """Generates samples from the posterior distribution of the hypermodel.
        For each sample, we need to assign a new set of hyperparameters.

        :param xnew: Input location of the samples
        :param num_samples: How many samples to draw
        :param predict_y: Whether to include the noise contribution from the likelihood
        :return: Tensor of shape num_samples x N where N is length of xnew.
        """

        samples = []
        for i in range(num_samples):
            index = tf.constant(i)
            sample = self.get_single_sample(Xnew, predict_y, index, full_cov)
            samples.append(sample)

        return tf.stack(samples)

    @tf.function
    def get_single_sample(self, Xnew, predict_y: bool, i: tf.constant, full_cov: bool) -> tf.Tensor:

        self.assign_hyperparameters(sample_number=i)
        output = self.model.predict_f_samples(Xnew, num_samples=1, full_cov=full_cov)
        sample = output[0]
        if predict_y:
            likelihood_standard_deviation = tf.sqrt(self.model.likelihood.variance)
            unit_noise_sample = self.noise_generator.sample()
            noise_sample = unit_noise_sample * likelihood_standard_deviation
            sample += noise_sample

        return sample

    def predict_y_samples(self, xnew, num_samples: int = 1, full_cov: bool = True) -> tf.Tensor:
        """Generates samples from the posterior distribution of the hypermodel,
        including the contribution from likelihood variance.

        :param xnew: Location of the samples
        :param num_samples: How many samples
        :param full_cov: Whether to include the noise contribution from the likelihood
        :return: Tensor of shape num_samples x N where N is length of xnew.
        """

        return self.predict_samples(xnew, num_samples, predict_y=True, full_cov=full_cov)

    def predict_f_samples(
        self, Xnew, num_samples: int = 1, full_cov: bool = True, full_output_cov: bool = False
    ) -> tf.Tensor:
        """Generates samples from the posterior distribution of the hypermodel,
        excluding the contribution from likelihood variance."""

        return self.predict_samples(Xnew, num_samples, predict_y=False, full_cov=full_cov)

    def predict_f(self, Xnew: tf.Tensor, full_cov: bool = False, full_output_cov: bool = False):
        r"""
        You probably shouldn't be using this function as the posterior is non-Gaussian.
        This method computes mean and variance of the posterior distribution at X \in R^{N \x D}
        input points.
        :param Xnew: Input points you shouldn't be passing in
        :param full_cov: Please don't use this function
        :param full_output_cov: I mean it.
        :return:
        """

        return evaluate_mean_and_variance_from_hypersamples(self.model, self.hypersamples, Xnew)

    def assign_hyperparameters(self, sample_number: int) -> None:
        """Sets the model to a particular set of hyperparameters.

        :param sample_number: Which sample to choose.
        """

        index = tf.math.floormod(sample_number, self.n_hypersamples)
        param_vector = self.hypersamples[index, :]
        assign_parameters_from_vector(self.model, param_vector)

    def maximum_log_likelihood_objective(self, *args, **kwargs) -> tf.constant:
        """The hypermodel potentially holds many likelihood values.
        Here we shall return the maximum likelihood value found."""

        return self.max_likelihood
