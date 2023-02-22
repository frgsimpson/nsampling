import numpy as np
import tensorflow as tf
from tensorflow_probability.python.distributions import Distribution, Uniform, LogNormal


class SplitPrior:
    """ Splices together two disjoint distributions so that they may be treated as one.
    Partially mimics a tf Distribution in that it supports quantile and sample methods. """

    def __init__(
        self, primary_dist: Distribution, secondary_dist: Distribution, split_point: float
    ):
        self.primary = primary_dist
        self.secondary = secondary_dist
        self.split = split_point
        self.event_shape = primary_dist.event_shape
        self.random_generator = Uniform(low=np.float64(0.0), high=np.float64(1.0))

    def quantile(self, x: tf.Tensor) -> tf.Tensor:
        """  Returns quantile evaluated at x. """

        is_primary = x < self.split
        primary_quantile = self.primary.quantile(x / self.split)
        secondary_index = (x - self.split) / (1 - self.split)
        secondary_quantile = self.secondary.quantile(secondary_index)

        return tf.where(is_primary, primary_quantile, secondary_quantile)

    def sample(self, sample_shape):
        """ Allows random samples to be drawn from the prior. """

        random_quantile_sample = self.random_generator.sample(sample_shape=sample_shape)
        return self.quantile(random_quantile_sample)


class MirrorPrior:
    """ Takes a prior distribution defined on positive x and mirrors it about zero. """

    def __init__(self, positive_distribution: Distribution):
        self.positive_distribution = positive_distribution
        self.event_shape = positive_distribution.event_shape
        self.random_generator = Uniform(low=np.float64(0.), high=np.float64(1.))

    def quantile(self, x: tf.Tensor) -> tf.Tensor:
        """  Returns quantile evaluated at x """

        x_mirror = tf.abs(2 * (x - 0.5))
        output = self.positive_distribution.quantile(x_mirror)

        return tf.where(x < 0.5, tf.math.negative(output), output)

    def sample(self, sample_shape):
        """ Allows random samples to be drawn from the prior. """

        random_quantile_sample = self.random_generator.sample(sample_shape=sample_shape)
        return self.quantile(random_quantile_sample)


class HalfLogNormal(LogNormal):
    """ Provides quantiles over the first half of a lognormal distribution. """

    def quantile(self, value, name='quantile', **kwargs):
        return super().quantile(value * 0.5,  **kwargs)