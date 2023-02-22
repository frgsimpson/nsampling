"""
This module provides functionality for training a GPflow model with the nested sampling package,
dynesty.
"""
from logging import warning
from typing import List, Optional, Sequence, Tuple

import dynesty as dy
import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import Parameter
from gpflow.models import GPModel

from hypermodel import HyperGPModel
from priors import set_default_priors_on_hyperparameters
from utils import (
    assign_parameters_from_vector,
    generate_equal_samples,
)

DEFAULT_MAX_ITER = 100_000
DEFAULT_MAX_CALL = 1_000_000
DEFAULT_DLOGZ = 0.01
N_POSTERIOR_SAMPLES = 1_000
MIN_LOG_LIKELIHOOD = -1e100


class NestedSamplingOptimizer:
    """
    This class integrates a GPflow model by utilising the dynamic nested sampling
    package, dynesty.
    """

    def __init__(self, n_live_points: int, method: str = 'auto', bound: str = 'multi',
                 n_runs: int = 1, dlogz: float = DEFAULT_DLOGZ, max_iter: int = DEFAULT_MAX_ITER,
                 max_call: int = DEFAULT_MAX_CALL):
        super().__init__()
        self.prior_functions: List[tfp.distributions.Distribution] = []
        self.posterior_samples: List[np.ndarray] = []
        self.max_log_likelihood = MIN_LOG_LIKELIHOOD
        self.n_live_points = n_live_points
        self.bound = bound
        self.method = method
        self._model: Optional[GPModel] = None
        self.n_runs = n_runs
        self.dlogz = dlogz
        self.max_iter = max_iter
        self.max_call = max_call

    def optimise(self, model: gpflow.models.GPModel) -> HyperGPModel:
        """
        Run the nested sampling algorithm and return the resulting "aggregate" model.

        See `run_nested_sampling` for details.
        """
        hyper_model, results = self.run_nested_sampling(model)
        print(f"Nested sampling complete. Model evidence: {results.logz[-1]}")
        # Store results inside the hypermodel so we can access them later
        hyper_model.results = results

        return hyper_model

    def run_nested_sampling(self, model: GPModel) -> Tuple[HyperGPModel, dy.results.Results]:
        """
        Use the sampling algorithm to integrate the priors of the parameters of the provided model.
        The results include samples from the posterior of the parameters, and the full marginal
        likelihood of the model.
        The sampler makes frequent calls to `prior_transform` and `loglikelihood`.

        :param model: The model to evaluate.
        :return: A hypermodel which contains many samples of the model hyperparameters,
            and a results dictionary as documented in [Results]
            (https://dynesty.readthedocs.io/en/latest/quickstart.html#results).
        """

        print(f"Initial model likelihood: {model.maximum_log_likelihood_objective()}")
        set_default_priors_on_hyperparameters(model)

        self._model = model
        self.prior_functions = self.define_list_of_priors(model.trainable_parameters)

        sampler = self.construct_sampler()
        results = self.collate_runs(sampler)

        hypermodel = self._build_hypermodel_from_results(results)
        self.max_log_likelihood = float(results.logl[-1])

        return hypermodel, results

    def construct_sampler(self) -> dy.NestedSampler:
        """Create either static or dynamic sampler."""

        n_parameters = len(self.prior_functions)

        print(
            f"Initiating static nested sampler over {n_parameters} parameters"
            f" and {self.n_live_points} live points"
        )
        sampler = dy.NestedSampler(
            loglikelihood=self.evaluate_log_likelihood_of_parameters,
            prior_transform=self.prior_map,
            ndim=n_parameters,
            bound=self.bound,
            sample=self.method,
            nlive=self.n_live_points,
            first_update={"min_eff": 3.0},
        )

        return sampler

    def collate_runs(self, sampler):
        """Perform a sequence of static sampling runs, and combine the results."""

        results_list = []

        for _ in range(self.n_runs):
            try:
                sampler.run_nested(
                    dlogz=self.dlogz,
                    maxiter=self.max_iter,
                    maxcall=self.max_call,
                )
                results_list.append(sampler.results)
                sampler.reset()
            except (RuntimeError, Exception) as e:
                warning(f"Nested sampling run failed: {e}")

        return dy.utils.merge_runs(results_list)

    def _build_hypermodel_from_results(self, results: dy.results.Results) -> HyperGPModel:
        """
        Construct a hypermodel based on equally weighted samples from the posterior
        distribution of the model's hyperparameters.
        """
        posterior_samples = generate_equal_samples(results, N_POSTERIOR_SAMPLES)
        return HyperGPModel(self._model, posterior_samples)

    def evaluate_log_likelihood_of_parameters(self, flat_parameters: np.ndarray) -> float:
        """
        The routine repeatedly called by dynesty, which first sets appropriate parameter values,
        before evaluating their corresponding log likelihood.

        :param flat_parameters: A vector of parameter values.
        :return: The log likelihood value.
        """

        try:
            param_tensor = tf.constant(flat_parameters)
            log_likelihood = float(self._evaluate_log_likelihood_of_parameter_tensor(param_tensor))
            if np.isnan(log_likelihood):
                log_likelihood = MIN_LOG_LIKELIHOOD
        except tf.errors.InvalidArgumentError:
            log_likelihood = MIN_LOG_LIKELIHOOD
            warning(f"Problematic parameters:{flat_parameters}")

        if log_likelihood > self.max_log_likelihood:
            self.max_log_likelihood = log_likelihood

        return log_likelihood

    @tf.function
    def _evaluate_log_likelihood_of_parameter_tensor(self, parameters: tf.Tensor) -> tf.Tensor:
        """Run rapid evaluation of the likelihood of the requested parameters."""

        assign_parameters_from_vector(self._model, parameters)
        return self._model.maximum_log_likelihood_objective()

    def evaluate_log_likelihood(self):
        """Does what it says on the tin."""

        try:
            log_likelihood = self._model.maximum_log_likelihood_objective()
        except tf.errors.InvalidArgumentError:
            log_likelihood = -1e100

        return log_likelihood

    def prior_map(self, cube: np.ndarray) -> np.ndarray:
        """
        Map the unit hypercube onto the prior distribution.

        :param cube: Dynesty will select values in the range 0 to 1 in each dimension...
        :return: ...which is then mapped to the corresponding parameter value.
        """
        hypercube_tensor = tf.constant(cube)

        return self._prior_map_with_tensors(hypercube_tensor).numpy()

    @tf.function
    def _prior_map_with_tensors(self, cube: tf.Tensor) -> tf.Tensor:
        """
        Map the unit hypercube onto the prior distribution.

        :param cube: Dynesty will select values in the range 0 to 1 in each dimension...
        :return: ..which is then mapped to the corresponding parameter value.
        """

        mapped_values = []

        for i, prior_function in enumerate(self.prior_functions):
            parameter_value = prior_function.quantile(cube[i])
            mapped_values.append(parameter_value)

        return tf.stack(mapped_values)

    @staticmethod
    def define_list_of_priors(parameters: Sequence[Parameter]) -> List[
        tfp.distributions.Distribution]:
        """
        Compile a list of priors to describe all parameters in the list.
        Here we assume that a parameter of size N holds a univariate prior that is common across
        each of the N (sub)parameters. Currently, more complex multivariate priors are not
        supported.
        """

        prior_list = []

        for parameter in parameters:
            prior = parameter.prior

            if prior is not None:
                for _ in range(np.size(parameter.numpy())):
                    prior_list.append(prior)

        return prior_list
