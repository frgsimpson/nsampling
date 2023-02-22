"""
Perform nested sampling on a spectral mixture kernel for a two-dimensional problem.
Recreates Figure 4 from https://arxiv.org/abs/2010.16344
"""

import gpflow as gf
import matplotlib.pyplot as plt
import numpy as np

from optimiser import NestedSamplingOptimizer
from pattern import make_gridded_test_data, make_train_data
from spectral_mixture import build_2d_spectral_mixture

N_TRAIN = 50
N_LIVE_POINTS = 200
N_COMPONENTS = 10

train_x, train_y = make_train_data(N_TRAIN)
test_x1, test_x2, test_z, test_x = make_gridded_test_data()
delta_x = np.max(train_x) - np.min(train_x)

# Normalise scaling of inputs for simplicity
norm_train_x = train_x / delta_x
norm_test_x = test_x / delta_x

# Set up model with max attainable frequency
FUND_FREQ = 1.0
NYQ_FREQ = 20 * FUND_FREQ
spectral_kernel = build_2d_spectral_mixture(n_components=N_COMPONENTS, fundamental_freq=FUND_FREQ, nyquist_freq=NYQ_FREQ)
model = gf.models.GPR(data=(norm_train_x, train_y), kernel=spectral_kernel)

# Train
optimiser = NestedSamplingOptimizer(n_live_points=N_LIVE_POINTS, method='rslice')
hypermodel = optimiser.optimise(model)

# Inference
mean, var = hypermodel.predict_f(norm_test_x)
reshaped_mean = mean.numpy().reshape(21, 21)


# Visualise results
def visualise(z, filename: str):
    plt.contourf(test_x1, test_x2, z, levels=50)
    plt.scatter(train_x[:, 0], train_x[:, 1], color='k')
    cbar = plt.colorbar()
    cbar.mappable.set_clim(-20, 20)

    savefile = './' + filename + '.png'
    plt.savefig(savefile)
    plt.clf()


visualise(reshaped_mean, 'nested_sampling')
visualise(test_z, 'ground_truth')
