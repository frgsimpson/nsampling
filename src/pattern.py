import numpy as np

DEFAULT_N_SAMPLES = 50


def make_train_data(n_samples: int = DEFAULT_N_SAMPLES):
    """ Train data are random samples from -6 to 6 """

    low = -6
    high = 6
    size = (n_samples, 2)
    x_train = np.random.uniform(low, high, size)
    y_train = guo_function(x_train)

    return x_train, y_train


def guo_function(x):
    """ Returns the function as defined in Figure 1 of https://arxiv.org/pdf/1806.04326.pdf

    :param x: Nx2 set of x coordinates
    :return: Nx1 array of y values
    """

    x1 = x[:, 0]
    x2 = x[:, 1]

    sqrt_term = np.sqrt(np.abs(x1 * x2))
    cos_term = np.cos(2. * x1) + np.cos(2. * x2)
    y = sqrt_term * cos_term

    return y[:, None]


def make_gridded_test_data(resolution: int = 21):

    x_min, x_max = -10, 10
    x1, x2 = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(x_min, x_max, resolution))
    X = np.vstack((x1.reshape(-1),
                   x2.reshape(-1))).T
    z = guo_function(X).reshape(21, 21)

    return x1, x2, z, X
