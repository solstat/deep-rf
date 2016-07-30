import numpy as np


def create_geometric_iterator(mu, multiplier=1.0, upper_bound=np.Inf):
    assert (mu >= 0 and upper_bound >= 0)

    while True:
        mu = min(upper_bound, mu)
        wait_time = np.random.geometric(1.0 / (mu + 1), 1) - 1
        while wait_time > 0:
            wait_time -= 1
            yield False
        yield True
        mu *= multiplier


def create_poisson_iterator(mu, multiplier=1.0, upper_bound=np.Inf):
    assert(mu >= 0 and upper_bound >= 0)

    while True:
        mu = min(upper_bound, mu)
        wait_time = np.random.poisson(mu, 1)
        while wait_time > 0:
            wait_time -= 1
            yield False
        yield True
        mu *= multiplier


def create_fixed_iterator(mu, multiplier=1.0, upper_bound=np.Inf):
    assert(mu >= 0 and upper_bound >= 0)

    while True:
        mu = min(upper_bound, mu)
        wait_time = np.ceil(mu)
        while wait_time > 0:
            wait_time -= 1
            yield False
        yield True
        mu *= multiplier