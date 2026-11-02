"""The simulated parameters, their prior, and the replay of the data factory's draws.

``scripts/generate_data.py`` draws four parameters for each of its 1,000 patterns
from independent uniform priors, in a fixed order, from a seeded generator. The
prior is therefore known exactly: a fact about the dataset rather than a
modelling choice.

The factory wrote only two of the four parameters to ``data/pattern_stats.npy``.
``replay_factory_draws`` reproduces all four by replaying the generator. Do not
change the number, order or distribution of the draws in either place without
regenerating the dataset: the replay depends on them, and it fails silently.
"""

import numpy as np

PARAMETER_NAMES = ("rho_c", "rho_b", "cr", "rb")

# The factory's draw ranges, in PARAMETER_NAMES order.
PRIOR_LOW = np.array([0.2, 0.0, 3.0, 0.0])
PRIOR_HIGH = np.array([1.0, 0.05, 15.0, 0.5])

# Mirrors scripts/generate_data.py exactly. Changing any of these breaks the replay.
FACTORY_SEED = 42
FACTORY_N_SIMS = 1000
FACTORY_PCP = 0.1


def replay_factory_draws(n_sims=FACTORY_N_SIMS, seed=FACTORY_SEED, pcp=FACTORY_PCP):
    """Reproduce the four parameter vectors in the factory's draw order."""
    rng = np.random.default_rng(seed)
    return {
        "rho_c": rng.uniform(low=pcp * 2, high=1.0, size=n_sims),
        "rho_b": rng.uniform(low=0.0, high=pcp * 0.5, size=n_sims),
        "cr": rng.uniform(low=3.0, high=15.0, size=n_sims),
        "rb": rng.uniform(low=0.0, high=0.5, size=n_sims),
    }
