# Data Factory Guide
Welcome to the walkthrough for the data_factory.py script!

**Note:** Please see the `clustersim_introduction.md` file for background information, including parameter definitions.

## Overview
This is script generates the point patterns that will then be used as the training data. First, we generate
the underlying point pattern (`upp`) and overlying point pattern (`opp`) that will be used in the `clustersim` function. 

We will generate and save `n_sims` point patterns.  Unless we want them
each to have the same parameters, we will make a vector of parameters to simulate. Here, I am using 
a fixed domain and intensity for the `upp` and `opp` and fixed overall guest concentration (`pcp`) while
varying `rho_c`, `rho_b`, `cr` and `rb`.

The point patterns are generated and then saved as `.npz` objects. An additional `pattern_stats.npy` object
which contains the actual, expected, and percent error for the `pcp`, `rho_c`, and  `rho_b` is saved and the pattern 
stats are also printed out.