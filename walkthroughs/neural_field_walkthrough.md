# Introduction
Welcome! This document is a walkthough for how to use this software to train a single neural field.

Note that this does not provide a walkthrough for the entire project, as the documentation is split up between files.  
For a walkthrough of the entire project, see the `README.md`. 
For the description of the `clustersim` algorithm, see `clustersim_introduction.md`. 

The goal of this is to train a single neural field to overfit a single point pattern. A neural field takes an input of
`(x, y, z)` and outputs a single value for a signal- in our case, probability of being inside a cluster.