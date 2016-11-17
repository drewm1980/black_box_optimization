# A Black Box Optimization Library

## Motivation
This is a simple black box (i.e. derivative free) optimization library in pure python. Its most distinctive feature is that it uses a graphs for representing the search space, making it suitable for search spaces with non-trivial topologies.

This might make sense to use if:
  * The objective function is VERY expensive to evaluate
  * Derivatives are unavailable, and too expensive to approximate with a bunch of function evaluations.
  * Parameter search space is fully discrete, with known adjacency. (Grid search is a special case of this)
  * You can't afford a full grid search (simpler codes like scipy.optimizize.brute can probably do this more efficiently) 
  * Your parameter space is not well modeled by a hypercube, i.e. subsets of your parameters live on a spherical manifold, torus, SO(3),etc... Graphs with dense samplings are a very expensive representations of high dimensional spaces, but are general enough to handle non-cartesian spaces cleanly.

## Dependencies

The code is pure python 3 and depends on only on the networkx graph library.

## Contact

It was developed at Robovision as part of a multiple view stereo benchmark, located at:

https://github.com/drewm1980/multi_view_stereo_benchmark
