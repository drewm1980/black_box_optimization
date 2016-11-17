#!/usr/bin/env python3

import numpy
import scipy
import networkx
import functools

from black_box_optimization import *


def test_flattening():
    xgrid = sequence([0], 'x')
    ygrid = sequence(range(1), 'y')
    zgrid = sequence(range(1), 'z')
    graph = cartesian_product((xgrid, ygrid, zgrid))
    assert (0, 0, 0) in graph.nodes()


def test_make_a_sequence():
    sequence([1, 2, 3], 'foo')


def test_make_a_cycle():
    cycle([1, 2, 3], 'bar')


def test_complete():
    complete([1, 2, 3], 'baz')


def test_make_cartesian_product():
    foo = sequence([1, 3, 5], 'foo')
    bar = sequence([2, 4, 6], 'bar')
    p = cartesian_product([foo, bar])
    assert p.number_of_nodes() == 9
    print(p.number_of_edges())


def toy_grid_search_problem():
    @functools.lru_cache(maxsize=None)
    def f(xyz):
        x,y,z = xyz
        return x * x + 3 * y * y + 4 * z * z
    w = 2 # half width 
    xgrid = sequence(range(-w, w + 1), 'x')
    ygrid = sequence(range(-w, w + 1), 'y')
    zgrid = sequence(range(-w, w + 1), 'z')
    grid = cartesian_product((xgrid, ygrid, zgrid))
    seed = (w, w, w)
    optimum = (0,0,0)
    return grid, f, seed, optimum

def test_best_neighbor_descent():
    grid,f,seed,optimum = toy_grid_search_problem()
    stopPoint = best_neighbor_descent(grid, f, seed)
    assert stopPoint == optimum


def test_greedy_neighbor_descent():
    grid,f,seed,optimum = toy_grid_search_problem()
    stopPoint = greedy_neighbor_descent(grid, f, seed)
    assert stopPoint == optimum

def test_exhaustive_search():
    grid,f,seed,optimum = toy_grid_search_problem()
    stopPoint = exhaustive_search(grid, f)
    assert stopPoint == optimum

if __name__ == '__main__':
    #test_flattening()
    #test_best_neighbor_descent()
    #test_exhaustive_search()
    test_greedy_neighbor_descent()
