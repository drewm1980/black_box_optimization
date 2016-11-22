#!/usr/bin/env python3

import numpy
import scipy
import networkx
import functools

#from black_box_optimization import black_box_optimization
from black_box_optimization import *

def test_make_a_sequence():
    g = sequence([1, 2, 3], 'foo')
    assert len(g.nodes()) == 3
    assert g.name == 'foo'


def test_make_a_cycle():
    g = cycle([1, 2, 3], 'bar')
    assert len(g.nodes()) == 3
    assert g.name == 'bar'


def test_complete():
    g = complete([1, 2, 3], 'baz')
    assert len(g.nodes()) == 3
    assert g.name == 'baz'


def test_flattening():
    xgrid = sequence([0], 'x')
    ygrid = sequence([0], 'y')
    zgrid = sequence([0], 'z')
    graph = cartesian_product((xgrid, ygrid, zgrid))
    assert len(graph.nodes()) > 0, 'cartesian_product returned an empty graph!'
    assert (0, 0, 0) in graph.nodes()


def test_make_cartesian_product():
    foo = sequence([1, 3, 5], 'foo')
    bar = sequence([2, 4, 6], 'bar')
    p = cartesian_product([foo, bar])
    assert p.number_of_nodes() == 9
    print(p.number_of_edges())

def test_product_with_a_scalar_graph1():
    foo = sequence([1, 3, 5], 'foo')
    bar = sequence([2], 'bar')
    assert all((g.name != '' for g in [foo,bar]))
    p = cartesian_product([foo,bar])
    assert p.name == 'foo,bar'

def test_product_with_a_scalar_graph2():
    foo = sequence([1, 3, 5], 'foo')
    bar = sequence([2], 'bar')
    p = cartesian_product([bar,foo])
    assert p.name == 'bar,foo'

def test_product_with_a_scalar_graph3():
    foo = sequence([1, 3, 5], 'foo')
    bar = sequence([2], 'bar')
    p = strong_product([bar,foo])
    assert p.name == 'bar,foo'

def test_product_with_more_than_two_graphs():
    foo = sequence([1, 3], 'foo')
    bar = sequence([2, 4], 'bar')
    baz = sequence([10, 20], 'baz')

    p = cartesian_product([foo, bar, baz])
    assert p.name == 'foo,bar,baz'
    assert (1,2,10) in p

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


def test_graph_node_to_dict():
    grid,f,seed,optimum = toy_grid_search_problem()
    d = graph_node_to_dict(grid, seed)
    assert d=={'x':seed[0],'y':seed[1],'z':seed[2]}


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
    # By default, run whatever test I'm currently working on fixing up.
    test_make_cartesian_product()
    #test_flattening()
    #test_best_neighbor_descent()
    #test_exhaustive_search()
    #test_greedy_neighbor_descent()
    #test_graph_node_to_dict()
    #test_product_with_a_scalar_graph()
