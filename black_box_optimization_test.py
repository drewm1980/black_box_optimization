#!/usr/bin/env python3

import numpy
import scipy
import networkx
import functools

#from black_box_optimization import black_box_optimization
from black_box_optimization import *

def test_make_a_sequence():
    g = sequence('foo', [1, 2, 3])
    assert len(g.nodes()) == 3
    assert g.name == 'foo'


def test_make_a_cycle():
    g = cycle('bar', [1, 2, 3])
    assert len(g.nodes()) == 3
    assert g.name == 'bar'


def test_complete():
    g = complete('baz', [1, 2, 3])
    assert len(g.nodes()) == 3
    assert g.name == 'baz'


def test_flattening():
    xgrid = sequence('x', [0])
    ygrid = sequence('y', [0])
    zgrid = sequence('z', [0])
    graph = cartesian_product((xgrid, ygrid, zgrid))
    assert len(graph.nodes()) > 0, 'cartesian_product returned an empty graph!'
    assert (0, 0, 0) in graph.nodes()


def test_make_cartesian_product():
    foo = sequence('foo', [1, 3, 5])
    bar = sequence('bar', [2, 4, 6])
    p = cartesian_product([foo, bar])
    assert p.number_of_nodes() == 9
    print(p.number_of_edges())

def test_product_with_a_scalar_graph1():
    foo = sequence('foo', [1, 3, 5])
    bar = sequence('bar', [2])
    assert all((g.name != '' for g in [foo,bar]))
    p = cartesian_product([foo,bar])
    assert p.name == 'foo,bar'

def test_product_with_a_scalar_graph2():
    foo = sequence('foo', [1, 3, 5])
    bar = sequence('bar', [2])
    p = cartesian_product([bar,foo])
    assert p.name == 'bar,foo'

def test_product_with_a_scalar_graph3():
    foo = sequence('foo', [1, 3, 5])
    bar = sequence('bar', [2])
    p = strong_product([bar,foo])
    assert p.name == 'bar,foo'

def test_product_with_more_than_two_graphs():
    foo = sequence('foo', [1, 3])
    bar = sequence('bar', [2, 4])
    baz = sequence('baz', [10, 20])
    p = cartesian_product([foo, bar, baz])
    assert p.name == 'foo,bar,baz'
    assert (1,2,10) in p

def test_product_with_mixed_types():
    foo = sequence('foo', [1, 3])
    bar = sequence('bar', [2.0, 4.0])
    baz = sequence('baz', ['a', 'b'])
    p = cartesian_product([foo, bar, baz])
    assert p.name == 'foo,bar,baz'
    assert (1,4.0,'a') in p
    assert (3,4.0,'a') in p


def test_product_with_none():
    foo = sequence('foo', [1, 3])
    bar = sequence('bar', [None])
    baz = sequence('baz', [2, 4])
    p = cartesian_product([foo, bar, baz])
    assert p.name == 'foo,bar,baz'
    assert (1,None,2) in p
    assert (3,None,4) in p


def test_product_with_long_list():
    listLength = 8
    l = [sequence('foo', [1, 2])] * listLength
    p = cartesian_product(l)
    assert tuple([2]*listLength) in p


def toy_grid_search_problem():
    @functools.lru_cache(maxsize=None)
    def f(xyz):
        x,y,z = xyz
        return x * x + 3 * y * y + 4 * z * z
    w = 2 # half width 
    xgrid = sequence('x', range(-w, w + 1))
    ygrid = sequence('y', range(-w, w + 1))
    zgrid = sequence('z', range(-w, w + 1))
    grid = cartesian_product((xgrid, ygrid, zgrid))
    seed = (w, w, w)
    optimum = (0,0,0)
    return grid, f, seed, optimum


def test_graph_node_to_dict():
    grid, f, seed, optimum = toy_grid_search_problem()
    d = graph_node_to_dict(grid, seed)
    assert d == {'x': seed[0], 'y': seed[1], 'z': seed[2]}


def test_best_neighbor_descent():
    grid, f, seed, optimum = toy_grid_search_problem()
    stopPoint = best_neighbor_descent(grid, f, seed)
    assert stopPoint == optimum


def test_greedy_neighbor_descent():
    grid, f, seed, optimum = toy_grid_search_problem()
    stopPoint = greedy_neighbor_descent(grid, f, seed)
    assert stopPoint == optimum

def test_exhaustive_search():
    grid,f,seed,optimum = toy_grid_search_problem()
    stopPoint = exhaustive_search(grid, f)
    assert stopPoint == optimum

if __name__ == '__main__':
    # By default, run an interesting test case
    #test_make_cartesian_product()
    #test_flattening()
    #test_best_neighbor_descent()
    #test_exhaustive_search()
    #test_greedy_neighbor_descent()
    #test_graph_node_to_dict()
    #test_product_with_a_scalar_graph()
    test_product_with_long_list()

    print('Remember, to run the full test suite run "py.test"!')
