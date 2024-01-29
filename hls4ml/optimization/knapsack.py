import sys
import time

import numpy as np


def solve_knapsack(values, weights, capacity, implementation='CBC_MIP', **kwargs):
    '''
    A function for solving the Knapsack problem

    Args:
        - values (np.array, float): A one-dimensional array, where each entry is the value of an item
        - weights (np.array, int): An matrix, each row represents the weights of every item, in a given knapsack
        - capacity (np.array, int): A one-dimensional array, each entry is the maximum weights of a Knapsack
        - implementation (string): Algorithm to solve Knapsack problem - dynamic programming, greedy, branch and bound
        - time_limit (float): Limit (in seconds) after which the CBC or Branch & Bound should
            stop looking for a solution and return optimal so far
        - scaling_factor (float): Scaling factor for floating points values in CBC or B&B

    Returns:
        tuple containing

        - optimal_value (float): The optimal values of elements in the knapsack
        - selected_items (list): A list of indices, corresponding to the selected elements

    Notes:
        - The general formulation of the Knapsack problem for N items and M knapsacks is:
                max v.T @ x
                s.t. A @ x <= W
                v ~ (N, 1) x ~ (N, 1) A ~ (M, N) W ~ (M, 1)
                x_{i, j} = {0, 1} and <= is the generalized, element-wise inequlaity for vectors

        - Supported implementations:
            - Dynamic programming:
                - Optimal solution
                - Time complexity: O(nW)
                - Suitable for single-dimensional constraints and a medium number of items, with integer weights
            - Branch and bound:
                - Optimal
                - Solved using Google OR-Tools
                - Suitable for multi-dimensional constraints and a large number of items
            - Branch and bound:
                - Solution sub-optimal, but often better than greeedy
                - Solved using Google OR-Tools, with the CBC MIP Solver
                - Suitable for multi-dimensional constraints and a very high number of items
            - Greedy:
                - Solution sub-optimal
                - Time complexity: O(mn)
                - Suitable for highly dimensional constraints or a very high number of items

        - Most implementations require integer values of weights and capacities;
            For pruning & weight sharing this is never a problem
            In case non-integer weights and capacities are requires,
            All of the values should be scaled by an appropriate scaling factor
    '''
    if implementation not in ('dynamic', 'greedy', 'branch_bound', 'CBC_MIP'):
        raise Exception('Unknown algorithm for solving Knapsack')

    if len(values.shape) != 1:
        raise Exception(
            'Current implementations of Knapsack optimization support single-objective problems. \
                        Values must be one-dimensional'
        )

    if len(weights.shape) != 2:
        raise Exception(
            'Current implementation of Knapsack assumes weight vector is 2-dimensional,'
            'to allow for multi-dimensional Knapsack problem. \
            If solving a one-dimensional Knapsack problem, extend dimensions of weights to a one-row matrix'
        )

    if values.shape[0] != weights.shape[1]:
        raise Exception('Uneven number of items and weights')

    if not (np.all(values >= 0) and np.all(weights >= 0)):
        raise Exception('Current implementation of Knapsack problem requires non-negative values and weights')

    if not np.all(np.equal(np.mod(capacity, 1), 0)) or not np.all(np.equal(np.mod(weights, 1), 0)):
        raise Exception('Current implementation of Knapsack problem requires integer weights and capacities')

    print(f'Starting to solve Knapsack problem with {values.shape[0]} variables and a weight constraint of {capacity}')
    start = time.time()

    # Special case, empty list
    if values.shape[0] == 0:
        return 0, []

    # Special case, the sum of all weights is less than al the capacity constraints
    if np.all(np.sum(weights, axis=1) <= capacity):
        return np.sum(values), list(range(0, values.shape[0]))

    # Special case, all the item weights per knapsack are equal, so we can greedily select the ones with the highest value
    if np.all([weights[i, :] == weights[i, 0] for i in range(weights.shape[0])]):
        return __solve_knapsack_equal_weights(values, weights, capacity)

    # General cases
    if implementation == 'dynamic':
        if weights.shape[0] == 1:
            optimal_value, selected_items = __solve_1d_knapsack_dp(values, weights[0], capacity[0])
        else:
            raise Exception('Solving Knapsack with dynamic programming requires single-dimensional constraints')
    elif implementation == 'branch_bound':
        optimal_value, selected_items = __solve_knapsack_branch_and_bound(values, weights, capacity, **kwargs)
    elif implementation == 'CBC_MIP':
        optimal_value, selected_items = __solve_knapsack_cbc_mip(values, weights, capacity, **kwargs)
    else:
        optimal_value, selected_items = __solve_knapsack_greedy(values, weights, capacity)

    print(f'Time taken to solve Knapsack {time.time() - start}s')
    return optimal_value, selected_items


def __solve_1d_knapsack_dp(values, weights, capacity):
    '''
    Helper function to solve the 1-dimensional Knapsack problem exactly through dynamic programming
    The dynamic programming approach is only suitable for one-dimensional weight constraints
    Furthermore, it has a high computational complexity and it is not suitable for highly-dimensional arrays
    NOTE: The weights and corresponding weight constraint need to be integers;
    If not, the they should be scaled and rounded beforehand
    '''
    assert len(weights.shape) == 1

    # Build look-up table in bottom-up approach
    N = values.shape[0]
    K = [[0 for w in range(capacity + 1)] for i in range(N + 1)]
    for i in range(1, N + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                K[i][w] = max(values[i - 1] + K[i - 1][w - weights[i - 1]], K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]

    # Reverse Knapsack to find selected groups
    i = N
    w = capacity
    res = K[N][capacity]
    selected = []
    while i >= 0 and res > 0:
        if res == K[i - 1][w]:
            pass
        else:
            selected.append(i - 1)
            res = res - values[i - 1]
            w = w - weights[i - 1]
        i = i - 1

    return K[N][capacity], selected


def __solve_knapsack_greedy(values, weights, capacity):
    '''
    Helper function that solves the n-dimensional Knapsack algorithm with a greedy algorithm
    The greedy approach should only be used for problems with many items or highly dimensional weights
    The solution can [and often will] be sub-optimal; otherwise, dynamic programming, branch & bound etc. should be used
    '''

    # For each item, calculate the value per weight ratio (this can be thought of as item efficiency)
    # The weights are scaled for every dimension, to avoid inherent bias towards large weights in a single dimension
    weights_rescaled = weights / np.max(weights, axis=1)[:, np.newaxis]
    ratios = values / weights_rescaled.sum(axis=0)
    indices = np.argsort(ratios)

    # Greedily select item with the highest ratio (efficiency)
    optimal = 0
    selected = []
    accum = np.zeros_like(capacity)
    for i in reversed(indices):
        if np.all((accum + weights[:, i]) <= capacity):
            selected.append(i)
            optimal += values[i]
            accum += weights[:, i]
        else:
            break

    # The greedy algorithm can be sub-optimal;
    # However, selecting the above elements or the next element that could not fit into the knapsack
    # Will lead to solution that is at most (1/2) of the optimal solution;
    # Therefore, take whichever is higher and satisfies the constraints
    if values[i] > optimal and np.all(weights[:, i]) <= capacity:
        return values[i], [i]
    else:
        return optimal, selected


def __solve_knapsack_branch_and_bound(values, weights, capacity, time_limit=sys.float_info.max, scaling_factor=10e4):
    '''
    Helper function to solve Knapsack problem using Branch and Bound;
    Implemented using Google OR-Tools [weights & capacities need to be integers]
    The algorithm explores the search space (a tree of all the posible combinations, 2^N nodes),
    But discards infeasible & sub-optimal solutions

    Additional args:
        - time_limit - Time limit in seconds
            After which B&B search should stop and return a sub-optimal solution
        - scaling_factor - Factor to scale floats in values arrays;
            OR-Tools requires all values & weights to be integers;
    '''
    try:
        from ortools.algorithms import pywrapknapsack_solver
    except ModuleNotFoundError:
        raise Exception('OR-Tools not found. Please insteal Google OR-Tools from pip.')

    solver = pywrapknapsack_solver.KnapsackSolver(
        pywrapknapsack_solver.KnapsackSolver.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'BB'
    )
    solver.set_time_limit(time_limit)
    solver.Init((values * scaling_factor).astype(int).tolist(), weights.astype(int).tolist(), capacity.astype(int).tolist())
    optimal = solver.Solve()
    selected = [i for i in range(values.shape[0]) if solver.BestSolutionContains(i)]
    return optimal / scaling_factor, selected


def __solve_knapsack_cbc_mip(values, weights, capacity, time_limit=sys.float_info.max, scaling_factor=10e4):
    '''
    Helper function to solve Knapsack problem using the CBC MIP solver using Google OR-Tools

    Additional args:
        - time_limit - Time limit (seconds) after which CBC solver should stop and return a sub-optimal solution
        - scaling_factor - Factor to scale floats in values arrays;
            OR-Tools requires all values & weights to be integers;
            So all of the values are scaled by a large number
    '''
    try:
        from ortools.algorithms import pywrapknapsack_solver
    except ModuleNotFoundError:
        raise Exception('OR-Tools not found. Please insteal Google OR-Tools from pip.')

    solver = pywrapknapsack_solver.KnapsackSolver(
        pywrapknapsack_solver.KnapsackSolver.KNAPSACK_MULTIDIMENSION_CBC_MIP_SOLVER, 'CBC'
    )
    solver.set_time_limit(time_limit)
    solver.Init((values * scaling_factor).astype(int).tolist(), weights.astype(int).tolist(), capacity.astype(int).tolist())
    optimal = solver.Solve()
    selected = [i for i in range(values.shape[0]) if solver.BestSolutionContains(i)]
    return optimal / scaling_factor, selected


def __solve_knapsack_equal_weights(values, weights, capacity):
    '''
    Helper function that solves the n-dimensional Knapsack algorithm with a greedy algorithm
    The assumption is that all the items have the same weight; while this seems a bit artificial
    It occurs often in pruning - e.g. in pattern pruning, each DSP block saves one DSP; however, as a counter-example
    In structured pruning, each structure can save a different amount of FLOPs (Conv2D filter vs Dense neuron)
    '''
    assert np.all([weights[i, :] == weights[i, 0] for i in range(weights.shape[0])])

    # Find items with the highest value
    indices = np.argsort(values)

    # Greedily select item with the highest ratio
    optimal = 0
    selected = []
    accum = np.zeros_like(capacity)
    for i in reversed(indices):
        if np.all((accum + weights[:, i]) <= capacity):
            selected.append(i)
            optimal += values[i]
            accum += weights[:, i]
        else:
            break

    return optimal, selected
