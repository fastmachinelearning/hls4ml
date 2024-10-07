import numpy as np
import pytest

from hls4ml.optimization.dsp_aware_pruning.knapsack import solve_knapsack


# In the simple case below, both implementations give the optimal answer
# In general, the greedy algorithm will not give the optimal solution
@pytest.mark.parametrize('implementation', ['dynamic', 'greedy', 'branch_bound', 'CBC_MIP'])
def test_knapsack_1d(implementation):
    values = np.array([4, 5, 6, 8, 3])
    weights = np.array([[2, 5, 3, 2, 5]])
    capacity = np.array([8])

    optimal, selected = solve_knapsack(values, weights, capacity, implementation=implementation)
    assert optimal == 18
    assert 0 in selected
    assert 2 in selected
    assert 3 in selected


@pytest.mark.parametrize('implementation', ['greedy', 'branch_bound', 'CBC_MIP'])
def test_multidimensional_knapsack(implementation):
    values = np.array([10, 2, 6, 12, 3])
    weights = np.array([[3, 1, 4, 5, 5], [3, 2, 4, 1, 2]])
    capacity = np.array([8, 7])

    optimal, selected = solve_knapsack(values, weights, capacity, implementation=implementation)
    assert optimal == 22
    assert 0 in selected
    assert 3 in selected


def test_knapsack_equal_weights():
    values = np.array([10, 2, 6, 8, 3])
    weights = np.array([[2, 2, 2, 2, 2], [3, 3, 3, 3, 3]])
    capacity = np.array([7, 7])

    optimal, selected = solve_knapsack(values, weights, capacity)
    assert optimal == 18
    assert 0 in selected
    assert 3 in selected


def test_knapsack_all_elements_fit():
    values = np.array([10, 2, 6, 12, 3])
    weights = np.array([[3, 1, 4, 5, 5], [3, 2, 4, 1, 2]])
    capacity = np.array([19, 12])

    optimal, selected = solve_knapsack(values, weights, capacity)
    assert optimal == 33
    assert selected == list(range(0, values.shape[0]))
