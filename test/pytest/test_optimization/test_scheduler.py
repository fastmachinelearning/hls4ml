import numpy as np  # Use np.testing.assert_allclose due to floating point rounding errors

from hls4ml.optimization.dsp_aware_pruning.scheduler import BinaryScheduler, ConstantScheduler, PolynomialScheduler


def test_constant_scheduler():
    initial_sparsity = 0.25
    update_step = 0.10
    target_sparsity = initial_sparsity + 2.5 * update_step

    # Assert initial sparsity correct
    scheduler = ConstantScheduler(initial_sparsity=initial_sparsity, final_sparsity=target_sparsity, update_step=update_step)
    np.testing.assert_allclose(scheduler.get_sparsity(), initial_sparsity)

    # Assert update step is correct
    np.testing.assert_allclose(scheduler.update_step(), (True, initial_sparsity + update_step))

    # Assert repair step is correct
    np.testing.assert_allclose(scheduler.repair_step(), (True, initial_sparsity + 2 * update_step))

    # Assert cannot update again, since it would go over target sparsity
    np.testing.assert_allclose(scheduler.update_step(), (False, initial_sparsity + 2 * update_step))

    # Assert final (achievable) sparsity is correct
    np.testing.assert_allclose(scheduler.get_sparsity(), initial_sparsity + 2 * update_step)


def test_binary_scheduler():
    initial_sparsity = 0.25
    target_sparsity = 0.5
    threshold = 0.05

    # Assert initial sparsity correct
    scheduler = BinaryScheduler(initial_sparsity=initial_sparsity, final_sparsity=target_sparsity, threshold=threshold)
    np.testing.assert_allclose(scheduler.get_sparsity(), initial_sparsity)

    # Assert 1st update step is correct
    s1 = 0.5 * (initial_sparsity + target_sparsity)
    np.testing.assert_allclose(scheduler.update_step(), (True, s1))

    # Assert 1st repair step is correct
    s2 = 0.5 * (initial_sparsity + s1)
    np.testing.assert_allclose(scheduler.repair_step(), (True, s2))

    # Assert 2nd update step is correct
    s3 = 0.5 * (s2 + s1)
    np.testing.assert_allclose(scheduler.update_step(), (True, s3))

    # Assert 2nd repair step doest not take place, difference < threshold
    np.testing.assert_allclose(scheduler.repair_step(), (False, s3))

    # Assert final (achievable) sparsity is correct
    np.testing.assert_allclose(scheduler.get_sparsity(), s3)


def test_polynomial_scheduler():
    decay_power = 2
    maximum_steps = 2
    initial_sparsity = 0.25
    target_sparsity = 0.5

    # Assert initial sparsity correct
    scheduler = PolynomialScheduler(
        maximum_steps, initial_sparsity=initial_sparsity, final_sparsity=target_sparsity, decay_power=decay_power
    )
    np.testing.assert_allclose(scheduler.get_sparsity(), initial_sparsity)

    # Assert 1st update step is correct
    s1 = target_sparsity + (initial_sparsity - target_sparsity) * ((1 - 1 / maximum_steps) ** decay_power)
    np.testing.assert_allclose(scheduler.update_step(), (True, s1))

    # Assert 1st repair step is correct
    s2 = target_sparsity + (initial_sparsity - target_sparsity) * ((1 - 2 / maximum_steps) ** decay_power)
    np.testing.assert_allclose(scheduler.repair_step(), (True, s2))

    # Assert 2nd update step does not occur, since current_step = maximum_steps
    np.testing.assert_allclose(scheduler.update_step(), (False, s2))

    # Assert final (achievable) sparsity is correct
    np.testing.assert_allclose(scheduler.get_sparsity(), target_sparsity)
