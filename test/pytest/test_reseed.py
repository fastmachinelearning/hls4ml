from importlib.metadata import entry_points


def test_reseed_entry_point_registered():
    eps = entry_points(group='pytest_randomly.random_seeder')
    assert any(ep.value == 'hls4ml:reseed' for ep in eps)
