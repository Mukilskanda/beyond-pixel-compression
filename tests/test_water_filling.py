import pytest

from profiler.water_filling import water_fill_allocate


def test_empty_batch_returns_empty():
    assert water_fill_allocate([], 1000, 100, 500) == []


def test_all_get_at_least_floor():
    allocs = water_fill_allocate([1.0, 4.0, 8.0], total_budget=900, min_per_image=100, max_per_image=500)
    assert all(a >= 100 for a in allocs)


def test_none_exceed_cap():
    allocs = water_fill_allocate([1.0, 4.0, 8.0], total_budget=900, min_per_image=100, max_per_image=500)
    assert all(a <= 500 for a in allocs)


def test_higher_entropy_gets_more_or_equal_budget():
    allocs = water_fill_allocate([1.0, 4.0, 8.0], total_budget=900, min_per_image=100, max_per_image=500)
    assert allocs[0] <= allocs[1] <= allocs[2]


def test_budget_fully_spent_when_pool_is_the_binding_constraint():
    allocs = water_fill_allocate([2.0, 2.0, 2.0], total_budget=900, min_per_image=100, max_per_image=500)
    assert sum(allocs) == 900


def test_equal_entropy_images_get_equal_allocation():
    allocs = water_fill_allocate([3.0, 3.0, 3.0, 3.0], total_budget=1200, min_per_image=100, max_per_image=500)
    assert len(set(allocs)) == 1


def test_pool_large_enough_to_cap_everyone():
    """If total_budget exceeds what every image needs to hit max_per_image,
    no image should exceed the cap even though budget is left unspent."""
    allocs = water_fill_allocate([8.0, 8.0], total_budget=100000, min_per_image=100, max_per_image=500)
    assert allocs == [500, 500]
    assert sum(allocs) < 100000


def test_zero_entropy_images_still_get_floor():
    allocs = water_fill_allocate([0.0, 8.0], total_budget=900, min_per_image=100, max_per_image=500)
    assert allocs[0] == 100  # zero entropy -> zero extra, only the floor
    assert allocs[1] > 100


def test_insufficient_budget_for_floor_raises():
    with pytest.raises(ValueError, match="cannot even cover"):
        water_fill_allocate([1.0, 1.0, 1.0], total_budget=100, min_per_image=100, max_per_image=500)


def test_min_greater_than_max_raises():
    with pytest.raises(ValueError, match="min_per_image must be"):
        water_fill_allocate([1.0], total_budget=1000, min_per_image=500, max_per_image=100)


def test_invalid_max_entropy_raises():
    with pytest.raises(ValueError):
        water_fill_allocate([1.0], total_budget=1000, min_per_image=100, max_per_image=500, max_entropy=0)
