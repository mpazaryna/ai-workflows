import pytest

from agents.hello import sum_numbers


def test_sum_numbers():
    assert sum_numbers(2, 3) == 5
    assert sum_numbers(-1, 1) == 0
    assert sum_numbers(0, 0) == 0


def test_sum_numbers_with_large_numbers():
    assert sum_numbers(1000000, 2000000) == 3000000


def test_sum_numbers_with_negative_numbers():
    assert sum_numbers(-10, -20) == -30
