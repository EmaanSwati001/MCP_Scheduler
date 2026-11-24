import pytest

from mcgsched.models import ClusterState
from mcgsched.mechanisms.load_balance import LoadBalance


def make_state():
    state = ClusterState.from_spec(
        {"clusters": [{"count": 3, "servers": 2, "gpus_per_server": 8}]}
    )
    state.allocate(1, 4, 0, 0)
    state.allocate(2, 8, 1, 0)
    return state


def test_index_matches_state():
    state = make_state()
    assert LoadBalance(state).index() == pytest.approx(state.load_balance_index())


def test_after_matches_state():
    state = make_state()
    lb = LoadBalance(state)
    for k in range(3):
        assert lb.after(k, 2) == pytest.approx(state.lb_index_after(k, 2))


def test_commit_matches_real_allocation():
    state = make_state()
    lb = LoadBalance(state)
    lb.commit(0, 2)
    state.allocate(3, 2, 0, 1)
    assert lb.index() == pytest.approx(state.load_balance_index())
    assert lb.after(2, 4) == pytest.approx(state.lb_index_after(2, 4))