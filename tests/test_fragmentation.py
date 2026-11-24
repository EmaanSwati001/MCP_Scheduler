import pytest

from mcgsched.models import Job
from mcgsched.mechanisms.fragmentation import DemandModel


def jobs_of(*sizes):
    return [Job(i, 0.0, g, 10.0) for i, g in enumerate(sizes)]


def global_model():
    m = DemandModel(lam=1.0, use_queue=False)
    m.fit(jobs_of(1, 1, 2, 4), 8)
    return m


def test_F_values():
    t = global_model().table(1)
    assert t[0] == 0
    assert t[1] == pytest.approx(0.5)
    assert t[2] == pytest.approx(0.5)
    assert t[3] == pytest.approx(0.75)
    assert t[4] == 0


def test_gradient():
    m = global_model()
    assert m.gradient(4, 1) == pytest.approx(0.75)
    assert m.gradient(4, 4) == pytest.approx(0)
    assert m.gradient(3, 1) == pytest.approx(-0.25)


def test_queue_blend_excludes_current_job():
    m = DemandModel(lam=1.0)
    m.fit(jobs_of(1, 1, 2, 4), 8)
    m.set_queue(jobs_of(2, 2, 4))
    assert m.table(2)[1] == pytest.approx(0.75)


def test_single_job_queue_uses_global_only():
    m = DemandModel(lam=1.0)
    m.fit(jobs_of(1, 1, 2, 4), 8)
    m.set_queue(jobs_of(2))
    assert m.table(2)[1] == pytest.approx(0.5)