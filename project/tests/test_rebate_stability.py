import numpy as np
from project.src.fairness.ot_rebate import rebate


def test_rebate_zero_mass():
    out = rebate(np.array([1.0, 2.0]), np.zeros(2), np.array([0.5]))
    assert np.all(out == 0)


def test_rebate_mass_conservation():
    donor_inc = np.array([1e6, 2e6, 5e6])
    recip_inc = np.array([1e3, 2e3, 3e3, 8e3])
    donor_tok = np.array([10., 20., 30.])
    out = rebate(donor_inc, donor_tok, recip_inc)
    assert abs(out.sum() - donor_tok.sum()) < 0.01 