import numpy as np, ot

def rebate(donor_income: np.ndarray,
           donor_tokens: np.ndarray,
           recip_income: np.ndarray,
           eps: float = 1e-3) -> np.ndarray:
    """
    Return € credit per recipient (same order as recip_income).
    Vectorised, safe against zero-mass and huge distances.
    """
    # ------------- early-exit on zero mass -------------
    mass = donor_tokens.sum()
    if mass < 1e-9 or len(recip_income) == 0:
        return np.zeros_like(recip_income, dtype=float)

    # ------------- rescale incomes to [0, 1] -----------
    scale = max(donor_income.max(), recip_income.max())
    d = donor_income / scale
    r = recip_income / scale

    a = donor_tokens / mass                       # probability on donors
    b = np.ones(len(r)) / len(r)                 # equal weight recipients
    M = ot.utils.dist(d[:, None], r[:, None], metric="euclidean")

    # Use the *stabilised* Sinkhorn variant – less overflow prone
    γ = ot.bregman.sinkhorn_stabilized(a, b, M, reg=eps, numItermax=1000)

    return γ.sum(0) * mass                       # back to € units
