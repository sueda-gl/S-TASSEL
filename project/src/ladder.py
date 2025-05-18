import numpy as np
from sklearn.isotonic import IsotonicRegression

def project_sorted_positive(p: np.ndarray) -> np.ndarray:
    """Isotonic projection onto 0 < p₁ < … < pₖ (O(K))."""
    iso = IsotonicRegression(increasing=True)
    p_sorted = iso.fit_transform(range(len(p)), p)
    # tiny ε to enforce strict monotonicity
    return np.maximum(p_sorted + 1e-6 * np.arange(1, len(p)+1), 1e-3)

def gini(eff: np.ndarray) -> float:
    n = len(eff)
    if n == 0:
        return 0.0
    return np.abs(eff[:, None] - eff[None, :]).sum() / (2 * n**2 * eff.mean())

def update_prices(p: np.ndarray,
                  sales: np.ndarray,
                  revenue: float,
                  g: float,
                  sold: float,
                  cfg) -> np.ndarray:
    grad = (-sales
            + cfg.beta * 2 * (g - cfg.gini_target)
            + cfg.zeta * 2 * (1 - sold))
    return project_sorted_positive(p - cfg.step_size * grad)
