import numpy as np
from project.src.auction.premium import resolve_tier

def test_truthful_vs_shaded():
    bids_truth  = [("A", 5.), ("B", 3.)]
    reserves    = {a: 1.0 for a,_ in bids_truth}
    _, pay_true = resolve_tier(bids_truth.copy(), reserves)

    # A shades slightly (still wins)
    bids_shaded = [("A", 4.), ("B", 3.)]
    _, pay_shade = resolve_tier(bids_shaded.copy(), reserves)

    assert pay_true <= pay_shade   # truthful weakly better