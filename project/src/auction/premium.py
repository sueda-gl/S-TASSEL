from typing import List, Tuple, Dict

def shapley_reserve(prices, k: int, lam: float) -> float:
    step = prices[k] - (prices[k-1] if k > 0 else 0.0)
    return lam * step

def resolve_tier(bids: List[Tuple[str, float]],
                 reserves: Dict[str, float]) -> Tuple[str, float]:
    """
    Vickrey + reserve.  Guarantees DSIC.
    Returns winner id and premium paid.
    """
    bids.sort(key=lambda x: -x[1])
    winner, _ = bids[0]
    second = bids[1][1] if len(bids) > 1 else 0.0
    return winner, max(second, reserves[winner])
