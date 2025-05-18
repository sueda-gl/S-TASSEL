import numpy as np
from ..auction.premium import shapley_reserve
from ..config import cfg, rng

def act(obs, shade_factor=0.7, randomize=True, seed=42):
    """
    Margin-seeking policy: strategically shades bids to maximize consumer surplus.
    
    Args:
        obs: Observation vector [prices, income, credit]
        shade_factor: Base shading multiplier (0.7 means bid 70% of surplus)
        randomize: Whether to apply random noise to the shading
        seed: Random seed for reproducibility
        
    Returns:
        (tier_id, bid) tuple
    """
    prices, inc, cred = obs[:-2], obs[-2], obs[-1]
    
    # Use a separate RNG for reproducibility
    local_rng = np.random.RandomState(seed)
    
    # Find affordable tiers (same logic as truthful)
    affordable_tiers = []
    for k in range(len(prices)-1, -1, -1):  # scan top→bottom
        reserve = shapley_reserve(prices, k, cfg.lambda_)
        if inc >= prices[k] + reserve - cred:
            surplus = inc - prices[k]  # true surplus
            
            # Apply shading: multiply surplus by a factor < 1
            if randomize:
                # Log-normal gives heavy-tailed multiplier centered below 1.0
                # μ=-0.5, σ=0.4 gives mode around shade_factor with positive skew
                noise = local_rng.lognormal(mean=-0.5, sigma=0.4)
                bid = surplus * shade_factor * noise
            else:
                bid = surplus * shade_factor
                
            # Ensure we don't bid below reserve
            reserve_floor = max(reserve, cfg.lambda_ * (prices[k] - (0 if k == 0 else prices[k-1])))
            bid = max(bid, reserve_floor + 0.01)  # tiny buffer to ensure it's > reserve
            
            affordable_tiers.append((k, bid, surplus - bid))  # (tier, bid, expected_margin)
    
    if not affordable_tiers:
        return len(prices), 0.0  # walk away
        
    # Choose tier with maximum expected margin
    best_tier, best_bid, _ = max(affordable_tiers, key=lambda x: x[2])
    return best_tier, float(best_bid)  # ensure we return Python float, not numpy float 