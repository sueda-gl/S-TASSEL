from ..auction.premium import shapley_reserve
from ..config import cfg

def act(obs, cfg=cfg):
    prices, inc, cred = obs[:-2], obs[-2], obs[-1]
    for k in range(len(prices)-1, -1, -1):          # scan top→bottom
        reserve = shapley_reserve(prices, k, cfg.lambda_)
        if inc >= prices[k] + reserve - cred:
            return k, inc - prices[k]               # truthful surplus bid
    return len(prices), 0.0                         # walk away
