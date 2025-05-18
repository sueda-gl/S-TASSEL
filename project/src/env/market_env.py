import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces
from collections import defaultdict
from ..auction.premium import shapley_reserve, resolve_tier
from ..tokens import ledger
from ..ladder import gini, update_prices
from ..fairness.ot_rebate import rebate
from ..config import cfg, rng

POVERTY_LINE = 1e4  # €10k income

class MarketplaceEnv(ParallelEnv):
    metadata = {"name": "S-TASSEL-v0"}

    def __init__(self, buyers, init_prices):
        self.buyers = buyers
        self.prices = np.array(init_prices, float)
        self.agents = list(buyers.keys())
        self.epoch  = 0
        K = len(self.prices)
        self.action_spaces = {a: spaces.Tuple(
            (spaces.Discrete(K+1), spaces.Box(0, 10, (1,))))
                              for a in self.agents}
        obs_len = K + 2
        self.observation_spaces = {a: spaces.Box(0, 1e6, (obs_len,)) for a in self.agents}
        self._reset_day()

    # ---------- PettingZoo API ----------
    def reset(self, seed=None, options=None):
        self._reset_day()
        return {a: self._obs(a) for a in self.agents}, {a: {} for a in self.agents}

    def step(self, actions):
        rewards, terms, truncs = {}, {}, {}
        for aid, (tier, bid) in actions.items():
            if tier == len(self.prices) or self.stock[tier] == 0:
                rewards[aid] = 0.; terms[aid]=truncs[aid]=True
                continue
            self.bids[tier].append((aid, float(bid)))
            rewards[aid] = 0.; terms[aid]=truncs[aid]=True
        if all(terms.values()):
            self._nightly_closure()
        return {a: self._obs(a) for a in self.agents}, rewards, terms, truncs, {}

    # ---------- internal helpers ----------
    def _reset_day(self):
        K = len(self.prices)
        self.stock = np.ones(K, int) * cfg.unit_stock
        self.sales = np.zeros(K)
        self.bids  = defaultdict(list)
        for b in self.buyers.values():
            b["credit"] = b.get("credit", 0.0)
        self.revenue = 0.

    def _obs(self, aid):
        inc  = self.buyers[aid]["income"]
        cred = self.buyers[aid]["credit"]
        return np.concatenate([self.prices, [inc, cred]])

    def _nightly_closure(self):
        # 1. resolve auctions
        for k, bids in self.bids.items():
            if not bids:
                continue
            reserves = {a: shapley_reserve(self.prices, k, cfg.lambda_) for a,_ in bids}
            winner, prem = resolve_tier(bids, reserves)
            if winner:                           # winner cannot be None here
                self.stock[k] -= 1
                self.sales[k] += 1
                self.revenue += self.prices[k]
                ledger.mint(self.epoch, winner, prem)

        # 2. OT rebate
        rows = ledger.load(self.epoch, cfg.token_expiry)
        if rows:
            donor_tok = np.array([t for _, t in rows])
            rec_ids = [a for a in self.agents if self.buyers[a]["income"] < POVERTY_LINE]

            # Skip OT call when no tokens to redistribute or no recipients
            if donor_tok.sum() > 0 and rec_ids:
                donor_inc = np.array([self.buyers[d]["income"] for d, _ in rows])
                recip_inc = np.array([self.buyers[r]["income"] for r in rec_ids])
                credits = rebate(donor_inc, donor_tok, recip_inc)
                for rid, c in zip(rec_ids, credits):
                    self.buyers[rid]["credit"] += c
        ledger.expire(self.epoch, cfg.token_expiry)

        # 3. adapt ladder
        eff = np.array([self.prices[self._tier(a)] - self.buyers[a]["credit"]
                        for a in self.agents if self._tier(a) is not None])
        g    = gini(eff)
        sold = 1 - self.stock.sum() / (cfg.unit_stock * cfg.K)
        self.prices = update_prices(self.prices, self.sales, self.revenue, g, sold, cfg)

        # 4. reset day
        self.sales[:] = 0
        self.bids.clear()
        self.epoch += 1

    def _tier(self, aid):
        inc, cred = self.buyers[aid]["income"], self.buyers[aid]["credit"]
        for k in range(len(self.prices)-1, -1, -1):
            if inc >= self.prices[k] - cred:
                return k
        return None
