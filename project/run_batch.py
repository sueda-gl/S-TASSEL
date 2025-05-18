import argparse, numpy as np, pandas as pd
from src.config import cfg, rng
from src.env.market_env import MarketplaceEnv
from src.policies.truthful import act

p = argparse.ArgumentParser(); p.add_argument("--epochs", type=int, default=30)
E = p.parse_args().epochs

buyers = {f"b{i}": {"income": rng.lognormal(3, 1)} for i in range(800)}
env    = MarketplaceEnv(buyers, [10, 15, 20, 25])

records = []
for epoch in range(E):
    obs,_ = env.reset(); done = {a: False for a in env.agents}
    while not all(done.values()):
        acts = {a: act(o) for a,o in obs.items() if not done[a]}
        obs,_,done,_,_ = env.step(acts)
    records.append({"epoch": epoch, "revenue": env.revenue})
pd.DataFrame(records).to_csv("kpis.csv", index=False)
print("batch run finished")
