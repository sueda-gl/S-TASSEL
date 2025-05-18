import os, sys
# Ensure project root is on the Python path so `src.*` imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
import pandas as pd

from src.config import cfg, rng  # cfg still holds parameters, rng for reproducibility
from src.env.market_env import MarketplaceEnv
from src.ladder import gini                              # <- new import
from src.policies.truthful import act

st.set_page_config(page_title="S-TASSEL Dashboard", layout="wide")
st.title("S-TASSEL Market Simulation")

# -------- sidebar controls --------
epochs   = st.sidebar.number_input("Epochs", min_value=1, max_value=1_000, value=30)
buyers_n = st.sidebar.number_input("# Buyers", min_value=10, max_value=5_000, value=800)
prices   = st.sidebar.text_input("Initial tier prices (comma-separated)", "10,15,20,25")
run_btn  = st.sidebar.button("Run simulation 🚀")

# -------- helper to run the model --------
@st.cache_data(show_spinner="Running simulation …", max_entries=5)
def run_sim(n_epochs: int, n_buyers: int, price_str: str) -> pd.DataFrame:
    price_list = [float(p.strip()) for p in price_str.split(",") if p.strip()]

    # fresh RNG so cached results are reproducible given the same inputs
    local_rng = np.random.default_rng(seed=42)

    buyers = {f"b{i}": {"income": local_rng.lognormal(3, 1)} for i in range(n_buyers)}
    env    = MarketplaceEnv(buyers, price_list)

    records = []
    for epoch in range(n_epochs):
        obs, _ = env.reset(); done = {a: False for a in env.agents}
        while not all(done.values()):
            acts = {a: act(o) for a, o in obs.items() if not done[a]}
            obs, _, done, _, _ = env.step(acts)
        # Compute effective price vector and its Gini coefficient
        eff = np.array([env.prices[env._tier(a)] - env.buyers[a]["credit"]
                        for a in env.agents if env._tier(a) is not None])
        rec = {"epoch": epoch,
               "revenue": env.revenue,
               "gini": gini(eff)}
        # store each tier price separately for later plotting
        for idx, price in enumerate(env.prices):
            rec[f"p{idx}"] = price
        records.append(rec)
    return pd.DataFrame(records)

# -------- run & display --------
if run_btn:
    df = run_sim(epochs, buyers_n, prices)

    st.subheader("Revenue per Epoch")
    st.line_chart(df.set_index("epoch")[["revenue"]]
                    .rename(columns={"revenue": "Revenue (€)"}))

    st.subheader("Gini per Epoch (effective prices)")
    st.line_chart(df.set_index("epoch")[["gini"]])

    # ----- price ladder drift -----
    price_cols = [c for c in df.columns if c.startswith("p")]
    if price_cols:
        st.subheader("Tier Prices over Epochs")
        st.line_chart(df.set_index("epoch")[price_cols])

    st.subheader("Raw KPIs")
    st.dataframe(df, use_container_width=True)

    # Offer download
    csv = df.to_csv(index=False).encode()
    st.download_button("Download CSV", data=csv, file_name="kpis.csv", mime="text/csv")
else:
    st.info("Set parameters in the sidebar and press **Run simulation** to begin.")
