import os, sys
# Ensure project root is on the Python path so `src.*` imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
import pandas as pd
import sqlite3
import plotly.express as px

from src.config import cfg, rng  # cfg still holds parameters, rng for reproducibility
from src.env.market_env import MarketplaceEnv
from src.ladder import gini                              # <- new import
from src.policies.truthful import act as truthful_act
from src.policies.margin import act as margin_act
from src.tokens import ledger

st.set_page_config(page_title="S-TASSEL Dashboard", layout="wide")
st.title("S-TASSEL Market Simulation")

# -------- sidebar controls --------
epochs   = st.sidebar.number_input("Epochs", min_value=1, max_value=1_000, value=30)
buyers_n = st.sidebar.number_input("# Buyers", min_value=10, max_value=5_000, value=800)
prices   = st.sidebar.text_input("Initial tier prices (comma-separated)", "10,15,20,25")

# Policy selection
policy_type = st.sidebar.radio("Agent Policy", 
                              ["Truthful", "Margin-seeking"], 
                              help="Truthful: agents bid their true surplus. Margin-seeking: agents shade bids to maximize surplus.")

# Margin policy parameters (shown only when margin-seeking is selected)
if policy_type == "Margin-seeking":
    shade_factor = st.sidebar.slider("Shade Factor", 0.1, 1.0, 0.7, 0.05, 
                                 help="Lower values mean more aggressive shading (bidding less).")
    use_randomization = st.sidebar.checkbox("Randomize bids", True, 
                                       help="Add noise to the shading factor for more realistic behavior.")

# Add income distribution controls
income_mean = st.sidebar.number_input("Income lognormal μ", 0.5, 5.0, 3.0, 0.1,
                                    help="Mean of lognormal distribution for income (in log-space)")
income_sigma = st.sidebar.number_input("Income lognormal σ", 0.1, 2.0, 1.0, 0.1,
                                     help="Sigma of lognormal distribution for income (in log-space)")

# Solidarity factor λ
lambda_val = st.sidebar.slider("Solidarity factor λ", 0.0, 1.0, float(cfg.lambda_), 0.05,
                               help="Share of each Shapley gap extracted as reserve (0=revenue-max, 1=fairness-max)")

run_btn  = st.sidebar.button("Run simulation 🚀")

# -------- helper to run the model --------
@st.cache_data(show_spinner="Running simulation …", max_entries=5)
def run_sim(n_epochs: int, n_buyers: int, price_str: str, policy: str, 
            margin_shade: float = 0.7, margin_random: bool = True,
            mu_inc: float = 3.0, sigma_inc: float = 1.0,
            lambda_val: float = cfg.lambda_) -> pd.DataFrame:
    price_list = [float(p.strip()) for p in price_str.split(",") if p.strip()]

    # fresh RNG so cached results are reproducible given the same inputs
    local_rng = np.random.default_rng(seed=42)

    # Clean the token database for a fresh simulation
    try:
        import pathlib
        db_path = pathlib.Path(__file__).parent.parent / "src" / "tokens" / "tokens.db"
        with sqlite3.connect(db_path) as con, con:
            con.execute("DELETE FROM vault")
    except Exception as e:
        print(f"Token DB cleanup failed: {e}")

    # set solidarity factor
    cfg.lambda_ = lambda_val

    # Act function selection
    if policy == "Truthful":
        agent_act = truthful_act
    else:  # Margin-seeking
        def agent_act(obs):
            return margin_act(obs, shade_factor=margin_shade, randomize=margin_random)

    buyers = {f"b{i}": {"income": local_rng.lognormal(mu_inc, sigma_inc)}
              for i in range(n_buyers)}
    env    = MarketplaceEnv(buyers, price_list)

    records = []
    for epoch in range(n_epochs):
        obs, _ = env.reset(); done = {a: False for a in env.agents}
        while not all(done.values()):
            acts = {a: agent_act(o) for a, o in obs.items() if not done[a]}
            obs, _, done, _, _ = env.step(acts)
        # Compute effective price vector and its Gini coefficient
        eff = np.array([env.prices[env._tier(a)] - env.buyers[a]["credit"]
                        for a in env.agents if env._tier(a) is not None])
        rec = {"epoch": epoch,
               "revenue": env.revenue,
               "gini": gini(eff)}
        # Get token vault balance before the next auction
        tokens_in_vault = 0
        try:
            rows = ledger.load(env.epoch, cfg.token_expiry)
            if rows:
                tokens_in_vault = sum(t for _, t in rows)
        except Exception as e:
            print(f"Token load failed: {e}")
        rec["token_balance"] = tokens_in_vault
        rec["minted"] = env.minted_last
        rec["expired"] = env.expired_last
        # store each tier price separately for later plotting
        for idx, price in enumerate(env.prices):
            rec[f"p{idx}"] = price
        records.append(rec)
    return pd.DataFrame(records)

# -------- run & display --------
if run_btn:
    if policy_type == "Truthful":
        df = run_sim(epochs, buyers_n, prices, policy_type,
                      mu_inc=income_mean, sigma_inc=income_sigma,
                      lambda_val=lambda_val)
    else:
        df = run_sim(epochs, buyers_n, prices, policy_type,
                     margin_shade=shade_factor, margin_random=use_randomization,
                     mu_inc=income_mean, sigma_inc=income_sigma,
                     lambda_val=lambda_val)

    st.subheader("Revenue per Epoch")
    st.line_chart(df.set_index("epoch")[["revenue"]]
                    .rename(columns={"revenue": "Revenue (€)"}))

    # Display token balance metric card
    st.subheader("Token Economy")
    cols = st.columns(2)
    with cols[0]:
        st.metric("Final Token Balance", f"{df['token_balance'].iloc[-1]:.1f} €")
    with cols[1]:
        st.metric("Peak Token Balance", f"{df['token_balance'].max():.1f} €")
    
    # Token balance chart
    st.line_chart(df.set_index("epoch")[["token_balance"]]
                   .rename(columns={"token_balance": "Tokens in Vault (€)"}))

    # Mint vs Expire chart
    st.subheader("Mint vs. Expire per Epoch")
    mint_exp_plot = px.bar(df, x="epoch", y=["minted", "expired"],
                           labels={"value": "Tokens (€)"}, barmode="relative")
    mint_exp_plot.update_layout(showlegend=True)
    st.plotly_chart(mint_exp_plot, use_container_width=True)

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
