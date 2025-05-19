# S-TASSEL

S-TASSEL is an economically principled multi-tier market simulation that implements a self-equalizing loop to balance revenue maximization with fairness goals.

## Live Demo

Try the S-TASSEL simulation online: [https://s-stl-simulation.streamlit.app](https://s-stl-simulation.streamlit.app)

## Running the Project Locally

The recommended way to run S-TASSEL locally is using Docker, which handles all dependencies automatically:

```bash
# Build the Docker image
docker build -t s-tassel .

# Run the Streamlit dashboard
docker run --rm -p 8501:8501 s-tassel
```

Then open http://localhost:8501 in your browser.

For batch simulation without the dashboard:
```bash
docker run --rm s-tassel python project/run_batch.py
```

To run tests:
```bash
docker run --rm s-tassel pytest
```

## Project Structure

- `project/src/`: Core implementation
  - `auction/`: Shapley-based auction mechanisms
  - `env/`: PettingZoo multi-agent environment
  - `fairness/`: Optimal transport rebate system
  - `tokens/`: FairToken ledger implementation
  - `policies/`: Agent bidding strategies
  - `config.py`: Simulation parameters
  - `ladder.py`: Price ladder optimization

- `project/dashboard/`: Interactive visualization
  - `app.py`: Streamlit dashboard

## Key Features

- **Truthful & margin-seeking policies**: Compare effects of different bidding strategies
- **Adaptive price ladder**: Self-adjusting tiers based on revenue and fairness criteria
- **Token-based subsidy**: Track premiums and credits through FairToken flows
- **Income distribution controls**: Test the mechanism under various market conditions
- **Gini coefficient tracking**: Monitor inequality of effective prices over time

## Mathematical Foundation

S-TASSEL combines concepts from:
- Cooperative game theory (Shapley value allocation)
- Auction design (Vickrey mechanism with credit adjustments)
- Optimal transport (redistribution with minimal distortion)
- Online convex optimization (mirrored descent with constraints)

See the dashboard for an interactive demonstration of the self-equalizing fairness loop in action.

## Requirements

All dependencies are handled by the Docker image. The core requirements are listed in `requirements.txt` and include numpy, pandas, pettingzoo, gymnasium, pot (Python Optimal Transport), and streamlit. 