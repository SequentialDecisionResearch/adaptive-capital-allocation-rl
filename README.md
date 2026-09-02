# Adaptive Capital Allocation with Reinforcement Learning

A research demo accompanying the Medium article:

**Beyond Stock Picking: Reinforcement Learning for Adaptive Capital Allocation**

This project explores a simple idea:

> A stock should be treated as a time-varying investment state rather than as a permanent ticker identity.

The system asks a practical sequential decision question:

> Where is the next dollar of capital most useful under the current market state and the current portfolio state?

## Main Ideas

The demo combines:

- time-varying stock states
- pooled cross-stock learning
- Bayesian linear modeling
- Thompson Sampling
- historical state similarity
- portfolio-aware marginal utility
- incremental monthly allocation
- strict walk-forward backtesting

The workflow is:

**Observe → Estimate → Rank → Allocate → Update**

## Data

The script automatically downloads adjusted market data from Yahoo Finance using `yfinance`.

The demonstration universe contains approximately 30 U.S. stocks together with SPY and QQQ benchmarks.

## Running the Demo

Install the required packages:

```bash
pip install -r requirements.txt

Then run:

python stocks_as_time_varying_investment_demo.py

The program downloads the data, constructs investment states, performs the walk-forward backtest, prints the main results, and saves figures and CSV outputs.

These results should not be interpreted as evidence of market-beating alpha.

The stock universe is a demonstration universe rather than a fully point-in-time survivorship-free universe. The main purpose of the experiment is to illustrate the decision framework.

Selected Figures
Dynamic allocation of new capital

Opportunity score versus next-dollar utility

Research Interpretation

The experiment illustrates several ideas:

the same ticker can represent very different investment states over time;
historically similar investment states may come from different companies;
the highest standalone opportunity score is not always the best destination for the next dollar;
portfolio state changes future allocation decisions;
Bayesian uncertainty can be incorporated without turning exploration into random trading.
Limitations

This is a research and educational demonstration, not a production trading system.

Important limitations include:

ex-post stock-universe selection;
simplified transaction-cost assumptions;
price-derived state variables in the public demo;
no formal off-policy evaluation estimator;
no claim of future investment performance.

Further Research

Natural extensions include:

point-in-time fundamental data;
a truly dynamic historical universe;
regime-aware nonstationarity;
formal offline policy evaluation;
constrained portfolio reinforcement learning;
state-transfer learning across stocks.

Medium article:

License

MIT License.
