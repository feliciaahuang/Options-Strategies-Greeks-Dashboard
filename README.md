
# Option Pricing and Greeks Visualization Dashboard

## Overview

**Option Pricing and Greeks Visualization Dashboard** is a Python-based analytical toolkit that prices European call and put options using both the **Black–Scholes–Merton model** and the **Cox–Ross–Rubinstein (binomial) model**.
Beyond pricing, the project provides **interactive visual dashboards** for exploring option behaviour, model convergence, and sensitivities (Greeks) across varying market conditions.

This repository is designed for students, quantitative finance practitioners, and analysts who want a clean, extensible, and visual framework for understanding equity options.

---

## Key Features

### 1. Market Data Integration

* Live price and historical volatility extraction via `yfinance`.
* Automatic computation of:

  * Daily log returns
  * Daily volatility
  * Annualised volatility
  * Real-time underlying asset price (S₀)

### 2. Analytical and Numerical Option Pricing

* **Black–Scholes–Merton** closed-form pricing for European calls and puts.
* **Binomial pricing model** with adjustable time steps (n).
* Full payoff and value trees for educational and verification purposes.

### 3. Model Convergence Analysis

* Demonstrates convergence of the binomial model toward the BSM analytical price as the number of steps increases.

### 4. Option Behaviour Visualisation

* Price vs time to maturity (Theta decay)
* Price vs underlying asset (S)
* Comparison between call and put profiles

### 5. Greeks Computation

Vectorised implementation of:

* Delta
* Gamma
* Vega
* Theta
* Rho

Computed using analytical BSM formulas.

### 6. Interactive Visualization Dashboard

Built using **Matplotlib Widgets**, enabling users to dynamically adjust:

* Underlying price (S₀)
* Strike price (K)
* Risk-free rate (r)
* Volatility (σ)
* Time to maturity (T)
* Option type (call/put)

All graphs update in real time, providing an intuitive understanding of option sensitivities.

---

## Project Structure

```
.
├── main.py                        # Full implementation of pricing, Greeks, and interactive visualisation
├── ci/run_tests.sh                # Placeholder for CI testing (optional)
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies
```

---

## Mathematical Models

### Black–Scholes–Merton Model

European option price:

```
C = S0 * N(d1) - K * e^{-rT} * N(d2)
P = K * e^{-rT} * N(-d2) - S0 * N(-d1)
```

### Cox–Ross–Rubinstein Binomial Model

* Underlying price evolves using up/down factors:

```
u = e^{σ√Δt}
d = e^{-σ√Δt}
p = (e^{rΔt} - d) / (u - d)
```

* Option value computed via backward induction.

The implementation ensures numerical robustness and correct handling of both calls and puts.

---

## Installation

### 1. Clone the repository

```
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create and activate a virtual environment (recommended)

```
python3 -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

If you do not have a requirements file, a typical environment includes:

```
numpy
scipy
matplotlib
yfinance
pandas
```

---

## Usage

Run the dashboard directly:

```
python main.py
```

The program will:

1. Fetch market data
2. Calculate volatility
3. Build option pricing models
4. Launch an **interactive GUI-style dashboard** inside a Matplotlib window

Use the sliders and input widgets to explore how option prices and Greeks behave under different market assumptions.

---

## Future Improvements

Potential enhancements include:

* Support for American options
* Monte Carlo pricing module
* Implied volatility surface visualisation
* Streamlit or Dash-based UI for web deployment
* Portfolio-level Greeks aggregation
