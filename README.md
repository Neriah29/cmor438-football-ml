# ⚽ football-ml — Machine Learning Engine for Football Prediction

> **Course**: CMOR 438 / INDE 577 — Data Science & Machine Learning  
> **Instructor**: Dr. Randy R. Davila · Rice University · Spring 2026  
> **Author**: [Neriah29](https://github.com/Neriah29)

---

## What This Is

This repository is the **machine learning core** of a larger football analytics platform. It contains from-scratch implementations of 14 machine learning algorithms — no scikit-learn under the hood — applied to a dataset of over 45,000 international football matches spanning 1872 to present.

Every algorithm is applied to the same domain question:

> *Can we use historical match data to understand, predict, and explain international football outcomes?*

The answers feed into a full-stack web application (in development) that will offer live match predictions, team strength ratings, tournament simulators, and — ultimately — a FIFA World Cup bracket predictor.

This repo is the engine. The web app is the car.

---

## The Bigger Picture

```
football-ml (this repo)
    ↓  trained models + prediction API
Full-stack web app
    ↓
Features:
  • Live match win probability
  • Team strength ratings over time
  • Head-to-head historical analysis
  • FIFA World Cup bracket simulator
  • Tournament outcome distributions
```

The ML algorithms here are not just course exercises — they are being evaluated for real predictive power. The best-performing models will be packaged as an API and integrated into the web app.

---

## Dataset

**Source**: [International Football Results 1872–2024](https://www.kaggle.com/) — Kaggle

| File | Contents |
|---|---|
| `results.csv` | Match-by-match results — home team, away team, scores, tournament, date, country |
| `goalscorers.csv` | Individual goalscorer records |
| `shootouts.csv` | Penalty shootout outcomes |

The data lives in `data/` (gitignored — download from Kaggle and place there).

### Features engineered for every match

| Feature | Description |
|---|---|
| `home_goals_rolling` | Home team's avg goals scored in last 10 matches |
| `away_goals_rolling` | Away team's avg goals scored in last 10 matches |
| `home_conceded_rolling` | Home team's avg goals conceded in last 10 matches |
| `away_conceded_rolling` | Away team's avg goals conceded in last 10 matches |
| `home_win_rate` | Home team's historical win rate |
| `away_win_rate` | Away team's historical win rate |
| `neutral` | Whether the match is on neutral ground |

Rolling averages use a 10-game window with `shift(1)` to prevent data leakage — the model never sees the result of the match it is predicting.

---

## Algorithms Implemented

### Supervised Learning — Predicting Match Outcomes

All classification algorithms predict **home win (1) vs draw/away win (0)**. Regression algorithms predict **goal difference**.

| # | Algorithm | Type | Football Question |
|---|---|---|---|
| 01 | Perceptron | Binary classifier | Can a linear threshold separate home wins from non-wins? |
| 02 | Linear Regression | Regression | Can we predict the goal difference from team form? |
| 03 | Logistic Regression | Binary classifier | What is the probability of a home win? |
| 04 | K-Nearest Neighbors | Binary classifier | Do similar past matches predict the same outcome? |
| 05 | Neural Network (MLP) | Binary classifier | Can non-linear patterns improve prediction? |
| 06 | Decision Tree | Binary classifier | What sequence of yes/no questions best separates outcomes? |
| 07 | Random Forest | Binary classifier | Does averaging 100 trees reduce prediction noise? |
| 07 | Gradient Boosting | Binary classifier | Can sequential error-correction improve accuracy? |
| 08 | Support Vector Machine | Binary classifier | What is the maximum-margin boundary between outcomes? |
| 09 | Naïve Bayes | Binary classifier | Can Bayesian probability estimate win likelihood? |
| 10 | Ridge / Lasso Regression | Regression | Does regularization improve goal difference prediction? |

### Unsupervised Learning — Discovering Structure

| # | Algorithm | Football Question |
|---|---|---|
| 11 | K-Means Clustering | Are there natural types of matches in the data? |
| 12 | DBSCAN | Which matches are truly anomalous or unusual? |
| 13 | PCA | What are the most important axes of variation in match data? |
| 14 | Hierarchical Clustering | How do matches naturally group at different levels of similarity? |

---

## Repository Structure

```
cmor438-football-ml/
│
├── src/
│   └── football_ml/               # Installable Python package
│       ├── supervised_learning/   # 10 supervised algorithm classes
│       ├── unsupervised_learning/ # 4 unsupervised algorithm classes
│       ├── utils/
│       │   └── logger.py          # Logging utility
│       └── exceptions.py          # Custom exceptions
│
├── notebooks/
│   ├── supervised_learning/       # One .ipynb per supervised algorithm
│   └── unsupervised_learning/     # One .ipynb per unsupervised algorithm
│
├── tests/
│   ├── conftest.py                # Shared fixtures + timeout benchmarks
│   └── unit/                      # pytest unit tests for every algorithm
│
├── data/                          # CSVs — gitignored, download from Kaggle
├── .github/workflows/ci.yml       # GitHub Actions CI — runs tests on every push
├── pyproject.toml
└── README.md
```

---

## Installation

**Prerequisites**: Python 3.10+

```bash
# Clone the repo
git clone git@github.com:Neriah29/cmor438-football-ml.git
cd cmor438-football-ml

# Install the package in editable mode
pip install -e .

# Install with development dependencies (for running tests)
pip install -e ".[dev]"
```

---

## Running the Tests

```bash
# Run all tests
pytest tests/unit/ -v

# Run a specific algorithm's tests
pytest tests/unit/test_logistic_regression.py -v

# Run with timing info
pytest tests/unit/ -v --durations=10
```

Tests are also run automatically on every push and pull request via GitHub Actions.

---

## Using the Package

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from football_ml.supervised_learning.logistic_regression import LogisticRegression
from football_ml.supervised_learning.ensemble import RandomForestClassifier
from football_ml.unsupervised_learning.kmeans import KMeans

# Load and prepare your features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Train a model
model = LogisticRegression(learning_rate=0.1, n_epochs=1000)
model.fit(X_train_sc, y_train)

# Predict
probs = model.predict_proba(X_test_sc)   # probability of home win
preds = model.predict(X_test_sc)         # hard label
acc   = model.score(X_test_sc, y_test)   # accuracy

print(f"Test accuracy: {acc:.3f}")
```

All models follow the same interface: `fit()`, `predict()`, `predict_proba()`, `score()`.

---

## Key Results

| Algorithm | Test Accuracy | Notes |
|---|---|---|
| Perceptron | ~58% | Linear only — limited by data structure |
| Logistic Regression | ~62% | Solid baseline |
| KNN | ~61% | Non-linear but slow |
| Neural Network (MLP) | ~63% | Best single-model accuracy |
| Random Forest | ~63% | Most robust, best generalisation |
| Gradient Boosting | ~63% | Competitive with tuning |
| Naïve Bayes | ~60% | Fast, interpretable baseline |
| SVM | ~61% | Strong but slow on large datasets |

> Football is inherently unpredictable — a 63% accuracy on binary outcome prediction is competitive with published academic benchmarks on similar datasets.

---

## What's Next

This repository will continue to grow as the course progresses and as the web app takes shape:

- [ ] Expose best models as a REST API (FastAPI)
- [ ] Add Elo rating system as an additional feature
- [ ] Incorporate player-level data for richer features
- [ ] Build FIFA World Cup bracket simulator using Monte Carlo methods
- [ ] Deploy prediction engine to cloud (target: AWS / Railway)
- [ ] Build full-stack frontend (React) consuming the prediction API

---

## Course Context

This project was built for **CMOR 438 / INDE 577 — Data Science & Machine Learning** at Rice University, Spring 2026, taught by Dr. Randy R. Davila. The course covers supervised learning, unsupervised learning, and introduces reinforcement learning. All algorithms are implemented from scratch using NumPy to build genuine understanding of the underlying mathematics — no black-box library calls.

---

## License

MIT License — see `LICENSE` for details.
