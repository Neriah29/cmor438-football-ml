# CMOR 438 Project — Session Context File
*Drop this file at the start of every new Claude session to restore full context.*

---

## Who I Am
- **Student**: Rice University, SWE Freshman, CMOR 438 / INDE 577 (Spring 2026)
- **Instructor**: Dr. Randy R. Davila (rrd6@rice.edu)
- **GitHub**: https://github.com/Neriah29
- **Repo**: https://github.com/Neriah29/cmor438-football-ml
- **Learning style**: Deep explanations, abstraction-first (understand *what* things do, not necessarily *how* derived), beginner Python level
- **Tools**: Python, NumPy, Pandas, Matplotlib, Seaborn, Jupyter Notebooks, GitHub (SSH), Cursor + Claude Code

---

## Knowledge Level Per Topic
- **Perceptron**: Some familiarity (covered with Gemini)
- **Linear Regression**: Some familiarity (covered with Gemini)
- **Logistic Regression**: Some familiarity — stopped here with Gemini
- **Everything else (KNN, Neural Nets, Decision Trees, Ensembles, SVM, Naive Bayes, Clustering, PCA, etc.)**: Ground zero — needs full explanation from scratch

---

## The Course
CMOR 438 = Data Science & Machine Learning. Covers supervised → unsupervised → (maybe) reinforcement learning. Equivalent to INDE 577. Instructor: Dr. Randy Davila.

### Algorithms to implement (ALL required in final repo):
**Supervised Learning:**
- [ ] Perceptron ← IMMEDIATE PRIORITY (professor wants this + 2 more by next class)
- [ ] Linear Regression (+ Gradient Descent)
- [ ] Logistic Regression ← IMMEDIATE PRIORITY
- [ ] K-Nearest Neighbors (KNN)
- [ ] Neural Networks (MLP)
- [ ] Decision Trees / Regression Trees
- [ ] Ensemble Methods (Random Forest, Gradient Boosting)
- [ ] Support Vector Machines (SVM)
- [ ] Naïve Bayes
- [ ] Ridge / Lasso Regression

**Unsupervised Learning:**
- [ ] K-Means Clustering
- [ ] DBSCAN
- [ ] Principal Component Analysis (PCA)
- [ ] Hierarchical Clustering
- [ ] t-SNE / Autoencoders (introductory, time permitting)

**Additional:**
- [ ] CNNs (introductory)
- [ ] Reinforcement Learning (time permitting)

---

## Final Project Requirements
**Deliverable**: A public GitHub repo with:
1. `src/football_ml/` — Custom Python ML package with algorithms as reusable classes
2. `notebooks/` — One Jupyter notebook per algorithm, on the football dataset, with Markdown explanations
3. `tests/unit/` — pytest unit tests for all algorithms
4. `README.md` — explains the package, installation, usage
5. Clean commit history (commit after each algorithm is done)

**Grading:**
- 40% — Functionality & implementation
- 20% — Documentation & readability
- 20% — Testing & reliability
- 10% — Examples & usability
- 10% — Repository quality

**Important**: Every notebook is part of the FINAL PROJECT — not practice. Build it properly every time (clean code, good Markdown, proper documentation).

---

## Dataset
**Primary**: International Football Results (Kaggle) — match-by-match results from 1872 to present
- Location: `data/` folder in the repo (NOT pushed to GitHub — in .gitignore)
- 4 CSV files: `results.csv` (main), `goalscorers.csv`, `shootouts.csv`, and one more
- `results.csv` is the main file — has home team, away team, scores, tournament, date, country
- **What we're predicting**: Match outcome (Win/Draw/Loss for home team) — good for classification algorithms
- For regression algorithms (e.g. Linear Regression): predict goal difference or total goals scored
- For clustering: group matches/teams by results patterns
- For dimensionality reduction (PCA): reduce engineered team-level features

**Summer project** (completely separate, don't build now): FIFA World Cup predictor/simulator

---

## Repo Structure
```
cmor438-football-ml/
├── src/
│   └── football_ml/         ← installable Python package
│       ├── __init__.py
│       ├── supervised_learning/
│       │   └── __init__.py
│       ├── unsupervised_learning/
│       │   └── __init__.py
│       └── processing/
│           └── __init__.py
├── notebooks/
│   ├── supervised_learning/    ← one .ipynb per algorithm
│   └── unsupervised_learning/
├── tests/
│   └── unit/                   ← pytest tests
├── data/                       ← CSVs live here (gitignored)
├── README.md
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

---

## Professor's Immediate Instruction
Before next class, implement and document (as part of the final project):
1. **Perceptron** ← START HERE
2. **Linear Regression**
3. **Logistic Regression**

For each algorithm, the notebook must include:
- What the algorithm does (plain English + intuition)
- The math/mechanics at a high level (no need to derive formulas)
- Implementation from scratch using NumPy (as a class in `src/football_ml/supervised_learning/`)
- Application to the football dataset with visualizations
- Discussion of whether/how this algorithm suits football prediction + its limitations
- Evaluation metrics (accuracy, loss curves, etc.)

**Key insight to document for Perceptron**: It's historically foundational but limited — only handles binary, linearly separable problems. Real football data is not linearly separable, so the perceptron alone is not reliable for predictions. It's a conceptual stepping stone to neural networks.

---

## Reference Repos (past CMOR 438 students)
1. https://github.com/rykerdolese/Data-Science-and-Machine-Learning — most complete, best structure
2. https://github.com/eridavlo1/CMOR-438 — clean modular package
3. https://github.com/ariaanthor/DataSci_and_MachineLearning_2025_Course — good tests
4. https://github.com/gwenfitz/fitzsimmons-cmor-438 — strong on notebook teaching style

---

## Session Log
### Session 1 — April 15, 2026
- Established full project context, learning style, knowledge level
- Reviewed syllabus, 4 reference repos, professor's lecture material (Lectures 1–10)
- Identified all algorithms needed for final repo
- Chose dataset: International football results (Kaggle, 4 CSVs in data/ folder)
- Set up repo structure via CLI in Cursor terminal
- Used SSH for GitHub authentication (not HTTPS — fine either way)
- Claude Code installed in Cursor CLI
- **Next session**: Start building — Perceptron notebook + src implementation + unit tests
