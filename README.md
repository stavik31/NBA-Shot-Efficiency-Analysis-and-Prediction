# 🏀 NBA Shot Efficiency Analysis & Prediction

## Overview

Traditional basketball statistics such as **Field Goal Percentage (FG%)** fail to capture the true difficulty and value of a shot. A wide-open layup and a contested three-pointer are treated as identical events, leading to misleading conclusions about player efficiency and shot quality.

This project develops an **end-to-end machine learning pipeline** to model **shot difficulty and efficiency** in the NBA by predicting the probability of a shot being made given its **spatial and situational context**. These probabilities are then used to derive:

- **Expected Points (xPts)**
- **Points Over Expectation (POE)**

This allows for fairer comparisons across players and teams by separating shot quality from shot-making ability.

---

## Objectives

- Predict NBA shot outcomes using spatial and situational data  
- Estimate **shot quality** using model-derived probabilities (xPts)  
- Separate **shot difficulty** from **shot making ability** (POE)  
- Demonstrate how probabilistic models support real-world team and player analysis  

---

## Data

- **Source:** NBA ShotChartDetail endpoint via the `nba_api` Python library  
- **Training Data:** All shots from the **2024–25 NBA season** (~200,000 shots)  
- **Case Study Data:** Early **2025–26 season** shots (used only for analysis, not training)  

### Feature Categories

- **Spatial Context:** Shot coordinates relative to the basket  
- **Situational Context:** Game time, quarter, clutch situations  
- **Action Type:** Simplified shot types (e.g., dunk, layup, jump shot)  
- **Target Variable:** Binary shot outcome (made / missed)  

---

## Data Pipeline

The project follows a complete machine learning workflow:

1. **Data Collection** – Extracted shot-level data using `nba_api`  
2. **Data Cleaning** – Removed missing values, standardized coordinates  
3. **Exploratory Data Analysis (EDA)** – Shot distribution, distance trends, efficiency by zone  
4. **Feature Engineering** – Derived context-aware features  
5. **Model Development** – Random Forest + MLP  
6. **Cross-Validation & Hyperparameter Tuning**  
7. **Post-Model Analysis** – xPts and POE calculations  

---

## Feature Engineering

To better capture shot difficulty, several domain-informed features were engineered:

- **Shot Angle** – Differentiates corner shots from center-floor attempts  
- **Pressure Indicator** – Flags late-game high-leverage situations  
- **Action Simplification** – Groups 50+ shot descriptions into 6 core shot types  
- **Distance Transformation** – Squared distance to model nonlinear difficulty growth  

Processing steps:

- One-hot encoding for categorical variables  
- Standard scaling for numerical features  
- Train-test separation before scaling to prevent leakage  

---

## Models

Two supervised classification models were trained and evaluated.

### 🌲 Random Forest (Baseline)

- Nonlinear ensemble method  
- Interpretable via feature importance  
- Strong performance on structured tabular data  

### 🧠 Multi-Layer Perceptron (MLP)

- Fully connected neural network  
- ReLU activations  
- Captures complex nonlinear relationships  
- Produces smoother probability outputs  

Hyperparameters were optimized using **GridSearchCV with 5-fold cross-validation**.

---

## Evaluation

Models were evaluated using:

- **Accuracy**
- **ROC-AUC**

| Model          | Accuracy | ROC-AUC |
|----------------|----------|---------|
| Random Forest  | ~62.8%   | ~0.65   |
| MLP            | ~62.7%   | ~0.65   |

Although shot outcomes are inherently noisy due to missing contextual variables (defender proximity, fatigue, etc.), both models significantly outperform random guessing and provide meaningful probability separation.

---

## From Prediction to Insight

### Expected Points (xPts)

```text
xPts = P(shot made) × shot value

This represents the league-average expected outcome for a given shot.

### Points Over Expectation (POE)

```text
POE = actual points − xPts
