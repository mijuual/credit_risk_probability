# 🏦 Bati Bank Credit Scoring Model

## Project Overview

Bati Bank is partnering with an emerging eCommerce platform to introduce a **Buy Now, Pay Later (BNPL)** service. As part of this initiative, our goal is to develop a **credit scoring model** that estimates the risk of customer default using transaction and behavioral data.

This repository outlines the complete pipeline for building the model — from **exploratory data analysis (EDA)** to **model development**, **risk scoring**, and **loan amount prediction**.

---

## 📊 Dataset Summary

- **Rows:** 95,662 transactions  
- **Columns:** 16 features  
- **Features:** Include customer ID, transaction amount, channel, product type, fraud flag, etc.  
- **Target:** No explicit default label. Will engineer a **proxy target variable** from user behavior.

---

## 📈 Key Steps

1. **Exploratory Data Analysis (EDA):**
   - Summary statistics
   - Feature distributions
   - Correlation heatmaps
   - Outlier detection
   - Missing value checks

2. **Feature Engineering:**
   - Aggregated RFM (Recency, Frequency, Monetary) variables
   - Proxy variable creation for customer risk
   - One-hot encoding and Weight of Evidence (WoE) encoding

3. **Modeling Approaches:**
   - Logistic Regression (with WoE)
   - Gradient Boosting (XGBoost, LightGBM)
   - Model evaluation using AUC, Gini, KS-statistic

4. **Scoring Pipeline:**
   - Risk probability prediction
   - Credit score transformation
   - Loan amount and duration recommendation

---

##  Credit Scoring Business Understanding

### 1. Basel II and the Need for Interpretability

The **Basel II Capital Accord** requires banks to maintain capital reserves based on the **riskiness of their lending**. This emphasizes:
- Rigorous **risk measurement** practices,
- Transparent, **auditable models**, and
- Clear documentation for regulatory review.

As a result, our models must be **interpretable** and **well-documented**. Regulators may reject complex "black-box" models that cannot explain why a customer was approved or denied credit.

---

### 2. The Proxy Variable: Necessity and Risks

Since the dataset lacks a direct `"default"` label, we must create a **proxy target variable** based on behavioral signals like fraud outcomes, refund rates, or RFM metrics.

- **Why it's needed:** Without a proxy, supervised learning is not possible.
- **Risk:** The proxy might **misclassify** some users — penalizing good customers or approving risky ones. This introduces potential **financial loss**, **bias**, or **compliance risks**.

Thus, proxy engineering must be both **data-driven** and **business-aware**.

---

### 3. Interpretable vs Complex Models: A Trade-off

| Simple Model (e.g., Logistic Regression with WoE) | Complex Model (e.g., Gradient Boosting) |
|--------------------------------------------------|-----------------------------------------|
|  Highly interpretable                         |  Often hard to explain                 |
|  Easy to justify to regulators                |  May require model documentation tools |
|  May have lower predictive power              |  Typically achieves better accuracy    |
|  Transparent scorecards for credit analysts   |  Needs advanced monitoring             |

**In a regulated context**, starting with an interpretable model is often preferred. Complex models can be layered in later stages with proper explanation tools (e.g., SHAP).

---

## 🛠 Requirements

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn

