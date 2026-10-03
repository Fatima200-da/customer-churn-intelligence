# 📊 Customer Churn Intelligence

AI-powered customer retention and churn risk analysis built with Python, Scikit-learn, SHAP, Plotly, and Streamlit.

## 🚀 Project Overview

Customer churn is a major business problem: identifying customers who may leave allows businesses to prioritize retention efforts and better understand customer behavior.

This project develops an end-to-end machine learning pipeline that predicts customer churn, explains individual predictions, segments customers by risk, and translates model outputs into practical retention insights.

### Key capabilities

* Customer churn prediction
* Risk segmentation
* Explainable AI with SHAP
* Business-oriented churn analysis
* Customer-level retention recommendations
* Interactive Streamlit dashboard
* High-risk customer identification

## 🎯 Business Problem

The goal is to answer four practical questions:

1. Which customers are more likely to churn?
2. What factors contribute most to the model's predictions?
3. Which customer segments show higher observed churn rates?
4. How can these insights support customer retention activities?

## 📊 Dataset

The project uses the Telco Customer Churn dataset containing customer demographic, service, contract, billing, and tenure information.

The dataset contains 7,043 customer records and 21 original columns.

The target variable is:

* `Churn = 1` → customer churned
* `Churn = 0` → customer did not churn

### Data preparation

Key preprocessing steps included:

* Converting `TotalCharges` to numeric format
* Handling missing values through the modeling pipeline
* Separating numerical and categorical features
* One-hot encoding categorical variables
* Standardizing numerical variables
* Stratified train/test splitting

## 🧠 Machine Learning

Three classification models were evaluated:

* Logistic Regression
* Random Forest
* Gradient Boosting

The final dashboard uses a Gradient Boosting model for customer-level churn probability estimation.

### Model Performance

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 80.27% |
| Precision | 66.44% |
| Recall    | 51.87% |
| F1 Score  | 58.26% |
| ROC-AUC   | 84.32% |

All metrics were calculated on a held-out test set.

## 🔍 Explainable AI

SHAP (SHapley Additive exPlanations) is used to understand which features contribute to individual model predictions.

Important model drivers included:

* Month-to-month contract
* Customer tenure
* Online security status
* Fiber optic internet service
* Monthly charges
* Technical support status
* Electronic check payment
* Total charges

SHAP values describe model contribution and should not be interpreted as causal effects.

## 🚦 Customer Risk Segmentation

Customers are divided into three illustrative risk groups based on predicted churn probability:

* **Low Risk:** probability below 40%
* **Medium Risk:** 40%–69.99%
* **High Risk:** 70% or higher

On the held-out test set:

* Low Risk: 999 customers
* Medium Risk: 292 customers
* High Risk: 118 customers

These thresholds are business segmentation rules rather than statistically validated intervention thresholds.

## 💡 Business Insights

The analysis identified several observed patterns in the dataset.

### Contract

Month-to-month customers showed an observed churn rate of **42.71%**, compared with **11.27%** for one-year contracts and **2.83%** for two-year contracts.

### Internet Service

Fiber optic customers showed an observed churn rate of **41.89%**, compared with **18.96%** for DSL customers and **7.40%** for customers without internet service.

### Payment Method

Electronic check customers showed an observed churn rate of **45.29%**, compared with **16.71%** for automatic bank transfer and **15.24%** for automatic credit card payments.

These are observed associations in the dataset and do not establish causal relationships.

## 💼 Retention Intelligence

The application translates customer-level predictions into practical retention suggestions.

Examples include:

* Offering longer-term contract incentives
* Prioritizing early-stage customer onboarding
* Promoting alternative automatic payment methods
* Offering technical support assistance
* Promoting online security services
* Reviewing the customer experience around specific service segments

Recommendations are rule-based and generated from customer characteristics alongside model predictions.

## 🖥️ Interactive Dashboard

The Streamlit application includes:

### Overview

High-level KPIs, churn distribution, risk segmentation, and executive business insights.

### Customer Prediction

Enter customer characteristics and receive:

* Churn probability
* Risk level
* SHAP-based risk factors
* Retention recommendations

### Model Performance

Interactive model evaluation including:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* ROC curve

### Business Insights

Visual analysis of churn patterns across:

* Contract type
* Internet service
* Payment method

### High-Risk Customers

A prioritized view of customers with the highest predicted churn probability, including downloadable results.

## 🛠️ Technology Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* SHAP
* Plotly
* Streamlit
* Joblib

## 📁 Project Structure

```text
customer-churn-intelligence/
│
├── app.py
├── churn_model.pkl
├── model_metrics.json
├── risk_distribution.csv
├── high_risk_customers.csv
├── business_insights.csv
├── roc_curve.csv
├── requirements.txt
├── README.md
└── data/
    └── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## ▶️ How to Run

Create and activate the project environment:

```bash
conda create -n churnapp python=3.11
conda activate churnapp
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## 📌 Important Notes

* Model performance is reported on a held-out test set.
* Risk thresholds are illustrative business segmentation rules.
* SHAP explanations describe model behavior, not causal relationships.
* Business insights represent associations observed in this dataset.
* The deployed environment should use the pinned Scikit-learn version used for model persistence.

## 👩‍💻 Project Focus

This project demonstrates an end-to-end approach to:

**Data → Machine Learning → Explainability → Risk Segmentation → Business Insights → Retention Intelligence**

It combines data science, machine learning, explainable AI, and application development into one practical customer analytics solution.
