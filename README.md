# Customer Churn Prediction Project

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-orange)](https://www.kaggle.com/datasets/mustafaoz158/telco-customer-churn)

---

## Problem Statement
The goal of this project is to predict whether a telecom customer will **churn** (leave the service) based on their demographics, account details, subscribed services, and billing information.

**Input:** Customer attributes (numerical + categorical features)  
**Output:** Churn status (**Yes / No**)  

**Purpose:** Predicting churn helps telecom companies **reduce customer attrition** by identifying at-risk customers and taking proactive measures.

---

## Dataset
**Source:** [Telco Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/mustafaoz158/telco-customer-churn)

**Overview:** The dataset contains 7,000+ customer records with features including:
📌Dataset Story
Telco churn data includes information about a fictitious telecom company that provided home phone and Internet services to 7,043 customers in California in the third quarter. It shows which customers left, stayed, or signed up for their service.

* 🆔CustomerId: Customer Id
* 👫Gender: Gender
* 👵SeniorCitizen: Whether the customer is elderly (1, 0)
* 👫Partner: Whether the customer has a partner (Yes, No)
* 👨‍👨‍👧‍👧Dependents: Whether the customer has dependents (Yes, No)
* 📜Tenure: Number of months the customer has been with the company
* ☎️PhoneService: Whether the customer has phone service (Yes, No)
* 📞MultipleLines: Whether the customer has more than one line (Yes, No, No phone service)
* 💻InternetService: Whether the customer has internet service provider (DSL, Fiber optic, No)
* ㊙️OnlineSecurity: Whether the customer has online security (Yes, No, No internet service)
* ◀️OnlineBackup: Whether the customer has online backup (Yes, No, No internet service)
* 🚫DeviceProtection: Whether the customer has device protection (Yes, No, No internet service)
* 🧢TechSupport: Whether the customer has technical support (Yes, No, No internet service)
* 📺StreamingTV: Whether the customer has streaming TV (Yes, No, No Internet service)
* 📽️StreamingMovies: Whether the customer streams movies (Yes, No, No internet service)
* 🗞️Contract: Whether the customer's contract term (Month-to-month, One year, Two years)
* 📰PaperlessBilling: Whether the customer has paperless billing (Yes, No)
* 💳PaymentMethod: Whether the customer's payment method (Electronic check, Postal check, Wire transfer (automatic), Credit card (automatic))
* 🤑MonthlyCharges: The amount charged to the customer monthly
* 💰TotalCharges: The total amount charged to the customer
* ❌Churn: Whether the customer uses (Yes or No* * * * * )
---

## Libraries Required

* import pandas as pd
* import numpy as np
* import matplotlib.pyplot as plt
* import seaborn as sns
* from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
* from sklearn.model_selection import train_test_split
* from sklearn.linear_model import LogisticRegression
* from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
* import joblib   



1. **Data Loading & Inspection**
   - Load dataset using `pandas`
   - Inspect first rows, data types, missing values, and unique value counts

2. **Data Cleaning & Preprocessing**
   - Convert `TotalCharges` to numeric and fill missing values with median
   - Encode categorical columns:
     - **LabelEncoder** for binary columns
     - **OneHotEncoder** for multi-category columns
   - Scale numeric columns (`tenure`, `MonthlyCharges`, `TotalCharges`) using **StandardScaler**

3. **Exploratory Data Analysis (EDA)**
   - **Churn Distribution:** Pie chart of churned vs non-churned customers
   - **Bar Charts:** Churn vs categorical features (`gender`, `Contract`, `PaymentMethod`, `InternetService`)
   - **Histograms & KDE:** Distribution of `MonthlyCharges` with churn
   - **Heatmap:** Correlation of service features with churn (`OnlineSecurity`, `TechSupport`, etc.)
   - **Scatter Plot:** `MonthlyCharges` vs `TotalCharges` colored by churn

4. **Model Building**
   - Split dataset into training and testing sets using `train_test_split`
   - Train **Logistic Regression** model
   - Evaluate using:
     - **Accuracy Score**
     - **Confusion Matrix**
     - **Classification Report**

5. **Model Saving**

   joblib.dump(Model, "churn_model.pkl")      
   joblib.dump(le, "label_encoder.pkl")       
   joblib.dump(ohe, "onehot_encoder.pkl")     
   joblib.dump(scaler, "scaler.pkl")

## Insights

* Customers without OnlineSecurity or TechSupport are more likely to churn
* Higher MonthlyCharges correlate with higher churn probability
* Contract type and payment method influence churn behavior

---
## Sample Charts:
**Heatmap**
<img width="950" height="782" alt="image" src="https://github.com/user-attachments/assets/cae16c3e-2a5f-42fa-8133-39d6ef641d50" />
**KDE Plot**
<img width="921" height="592" alt="image" src="https://github.com/user-attachments/assets/e6571528-1fc1-4714-aa87-0d805c907f3a" />
**Histogram**
<img width="866" height="582" alt="image" src="https://github.com/user-attachments/assets/d10b10be-205a-4b2c-974f-262de5a1a155" />

**And many more...**

---

## Author

Isbah Ali - Data Analyst 
