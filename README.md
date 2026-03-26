Credit Default Risk Scoring System
An end-to-end machine learning pipeline for predicting loan default risk built on a real-world 10-file, 3GB financial dataset with 300,000+ loan applications.
Overview
This project tackles a binary classification problem — predicting whether a client will have difficulty repaying their loan. The dataset contains highly imbalanced classes (~8% default rate) and requires complex multi-table feature engineering across bureau records, payment histories, and previous loan applications.
Pipeline

Data exploration — multi-table schema analysis, missing value profiling, anomaly detection
Feature engineering — 500+ features including credit risk ratios, behavioral payment indicators, and external score interactions
Multi-table aggregation — bureau, previous applications, POS cash, installments, and credit card histories
Modeling — LightGBM with Bayesian hyperparameter optimization via Optuna
Evaluation — Stratified 5-fold cross validation with AUC-ROC

Tech Stack
Python, Pandas, NumPy, LightGBM, Scikit-learn, Optuna
