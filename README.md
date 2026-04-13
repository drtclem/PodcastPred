# Podcast Listening Time Predictor

A regression pipeline for predicting podcast episode listening time, built end-to-end in Python using a 750,000-record dataset.

## Overview

This project walks through a full supervised learning workflow — from raw data ingestion and feature engineering through EDA, baseline modeling, regularization, and hyperparameter tuning — with the goal of predicting how long a listener will engage with a podcast episode.

See [`final.ipynb`](./final.ipynb) for the full walkthrough and all project work.

## Dataset

750,000 training records with the following features:

- `Podcast_Name`, `Episode_Title`, `Genre`
- `Episode_Length_minutes`
- `Host_Popularity_percentage`, `Guest_Popularity_percentage`
- `Publication_Day`, `Publication_Time`
- `Number_of_Ads`, `Episode_Sentiment`
- **Target:** `Listening_Time_minutes`

## Workflow

**1. Data Preparation**
- Load train/test CSVs and inspect structure
- Encode categorical variables (genre dummies, sentiment mapping)
- Engineer cyclical time features (sin/cos encoding for day and time of day)
- Handle missing values and drop low-signal columns

**2. Exploratory Data Analysis**
- Distribution plots for key numeric features
- Correlation heatmap against target variable
- Listening time by genre, sentiment, host/guest popularity, and time of day

**3. Modeling**
- Baseline Linear Regression
- Ridge and Lasso regularization with GridSearchCV tuning
- Random Forest Regressor
- XGBoost with GridSearchCV and RandomizedSearchCV hyperparameter tuning
- Final model evaluation: RMSE and R²
- Actual vs. predicted scatter plot

## Tech Stack

- Python
- Scikit-learn
- XGBoost
- NumPy, Pandas
- Matplotlib, Seaborn
- Joblib (model persistence)

## Results

Iterative model development from baseline linear regression through tuned XGBoost, with cross-validated RMSE and R² tracked at each stage.

## Author

Taylor Clements, PhD
