Technical Document: AVM Workflow and Code Explanation

The goal is a reproducible Python Automated Valuation Model (AVM) for Winnipeg property assessments using publicly available property data .I use supervised learning because historical assessed values are available.

1\. Data Acquisition & Cleaning

* Fetch: I pull data from the Socrata API, with an OData fallback to ensure robust access.  
* Inspection & Conversion: I inspect feature target relationships and missing rates. Numeric fields (e.g., living area, land area, year built, assessment values) are coerced to numeric types after stripping “$” and “,”.  
* Data is visualized to find correlation and relevance with the target variable (total\_assessment\_value)

2\. Feature Selection & Engineering

* Selection: I keep features with clear relevance—location, size, structure, amenity, and property codes, and drop those with excessive null values or redundancy.  
* Feature Engineering, new features are created:  
  * *amenity\_score* weights binary amenities (garage, pool, etc.) on a 0–10 scale according to importance of the amenity.  
  * *property\_age* is derived from year built (current year minus build year).  
  * *has\_garage* flags any garage presence.  
  * *luxury\_score* sums luxury amenities (pool, fireplace).  
    Original amenity columns are dropped to avoid multicollinearity.

3\. Preprocessing

* Imputation & Scaling:  
  * Numeric features use KNN Imputer (k=5) to leverage nearby property values and StandardScaler for variance normalization.  
  * Categorical features use “most frequent” imputation and OneHotEncoder to convert nominal data into binary vectors.  
* Percentile Trimming: I remove assessments outside the 1st–99th percentile, which is less aggressive than IQR and preserves high-value cases.Extreme assessment values outside the 1st–99th percentiles are trimmed to mitigate the effect of a long right tail.

4\. Model Training:

* Train/Test Split: An 80/20 split with fixed random state is used to ensure reproducibility.  
* Model Definitions: I configure two regressors with parameter grids:  
  * *RandomForestRegressor* (trees, depth, min samples split)  
  * *XGBRegressor* (boosting rounds, learning rate, max depth)  
* Pipeline & Hyperparameter Tuning: A pipeline couples preprocessing and the regressor. Randomized search cross validation (3-fold, 5 iterations) optimizes negative MSE. The best estimator and parameters are selected.  
* Evaluation metrics: Predictions on the test set yield R², RMSE, MAE, and MAPE. Results and metadata are stored in a dictionary, and the model with the highest R² is tracked.

5\. Model Evaluation & Summary

* evaluate\_best\_model: Identifies and logs the best model’s parameters and metrics. Computes accuracy within 5%, 10%, 15% of actual values. Splits performance by low/mid/high terciles to report segment MAEs.  
* model\_summary: Formats a summary table of all models’ R², RMSE, MAE, and MAPE, and highlights the champion model’s key statistics.

6: Results

* XGBoost with the following parameters : \[**regressor\_\_n\_estimators:** 200, **regressor\_\_learning\_rate:** 0.1, **regressor\_\_max\_depth:** 10\] achieved the best performance:  
  * R²=0.83,   
  * MAE=$42K  
  * MAPE=10.3%.  
* It delivers 46.6% of estimates within ±5%, and 72.0% within ±10% of actual values which are comparable to leading commercial AVMs.

7: The following assumptions are made:

Assumptions:

* The City’s dataset schema remains stable.  
* Internet access is available for live data fetch.  
* Open source libraries listed in ‘requirements.txt’ are installed.

This pipeline: data fetch, cleaning, feature engineering, robust preprocessing, hyperparameter-tuned modeling, and comprehensive evaluation—follows industry best practices for a transparent, reproducible AVM. The logging is saved to the working

directory under ‘winnipeg\_avm.log.’