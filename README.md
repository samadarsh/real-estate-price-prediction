# Real Estate Price Prediction

## Overview
This project predicts real estate prices in Bangalore. It uses a dataset containing various features like location, size, total square footage, number of bathrooms, and balconies. It walks through the end-to-end process of building a machine learning model, covering data loading, data cleaning, feature engineering, outlier removal, model building/tuning using GridSearchCV, and exporting the trained model.

## Tech Stack
- **Python** (Pandas, Numpy, Matplotlib)
- **Machine Learning**: Scikit-Learn (Linear Regression, Lasso, Decision Tree Regressor)
- **Model Deployment Prep**: Pickle, JSON

## Project Structure
- `real_estate_price_pred.ipynb`: The primary Jupyter Notebook containing the entire machine learning pipeline.

*(Note: When the notebook is fully run, it will also generate `banglore_home_prices_model.pickle` and `columns.json` which contain the trained model and features.)*

## Key Steps 
1. **Data Loading:** Read the dataset into a pandas dataframe.
2. **Data Cleaning:** Handle missing values (`NA`) and drop unnecessary columns.
3. **Feature Engineering:** Create a new integer feature `bhk` from the `size` column. Handle range values in the `total_sqft` column by averaging them.
4. **Dimensionality Reduction:** Categorize locations with few data points into a generic 'other' category to reduce complexity.
5. **Outlier Removal:** Use standard deviation and domain knowledge (e.g., minimum square footage per bedroom) to remove anomalies in the data.
6. **Model Building:** Train multiple regression models (Linear Regression, Lasso, and Decision Tree).
7. **Hyperparameter Tuning:** Use `GridSearchCV` and K-Fold cross-validation to find the best performing model.
8. **Export Model:** Save the trained Linear Regression model to a pickle file and the feature names to a JSON file.
