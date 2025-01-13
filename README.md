# 7edge revenue predictions
 

# SARIMAX Revenue Forecasting

This project provides a revenue forecasting system using the Seasonal Autoregressive Integrated Moving Average with eXogenous factors (SARIMAX) model. It is designed to analyze time series data and predict future revenues based on historical data. The system includes data preparation, parameter optimization, evaluation metrics, and visualization of results.

---

## Table of Contents
- [Features](#features)
- [Dependencies](#dependencies)
- [Dataset Format](#dataset-format)
- [Usage](#usage)
- [Methodology](#methodology)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [File Descriptions](#file-descriptions)
- [License](#license)

---

## Features
- **Time Series Analysis**: Processes daily revenue data with moving averages and backfilling missing values.
- **SARIMAX Model Optimization**: Optimizes SARIMAX parameters for the highest accuracy using grid search.
- **Evaluation Metrics**: Calculates RMSE, MAE, MAPE, and prediction accuracy.
- **Future Forecasting**: Provides detailed forecasts for the next 30 days.
- **Visualization**: Plots actual data, predictions, and future forecasts for better understanding.

---

## Dependencies
The following Python libraries are required:
- `pandas`
- `numpy`
- `statsmodels`
- `scikit-learn`
- `matplotlib`

Install dependencies using:
```bash
pip install pandas numpy statsmodels scikit-learn matplotlib
```

---

## Dataset Format
The input dataset should be a CSV file with the following structure:
- **Date**: Dates in `dd/mm/yyyy` format.
- **Revenue**: Daily revenue values.

Example:
```csv
Date,Revenue
01/01/2020,1500
02/01/2020,1600
03/01/2020,1450
```

---

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/username/revenue-forecasting.git
   ```
2. Place your dataset at `7ege/synthetic_revenue_data.csv`.
3. Run the script:
   ```bash
   python main.py
   ```
4. View the generated revenue predictions in the terminal and the saved file `revenue_predictions.csv`.

---

## Methodology
1. **Data Preparation**:
   - Converts the `Date` column to datetime format and sets it as the index.
   - Applies daily frequency and calculates moving averages (7-day and 30-day).
   - Fills missing data with backfill methods.

2. **Model Optimization**:
   - Tests multiple SARIMAX parameter combinations.
   - Selects the model with the highest prediction accuracy.

3. **Evaluation**:
   - Compares predictions with actual test data using metrics like RMSE, MAE, and MAPE.

4. **Forecasting**:
   - Predicts test data and 30 days of future revenues.

5. **Visualization**:
   - Plots the training data, test data, predictions, and future forecasts.

---

## Evaluation Metrics
- **Root Mean Squared Error (RMSE)**: Measures prediction error magnitude.
- **Mean Absolute Error (MAE)**: Average absolute difference between actual and predicted values.
- **Mean Absolute Percentage Error (MAPE)**: Percentage error relative to actual values.
- **Accuracy**: Percentage of predictions within acceptable error margins.

---

## Results
After running the script, key results include:
- **Prediction Accuracy**: Model accuracy as a percentage.
- **Future Revenue Forecast**: Revenue predictions for the next 30 days.
- **Summary Statistics**: Average, minimum, and maximum predicted revenues.

---

## File Descriptions
- **`main.py`**: Main script for data preparation, model optimization, forecasting, and visualization.
- **`revenue_predictions.csv`**: Output file with 30-day revenue predictions.
- **`README.md`**: Project documentation (this file).

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

