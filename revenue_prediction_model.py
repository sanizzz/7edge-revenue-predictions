import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import itertools
import warnings
warnings.filterwarnings('ignore')

def prepare_data():
    df = pd.read_csv('7ege/synthetic_revenue_data.csv')
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df.set_index('Date', inplace=True)
    df = df.asfreq('D')
    df['MA7'] = df['Revenue'].rolling(window=7).mean()
    df['MA30'] = df['Revenue'].rolling(window=30).mean()
    df = df.fillna(method='bfill')
    return df

def evaluate_predictions(actual, predictions):
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mae = mean_absolute_error(actual, predictions)
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100
    accuracy = 100 - mape
    within_1_percent = np.mean(np.abs((actual - predictions) / actual) <= 0.01) * 100
    within_5_percent = np.mean(np.abs((actual - predictions) / actual) <= 0.05) * 100
    within_10_percent = np.mean(np.abs((actual - predictions) / actual) <= 0.10) * 100
    return rmse, mae, mape, accuracy, within_1_percent, within_5_percent, within_10_percent

def optimize_sarimax_parameters(data, train_size):
    train = data[:train_size]
    test = data[train_size:]
    
    parameter_combinations = [
        ((3,1,3), (1,1,1,7)),
        ((4,1,3), (1,1,1,7)),
        ((5,1,3), (1,1,1,7)),
        ((3,1,3), (2,1,2,7)),
        ((4,1,3), (2,1,2,7)),
        ((3,1,3), (1,1,1,14)),
        ((4,1,3), (2,1,2,14)),
        ((5,1,3), (1,1,1,30)),
        ((6,1,3), (1,1,1,30))
    ]
    
    best_accuracy = 0
    best_params = None
    best_model = None
    best_metrics = None
    all_results = []
    
    for order, seasonal_order in parameter_combinations:
        try:
            model = SARIMAX(train,
                          order=order,
                          seasonal_order=seasonal_order,
                          enforce_stationarity=False,
                          enforce_invertibility=False)
            results = model.fit(disp=False, method='powell', maxiter=200)
            
            predictions = results.forecast(steps=len(test))
            metrics = evaluate_predictions(test, predictions)
            rmse, mae, mape, accuracy, within_1, within_5, within_10 = metrics
            
            all_results.append({
                'order': order,
                'seasonal_order': seasonal_order,
                'accuracy': accuracy,
                'metrics': metrics,
                'model': results
            })
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = (order, seasonal_order)
                best_model = results
                best_metrics = metrics
            
        except:
            continue
    
    all_results.sort(key=lambda x: -x['accuracy'])
    best_model = all_results[0]['model']
    best_metrics = all_results[0]['metrics']
    best_params = (all_results[0]['order'], all_results[0]['seasonal_order'])
    
    return best_params, best_metrics, best_model

def plot_results(train, test, predictions, future_predictions):
    plt.figure(figsize=(15, 7))
    plt.plot(train.index, train, label='Actual Data', color='blue', alpha=0.7)
    plt.plot(test.index, test, label='Actual Test Data', color='green', alpha=0.7)
    plt.plot(test.index, predictions, label='Predictions', color='red', alpha=0.7)
    
    future_dates = pd.date_range(test.index[-1] + pd.Timedelta(days=1), periods=30)
    plt.plot(future_dates, future_predictions, label='Future Predictions', 
             color='purple', linestyle='--', alpha=0.7)
    
    plt.title('Revenue Predictions')
    plt.xlabel('Date')
    plt.ylabel('Revenue ($)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    df = prepare_data()
    train_size = int(len(df) * 0.8)
    train = df['Revenue'][:train_size]
    test = df['Revenue'][train_size:]
    
    best_params, best_metrics, best_model = optimize_sarimax_parameters(df['Revenue'], train_size)
    rmse, mae, mape, accuracy, within_1, within_5, within_10 = best_metrics
    
    predictions = best_model.forecast(steps=len(test))
    future_predictions = best_model.forecast(steps=30)
    
    plot_results(train, test, predictions, future_predictions)
    
    print("\nPrediction Results:")
    print("=" * 50)
    print(f"Model Accuracy: {accuracy:.2f}%")
    print(f"Average Error: ${mae:.2f}")
    print("\nDetailed Future Revenue Predictions:")
    print("=" * 50)
    
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=30)
    predictions_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Revenue': future_predictions
    })
    
    print("\nNext 30 Days Revenue Predictions:")
    print("=" * 50)
    for _, row in predictions_df.iterrows():
        print(f"{row['Date'].strftime('%Y-%m-%d')}: ${row['Predicted Revenue']:,.2f}")
    
    print("\nSummary Statistics:")
    print("=" * 50)
    print(f"Average Predicted Revenue: ${future_predictions.mean():,.2f}")
    print(f"Minimum Predicted Revenue: ${future_predictions.min():,.2f}")
    print(f"Maximum Predicted Revenue: ${future_predictions.max():,.2f}")
    
    predictions_df.to_csv('revenue_predictions.csv', index=False)
    print("\nPredictions have been saved to 'revenue_predictions.csv'")

if __name__ == "__main__":
    main() 