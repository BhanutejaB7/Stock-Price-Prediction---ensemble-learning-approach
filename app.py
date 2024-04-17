import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, BaggingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt

app = Flask(__name__, static_url_path='/static')

def validate_company(company_code):
    try:
        # Check if the company exists
        yf.Ticker(company_code)
        return True
    except ValueError:
        return False

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        company = request.form.get('company')
        custom_company = request.form.get('custom_company')
        days = int(request.form.get('days'))

        # Check if custom_company is provided, otherwise use company
        company_code = custom_company if custom_company else company

        if company_code is None or company_code.strip() == "":
            return render_template('index.html', error="Please select or enter a company.")

        if not validate_company(company_code):
            return render_template('index.html', error="Company does not exist")

        # Fetch historical stock data for the selected company
        start_date = datetime.now() - timedelta(days=365)  # One year ago
        end_date = datetime.now()
        stock_data = yf.download(company_code, start=start_date, end=end_date)

        if stock_data.empty:
            # If no data found for the company, display error message
            return render_template('index.html', error="No data available for the selected company")

        # Prepare the data
        stock_data['Date'] = stock_data.index
        stock_data.reset_index(drop=True, inplace=True)

        # Add features
        stock_data['Day'] = stock_data['Date'].dt.day
        stock_data['Month'] = stock_data['Date'].dt.month
        stock_data['Year'] = stock_data['Date'].dt.year
        stock_data['Weekday'] = stock_data['Date'].dt.weekday

        # Create lag features for prediction
        for i in range(1, 6):
            stock_data[f'Close_Lag_{i}'] = stock_data['Close'].shift(i)

        # Drop rows with NaN values
        stock_data.dropna(inplace=True)

        # Split data into features and target variable
        X = stock_data.drop(['Date', 'Close', 'Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)
        y = stock_data['Close']

        # Train ensemble learning models
        models = {
            'LightGBM': lgb.LGBMRegressor(force_col_wise=True, n_estimators=150, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=150, random_state=42),
            'Extra Trees': ExtraTreesRegressor(n_estimators=150, random_state=42),
            'Bagging': BaggingRegressor(n_estimators=150, random_state=42),
            'CatBoost': CatBoostRegressor(iterations=150, random_state=42)
        }

        future_predictions = {}
        plot_paths = {}

        # Make predictions for the next 'days' days using each model
        future_dates = [stock_data['Date'].iloc[-1] + timedelta(days=i) for i in range(0, days)]
        future_data = pd.DataFrame({'Date': future_dates})
        future_data['Day'] = future_data['Date'].dt.day
        future_data['Month'] = future_data['Date'].dt.month
        future_data['Year'] = future_data['Date'].dt.year
        future_data['Weekday'] = future_data['Date'].dt.weekday

        for i in range(1, 6):
            future_data[f'Close_Lag_{i}'] = stock_data['Close'].iloc[-i]

        for name, model in models.items():
            model.fit(X, y)
            future_predictions[name] = model.predict(future_data.drop('Date', axis=1))

            # Plot future predictions
            plt.figure(figsize=(10, 6))
            plt.plot(stock_data['Date'], stock_data['Close'], label='Historical Data')
            plt.plot(future_dates, future_predictions[name], 'r--', label=f'Future Predictions ({name})')
            plt.title(f'Stock Price Prediction for {company_code} for the next {days} days ({name})')
            plt.xlabel('Date')
            plt.ylabel('Close Price')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            plot_path = f'static/plot_{name.title().replace(" ", "_")}.png'
            plt.savefig(plot_path)
            plt.close()
            plot_paths[name.title()] = plot_path

        # Collect predicted values for each model
        predicted_values = {}
        for name, preds in future_predictions.items():
            predicted_values[name.title()] = [(future_dates[i], preds[i]) for i in range(len(future_dates))]

        return render_template('index.html',
                                plot_paths=plot_paths,
                                future_dates=future_dates,
                                future_predictions=predicted_values)

    return render_template('index.html', error=None)

if __name__ == '__main__':
    app.run(debug=True)
