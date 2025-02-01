import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

class SalesProfitPredictor:
    def __init__(self, mongodb_uri='mongodb://localhost:27017/'):
        self.client = MongoClient(mongodb_uri)
        self.db = self.client['indian_retail']
    
    def prepare_sales_data(self, start_date=None, end_date=None) -> pd.DataFrame:
        """Previous implementation remains the same"""
        query = {}
        if start_date:
            query['date'] = {'$gte': start_date}
        if end_date:
            query['date'] = query.get('date', {})
            query['date']['$lte'] = end_date
        
        transactions = list(self.db.transactions.find(query))
        sales_data = []
        for txn in transactions:
            sales_data.append({
                'date': pd.to_datetime(txn['date']),
                'total_amount': float(txn['payment']['final_amount']),
                'transaction_count': 1
            })
        
        sales_df = pd.DataFrame(sales_data)
        daily_sales = sales_df.groupby(pd.Grouper(key='date', freq='D')).agg({
            'total_amount': 'sum',
            'transaction_count': 'sum'
        }).reset_index()
        
        date_range = pd.date_range(start=daily_sales['date'].min(), end=daily_sales['date'].max())
        daily_sales = daily_sales.set_index('date')
        daily_sales = daily_sales.reindex(date_range, fill_value=0)
        
        return daily_sales.reset_index().rename(columns={'index': 'date'})

    def extract_patterns(self, sales_series):
        """
        Extract daily and weekly patterns from the sales data
        """
        # Calculate daily pattern
        daily_pattern = sales_series.groupby(sales_series.index.dayofweek).mean()
        daily_pattern = daily_pattern / daily_pattern.mean()
        
        # Calculate weekly pattern
        weekly_sales = sales_series.resample('W').mean()
        weekly_pattern = weekly_sales.rolling(window=4, center=True).mean()
        
        return daily_pattern, weekly_pattern

    def forecast_sales(self, forecast_steps=30):
        """Forecast sales with preserved seasonal patterns"""
        # Prepare sales data
        sales_df = self.prepare_sales_data()
        sales_series = sales_df.set_index('date')['total_amount']
        
        # Handle outliers
        Q1 = sales_series.quantile(0.25)
        Q3 = sales_series.quantile(0.75)
        IQR = Q3 - Q1
        cleaned_series = sales_series.copy()
        cleaned_series[cleaned_series > Q3 + 1.5 * IQR] = Q3 + 1.5 * IQR
        cleaned_series[cleaned_series < Q1 - 1.5 * IQR] = Q1 - 1.5 * IQR
        
        # Extract patterns
        daily_pattern, weekly_pattern = self.extract_patterns(cleaned_series)
        
        try:
            # Decompose the series
            decomposition = seasonal_decompose(
                cleaned_series, 
                period=7,  # Weekly seasonality
                extrapolate_trend='freq'
            )
            
            # Forecast trend
            trend_model = ARIMA(decomposition.trend.dropna(), order=(1, 1, 1))
            trend_fit = trend_model.fit()
            trend_forecast = trend_fit.forecast(steps=forecast_steps)
            
            # Create forecast dates
            forecast_dates = pd.date_range(
                start=sales_series.index[-1] + timedelta(days=1),
                periods=forecast_steps
            )
            
            # Apply patterns to create the final forecast
            forecast_values = []
            for i, date in enumerate(forecast_dates):
                # Get base trend value
                trend_value = trend_forecast[i]
                
                # Apply daily pattern
                day_factor = daily_pattern[date.dayofweek]
                
                # Calculate forecast with patterns
                forecast_value = trend_value * day_factor
                
                # Add some controlled randomness to mimic historical volatility
                volatility = cleaned_series.std() * 0.3
                random_factor = np.random.normal(1, 0.1)  # 10% random variation
                forecast_value *= random_factor
                
                forecast_values.append(forecast_value)
            
            forecast_values = np.array(forecast_values)
            
            # Calculate confidence intervals
            std_dev = cleaned_series.std() * 0.5
            forecast_std = std_dev * np.sqrt(np.arange(1, forecast_steps + 1)) * 0.3
            
            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'forecasted_sales': forecast_values,
                'lower_ci': forecast_values - forecast_std,
                'upper_ci': forecast_values + forecast_std
            })
            
            # Ensure non-negative values
            forecast_df['lower_ci'] = forecast_df['lower_ci'].clip(lower=0)
            
            return forecast_df
            
        except Exception as e:
            print(f"Error in forecasting: {e}")
            return self._fallback_forecast(cleaned_series, forecast_steps)
    
    def _fallback_forecast(self, sales_series, forecast_steps):
        """Fallback forecasting method with patterns"""
        # Extract patterns
        daily_pattern, _ = self.extract_patterns(sales_series)
        
        # Calculate base level and trend
        recent_data = sales_series[-30:]
        base_level = recent_data.mean()
        
        # Generate forecast dates
        forecast_dates = pd.date_range(
            start=sales_series.index[-1] + timedelta(days=1),
            periods=forecast_steps
        )
        
        # Generate forecast with patterns
        forecast_values = []
        for date in forecast_dates:
            # Apply daily pattern
            day_factor = daily_pattern[date.dayofweek]
            forecast_value = base_level * day_factor
            
            # Add controlled randomness
            volatility = sales_series.std() * 0.3
            random_factor = np.random.normal(1, 0.1)
            forecast_value *= random_factor
            
            forecast_values.append(forecast_value)
        
        forecast_values = np.array(forecast_values)
        
        # Calculate confidence intervals
        std = sales_series.std() * 0.4
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecasted_sales': forecast_values,
            'lower_ci': forecast_values - std,
            'upper_ci': forecast_values + std
        })
        
        # Ensure non-negative values
        forecast_df['lower_ci'] = forecast_df['lower_ci'].clip(lower=0)
        
        return forecast_df
    
    def visualize_sales_forecast(self, forecast_df):
        """Previous visualization implementation with adjusted y-axis"""
        sales_df = self.prepare_sales_data()
        historical_sales = sales_df.set_index('date')['total_amount']
        
        plt.figure(figsize=(15, 8))
        
        # Plot historical data
        plt.plot(historical_sales.index, historical_sales.values,
                label='Historical Sales', color='blue', alpha=0.6)
        
        # Plot forecast
        plt.plot(forecast_df['date'], forecast_df['forecasted_sales'],
                color='red', label='Forecasted Sales', linewidth=2)
        
        # Plot confidence intervals
        plt.fill_between(
            forecast_df['date'],
            forecast_df['lower_ci'],
            forecast_df['upper_ci'],
            color='red', alpha=0.2,
            label='95% Confidence Interval'
        )
        
        # Add rolling average
        rolling_avg = historical_sales.rolling(window=7, center=True).mean()
        plt.plot(rolling_avg.index, rolling_avg.values,
                color='green', label='7-day Moving Average',
                linestyle='--', alpha=0.5)
        
        # Set y-axis limits based on data
        all_values = np.concatenate([
            historical_sales.values,
            forecast_df['upper_ci'].values,
            forecast_df['lower_ci'].values
        ])
        plt.ylim(0, np.percentile(all_values, 99))
        
        plt.title('Sales Forecast with Confidence Intervals')
        plt.xlabel('Date')
        plt.ylabel('Sales Amount')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('sales_forecast.png')
        plt.close()

def main():
    predictor = SalesProfitPredictor()
    print("Forecasting Sales...")
    sales_forecast = predictor.forecast_sales(forecast_steps=30)
    print("\nForecast Summary:")
    print(sales_forecast)
    predictor.visualize_sales_forecast(sales_forecast)

if __name__ == "__main__":
    main()