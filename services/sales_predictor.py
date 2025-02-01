import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from pymongo import MongoClient

class SalesProfitPredictor:
    def __init__(self, mongodb_uri='mongodb://localhost:27017/'):
        self.client = MongoClient(mongodb_uri)
        self.db = self.client['indian_retail']
    
    def prepare_sales_data(self, start_date=None, end_date=None, query_filter=None) -> pd.DataFrame:
        """Prepare daily sales data from transactions"""
        query = {}
        if start_date or end_date:
            query['date'] = {}
            if start_date:
                query['date']['$gte'] = start_date
            if end_date:
                query['date']['$lte'] = end_date
        
        if query_filter:
            query.update(query_filter)
        
        transactions = list(self.db.transactions.find(query))
        sales_data = []
        for txn in transactions:
            sales_data.append({
                'date': pd.to_datetime(txn['date']),
                'total_amount': float(txn['payment']['final_amount']),
                'transaction_count': 1
            })
        
        if not sales_data:
            raise ValueError("No sales data found for the specified criteria")
            
        sales_df = pd.DataFrame(sales_data)
        daily_sales = sales_df.groupby(pd.Grouper(key='date', freq='D')).agg({
            'total_amount': 'sum',
            'transaction_count': 'sum'
        }).reset_index()
        
        date_range = pd.date_range(
            start=daily_sales['date'].min(),
            end=daily_sales['date'].max()
        )
        daily_sales = daily_sales.set_index('date')
        daily_sales = daily_sales.reindex(date_range, fill_value=0)
        
        return daily_sales.reset_index().rename(columns={'index': 'date'})
    
    def extract_patterns(self, sales_series):
        """Extract daily and weekly patterns from the sales data"""
        daily_pattern = sales_series.groupby(sales_series.index.dayofweek).mean()
        daily_pattern = daily_pattern / daily_pattern.mean()
        
        weekly_sales = sales_series.resample('W').mean()
        weekly_pattern = weekly_sales.rolling(window=4, center=True).mean()
        
        return daily_pattern, weekly_pattern

    def forecast_sales(self, forecast_steps=30):
        """
        Forecast sales with preserved seasonal patterns and include historical data
        Returns both historical and forecasted data in a single DataFrame
        """
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
        
        try:
            # Decompose the series
            decomposition = seasonal_decompose(
                cleaned_series, 
                period=7,
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
            
            # Generate forecast values with patterns
            daily_pattern, _ = self.extract_patterns(cleaned_series)
            forecast_values = []
            for i, date in enumerate(forecast_dates):
                trend_value = trend_forecast[i]
                day_factor = daily_pattern[date.dayofweek]
                forecast_value = trend_value * day_factor
                random_factor = np.random.normal(1, 0.1)
                forecast_value *= random_factor
                forecast_values.append(forecast_value)
            
            forecast_values = np.array(forecast_values)
            
            # Calculate confidence intervals
            std_dev = cleaned_series.std() * 0.5
            forecast_std = std_dev * np.sqrt(np.arange(1, forecast_steps + 1)) * 0.3
            
            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'sales': forecast_values,
                'lower_ci': forecast_values - forecast_std,
                'upper_ci': forecast_values + forecast_std,
                'is_forecast': True
            })
            
            # Create historical dataframe
            historical_df = pd.DataFrame({
                'date': sales_series.index,
                'sales': sales_series.values,
                'lower_ci': None,
                'upper_ci': None,
                'is_forecast': False
            })
            
            # Combine historical and forecast data
            combined_df = pd.concat([historical_df, forecast_df], ignore_index=True)
            
            # Ensure non-negative values
            combined_df['lower_ci'] = combined_df['lower_ci'].clip(lower=0)
            combined_df['sales'] = combined_df['sales'].clip(lower=0)
            
            return combined_df
            
        except Exception as e:
            print(f"Error in forecasting: {e}")
            return self._fallback_forecast(cleaned_series, forecast_steps)
    
    def _fallback_forecast(self, sales_series, forecast_steps):
        """Fallback forecasting method with patterns, including historical data"""
        daily_pattern, _ = self.extract_patterns(sales_series)
        recent_data = sales_series[-30:]
        base_level = recent_data.mean()
        
        # Generate forecast dates
        forecast_dates = pd.date_range(
            start=sales_series.index[-1] + timedelta(days=1),
            periods=forecast_steps
        )
        
        # Generate forecast values
        forecast_values = []
        for date in forecast_dates:
            day_factor = daily_pattern[date.dayofweek]
            forecast_value = base_level * day_factor
            random_factor = np.random.normal(1, 0.1)
            forecast_value *= random_factor
            forecast_values.append(forecast_value)
        
        forecast_values = np.array(forecast_values)
        std = sales_series.std() * 0.4
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'sales': forecast_values,
            'lower_ci': forecast_values - std,
            'upper_ci': forecast_values + std,
            'is_forecast': True
        })
        
        # Create historical dataframe
        historical_df = pd.DataFrame({
            'date': sales_series.index,
            'sales': sales_series.values,
            'lower_ci': None,
            'upper_ci': None,
            'is_forecast': False
        })
        
        # Combine historical and forecast data
        combined_df = pd.concat([historical_df, forecast_df], ignore_index=True)
        
        # Ensure non-negative values
        combined_df['lower_ci'] = combined_df['lower_ci'].clip(lower=0)
        combined_df['sales'] = combined_df['sales'].clip(lower=0)
        
        return combined_df

class SalesPredictorService:
    def __init__(self, mongodb_uri='mongodb://localhost:27017/'):
        self.predictor = SalesProfitPredictor(mongodb_uri)

    async def get_sales_forecast(self, store_id: str = None, days: int = 30):
        """Get sales forecast with historical data for specific store or all stores"""
        try:
            # Prepare data based on store_id
            if store_id:
                sales_df = self.predictor.prepare_sales_data(
                    query_filter={'store_id': store_id}
                )
            else:
                sales_df = self.predictor.prepare_sales_data()

            # Get forecast with historical data
            combined_df = self.predictor.forecast_sales(forecast_steps=days)
            
            # Convert to response format
            forecasts = [
                {
                    'date': row['date'],
                    'sales': float(row['sales']),
                    'lower_ci': float(row['lower_ci']) if pd.notna(row['lower_ci']) else None,
                    'upper_ci': float(row['upper_ci']) if pd.notna(row['upper_ci']) else None,
                    'is_forecast': bool(row['is_forecast'])
                }
                for _, row in combined_df.iterrows()
            ]
            
            # Calculate summary statistics
            forecast_data = combined_df[combined_df['is_forecast']]
            summary = {
                'mean_forecast': float(forecast_data['sales'].mean()),
                'min_forecast': float(forecast_data['sales'].min()),
                'max_forecast': float(forecast_data['sales'].max()),
                'total_forecast': float(forecast_data['sales'].sum()),
                'confidence_width': float(
                    (forecast_data['upper_ci'] - forecast_data['lower_ci']).mean()
                )
            }
            
            return {
                'forecasts': forecasts,
                'summary': summary
            }
        except Exception as e:
            raise Exception(f"Error generating sales forecast: {str(e)}")