import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, IsolationForest
import lightgbm as lgb
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class MLStoreAnalyzer:
    def __init__(self, mongodb_uri='mongodb://localhost:27017/'):
        self.client = MongoClient(mongodb_uri)
        self.db = self.client['indian_retail']
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _prepare_timeseries_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare time series features for a DataFrame with a date index"""
        df = data.copy()
        
        # Basic time features
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        
        # Rolling statistics (7-day and 30-day)
        if 'sales' in df.columns:
            windows = [7, 30]
            for window in windows:
                df[f'rolling_mean_{window}d'] = df['sales'].rolling(window=window, min_periods=1).mean()
                df[f'rolling_std_{window}d'] = df['sales'].rolling(window=window, min_periods=1).std()
        
        return df

    def _predict_future_sales(self, txn_df: pd.DataFrame) -> Dict:
        """Predict future sales using time series features"""
        # Create daily sales data
        sales_df = txn_df.groupby('date')['payment_final_amount'].sum().reset_index()
        sales_df = sales_df.set_index('date')
        sales_df = sales_df.rename(columns={'payment_final_amount': 'sales'})
        
        # Prepare features for historical data
        features_df = self._prepare_timeseries_features(sales_df)
        
        # Define feature columns for model
        feature_cols = ['year', 'month', 'day', 'dayofweek', 'quarter', 
                       'is_weekend', 'is_month_start', 'is_month_end']
        
        if len(features_df) >= 7:  # Only add rolling features if we have enough data
            feature_cols.extend(['rolling_mean_7d', 'rolling_std_7d'])
        if len(features_df) >= 30:  # Only add 30-day features if we have enough data
            feature_cols.extend(['rolling_mean_30d', 'rolling_std_30d'])
        
        # Train model
        model = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42
        )
        
        X = features_df[feature_cols]
        y = features_df['sales']
        
        model.fit(X, y)
        
        # Generate future dates
        last_date = features_df.index[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=30,
            freq='D'
        )
        
        # Create features for future dates
        future_df = pd.DataFrame(index=future_dates)
        future_df['sales'] = np.nan  # Placeholder for rolling calculations
        future_df = self._prepare_timeseries_features(future_df)
        
        # For rolling statistics in future data, use the last known values
        if 'rolling_mean_7d' in feature_cols:
            future_df['rolling_mean_7d'] = features_df['rolling_mean_7d'].iloc[-1]
            future_df['rolling_std_7d'] = features_df['rolling_std_7d'].iloc[-1]
        if 'rolling_mean_30d' in feature_cols:
            future_df['rolling_mean_30d'] = features_df['rolling_mean_30d'].iloc[-1]
            future_df['rolling_std_30d'] = features_df['rolling_std_30d'].iloc[-1]
        
        # Make predictions
        predictions = model.predict(future_df[feature_cols])
        
        # Calculate confidence score based on model performance
        confidence_score = min(0.95, model.score(X, y))
        
        return {
            'dates': future_dates.tolist(),
            'predictions': predictions.tolist(),
            'confidence_score': confidence_score,
            'feature_importance': dict(zip(feature_cols, model.feature_importances_))
        }

    def analyze_store_performance(self, store_id: str) -> Dict:
        """Analyze store performance with ML insights"""
        # Get store details
        store = self.db.stores.find_one({'store_id': store_id})
        if not store:
            raise ValueError(f"Store with ID {store_id} not found")
            
        # Get all transactions for this store
        store_transactions = list(self.db.transactions.find({'store_id': store_id}))
        if not store_transactions:
            raise ValueError(f"No transactions found for store {store_id}")
            
        # Process transactions
        txn_df = pd.DataFrame([self._flatten_dict(txn) for txn in store_transactions])
        txn_df['date'] = pd.to_datetime(txn_df['date'])
        
        # Generate insights
        analysis = {
            'basic_metrics': self._calculate_basic_metrics(txn_df),
            'sales_prediction': self._predict_future_sales(txn_df),
            'customer_insights': self._analyze_customers(txn_df),
            'product_insights': self._analyze_products(txn_df)
        }
        
        return analysis

    def _calculate_basic_metrics(self, txn_df: pd.DataFrame) -> Dict:
        """Calculate basic performance metrics"""
        current_month = txn_df['date'].max().replace(day=1)
        last_month = current_month - pd.DateOffset(months=1)
        
        current_month_sales = txn_df[txn_df['date'] >= current_month]['payment_final_amount'].sum()
        last_month_sales = txn_df[(txn_df['date'] >= last_month) & 
                                 (txn_df['date'] < current_month)]['payment_final_amount'].sum()
        
        return {
            'total_sales': txn_df['payment_final_amount'].sum(),
            'total_transactions': len(txn_df),
            'avg_transaction_value': txn_df['payment_final_amount'].mean(),
            'current_month_sales': current_month_sales,
            'last_month_sales': last_month_sales,
            'mom_growth': ((current_month_sales - last_month_sales) / last_month_sales * 100) 
                          if last_month_sales > 0 else 0
        }

    def _analyze_customers(self, txn_df: pd.DataFrame) -> Dict:
        """Analyze customer behavior"""
        customer_metrics = txn_df.groupby('customer_id').agg({
            'payment_final_amount': ['count', 'mean', 'sum'],
            'date': [lambda x: (x.max() - x.min()).days]
        }).reset_index()
        
        customer_metrics.columns = ['customer_id', 'visit_count', 'avg_spend', 
                                  'total_spend', 'days_active']
        
        return {
            'total_customers': len(customer_metrics),
            'avg_customer_spend': customer_metrics['avg_spend'].mean(),
            'loyal_customers': len(customer_metrics[customer_metrics['visit_count'] > 5])
        }

    def _analyze_products(self, txn_df: pd.DataFrame) -> Dict:
        """Analyze product performance"""
        # Extract items from transactions
        all_items = []
        for _, row in txn_df.iterrows():
            for item in row['items']:
                item_dict = self._flatten_dict(item)
                item_dict['transaction_date'] = row['date']
                all_items.append(item_dict)
        
        items_df = pd.DataFrame(all_items)
        
        return {
            'total_items_sold': items_df['quantity'].sum(),
            'avg_item_price': (items_df['total'] / items_df['quantity']).mean(),
            'top_products': items_df.groupby('product_id')['quantity'].sum().nlargest(5).to_dict()
        }

def main():
    try:
        analyzer = MLStoreAnalyzer()
        
        # Get example store_id
        store = analyzer.db.stores.find_one()
        if not store:
            print("No stores found in database")
            return
            
        store_id = store['store_id']
        print(f"\nAnalyzing store: {store_id}")
        
        # Perform analysis
        analysis = analyzer.analyze_store_performance(store_id)
        
        # Print results
        print("\nBasic Metrics:")
        metrics = analysis['basic_metrics']
        print(f"Total Sales: ₹{metrics['total_sales']:,.2f}")
        print(f"Average Transaction: ₹{metrics['avg_transaction_value']:,.2f}")
        print(f"Month-over-Month Growth: {metrics['mom_growth']:.1f}%")
        
        print("\nSales Predictions:")
        predictions = analysis['sales_prediction']
        avg_predicted = np.mean(predictions['predictions'])
        print(f"Average Predicted Daily Sales: ₹{avg_predicted:,.2f}")
        print(f"Prediction Confidence: {predictions['confidence_score']*100:.1f}%")
        
        print("\nCustomer Insights:")
        customers = analysis['customer_insights']
        print(f"Total Customers: {customers['total_customers']}")
        print(f"Loyal Customers: {customers['loyal_customers']}")
        print(f"Average Customer Spend: ₹{customers['avg_customer_spend']:,.2f}")
        
        print("\nProduct Insights:")
        products = analysis['product_insights']
        print(f"Total Items Sold: {products['total_items_sold']}")
        print(f"Average Item Price: ₹{products['avg_item_price']:,.2f}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()