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
from config import settings
warnings.filterwarnings('ignore')

class MLStoreAnalyzer:
    def __init__(self, mongodb_uri=settings.mongodb_uri):
        self.client = MongoClient(mongodb_uri)
        self.db = self.client['indian_retail']
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
        df = data.copy()
        
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        
        if 'sales' in df.columns:
            windows = [7, 30]
            for window in windows:
                df[f'rolling_mean_{window}d'] = df['sales'].rolling(window=window, min_periods=1).mean()
                df[f'rolling_std_{window}d'] = df['sales'].rolling(window=window, min_periods=1).std()
        
        return df

    def _predict_future_sales(self, txn_df: pd.DataFrame) -> Dict:
        sales_df = txn_df.groupby('date')['payment_final_amount'].sum().reset_index()
        sales_df = sales_df.set_index('date')
        sales_df = sales_df.rename(columns={'payment_final_amount': 'sales'})
        
        features_df = self._prepare_timeseries_features(sales_df)
        
        feature_cols = ['year', 'month', 'day', 'dayofweek', 'quarter', 
                       'is_weekend', 'is_month_start', 'is_month_end']
        
        if len(features_df) >= 7:
            feature_cols.extend(['rolling_mean_7d', 'rolling_std_7d'])
        if len(features_df) >= 30:
            feature_cols.extend(['rolling_mean_30d', 'rolling_std_30d'])
        
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
        
        last_date = features_df.index[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=30,
            freq='D'
        )
        
        future_df = pd.DataFrame(index=future_dates)
        future_df['sales'] = np.nan
        future_df = self._prepare_timeseries_features(future_df)
        
        if 'rolling_mean_7d' in feature_cols:
            future_df['rolling_mean_7d'] = features_df['rolling_mean_7d'].iloc[-1]
            future_df['rolling_std_7d'] = features_df['rolling_std_7d'].iloc[-1]
        if 'rolling_mean_30d' in feature_cols:
            future_df['rolling_mean_30d'] = features_df['rolling_mean_30d'].iloc[-1]
            future_df['rolling_std_30d'] = features_df['rolling_std_30d'].iloc[-1]
        
        predictions = model.predict(future_df[feature_cols])
        confidence_score = min(0.95, model.score(X, y))
        
        return {
            'dates': future_dates.tolist(),
            'predictions': predictions.tolist(),
            'confidence_score': confidence_score,
            'feature_importance': dict(zip(feature_cols, model.feature_importances_))
        }

    def analyze_store_performance(self, store_id: str) -> Dict:
        store = self.db.stores.find_one({'store_id': store_id})
        if not store:
            raise ValueError(f"Store with ID {store_id} not found")
            
        store_transactions = list(self.db.transactions.find({'store_id': store_id}))
        if not store_transactions:
            raise ValueError(f"No transactions found for store {store_id}")
            
        txn_df = pd.DataFrame([self._flatten_dict(txn) for txn in store_transactions])
        txn_df['date'] = pd.to_datetime(txn_df['date'])
        
        return {
            'basic_metrics': self._calculate_basic_metrics(txn_df),
            'sales_prediction': self._predict_future_sales(txn_df),
            'customer_insights': self._analyze_customers(txn_df),
            'product_insights': self._analyze_products(txn_df)
        }

    def _calculate_basic_metrics(self, txn_df: pd.DataFrame) -> Dict:
        current_month = txn_df['date'].max().replace(day=1)
        last_month = current_month - pd.DateOffset(months=1)
        
        current_month_sales = txn_df[txn_df['date'] >= current_month]['payment_final_amount'].sum()
        last_month_sales = txn_df[(txn_df['date'] >= last_month) & 
                                 (txn_df['date'] < current_month)]['payment_final_amount'].sum()
        
        return {
            'total_sales': float(txn_df['payment_final_amount'].sum()),
            'total_transactions': int(len(txn_df)),
            'avg_transaction_value': float(txn_df['payment_final_amount'].mean()),
            'current_month_sales': float(current_month_sales),
            'last_month_sales': float(last_month_sales),
            'mom_growth': float(((current_month_sales - last_month_sales) / last_month_sales * 100)
                              if last_month_sales > 0 else 0)
        }

    def _analyze_customers(self, txn_df: pd.DataFrame) -> Dict:
        customer_metrics = txn_df.groupby('customer_id').agg({
            'payment_final_amount': ['count', 'mean', 'sum'],
            'date': [lambda x: (x.max() - x.min()).days]
        }).reset_index()
        
        customer_metrics.columns = ['customer_id', 'visit_count', 'avg_spend', 
                                  'total_spend', 'days_active']
        
        return {
            'total_customers': int(len(customer_metrics)),
            'avg_customer_spend': float(customer_metrics['avg_spend'].mean()),
            'loyal_customers': int(len(customer_metrics[customer_metrics['visit_count'] > 5]))
        }

    def _analyze_products(self, txn_df: pd.DataFrame) -> Dict:
        all_items = []
        for _, row in txn_df.iterrows():
            for item in row['items']:
                item_dict = self._flatten_dict(item)
                item_dict['transaction_date'] = row['date']
                all_items.append(item_dict)
        
        items_df = pd.DataFrame(all_items)
        
        return {
            'total_items_sold': int(items_df['quantity'].sum()),
            'avg_item_price': float((items_df['total'] / items_df['quantity']).mean()),
            'top_products': {str(k): int(v) for k, v 
                           in items_df.groupby('product_id')['quantity'].sum().nlargest(5).items()}
        }
