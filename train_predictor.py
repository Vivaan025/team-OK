import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timedelta
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import Settings
# model_trainer.py


class ProductRevenueModelTrainer:
    @staticmethod
    def train_model(
        mongodb_uri="mongodb://localhost:27017/", 
        model_save_path="retail_optimized_model_1.txt",
        preprocessing_save_path="retail_optimized_model_preprocessing_1.joblib"
    ):
        """
        Train the product revenue prediction model
        """
        # MongoDB Connection
        client = MongoClient(mongodb_uri)
        db = client[Settings.DATABASE_NAME]
        
        # Extract training data
        pipeline = [
            {
                '$unwind': '$items'
            },
            {
                '$lookup': {
                    'from': 'products',
                    'localField': 'items.product_id',
                    'foreignField': 'product_id',
                    'as': 'product_details'
                }
            },
            {
                '$unwind': '$product_details'
            },
            {
                '$group': {
                    '_id': '$items.product_id',
                    'total_revenue': {'$sum': '$items.total'},
                    'total_quantity': {'$sum': '$items.quantity'},
                    'transaction_count': {'$sum': 1},
                    'unique_stores': {'$addToSet': '$store_id'},
                    'product_details': {'$first': '$product_details'}
                }
            }
        ]
        
        # Execute aggregation
        results = list(db.transactions.aggregate(pipeline))
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Feature engineering
        df['category'] = df['product_details'].apply(lambda x: x['category'])
        df['subcategory'] = df['product_details'].apply(lambda x: x.get('subcategory', 'Unknown'))
        df['base_price'] = df['product_details'].apply(lambda x: x['pricing']['base_price'])
        df['current_stock'] = df['product_details'].apply(lambda x: x['inventory']['total_stock'])
        
        # Preprocessing
        preprocessing = {
            'label_encoders': {}
        }
        
        # Encode categorical features
        categorical_cols = ['category', 'subcategory']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            preprocessing['label_encoders'][col] = le
        
        # Select features
        features = [
            'category', 'subcategory', 'base_price', 'current_stock', 
            'total_quantity', 'unique_stores', 'transaction_count'
        ]
        
        X = df[features]
        y = df['total_revenue']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Prepare LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test)
        
        # Model parameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[test_data]
        )
        
        # Save model and preprocessing
        model.save_model(model_save_path)
        joblib.dump(preprocessing, preprocessing_save_path)
        
        # Print evaluation metrics
        y_pred = model.predict(X_test)
        print("Model Performance:")
        print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
        print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
        print(f"R-squared: {r2_score(y_test, y_pred)}")
        
        return model
    
if __name__ == "main": 
    model = ProductRevenueModelTrainer.train_model()