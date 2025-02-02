import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timedelta
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

class ProductRevenueModelTrainer:
    def __init__(self, mongodb_uri='mongodb://localhost:27017/'):
        """
        Initialize the model trainer with MongoDB connection
        """
        self.client = MongoClient(mongodb_uri)
        self.db = self.client['indian_retail']
        
        # Preprocessing objects to be saved
        self.preprocessing = {
            'label_encoders': {},
            'scalers': {}
        }
    
    def _extract_historical_features(self, historical_period: int = 365):
        """
        Extract comprehensive features from historical transaction data
        
        Parameters:
        - historical_period: Number of days to analyze historical data
        """
        # Get date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=historical_period)
        
        # Aggregate transaction data
        pipeline = [
            {
                '$match': {
                    'date': {'$gte': start_date, '$lte': end_date}
                }
            },
            {
                '$unwind': '$items'
            },
            {
                '$group': {
                    '_id': '$items.product_id',
                    'total_quantity': {'$sum': '$items.quantity'},
                    'total_revenue': {'$sum': {'$multiply': ['$items.quantity', '$items.final_price']}},
                    'transaction_count': {'$sum': 1},
                    'unique_stores': {'$addToSet': '$store_id'},
                    'avg_transaction_value': {'$avg': {'$multiply': ['$items.quantity', '$items.final_price']}},
                    'month_dist': {'$addToSet': {'$month': '$date'}}
                }
            },
            {
                '$lookup': {
                    'from': 'products',
                    'localField': '_id',
                    'foreignField': 'product_id',
                    'as': 'product_details'
                }
            },
            {
                '$unwind': {'path': '$product_details', 'preserveNullAndEmptyArrays': True}
            }
        ]
        
        # Execute aggregation
        results = list(self.db.transactions.aggregate(pipeline))
        
        # Prepare features
        features_data = []
        for result in results:
            try:
                # Safely extract nested values
                product_details = result.get('product_details', {})
                pricing = product_details.get('pricing', {})
                inventory = product_details.get('inventory', {})
                
                features = {
                    'product_id': result['_id'],
                    'category': product_details.get('category', 'Unknown'),
                    'subcategory': product_details.get('subcategory', 'Unknown'),
                    
                    # Pricing features
                    'base_price': pricing.get('base_price', 0),
                    'current_stock': inventory.get('total_stock', 0),
                    
                    # Transaction-based features
                    'total_quantity': result.get('total_quantity', 0),
                    'total_revenue': result.get('total_revenue', 0),
                    'transaction_count': result.get('transaction_count', 0),
                    'unique_store_count': len(result.get('unique_stores', [])),
                    'avg_transaction_value': result.get('avg_transaction_value', 0),
                    
                    # Seasonal features
                    'seasonal_diversity': len(result.get('month_dist', [])),
                    
                    # Derived features
                    'avg_price': (result.get('total_revenue', 0) / 
                                  max(result.get('total_quantity', 1), 1))
                }
                features_data.append(features)
            except Exception as e:
                print(f"Error processing product {result.get('_id', 'Unknown')}: {e}")
        
        # Convert to DataFrame
        df = pd.DataFrame(features_data)
        
        # Diagnostic information
        print("\nData Extraction Diagnostic:")
        print(f"Total products processed: {len(df)}")
        print("Columns extracted:", list(df.columns))
        
        return df
    
    def _preprocess_features(self, df):
        """
        Preprocess features for model training
        """
        # Categorical encoding
        categorical_cols = ['category', 'subcategory']
        for col in categorical_cols:
            # Fill missing values
            df[col] = df[col].fillna('Unknown')
            
            # Label Encoding
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.preprocessing['label_encoders'][col] = le
        
        # Numeric columns to scale
        numeric_cols = [
            'base_price', 'current_stock', 'total_quantity', 
            'total_revenue', 'transaction_count', 
            'unique_store_count', 'avg_transaction_value', 
            'seasonal_diversity', 'avg_price'
        ]
        
        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        # Scale numeric features
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        self.preprocessing['scalers']['numeric'] = scaler
        
        return df
    
    def train_model(self, target_column='total_revenue', test_size=0.2):
        """
        Train LightGBM model for product revenue prediction
        
        Parameters:
        - target_column: Column to predict (default: total_revenue)
        - test_size: Proportion of data to use for testing
        """
        # Extract features
        df = self._extract_historical_features()
        
        # Preprocess features
        df = self._preprocess_features(df)
        
        # Prepare feature matrix and target
        features = [
            'category', 'subcategory', 'base_price', 'current_stock', 
            'total_quantity', 'avg_price', 'transaction_count', 
            'unique_store_count', 'seasonal_diversity', 
            'avg_transaction_value'
        ]
        
        X = df[features]
        y = df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
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
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[test_data],
            early_stopping_rounds=10
        )
        
        # Evaluate model
        y_pred = model.predict(X_test)
        
        print("\nModel Performance Metrics:")
        print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
        print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
        print(f"R-squared Score: {r2_score(y_test, y_pred)}")
        
        # Save model and preprocessing objects
        model.save_model('product_revenue_model.lgb')
        joblib.dump(self.preprocessing, 'product_revenue_preprocessing.joblib')
        
        return model
    
    def predict_sample(self, model, sample_features):
        """
        Make a sample prediction to validate the model
        
        Parameters:
        - model: Trained LightGBM model
        - sample_features: Features for prediction
        """
        # Preprocess sample features
        sample_df = pd.DataFrame([sample_features])
        sample_df = self._preprocess_features(sample_df)
        
        features = [
            'category', 'subcategory', 'base_price', 'current_stock', 
            'total_quantity', 'avg_price', 'transaction_count', 
            'unique_store_count', 'seasonal_diversity', 
            'avg_transaction_value'
        ]
        
        # Predict
        prediction = model.predict(sample_df[features])
        return prediction[0]

# Usage example
if __name__ == "__main__":
    # Initialize and train the model
    trainer = ProductRevenueModelTrainer()
    trained_model = trainer.train_model()
    
    # Example of making a sample prediction
    sample_features = {
        'category': 'Electronics',
        'subcategory': 'Mobile Accessories',
        'base_price': 1000,
        'current_stock': 50,
        'total_quantity': 100,
        'avg_price': 1200,
        'transaction_count': 20,
        'unique_store_count': 5,
        'seasonal_diversity': 3,
        'avg_transaction_value': 1500
    }
    
    prediction = trainer.predict_sample(trained_model, sample_features)
    print(f"Sample Revenue Prediction: {prediction}")