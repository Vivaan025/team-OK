import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RetailPredictor:
    def __init__(self):
        # Load trained models and preprocessors
        print("Loading trained models...")
        try:
            with open('outlet_preference_model.pkl', 'rb') as f:
                self.outlet_preference_model = pickle.load(f)
            
            with open('sales_prediction_model.pkl', 'rb') as f:
                self.sales_prediction_model = pickle.load(f)
            
            with open('customer_segment_model.pkl', 'rb') as f:
                self.customer_segment_model = pickle.load(f)
            
            with open('label_encoders.pkl', 'rb') as f:
                self.label_encoders = pickle.load(f)
            
            with open('scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
        except FileNotFoundError as e:
            print(f"Error loading models: {e}")
            raise
        
        self.customer_features = ['age', 'income', 'family_size']
        self.outlet_features = ['commercial_viability', 'competition_score', 'luxury_index']
        self.location_features = ['commercial_score', 'public_transport_accessibility', 'population_density']
    
    def validate_data(self, customer_data, outlet_data, location_data):
        """Validate input data consistency"""
        
        # Check required columns
        for df, name, required_cols in [
            (customer_data, 'Customer data', self.customer_features),
            (outlet_data, 'Outlet data', ['location_id'] + self.outlet_features),
            (location_data, 'Location data', ['location_id'] + self.location_features)
        ]:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"{name} is missing required columns: {missing_cols}")
        
        # Validate location references
        invalid_locations = outlet_data[~outlet_data['location_id'].isin(location_data['location_id'])]
        if not invalid_locations.empty:
            print(f"Warning: Found {len(invalid_locations)} outlets with invalid location IDs")
            print("Removing invalid entries...")
            outlet_data = outlet_data[outlet_data['location_id'].isin(location_data['location_id'])]
        
        if outlet_data.empty:
            raise ValueError("No valid outlets remain after validation")
        
        return customer_data, outlet_data, location_data

    def preprocess_input(self, customer_data, outlet_data=None, location_data=None):
        """Preprocess input data using trained encoders and scaler"""
        
        print("Preprocessing input data...")
        
        # Create copies to avoid modifying original data
        customer_data = customer_data.copy()
        if outlet_data is not None:
            outlet_data = outlet_data.copy()
        if location_data is not None:
            location_data = location_data.copy()
        
        try:
            # Validate data consistency
            customer_data, outlet_data, location_data = self.validate_data(
                customer_data, outlet_data, location_data
            )
            
            # Encode categorical variables in customer data
            if 'gender' in customer_data.columns:
                customer_data['gender'] = self.label_encoders['customers_gender'].transform(customer_data['gender'])
            if 'occupation' in customer_data.columns:
                customer_data['occupation'] = self.label_encoders['customers_occupation'].transform(customer_data['occupation'])
            if 'marital_status' in customer_data.columns:
                customer_data['marital_status'] = self.label_encoders['customers_marital_status'].transform(customer_data['marital_status'])
            if 'residential_area' in customer_data.columns:
                customer_data['residential_area'] = self.label_encoders['customers_residential_area'].transform(customer_data['residential_area'])
            
            # Encode outlet data if provided
            if outlet_data is not None:
                if 'category' in outlet_data.columns:
                    outlet_data['category'] = self.label_encoders['outlets_category'].transform(outlet_data['category'])
                if 'target_demographic' in outlet_data.columns:
                    outlet_data['target_demographic'] = self.label_encoders['outlets_target_demographic'].transform(outlet_data['target_demographic'])
                if 'outlet_size' in outlet_data.columns:
                    outlet_data['outlet_size'] = self.label_encoders['outlets_outlet_size'].transform(outlet_data['outlet_size'])
            
            print("Preprocessing completed successfully")
            return customer_data, outlet_data, location_data
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            raise

    def predict_outlet_preference(self, customer_data, outlet_data, location_data):
        """Predict preferred outlets for customers"""
        
        print("Preparing features for outlet preference prediction...")
        features = []
        feature_maps = []  # To keep track of customer-outlet mappings
        
        # Join outlet data with location data
        outlet_data = outlet_data.merge(
            location_data[['location_id'] + self.location_features],
            on='location_id',
            how='inner'
        )
        
        if outlet_data.empty:
            raise ValueError("No valid outlet-location combinations found")
        
        for _, customer in customer_data.iterrows():
            for _, outlet in outlet_data.iterrows():
                # Current hour and day features
                current_time = datetime.now()
                hour = current_time.hour
                day_of_week = current_time.weekday()
                month = current_time.month
                weekend_flag = 1 if day_of_week >= 5 else 0
                
                # Combine all features
                row_features = (
                    [customer[feat] for feat in self.customer_features] +
                    [outlet[feat] for feat in self.outlet_features] +
                    [outlet[feat] for feat in self.location_features] +
                    [hour, day_of_week, month, weekend_flag]
                )
                features.append(row_features)
                feature_maps.append({
                    'customer_id': customer['customer_id'],
                    'outlet_id': outlet['outlet_id'],
                    'outlet_name': outlet['outlet_name'] if 'outlet_name' in outlet else outlet['outlet_id']
                })
        
        if not features:
            raise ValueError("No features generated for prediction")
        
        # Scale features
        X = self.scaler.transform(features)
        
        # Get predictions and probabilities
        print("Making predictions...")
        preferences = self.outlet_preference_model.predict(X)
        probabilities = self.outlet_preference_model.predict_proba(X)
        
        # Prepare results
        results = []
        for i, mapping in enumerate(feature_maps):
            results.append({
                **mapping,
                'preference_score': probabilities[i].max(),
                'is_preferred': bool(preferences[i])
            })
        
        return pd.DataFrame(results)

    def predict_sales(self, customer_data, outlet_data, location_data):
        """Predict sales for given customer-outlet combinations"""
        
        print("Preparing features for sales prediction...")
        
        # Join outlet data with location data first
        outlet_data = outlet_data.merge(
            location_data[['location_id'] + self.location_features],
            on='location_id',
            how='inner'
        )
        
        features = []
        feature_maps = []
        
        for _, customer in customer_data.iterrows():
            for _, outlet in outlet_data.iterrows():
                current_time = datetime.now()
                hour = current_time.hour
                day_of_week = current_time.weekday()
                month = current_time.month
                weekend_flag = 1 if day_of_week >= 5 else 0
                
                row_features = (
                    [customer[feat] for feat in self.customer_features] +
                    [outlet[feat] for feat in self.outlet_features] +
                    [outlet[feat] for feat in self.location_features] +
                    [hour, day_of_week, month, weekend_flag]
                )
                features.append(row_features)
                feature_maps.append({
                    'customer_id': customer['customer_id'],
                    'outlet_id': outlet['outlet_id'],
                    'outlet_name': outlet['outlet_name'] if 'outlet_name' in outlet else outlet['outlet_id']
                })
        
        X = self.scaler.transform(features)
        
        print("Predicting sales...")
        predicted_sales = self.sales_prediction_model.predict(X)
        
        results = []
        for i, mapping in enumerate(feature_maps):
            results.append({
                **mapping,
                'predicted_sales': predicted_sales[i]
            })
        
        return pd.DataFrame(results)

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = RetailPredictor()
    
    try:
        # Load test data
        print("Loading test data...")
        test_customers = pd.read_csv('test_customers.csv')
        test_outlets = pd.read_csv('test_outlets.csv')
        test_locations = pd.read_csv('mumbai_locations.csv')  # Use original Mumbai locations
        
        # Preprocess test data
        test_customers, test_outlets, test_locations = predictor.preprocess_input(
            test_customers, test_outlets, test_locations
        )
        
        # Make predictions
        print("\nPredicting outlet preferences...")
        preferences = predictor.predict_outlet_preference(
            test_customers, test_outlets, test_locations
        )
        print("\nTop outlet preferences:")
        print(preferences.sort_values('preference_score', ascending=False).head())
        
        print("\nPredicting sales...")
        sales_predictions = predictor.predict_sales(
            test_customers, test_outlets, test_locations
        )
        print("\nSales predictions:")
        print(sales_predictions.head())
        
    except Exception as e:
        print(f"\nError in prediction pipeline: {str(e)}")
        raise