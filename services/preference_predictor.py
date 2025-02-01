import pandas as pd
import numpy as np
from pymongo import MongoClient
import lightgbm as lgb
import joblib
from collections import defaultdict
from datetime import datetime
import random
from config import settings

class RetailPreferencePredictor:
    def __init__(self, 
                 model_path=settings.model_path,
                 preprocessing_path=settings.preprocessing_path,
                 mongodb_uri=settings.mongodb_uri):
        # Load model and preprocessing objects
        self.model = lgb.Booster(model_file=model_path)
        self.preprocessing = joblib.load(preprocessing_path)
        
        # Initialize MongoDB connection
        self.client = MongoClient(mongodb_uri)
        self.db = self.client['indian_retail']
        
        # Cache store and product data
        self.store_categories = {s['store_id']: s['category'] 
                               for s in self.db.stores.find()}
        self.product_categories = {p['product_id']: p['category'] 
                                 for p in self.db.products.find()}
        
        # Print expected features
        print("\nExpected features from model:")
        print(self.model.feature_name())

    def _extract_customer_features(self, customer_info, recent_transactions):
        """Extract features from customer info and transactions"""
        features = {
            'age': customer_info['age'],
            'gender': customer_info['gender'],
            'city': customer_info['city'],
            'state': customer_info['state'],
            'membership_tier': customer_info['membership_tier']
        }
        
        if not recent_transactions:
            # Set default values if no transactions
            features.update({
                'recent_avg_amount': 0,
                'recent_max_amount': 0,
                'recent_min_amount': 0,
                'recent_std_amount': 0,
                'morning_ratio': 0,
                'afternoon_ratio': 0,
                'evening_ratio': 0,
                'night_ratio': 0,
                'festival_shopping_ratio': 0
            })
            
            # Set default store ratios
            store_types = ['express_store', 'supermarket', 'hypermarket', 
                          'premium_store', 'wholesale_club']
            for store_type in store_types:
                features[f'recent_{store_type}_ratio'] = 0
            
            # Set default category ratios
            categories = ['groceries', 'electronics', 'fashion', 
                         'household', 'personal_care']
            for category in categories:
                features[f'recent_{category}_spend_ratio'] = 0
                features[f'recent_{category}_quantity_ratio'] = 0
            
            # Set default payment ratios
            payment_types = ['upi', 'credit_card', 'debit_card', 
                           'cash', 'net_banking']
            for payment in payment_types:
                features[f'payment_{payment}_ratio'] = 0
                
            return features
        
        # Process transactions
        amounts = [t['payment']['final_amount'] for t in recent_transactions]
        features.update({
            'recent_avg_amount': np.mean(amounts),
            'recent_max_amount': max(amounts),
            'recent_min_amount': min(amounts),
            'recent_std_amount': np.std(amounts) if len(amounts) > 1 else 0
        })
        
        # Time-based features
        times = [pd.to_datetime(t['date']).hour for t in recent_transactions]
        total_txns = len(times)
        features.update({
            'morning_ratio': len([t for t in times if 6 <= t < 12]) / total_txns,
            'afternoon_ratio': len([t for t in times if 12 <= t < 17]) / total_txns,
            'evening_ratio': len([t for t in times if 17 <= t < 22]) / total_txns,
            'night_ratio': len([t for t in times if t >= 22 or t < 6]) / total_txns
        })
        
        # Store categories
        store_visits = [self.store_categories[t['store_id']] for t in recent_transactions]
        store_counts = pd.Series(store_visits).value_counts()
        
        for store_type in set(self.store_categories.values()):
            ratio = store_counts.get(store_type, 0) / total_txns
            features[f'recent_{store_type.lower().replace(" ", "_")}_ratio'] = ratio
        
        # Product categories
        category_amounts = defaultdict(float)
        category_quantities = defaultdict(int)
        total_amount = 0
        total_items = 0
        
        for txn in recent_transactions:
            for item in txn['items']:
                category = self.product_categories[item['product_id']]
                amount = item['quantity'] * item['final_price']
                category_amounts[category] += amount
                category_quantities[category] += item['quantity']
                total_amount += amount
                total_items += item['quantity']
        
        for category in set(self.product_categories.values()):
            cat_key = category.lower().replace(" ", "_")
            if total_amount > 0:
                features[f'recent_{cat_key}_spend_ratio'] = (
                    category_amounts[category] / total_amount
                )
                features[f'recent_{cat_key}_quantity_ratio'] = (
                    category_quantities[category] / total_items if total_items > 0 else 0
                )
            else:
                features[f'recent_{cat_key}_spend_ratio'] = 0
                features[f'recent_{cat_key}_quantity_ratio'] = 0
        
        # Payment methods
        payment_methods = [t['payment']['method'] for t in recent_transactions]
        payment_counts = pd.Series(payment_methods).value_counts()
        
        for method in ['UPI', 'Credit Card', 'Debit Card', 'Cash', 'Net Banking']:
            method_key = method.lower().replace(" ", "_")
            features[f'payment_{method_key}_ratio'] = (
                payment_counts.get(method, 0) / total_txns
            )
        
        # Festival shopping
        festival_txns = [t for t in recent_transactions 
                        if t['festivals'] and len(t['festivals']) > 0]
        features['festival_shopping_ratio'] = len(festival_txns) / total_txns
        
        return features

    def predict(self, customer_info, recent_transactions=None):
        """
        Predict store preference for a customer
        """
        # Extract features
        features = self._extract_customer_features(customer_info, recent_transactions)
        
        # Create DataFrame with single row
        df = pd.DataFrame([features])
        
        # Get expected feature names from model
        expected_features = self.model.feature_name()
        
        # Check for missing features and add them with default values
        missing_features = set(expected_features) - set(df.columns)
        if missing_features:
            print(f"Adding missing features: {missing_features}")
            for feature in missing_features:
                df[feature] = 0.0
        
        # Ensure correct feature order
        df = df[expected_features]
        
        # Encode categorical variables
        categorical_cols = ['gender', 'city', 'state', 'membership_tier']
        for col in categorical_cols:
            if col in df.columns and col in self.preprocessing['label_encoders']:
                encoder = self.preprocessing['label_encoders'][col]
                try:
                    df[col] = encoder.transform(df[col])
                except ValueError as e:
                    print(f"Warning: Unknown category in {col}: {df[col].iloc[0]}")
                    # Set to most common category
                    df[col] = encoder.transform([encoder.classes_[0]])
        
        # Make prediction
        probabilities = self.model.predict(df)
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[0][predicted_class]
        
        # Convert predicted class back to store category
        store_encoder = self.preprocessing['label_encoders']['preferred_store']
        predicted_category = store_encoder.inverse_transform([predicted_class])[0]
        
        # Find best matching store within the predicted category
        best_store = self._find_best_store(customer_info, predicted_category)
        
        return predicted_category, best_store, confidence
        
    def _find_best_store(self, customer_info, predicted_category):
        """Find the best matching store within the predicted category"""
        # Get all stores in the predicted category
        matching_stores = [store for store in self.db.stores.find({
            'category': predicted_category
        })]
        
        if not matching_stores:
            return None
            
        # Score each store based on various factors
        store_scores = []
        customer_city = customer_info['city']
        customer_state = customer_info['state']
        
        for store in matching_stores:
            score = 0
            
            # Location matching
            if store['location']['city'] == customer_city:
                score += 10  # High priority for same city
            elif store['location']['state'] == customer_state:
                score += 5  # Medium priority for same state
                
            # Store ratings (if available)
            if 'ratings' in store:
                score += store['ratings'].get('overall', 0) * 2
                
            # Store amenities (if available)
            if 'amenities' in store:
                score += len(store['amenities']) * 0.5
                
            store_scores.append((store, score))
        
        # Get the store with highest score
        best_store = max(store_scores, key=lambda x: x[1])[0]
        
        # Return store details
        return {
            'name': best_store['name'],
            'address': best_store['location']['address'],
            'city': best_store['location']['city'],
            'pincode': best_store['location']['pincode'],
            'ratings': best_store.get('ratings', {}),
            'amenities': best_store.get('amenities', [])
        }
