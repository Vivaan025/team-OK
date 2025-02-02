import pandas as pd
import numpy as np
from pymongo import MongoClient
import lightgbm as lgb
import joblib
from collections import defaultdict
from datetime import datetime, timedelta
import random
from config import settings
from typing import List, Dict, Any, Optional

class TransactionProcessor:
    def __init__(self, mongo_client):
        """
        Initialize transaction processor with MongoDB client
        
        Args:
            mongo_client: Initialized MongoDB client
        """
        self.db = mongo_client['indian_retail']

    def get_customer_transactions(
        self, 
        customer_id: str, 
        days_back: int = 90
    ) -> List[Dict[str, Any]]:
        """
        Retrieve and process transactions for a specific customer
        
        Args:
            customer_id: Unique identifier for the customer
            days_back: Number of days to look back for transactions
        
        Returns:
            List of processed transactions
        """
        # Set date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Retrieve transactions
        print(f"\n--- Fetching Transactions for Customer {customer_id} ---")
        
        transactions = list(self.db.transactions.find({
            'customer_id': customer_id,
            'date': {
                '$gte': start_date,
                '$lte': end_date
            }
        }))
        
        print(f"Raw Transactions Found: {len(transactions)}")
        
        # Process and validate transactions
        processed_transactions = []
        for txn in transactions:
            try:
                processed_txn = self._validate_transaction(txn)
                if processed_txn:
                    processed_transactions.append(processed_txn)
                    print(f"Processed Transaction: {processed_txn.get('transaction_id')}")
            except Exception as e:
                print(f"Error processing transaction {txn.get('transaction_id')}: {e}")
        
        print(f"Processed Transactions: {len(processed_transactions)}")
        
        # Detailed transaction logging
        for txn in processed_transactions:
            print("\nTransaction Details:")
            print(f"ID: {txn.get('transaction_id')}")
            print(f"Date: {txn.get('date')}")
            print(f"Store ID: {txn.get('store_id')}")
            print(f"Payment Method: {txn['payment'].get('method')}")
            print(f"Total Amount: {txn['payment'].get('final_amount')}")
            print("Items:")
            for item in txn.get('items', []):
                print(f"  - Product ID: {item.get('product_id')}")
                print(f"    Quantity: {item.get('quantity')}")
                print(f"    Final Price: {item.get('final_price')}")
            print(f"Festivals: {txn.get('festivals', [])}")
        
        return processed_transactions

    def _validate_transaction(self, transaction: Dict) -> Optional[Dict]:
        """
        Validate and standardize a single transaction
        
        Args:
            transaction: Raw transaction dictionary
        
        Returns:
            Processed transaction or None if invalid
        """
        # Validate basic transaction structure
        if not transaction or not isinstance(transaction, dict):
            print("Transaction validation failed: Invalid transaction structure")
            return None

        # Validate date
        try:
            date = pd.to_datetime(transaction.get('date'))
        except:
            print("Transaction validation failed: Invalid date")
            return None

        # Validate payment information
        payment = transaction.get('payment', {})
        if not isinstance(payment, dict):
            print("Transaction validation failed: Invalid payment structure")
            payment = {}

        # Standardize payment method
        payment_method = str(payment.get('method', 'Unknown')).lower()
        payment_methods_map = {
            'upi': 'UPI',
            'credit card': 'Credit Card', 
            'debit card': 'Debit Card',
            'cash': 'Cash',
            'net banking': 'Net Banking'
        }
        # Normalize payment method
        normalized_method = next(
            (v for k, v in payment_methods_map.items() if k in payment_method), 
            'Unknown'
        )

        # Process items
        processed_items = []
        total_amount = 0
        for item in transaction.get('items', []):
            try:
                # Validate item structure
                if not isinstance(item, dict):
                    print(f"Skipping invalid item: {item}")
                    continue

                # Extract and validate item details
                quantity = float(item.get('quantity', 0))
                unit_price = float(item.get('unit_price', 0))
                discount = float(item.get('discount', 0))
                
                # Calculate final price
                final_price = unit_price * (1 - discount)
                item_total = quantity * final_price

                processed_item = {
                    'product_id': item.get('product_id'),
                    'quantity': quantity,
                    'unit_price': unit_price,
                    'final_price': final_price,
                    'discount': discount
                }
                
                processed_items.append(processed_item)
                total_amount += item_total

            except Exception as e:
                print(f"Error processing item: {e}")

        # Check if transaction is valid
        if not processed_items:
            print("Transaction validation failed: No valid items")
            return None

        # Determine if transaction involves a festival
        festivals = transaction.get('festivals', [])
        if not isinstance(festivals, list):
            festivals = []

        # Construct processed transaction
        processed_transaction = {
            'date': date,
            'transaction_id': transaction.get('transaction_id'),
            'store_id': transaction.get('store_id'),
            'payment': {
                'method': normalized_method,
                'final_amount': total_amount
            },
            'items': processed_items,
            'festivals': festivals
        }

        return processed_transaction
    
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
        
        # Initialize Transaction Processor
        self.transaction_processor = TransactionProcessor(self.client)
        
        # Cache store and product data
        self.store_categories = {s['store_id']: s['category'] 
                               for s in self.db.stores.find()}
        self.product_categories = {p['product_id']: p['category'] 
                                 for p in self.db.products.find()}
        
        # Print expected features from model
        print("\nExpected features from model:")
        print(self.model.feature_name())

    def _extract_customer_features(self, customer_info, customer_id=None, recent_transactions=None):
        """
        Extract features from customer info and transactions
        
        Args:
            customer_info: Dictionary of customer basic information
            customer_id: Optional customer ID to fetch transactions
            recent_transactions: Optional pre-processed transactions
        """
        # If transactions not provided, try to fetch
        if recent_transactions is None and customer_id:
            recent_transactions = self.transaction_processor.get_customer_transactions(customer_id)
        
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
                'festival_shopping_ratio': 0,
                'primary_payment': 'Unknown'  # Add default primary payment
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
        times = [t['date'].hour for t in recent_transactions]
        total_txns = len(times)
        features.update({
            'morning_ratio': len([t for t in times if 6 <= t < 12]) / total_txns,
            'afternoon_ratio': len([t for t in times if 12 <= t < 17]) / total_txns,
            'evening_ratio': len([t for t in times if 17 <= t < 22]) / total_txns,
            'night_ratio': len([t for t in times if t >= 22 or t < 6]) / total_txns
        })
        
        # Store categories
        store_visits = [self.store_categories.get(t['store_id'], 'Unknown') for t in recent_transactions]
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
                category = self.product_categories.get(item['product_id'], 'Unknown')
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
        
        # Determine primary payment method
        if payment_methods:
            features['primary_payment'] = max(set(payment_methods), key=payment_methods.count)
        else:
            features['primary_payment'] = 'Unknown'
        
        for method in ['UPI', 'Credit Card', 'Debit Card', 'Cash', 'Net Banking']:
            method_key = method.lower().replace(" ", "_")
            features[f'payment_{method_key}_ratio'] = (
                payment_counts.get(method, 0) / total_txns
            )
        
        # Festival shopping
        festival_txns = [t for t in recent_transactions if t['festivals']]
        features['festival_shopping_ratio'] = len(festival_txns) / total_txns
        
        return features

    def predict(self, customer_info, customer_id=None, recent_transactions=None):
        """
        Predict store preference for a customer
        
        Args:
            customer_info: Dictionary of customer basic information
            customer_id: Optional customer ID to fetch transactions
            recent_transactions: Optional pre-processed transactions
        """
        # Extract features
        features = self._extract_customer_features(
            customer_info, 
            customer_id, 
            recent_transactions
        )
        
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
        categorical_cols = ['gender', 'city', 'state', 'membership_tier', 'primary_payment']
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
        """
        Find the best matching store within the predicted category with strict city matching
        and comprehensive scoring mechanism
        """
        # Get all stores in the predicted category in the customer's exact city
        matching_stores = [store for store in self.db.stores.find({
            'category': predicted_category,
            'location.city': customer_info['city']
        })]
        
        # If no stores in the exact city, expand to state-level matching
        if not matching_stores:
            matching_stores = [store for store in self.db.stores.find({
                'category': predicted_category,
                'location.state': customer_info['state']
            })]
            
            # If still no matching stores, return None
            if not matching_stores:
                return None
        
        # Comprehensive store scoring mechanism
        def calculate_store_score(store):
            score = 0
            
            # Location Scoring (Highest Priority)
            # Exact city match gets maximum points
            if store['location']['city'] == customer_info['city']:
                score += 50  # Highest priority for exact city match
            
            # Demographics Matching
            if 'customer_demographics' in store:
                # Match customer age group
                if 'preferred_age_groups' in store['customer_demographics']:
                    # Assume age groups like 'young_adults', 'middle_aged', 'seniors'
                    age = customer_info['age']
                    if (age < 30 and 'young_adults' in store['customer_demographics']['preferred_age_groups']) or \
                    (30 <= age < 50 and 'middle_aged' in store['customer_demographics']['preferred_age_groups']) or \
                    (age >= 50 and 'seniors' in store['customer_demographics']['preferred_age_groups']):
                        score += 15
                
                # Match customer gender preferences
                if 'preferred_gender' in store['customer_demographics']:
                    if customer_info['gender'] in store['customer_demographics']['preferred_gender']:
                        score += 10
            
            # Store Ratings and Reviews
            if 'ratings' in store:
                # Overall rating
                score += store['ratings'].get('overall', 0) * 5
                
                # Variety of rating sources
                rating_sources = store['ratings'].get('sources', [])
                score += len(rating_sources) * 2
            
            # Store Amenities
            if 'amenities' in store:
                amenities = store['amenities']
                # Bonus points for various amenities
                amenity_points = {
                    'parking': 5,
                    'free_wifi': 3,
                    'accessibility': 4,
                    'digital_payment': 3,
                    'personal_shopping_assistant': 5
                }
                
                for amenity in amenities:
                    score += amenity_points.get(amenity, 1)
            
            # Proximity and Accessibility Factors
            if 'location_details' in store:
                # Additional points for accessibility
                if store['location_details'].get('easily_accessible', False):
                    score += 5
                
                # Points for proximity to public transport
                if store['location_details'].get('near_public_transport', False):
                    score += 4
            
            # Membership Tier Alignment
            if 'membership_benefits' in store:
                customer_tier = customer_info['membership_tier']
                if customer_tier in store.get('membership_benefits', {}):
                    score += 7
            
            # Inventory Diversity
            if 'inventory_specialties' in store:
                # Check if store specialties align with customer's past purchase categories
                specialty_points = 3
                score += specialty_points
            
            return score
        
        # Sort stores by comprehensive score
        scored_stores = [
            (store, calculate_store_score(store)) 
            for store in matching_stores
        ]
        
        # Sort by score in descending order
        scored_stores.sort(key=lambda x: x[1], reverse=True)
        
        # Select the top-scoring store
        best_store = scored_stores[0][0]
        
        # Return store details
        return {
            'name': best_store['name'],
            'address': best_store['location']['address'],
            'city': best_store['location']['city'],
            'pincode': best_store['location']['pincode'],
            'ratings': best_store.get('ratings', {}),
            'amenities': best_store.get('amenities', []),
            'match_score': scored_stores[0][1]  # Include the calculated match score
        }