import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
        transactions = list(self.db.transactions.find({
            'customer_id': customer_id,
            'date': {
                '$gte': start_date,
                '$lte': end_date
            }
        }))

        # Process and validate transactions
        processed_transactions = []
        for txn in transactions:
            try:
                processed_txn = self._validate_transaction(txn)
                if processed_txn:
                    processed_transactions.append(processed_txn)
            except Exception as e:
                print(f"Error processing transaction {txn.get('transaction_id')}: {e}")
        
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
            return None

        # Validate date
        try:
            date = pd.to_datetime(transaction.get('date'))
        except:
            return None

        # Validate payment information
        payment = transaction.get('payment', {})
        if not isinstance(payment, dict):
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

    def extract_payment_features(self, transactions: List[Dict]) -> Dict[str, float]:
        """
        Extract payment-related features from transactions
        
        Args:
            transactions: List of processed transactions
        
        Returns:
            Dictionary of payment feature ratios
        """
        if not transactions:
            return {
                'total_transactions': 0,
                'primary_payment': 'Unknown',
                'payment_upi_ratio': 0,
                'payment_credit_card_ratio': 0,
                'payment_debit_card_ratio': 0,
                'payment_cash_ratio': 0,
                'payment_net_banking_ratio': 0,
                'recent_avg_amount': 0,
                'recent_max_amount': 0,
                'recent_min_amount': 0,
                'recent_std_amount': 0
            }

        # Calculate payment method ratios
        total_transactions = len(transactions)
        payment_methods = [txn['payment']['method'] for txn in transactions]
        
        # Compute method ratios
        method_counts = {}
        for method in set(payment_methods):
            method_key = f'payment_{method.lower().replace(" ", "_")}_ratio'
            method_counts[method_key] = payment_methods.count(method) / total_transactions

        # Determine primary payment method
        primary_payment = max(set(payment_methods), key=payment_methods.count)

        # Calculate transaction amounts
        amounts = [txn['payment']['final_amount'] for txn in transactions]
        
        return {
            'total_transactions': total_transactions,
            'primary_payment': primary_payment,
            **method_counts,
            'recent_avg_amount': np.mean(amounts),
            'recent_max_amount': max(amounts) if amounts else 0,
            'recent_min_amount': min(amounts) if amounts else 0,
            'recent_std_amount': np.std(amounts) if len(amounts) > 1 else 0
        }

    def extract_time_features(self, transactions: List[Dict]) -> Dict[str, float]:
        """
        Extract time-based features from transactions
        
        Args:
            transactions: List of processed transactions
        
        Returns:
            Dictionary of time-based feature ratios
        """
        if not transactions:
            return {
                'morning_ratio': 0,
                'afternoon_ratio': 0,
                'evening_ratio': 0,
                'night_ratio': 0,
                'festival_shopping_ratio': 0
            }

        # Extract transaction times
        times = [txn['date'].hour for txn in transactions]
        total_txns = len(times)

        # Time period ratios
        time_features = {
            'morning_ratio': len([t for t in times if 6 <= t < 12]) / total_txns,
            'afternoon_ratio': len([t for t in times if 12 <= t < 17]) / total_txns,
            'evening_ratio': len([t for t in times if 17 <= t < 22]) / total_txns,
            'night_ratio': len([t for t in times if t >= 22 or t < 6]) / total_txns
        }

        # Festival shopping ratio
        festival_txns = [txn for txn in transactions if txn.get('festivals')]
        time_features['festival_shopping_ratio'] = len(festival_txns) / total_txns

        return time_features

    def extract_store_features(self, transactions: List[Dict]) -> Dict[str, float]:
        """
        Extract store-related features from transactions
        
        Args:
            transactions: List of processed transactions
        
        Returns:
            Dictionary of store feature ratios
        """
        if not transactions:
            # Default values for store ratios
            store_types = ['premium_store', 'express_store', 'hypermarket', 
                           'supermarket', 'wholesale_club']
            return {f'recent_{store}_ratio': 0.0 for store in store_types}

        # Retrieve store details
        store_details = {}
        for txn in transactions:
            store = self.db.stores.find_one({'store_id': txn['store_id']})
            if store:
                store_details.setdefault(store['category'], 0)
                store_details[store['category']] += 1

        # Normalize ratios
        total_txns = len(transactions)
        store_features = {
            f'recent_{k.lower().replace(" ", "_")}_ratio': v / total_txns 
            for k, v in store_details.items()
        }

        return store_features

    def extract_category_features(self, transactions: List[Dict]) -> Dict[str, float]:
        """
        Extract product category-related features from transactions
        
        Args:
            transactions: List of processed transactions
        
        Returns:
            Dictionary of category feature ratios
        """
        if not transactions:
            # Default values for category ratios
            categories = ['groceries', 'electronics', 'fashion', 
                          'household', 'personal_care']
            return {
                f'recent_{cat}_spend_ratio': 0.0 
                for cat in categories
            } | {
                f'recent_{cat}_quantity_ratio': 0.0 
                for cat in categories
            }

        # Aggregate category information
        category_amounts = {}
        category_quantities = {}
        total_amount = 0
        total_quantity = 0

        for txn in transactions:
            for item in txn['items']:
                # Retrieve product category
                product = self.db.products.find_one({'product_id': item['product_id']})
                if not product:
                    continue

                category = product['category'].lower()
                amount = item['final_price'] * item['quantity']
                
                category_amounts[category] = category_amounts.get(category, 0) + amount
                category_quantities[category] = category_quantities.get(category, 0) + item['quantity']
                
                total_amount += amount
                total_quantity += item['quantity']

        # Compute category ratios
        category_features = {}
        for category in set(category_amounts.keys()):
            category_key = category.lower().replace(" ", "_")
            
            # Spend ratio
            category_features[f'recent_{category_key}_spend_ratio'] = (
                category_amounts.get(category, 0) / total_amount if total_amount > 0 else 0
            )
            
            # Quantity ratio
            category_features[f'recent_{category_key}_quantity_ratio'] = (
                category_quantities.get(category, 0) / total_quantity if total_quantity > 0 else 0
            )

        return category_features

    def generate_customer_features(self, customer_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive features for a customer
        
        Args:
            customer_id: Unique identifier for the customer
        
        Returns:
            Dictionary of customer features
        """
        # Retrieve customer transactions
        transactions = self.get_customer_transactions(customer_id)

        # Extract various features
        features = {}
        
        # Payment features
        features.update(self.extract_payment_features(transactions))
        
        # Time-based features
        features.update(self.extract_time_features(transactions))
        
        # Store features
        features.update(self.extract_store_features(transactions))
        
        # Category features
        features.update(self.extract_category_features(transactions))

        return features