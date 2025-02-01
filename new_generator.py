import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import random
from datetime import datetime, timedelta

# Define Mumbai areas with realistic characteristics
mumbai_areas = {
    'Ghatkopar East': {
        'income_range': {
            'lower_class': (25000, 40000, 0.2),    # 20% population
            'middle_class': (40000, 100000, 0.5),  # 50% population
            'upper_middle': (100000, 200000, 0.25), # 25% population
            'affluent': (200000, 500000, 0.05)     # 5% population
        },
        'property_rates': (15000, 25000),  # per sq ft in INR
        'population_density': 45000,        # per sq km
        'commercial_score': 8,
        'residential_score': 7,
        'transit_score': 9,                # Metro + Railway station
        'prime_locations': ['Pant Nagar', 'Rajawadi', 'Amrut Nagar'],
        'market_types': ['Street Market', 'Shopping Complex', 'Mall']
    },
    'Powai': {
        'income_range': {
            'lower_class': (30000, 45000, 0.15),
            'middle_class': (45000, 120000, 0.45),
            'upper_middle': (120000, 300000, 0.3),
            'affluent': (300000, 800000, 0.1)
        },
        'property_rates': (18000, 32000),
        'population_density': 35000,
        'commercial_score': 7,
        'residential_score': 9,
        'transit_score': 6,
        'prime_locations': ['Hiranandani Gardens', 'IIT Area', 'Lake Side'],
        'market_types': ['Mall', 'Shopping Complex', 'Premium Retail']
    },
    'Andheri West': {
        'income_range': {
            'lower_class': (28000, 42000, 0.2),
            'middle_class': (42000, 110000, 0.5),
            'upper_middle': (110000, 250000, 0.25),
            'affluent': (250000, 600000, 0.05)
        },
        'property_rates': (20000, 35000),
        'population_density': 42000,
        'commercial_score': 9,
        'residential_score': 8,
        'transit_score': 9,
        'prime_locations': ['Lokhandwala', 'Versova', 'Seven Bungalows'],
        'market_types': ['Mall', 'Street Market', 'Premium Retail']
    },
    'Bandra West': {
        'income_range': {
            'lower_class': (35000, 50000, 0.1),
            'middle_class': (50000, 150000, 0.4),
            'upper_middle': (150000, 400000, 0.35),
            'affluent': (400000, 1500000, 0.15)
        },
        'property_rates': (35000, 75000),
        'population_density': 38000,
        'commercial_score': 9,
        'residential_score': 8,
        'transit_score': 8,
        'prime_locations': ['Pali Hill', 'Carter Road', 'Linking Road'],
        'market_types': ['High Street', 'Premium Retail', 'Mall']
    },
    'Mulund West': {
        'income_range': {
            'lower_class': (25000, 40000, 0.2),
            'middle_class': (40000, 90000, 0.55),
            'upper_middle': (90000, 180000, 0.2),
            'affluent': (180000, 400000, 0.05)
        },
        'property_rates': (13000, 22000),
        'population_density': 40000,
        'commercial_score': 7,
        'residential_score': 8,
        'transit_score': 7,
        'prime_locations': ['Sarvodaya Nagar', 'Vardhaman Nagar', 'Nahur'],
        'market_types': ['Shopping Complex', 'Mall', 'Street Market']
    }
}

# Realistic product categories with actual price ranges and seasonality
product_categories = {
    'Electronics': {
        'price_range': {
            'budget': (1000, 15000),
            'mid_range': (15000, 50000),
            'premium': (50000, 200000)
        },
        'store_types': ['Mall', 'Standalone Store', 'Shopping Complex'],
        'peak_seasons': ['Diwali', 'Christmas'],
        'markup_range': (0.15, 0.35),  # 15-35% markup
        'average_purchase_frequency': 180  # days
    },
    'Fashion': {
        'price_range': {
            'budget': (500, 3000),
            'mid_range': (3000, 10000),
            'premium': (10000, 50000)
        },
        'store_types': ['Mall', 'Boutique', 'High Street'],
        'peak_seasons': ['Diwali', 'Wedding', 'End of Season'],
        'markup_range': (0.5, 1.2),  # 50-120% markup
        'average_purchase_frequency': 60
    },
    'Grocery': {
        'price_range': {
            'budget': (100, 1000),
            'mid_range': (1000, 3000),
            'premium': (3000, 10000)
        },
        'store_types': ['Supermarket', 'Local Kirana', 'Hypermarket'],
        'peak_seasons': ['Diwali', 'Monthly Start'],
        'markup_range': (0.1, 0.25),  # 10-25% markup
        'average_purchase_frequency': 7
    }
}

def get_income_bracket(amount, area_info):
    """Determine income bracket based on amount and area demographics"""
    for bracket, (min_val, max_val, _) in area_info['income_range'].items():
        if min_val <= amount <= max_val:
            return bracket
    return 'middle_class'  # default fallback

def generate_customer_data(num_customers):
    customers = []
    age_ranges = {
        '18-25': 0.2,
        '26-35': 0.35,
        '36-50': 0.3,
        '50+': 0.15
    }
    
    for _ in range(num_customers):
        area = random.choice(list(mumbai_areas.keys()))
        area_info = mumbai_areas[area]
        
        # Select income bracket based on area demographics
        bracket = random.choices(
            list(area_info['income_range'].keys()),
            weights=[x[2] for x in area_info['income_range'].values()]
        )[0]
        min_income, max_income, _ = area_info['income_range'][bracket]
        
        # Generate age based on demographic distribution
        age_group = random.choices(
            list(age_ranges.keys()),
            weights=list(age_ranges.values())
        )[0]
        
        customer = {
            'customer_id': f'CUST_{str(random.randint(1000, 9999))}',
            'age_group': age_group,
            'area': area,
            'sub_area': random.choice(area_info['prime_locations']),
            'income': round(random.uniform(min_income, max_income), -2),  # Round to nearest 100
            'income_bracket': bracket,
            'preferred_payment': random.choices(
                ['Cash', 'Credit Card', 'UPI', 'Debit Card'],
                weights=[0.2, 0.3, 0.35, 0.15]  # Modern Mumbai payment preferences
            )[0],
            'shopping_frequency': random.choices(
                ['Weekly', 'Bi-weekly', 'Monthly'],
                weights=[0.4, 0.35, 0.25]
            )[0],
            'preferred_shopping_time': random.choices(
                ['Morning', 'Afternoon', 'Evening', 'Night'],
                weights=[0.2, 0.25, 0.4, 0.15]
            )[0],
            'online_shopping_preference': random.choices(
                ['Low', 'Medium', 'High'],
                weights=[0.3, 0.4, 0.3]
            )[0]
        }
        customers.append(customer)
    
    return pd.DataFrame(customers)

def generate_store_data(num_stores):
    stores = []
    
    for _ in range(num_stores):
        area = random.choice(list(mumbai_areas.keys()))
        area_info = mumbai_areas[area]
        category = random.choice(list(product_categories.keys()))
        store_type = random.choice(product_categories[category]['store_types'])
        
        # Calculate rent based on area's property rates
        min_rate, max_rate = area_info['property_rates']
        store_size = random.randint(500, 10000)
        monthly_rent = (random.uniform(min_rate, max_rate) * store_size) / 100  # Approximate monthly rent
        
        store = {
            'store_id': f'STR_{str(random.randint(1000, 9999))}',
            'area': area,
            'sub_area': random.choice(area_info['prime_locations']),
            'category': category,
            'store_type': store_type,
            'store_size_sqft': store_size,
            'monthly_rent': round(monthly_rent, -2),  # Round to nearest 100
            'parking_available': random.choices([True, False], weights=[0.7, 0.3])[0],
            'years_in_operation': random.randint(1, 20),
            'commercial_score': area_info['commercial_score'],
            'transit_score': area_info['transit_score'],
            'population_density': area_info['population_density'],
            'target_customer_segment': random.choices(
                ['budget', 'mid_range', 'premium'],
                weights=[0.4, 0.4, 0.2]
            )[0],
            'air_conditioned': random.choices([True, False], weights=[0.8, 0.2])[0],
            'weekday_hours': f"{random.randint(9,11)}:00-{random.randint(20,22)}:00",
            'weekend_hours': f"{random.randint(9,11)}:00-{random.randint(21,23)}:00"
        }
        stores.append(store)
    
    return pd.DataFrame(stores)

def generate_sales_data(customers_df, stores_df, num_transactions):
    transactions = []
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    # Festival dates
    festivals = {
        'Diwali': [(2023, 11, 12), (2024, 10, 31)],
        'Christmas': [(2023, 12, 25), (2024, 12, 25)],
        'End of Season': [(2023, 7, 1), (2023, 12, 1), (2024, 7, 1), (2024, 12, 1)]
    }
    
    def is_festival_period(date):
        for festival, dates in festivals.items():
            for fest_date in dates:
                fest_datetime = datetime(*fest_date)
                # Consider 15 days before and after as festival period
                if abs((date - fest_datetime).days) <= 15:
                    return True
        return False
    
    for _ in range(num_transactions):
        customer = customers_df.sample(1).iloc[0]
        store = stores_df.sample(1).iloc[0]
        category = store['category']
        
        # Generate transaction date
        random_days = random.randint(0, (end_date - start_date).days)
        transaction_date = start_date + timedelta(days=random_days)
        
        # Determine price range based on customer income bracket
        if customer['income_bracket'] in ['affluent', 'upper_middle']:
            price_category = random.choices(['premium', 'mid_range'], weights=[0.6, 0.4])[0]
        elif customer['income_bracket'] == 'middle_class':
            price_category = random.choices(['mid_range', 'budget'], weights=[0.7, 0.3])[0]
        else:
            price_category = 'budget'
            
        price_range = product_categories[category]['price_range'][price_category]
        base_amount = random.uniform(*price_range)
        
        # Apply festival season markup
        if is_festival_period(transaction_date):
            base_amount *= random.uniform(1.1, 1.3)  # 10-30% festival markup
            
        # Apply weekend markup
        if transaction_date.weekday() >= 5:  # Weekend
            base_amount *= 1.1  # 10% weekend markup
            
        transaction = {
            'transaction_id': f'TXN_{str(random.randint(10000, 99999))}',
            'customer_id': customer['customer_id'],
            'store_id': store['store_id'],
            'date': transaction_date,
            'amount': round(base_amount, 2),
            'category': category,
            'price_segment': price_category,
            'payment_method': customer['preferred_payment'],
            'time_of_day': customer['preferred_shopping_time'],
            'is_festival_period': is_festival_period(transaction_date),
            'is_weekend': transaction_date.weekday() >= 5
        }
        transactions.append(transaction)
    
    return pd.DataFrame(transactions)

def generate_full_dataset(num_customers=1000, num_stores=100, num_transactions=5000):
    print("Generating customer data...")
    customers_df = generate_customer_data(num_customers)
    
    print("Generating store data...")
    stores_df = generate_store_data(num_stores)
    
    print("Generating transaction data...")
    transactions_df = generate_sales_data(customers_df, stores_df, num_transactions)
    
    # Save to CSV files
    customers_df.to_csv('mumbai_retail_customers.csv', index=False)
    stores_df.to_csv('mumbai_retail_stores.csv', index=False)
    transactions_df.to_csv('mumbai_retail_transactions.csv', index=False)
    
    # Generate summary statistics
    print("\nDataset Statistics:")
    print("\nCustomer Demographics:")
    print(customers_df['income_bracket'].value_counts(normalize=True))
    print("\nArea-wise Store Distribution:")
    print(stores_df['area'].value_counts())
    print("\nTransaction Amount Statistics by Category:")
    print(transactions_df.groupby('category')['amount'].describe())
    
    return customers_df, stores_df, transactions_df


class RetailDataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def prepare_features(self, customers_df, stores_df, transactions_df):
        """Prepare features for ML model"""
        
        # Aggregate customer purchase patterns
        customer_patterns = transactions_df.groupby('customer_id').agg({
            'amount': ['mean', 'sum', 'std', 'count'],
            'is_festival_period': 'mean',
            'is_weekend': 'mean',
            'category': lambda x: x.mode().iloc[0] if not x.empty else None,
            'price_segment': lambda x: x.mode().iloc[0] if not x.empty else None
        }).reset_index()
        
        customer_patterns.columns = [
            'customer_id', 'avg_purchase', 'total_spend', 
            'spend_std', 'transaction_count', 'festival_shopping_ratio',
            'weekend_shopping_ratio', 'preferred_category', 'preferred_segment'
        ]
        
        # Merge with customer demographics
        customer_features = customers_df.merge(
            customer_patterns,
            on='customer_id',
            how='left'
        )
        
        # Calculate store performance metrics
        store_performance = transactions_df.groupby('store_id').agg({
            'amount': ['mean', 'sum', 'count'],
            'customer_id': 'nunique'
        }).reset_index()
        
        store_performance.columns = [
            'store_id', 'avg_transaction', 'total_revenue',
            'transaction_count', 'unique_customers'
        ]
        
        # Merge with store information
        store_features = stores_df.merge(
            store_performance,
            on='store_id',
            how='left'
        )
        
        # Create features for prediction
        def encode_categorical(df, columns):
            for col in columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
            return df
        
        # Encode categorical variables
        categorical_cols = ['area', 'income_bracket', 'age_group', 'preferred_payment',
                          'shopping_frequency', 'preferred_shopping_time']
        customer_features = encode_categorical(customer_features, categorical_cols)
        
        store_categorical_cols = ['area', 'category', 'store_type', 'target_customer_segment']
        store_features = encode_categorical(store_features, store_categorical_cols)
        
        # Create final feature matrix
        customer_features['spending_power'] = (
            customer_features['total_spend'] / customer_features['transaction_count']
        )
        customer_features['festival_preference'] = (
            customer_features['festival_shopping_ratio'] * customer_features['avg_purchase']
        )
        
        store_features['revenue_per_customer'] = (
            store_features['total_revenue'] / store_features['unique_customers']
        )
        store_features['efficiency_score'] = (
            store_features['total_revenue'] / (store_features['store_size_sqft'] * store_features['monthly_rent'])
        )
        
        return customer_features, store_features
        
    def create_training_data(self, customer_features, store_features, transactions_df):
        """Create training data for store preference prediction"""
        
        # Create customer-store interactions
        interactions = transactions_df.groupby(['customer_id', 'store_id']).agg({
            'amount': ['count', 'sum'],
            'is_festival_period': 'mean',
            'is_weekend': 'mean'
        }).reset_index()
        
        interactions.columns = [
            'customer_id', 'store_id', 'visit_count', 
            'total_spend', 'festival_visit_ratio', 'weekend_visit_ratio'
        ]
        
        # Calculate preference score (normalized visits and spending)
        interactions['preference_score'] = (
            0.7 * (interactions['visit_count'] / interactions.groupby('customer_id')['visit_count'].transform('max')) +
            0.3 * (interactions['total_spend'] / interactions.groupby('customer_id')['total_spend'].transform('max'))
        )
        
        # Merge customer and store features
        training_data = interactions.merge(
            customer_features, on='customer_id', how='left'
        ).merge(
            store_features, on='store_id', how='left'
        )
        
        # Create distance feature (dummy for now, can be replaced with actual distances)
        training_data['customer_store_distance'] = np.random.uniform(0, 5, len(training_data))
        
        # Select features for model
        feature_columns = [
            'avg_purchase', 'total_spend', 'spend_std', 'transaction_count',
            'festival_shopping_ratio', 'weekend_shopping_ratio',
            'area_encoded_x', 'income_bracket_encoded', 'age_group_encoded',
            'store_size_sqft', 'monthly_rent', 'commercial_score',
            'transit_score', 'population_density', 'area_encoded_y',
            'revenue_per_customer', 'efficiency_score', 'customer_store_distance'
        ]
        
        X = training_data[feature_columns]
        y = training_data['preference_score']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def create_ml_ready_dataset(customers_df, stores_df, transactions_df):
    """Create ML-ready dataset from raw data"""
    
    preprocessor = RetailDataPreprocessor()
    
    # Prepare features
    customer_features, store_features = preprocessor.prepare_features(
        customers_df, stores_df, transactions_df
    )
    
    # Create training data
    X_train, X_test, y_train, y_test = preprocessor.create_training_data(
        customer_features, store_features, transactions_df
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'customer_features': customer_features,
        'store_features': store_features,
        'preprocessor': preprocessor
    }

if __name__ == "__main__":
    # Generate the dataset
    customers_df, stores_df, transactions_df = generate_full_dataset(
        num_customers=2000,
        num_stores=200,
        num_transactions=10000
    )
    
    # Prepare data for ML
    ml_data = create_ml_ready_dataset(customers_df, stores_df, transactions_df)
    
    print("\nML Dataset prepared successfully!")
    print(f"Training set size: {ml_data['X_train'].shape}")
    print(f"Test set size: {ml_data['X_test'].shape}")
    
    # Save processed features
    ml_data['customer_features'].to_csv('processed_customer_features.csv', index=False)
    ml_data['store_features'].to_csv('processed_store_features.csv', index=False)
    
    print("\nFeature importance analysis:")
    feature_names = [
        'avg_purchase', 'total_spend', 'spend_std', 'transaction_count',
        'festival_ratio', 'weekend_ratio', 'area_customer', 'income_bracket',
        'age_group', 'store_size', 'monthly_rent', 'commercial_score',
        'transit_score', 'population_density', 'area_store', 'revenue_per_customer',
        'efficiency_score', 'distance'
    ]
    
    # Print basic statistics about the features
    print("\nFeature Statistics:")
    for i, feature in enumerate(feature_names):
        print(f"{feature}:")
        print(f"  Mean: {ml_data['X_train'][:, i].mean():.2f}")
        print(f"  Std: {ml_data['X_train'][:, i].std():.2f}")
        print(f"  Min: {ml_data['X_train'][:, i].min():.2f}")
        print(f"  Max: {ml_data['X_train'][:, i].max():.2f}")