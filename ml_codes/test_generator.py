import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_test_data(num_samples=50):
    # Define possible values
    store_types = ['Hypermarket', 'Supermarket', 'Convenience']
    payment_modes = ['UPI', 'Credit Card', 'Debit Card', 'Cash']
    income_categories = ['Low', 'Middle', 'Upper Middle', 'High']
    location_types = ['Urban', 'Semi-Urban', 'Rural']
    tiers = ['Tier 1', 'Tier 2', 'Tier 3']
    sizes = ['Large', 'Medium', 'Small']
    
    # Generate random data
    data = {
        'transaction_date': [
            datetime.now() - timedelta(days=np.random.randint(1, 30))
            for _ in range(num_samples)
        ],
        'store_id': np.random.randint(1, 51, num_samples),
        'customer_id': np.random.randint(1, 1001, num_samples),
        'num_items': np.random.randint(1, 20, num_samples),
        'total_amount': np.random.uniform(100, 50000, num_samples),  # Add actual amount for testing
        'payment_mode': np.random.choice(payment_modes, num_samples),
        'store_type': np.random.choice(store_types, num_samples),
        'tier': np.random.choice(tiers, num_samples),
        'size': np.random.choice(sizes, num_samples),
        'age': np.random.randint(18, 75, num_samples),
        'loyalty_score': np.random.randint(1, 101, num_samples),
        'income_category': np.random.choice(income_categories, num_samples),
        'location_type': np.random.choice(location_types, num_samples),
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add derived features
    df['avg_transaction'] = np.random.uniform(500, 5000, num_samples)
    df['std_transaction'] = np.random.uniform(100, 1000, num_samples)
    df['transaction_count'] = np.random.randint(1, 50, num_samples)
    df['avg_items'] = np.random.uniform(1, 15, num_samples)
    df['max_items'] = df['avg_items'] + np.random.uniform(1, 10, num_samples)
    df['store_avg_sale'] = np.random.uniform(1000, 10000, num_samples)
    df['store_std_sale'] = np.random.uniform(200, 2000, num_samples)
    df['store_sale_count'] = np.random.randint(50, 500, num_samples)
    
    # Add time-based features
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['day_of_week'] = df['transaction_date'].dt.dayofweek
    df['month'] = df['transaction_date'].dt.month
    
    return df

if __name__ == "__main__":
    # Generate test data
    print("Generating test data...")
    test_df = generate_test_data()
    
    # Save to CSV
    test_df.to_csv('test_data.csv', index=False)
    print("Test data saved to test_data.csv")
    print(f"Generated {len(test_df)} test records")
    
    # Print sample
    print("\nSample of test data:")
    print(test_df.head())
    
    # Print data info
    print("\nDataset information:")
    print(test_df.info())