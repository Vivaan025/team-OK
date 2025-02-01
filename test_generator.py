import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm, beta, gamma

class MumbaiTestDataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.current_date = datetime.now()
        
        # Load actual Mumbai locations data
        self.locations = pd.read_csv('mumbai_locations.csv')  # Original training locations
    
    def generate_test_customers(self, num_customers=10):
        """Generate test customer data based on Mumbai demographics"""
        
        # Adjust income distribution based on Mumbai tiers
        high_income_prob = len(self.locations[self.locations['primary_demographic'] == 'High Income']) / len(self.locations)
        mid_income_prob = len(self.locations[self.locations['primary_demographic'] == 'Middle Income']) / len(self.locations)
        
        customers = {
            'customer_id': [f'TEST_CUST{i:03d}' for i in range(num_customers)],
            'age': np.random.randint(18, 70, num_customers),
            'income': [],
            'gender': np.random.choice(['M', 'F', 'Other'], num_customers, p=[0.48, 0.48, 0.04]),
            'occupation': [],
            'marital_status': np.random.choice(['Single', 'Married'], num_customers),
            'family_size': np.random.randint(1, 6, num_customers),
            'residential_area': []
        }
        
        for i in range(num_customers):
            # Generate income based on Mumbai demographics
            income_category = np.random.choice(['High', 'Middle', 'Low'], 
                                             p=[high_income_prob, mid_income_prob, 1-high_income_prob-mid_income_prob])
            if income_category == 'High':
                income = np.random.uniform(100000, 200000)
            elif income_category == 'Middle':
                income = np.random.uniform(50000, 100000)
            else:
                income = np.random.uniform(20000, 50000)
            customers['income'].append(income)
            
            # Set occupation based on income
            if income > 100000:
                occupation_choices = ['Professional', 'Self-employed']
                area_choices = ['Urban']
            elif income > 50000:
                occupation_choices = ['Professional', 'Service', 'Self-employed']
                area_choices = ['Urban', 'Suburban']
            else:
                occupation_choices = ['Service', 'Others']
                area_choices = ['Suburban', 'Rural']
            
            customers['occupation'].append(np.random.choice(occupation_choices))
            customers['residential_area'].append(np.random.choice(area_choices))
        
        return pd.DataFrame(customers)
    
    def generate_test_outlets(self, num_outlets=5):
        """Generate test outlet data using actual Mumbai locations"""
        
        store_categories = {
            'Luxury_Automotive': {
                'brands': ['BMW', 'Mercedes-Benz', 'Audi'],
                'locations': self.locations[self.locations['primary_demographic'] == 'High Income']['location_id'].tolist()
            },
            'Mid_Range_Automotive': {
                'brands': ['Honda', 'Toyota', 'Hyundai'],
                'locations': self.locations[self.locations['primary_demographic'].isin(['High Income', 'Middle Income'])]['location_id'].tolist()
            },
            'Budget_Automotive': {
                'brands': ['Maruti Suzuki', 'Tata', 'Renault'],
                'locations': self.locations[self.locations['primary_demographic'] != 'High Income']['location_id'].tolist()
            },
            'Luxury_Fashion': {
                'brands': ['Gucci', 'Louis Vuitton', 'Prada'],
                'locations': self.locations[self.locations['tier'] == 1]['location_id'].tolist()
            },
            'Premium_Electronics': {
                'brands': ['Apple', 'Samsung Premium', 'Sony'],
                'locations': self.locations[self.locations['commercial_score'] > 0.7]['location_id'].tolist()
            }
        }
        
        outlets = {
            'outlet_id': [f'TEST_OUT{i:03d}' for i in range(num_outlets)],
            'outlet_name': [],
            'category': [],
            'brand': [],
            'location_id': [],
            'target_demographic': [],
            'avg_transaction_value': [],
            'outlet_size': [],
            'parking_available': [],
            'area_type': [],
            'location_tier': [],
            'monthly_footfall': [],
            'conversion_rate': [],
            'customer_satisfaction': [],
            'luxury_index': [],
            'commercial_viability': [],
            'competition_score': []
        }
        
        for i in range(num_outlets):
            # Select category and suitable location
            category = np.random.choice(list(store_categories.keys()))
            category_info = store_categories[category]
            
            # Select location based on category requirements
            location_id = np.random.choice(category_info['locations'])
            location = self.locations[self.locations['location_id'] == location_id].iloc[0]
            
            brand = np.random.choice(category_info['brands'])
            is_luxury = category.startswith('Luxury')
            
            outlets['outlet_name'].append(f"{brand} - {location['area_name']}")
            outlets['category'].append(category)
            outlets['brand'].append(brand)
            outlets['location_id'].append(location_id)
            outlets['target_demographic'].append(
                'High Income' if is_luxury else 'Middle Income'
            )
            outlets['avg_transaction_value'].append(
                np.random.gamma(2.0, scale=50000 if is_luxury else 5000)
            )
            outlets['outlet_size'].append(
                'Large' if is_luxury else np.random.choice(['Medium', 'Small'])
            )
            outlets['parking_available'].append(True if is_luxury else np.random.choice([True, False], p=[0.7, 0.3]))
            outlets['area_type'].append(location['primary_demographic'])
            outlets['location_tier'].append(location['tier'])
            outlets['monthly_footfall'].append(
                int(np.random.normal(1000 if is_luxury else 3000, 200) * location['commercial_score'])
            )
            outlets['conversion_rate'].append(
                min(1.0, np.random.beta(8 if is_luxury else 5, 2) * location['commercial_score'])
            )
            outlets['customer_satisfaction'].append(
                min(5.0, np.random.normal(4.5 if is_luxury else 4.0, 0.3) * location['commercial_score'])
            )
            outlets['luxury_index'].append(
                90 if is_luxury else 50 * location['commercial_score']
            )
            outlets['commercial_viability'].append(location['commercial_score'] * 100)
            
            # Calculate competition score based on existing outlets in the area
            similar_outlets = len([
                out for out in outlets['category'][:i]
                if out == category and outlets['location_id'][outlets['category'].index(out)] == location_id
            ])
            competition_score = 100 - (similar_outlets * 20)
            outlets['competition_score'].append(max(0, competition_score))
        
        return pd.DataFrame(outlets)
    
    def generate_test_transactions(self, customers_df, outlets_df, num_transactions=20):
        """Generate test transaction data with Mumbai-specific patterns"""
        
        transactions = {
            'transaction_id': [f'TEST_TRX{i:06d}' for i in range(num_transactions)],
            'customer_id': [],
            'outlet_id': [],
            'location_id': [],
            'transaction_date': [],
            'purchase_amount': [],
            'product_category': [],
            'is_premium_purchase': [],
            'payment_method': [],
            'discount_applied': [],
            'customer_rating': [],
            'weekend_flag': [],
            'time_of_day': [],
            'season': []
        }
        
        # Generate dates with Mumbai festival season weights
        dates = []
        for _ in range(num_transactions):
            # Higher weights during Diwali (Oct-Nov) and Christmas (Dec)
            month_weights = [0.7, 0.7, 0.8, 0.9, 1.0, 0.8,  # Jan-Jun
                           0.7, 0.9, 1.2, 1.3, 1.3, 1.2]    # Jul-Dec
            target_month = np.random.choice(12, p=np.array(month_weights)/sum(month_weights))
            target_date = self.current_date - timedelta(
                days=np.random.randint(0, 365)
            )
            target_date = target_date.replace(month=target_month+1)
            dates.append(target_date)
        
        dates.sort()
        transactions['transaction_date'] = dates
        
        for i in range(num_transactions):
            # Select customer based on income
            customer = customers_df.sample(n=1).iloc[0]
            
            # Select suitable outlet based on customer income
            if customer['income'] > 100000:
                suitable_outlets = outlets_df[outlets_df['target_demographic'] == 'High Income']
            else:
                suitable_outlets = outlets_df[outlets_df['target_demographic'] != 'High Income']
            
            if len(suitable_outlets) == 0:
                suitable_outlets = outlets_df
            
            outlet = suitable_outlets.sample(n=1).iloc[0]
            is_luxury = outlet['category'].startswith('Luxury')
            
            transactions['customer_id'].append(customer['customer_id'])
            transactions['outlet_id'].append(outlet['outlet_id'])
            transactions['location_id'].append(outlet['location_id'])
            transactions['purchase_amount'].append(
                outlet['avg_transaction_value'] * np.random.normal(1, 0.2)
            )
            transactions['product_category'].append(outlet['category'])
            transactions['is_premium_purchase'].append(is_luxury)
            
            # Payment method based on transaction value
            if outlet['avg_transaction_value'] > 50000:
                payment_probs = [0.6, 0.3, 0.05, 0.05, 0]  # Prefer credit cards for luxury
            else:
                payment_probs = [0.3, 0.3, 0.2, 0.15, 0.05]
            
            transactions['payment_method'].append(
                np.random.choice(
                    ['Credit Card', 'Debit Card', 'Cash', 'UPI', 'Wallet'],
                    p=payment_probs
                )
            )
            
            transactions['discount_applied'].append(
                np.random.beta(2, 8) if is_luxury else np.random.beta(2, 5)
            )
            transactions['customer_rating'].append(
                min(5.0, np.random.normal(outlet['customer_satisfaction'], 0.3))
            )
            transactions['weekend_flag'].append(
                1 if dates[i].weekday() >= 5 else 0
            )
            
            hour = np.random.choice(
                list(range(9, 22)),
                p=self._get_hour_distribution(dates[i].weekday())
            )
            transactions['time_of_day'].append(
                'Morning' if hour < 12 else
                'Afternoon' if hour < 17 else
                'Evening'
            )
            
            transactions['season'].append(
                'Summer' if dates[i].month in [3,4,5,6] else
                'Monsoon' if dates[i].month in [7,8,9] else
                'Winter' if dates[i].month in [11,12,1,2] else
                'Festival'  # October (Diwali season)
            )
        
        return pd.DataFrame(transactions)
    
    def _get_hour_distribution(self, weekday):
        """Get hourly distribution based on day of week"""
        hours = np.zeros(13)  # 9 AM to 9 PM
        
        if weekday >= 5:  # Weekend
            # Peak hours: 11 AM - 1 PM and 4 PM - 7 PM
            hours[2:4] = 0.15  # 11 AM - 1 PM
            hours[7:10] = 0.15  # 4 PM - 7 PM
            hours[4:7] = 0.1   # Other hours
            hours[0:2] = 0.05  # Early hours
            hours[10:] = 0.05  # Late hours
        else:  # Weekday
            # Peak hours: 10 AM - 12 PM and 5 PM - 8 PM
            hours[1:3] = 0.1   # 10 AM - 12 PM
            hours[8:11] = 0.15 # 5 PM - 8 PM
            hours[3:8] = 0.08  # Other hours
            hours[0] = 0.05    # Early hours
            hours[11:] = 0.05  # Late hours
        
        return hours / hours.sum()

def generate_test_files():
    """Generate all test data files"""
    
    generator = MumbaiTestDataGenerator()
    
    print("Generating test customer data...")
    customers = generator.generate_test_customers(20)
    
    print("Generating test outlet data...")
    outlets = generator.generate_test_outlets(10)
    
    print("Loading Mumbai location data...")
    locations = generator.locations
    
    print("Generating test transaction data...")
    transactions = generator.generate_test_transactions(customers, outlets, 100)
    
    # Save to CSV files
    customers.to_csv('test_customers.csv', index=False)
    outlets.to_csv('test_outlets.csv', index=False)
    transactions.to_csv('test_transactions.csv', index=False)
    
    print("\nTest data files generated:")
    print("- test_customers.csv")
    print("- test_outlets.csv")
    print("- test_transactions.csv")
    
    return customers, outlets, locations, transactions

if __name__ == "__main__":
    customers, outlets, locations, transactions = generate_test_files()
    
    print("\nTest Data Summary:")
    print(f"\nNumber of test customers: {len(customers)}")
    print(f"Number of test outlets: {len(outlets)}")
    print(f"Number of test transactions: {len(transactions)}")
    
    print("\nOutlet Distribution by Area:")
    print(outlets.groupby('area_type')['category'].value_counts())
    
    print("\nTransaction Summary by Category:")
    print(transactions.groupby('product_category')['purchase_amount'].agg(['count', 'mean']))