from pymongo import MongoClient
from faker import Faker
import random
from datetime import datetime, timedelta
import uuid
from typing import List, Dict
import numpy as np

# Initialize Faker with Indian locale
fake = Faker(['en_IN'])

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['indian_retail']

# Indian cities with state mapping
INDIAN_CITIES = {
    'Maharashtra': ['Mumbai', 'Pune', 'Nagpur', 'Thane', 'Nashik'],
    'Delhi': ['New Delhi', 'Noida', 'Gurgaon', 'Faridabad', 'Ghaziabad'],
    'Karnataka': ['Bangalore', 'Mysore', 'Hubli', 'Mangalore', 'Belgaum'],
    'Tamil Nadu': ['Chennai', 'Coimbatore', 'Madurai', 'Salem', 'Trichy'],
    'Telangana': ['Hyderabad', 'Warangal', 'Nizamabad', 'Karimnagar', 'Khammam']
}

# Store categories
STORE_CATEGORIES = ['Hypermarket', 'Supermarket', 'Premium Store', 'Express Store', 'Wholesale Club']

# Store amenities
STORE_AMENITIES = [
    'Parking', 'AC', 'Home Delivery', 'Digital Payments', 'Loyalty Program',
    'Fresh Produce', 'Pharmacy', 'Customer Service Desk', 'Return Policy',
    'Gift Wrapping', 'Senior Citizen Benefits', 'Weekend Discounts'
]

# Product categories with subcategories
PRODUCT_CATEGORIES = {
    'Groceries': [
        'Staples', 'Pulses', 'Rice', 'Atta & Flour', 'Masalas & Spices',
        'Oil & Ghee', 'Dry Fruits', 'Snacks', 'Beverages', 'Breakfast Items'
    ],
    'Personal Care': [
        'Soap & Body Wash', 'Hair Care', 'Oral Care', 'Skin Care',
        'Cosmetics', 'Deodorants', 'Feminine Care', 'Men\'s Grooming'
    ],
    'Household': [
        'Cleaning Supplies', 'Laundry', 'Kitchen Tools', 'Storage',
        'Disposables', 'Pooja Needs', 'Home Decor', 'Furnishing'
    ],
    'Electronics': [
        'Mobile Accessories', 'Batteries', 'Light Bulbs', 'Small Appliances',
        'Computer Accessories', 'Audio', 'Cables & Chargers'
    ],
    'Fashion': [
        'Men\'s Wear', 'Women\'s Wear', 'Kids\' Wear', 'Footwear',
        'Accessories', 'Traditional Wear', 'Innerwear', 'Winter Wear'
    ]
}

# Indian festivals and holidays
INDIAN_FESTIVALS = {
    'Diwali': {'month': 11, 'discount': 0.25, 'popularity': 0.9},
    'Dussehra': {'month': 10, 'discount': 0.20, 'popularity': 0.8},
    'Eid': {'month': 6, 'discount': 0.15, 'popularity': 0.7},
    'Christmas': {'month': 12, 'discount': 0.15, 'popularity': 0.6},
    'Holi': {'month': 3, 'discount': 0.10, 'popularity': 0.7},
    'Raksha Bandhan': {'month': 8, 'discount': 0.15, 'popularity': 0.6},
    'Independence Day': {'month': 8, 'discount': 0.20, 'popularity': 0.5},
    'Republic Day': {'month': 1, 'discount': 0.20, 'popularity': 0.5}
}

class IndianRetailSeeder:
    def __init__(self):
        # Clear existing collections
        db.stores.drop()
        db.products.drop()
        db.customers.drop()
        db.transactions.drop()
        
        # Create indexes
        db.stores.create_index("store_id", unique=True)
        db.products.create_index("product_id", unique=True)
        db.customers.create_index("customer_id", unique=True)
        db.transactions.create_index("transaction_id", unique=True)
        
        self.store_ids = []
        self.product_ids = []
        self.customer_ids = []

    def generate_stores(self, num_stores: int = 50) -> List[Dict]:
        """Generate store data with realistic Indian context"""
        stores = []
        
        for _ in range(num_stores):
            state = random.choice(list(INDIAN_CITIES.keys()))
            city = random.choice(INDIAN_CITIES[state])
            store_id = str(uuid.uuid4())
            self.store_ids.append(store_id)
            
            store = {
                'store_id': store_id,
                'name': f"IndianMart {city} {fake.random_int(1, 5)}",
                'category': random.choice(STORE_CATEGORIES),
                'location': {
                    'state': state,
                    'city': city,
                    'address': fake.address(),
                    'pincode': fake.postcode(),
                    'coordinates': {
                        'latitude': float(fake.latitude()),
                        'longitude': float(fake.longitude())
                    }
                },
                'contact': {
                    'phone': fake.phone_number(),
                    'email': fake.company_email(),
                    'manager_name': fake.name()
                },
                'amenities': random.sample(STORE_AMENITIES, 
                                         random.randint(5, len(STORE_AMENITIES))),
                'ratings': {
                    'overall': round(random.uniform(3.5, 4.8), 1),
                    'service': round(random.uniform(3.5, 4.8), 1),
                    'cleanliness': round(random.uniform(3.5, 4.8), 1)
                },
                'opening_hours': {
                    'weekday': '9:00 AM - 10:00 PM',
                    'weekend': '9:00 AM - 11:00 PM'
                },
                'established_date': fake.date_between(
                    start_date='-10y', end_date='-1y').strftime('%Y-%m-%d')
            }
            stores.append(store)
        
        db.stores.insert_many(stores)
        return stores

    def generate_products(self, num_products: int = 1000) -> List[Dict]:
        """Generate product data with Indian context"""
        products = []
        
        for _ in range(num_products):
            category = random.choice(list(PRODUCT_CATEGORIES.keys()))
            subcategory = random.choice(PRODUCT_CATEGORIES[category])
            product_id = str(uuid.uuid4())
            self.product_ids.append(product_id)
            
            # Base price logic based on category
            base_price_ranges = {
                'Groceries': (20, 1000),
                'Personal Care': (50, 2000),
                'Household': (100, 5000),
                'Electronics': (500, 15000),
                'Fashion': (200, 5000)
            }
            
            base_price = round(random.uniform(
                base_price_ranges[category][0], 
                base_price_ranges[category][1]
            ), 2)
            
            product = {
                'product_id': product_id,
                'name': f'{fake.word().title()} {subcategory}',
                'category': category,
                'subcategory': subcategory,
                'brand': fake.company(),
                'pricing': {
                    'base_price': base_price,
                    'current_price': base_price,
                    'discount_percentage': 0,
                    'tax_rate': 0.18 if category in ['Electronics'] else 0.12
                },
                'inventory': {
                    'total_stock': random.randint(100, 1000),
                    'min_stock_threshold': random.randint(10, 50),
                    'reorder_quantity': random.randint(50, 200)
                },
                'ratings': {
                    'average': round(random.uniform(3.5, 4.8), 1),
                    'count': random.randint(10, 1000)
                },
                'specifications': {
                    'weight': f'{random.randint(100, 1000)}g',
                    'dimensions': f'{random.randint(5, 30)}x{random.randint(5, 30)}x{random.randint(5, 30)} cm',
                    'manufacturer': fake.company(),
                    'country_of_origin': 'India' if random.random() < 0.7 else random.choice(['China', 'USA', 'UK', 'Japan'])
                },
                'features': [fake.word() for _ in range(random.randint(3, 6))],
                'launch_date': fake.date_between(
                    start_date='-2y', end_date='today').strftime('%Y-%m-%d')
            }
            products.append(product)
        
        db.products.insert_many(products)
        return products

    def generate_customers(self, num_customers: int = 500) -> List[Dict]:
        """Generate customer data with Indian context"""
        customers = []
        
        for _ in range(num_customers):
            state = random.choice(list(INDIAN_CITIES.keys()))
            city = random.choice(INDIAN_CITIES[state])
            customer_id = str(uuid.uuid4())
            self.customer_ids.append(customer_id)
            
            # Generate age with realistic distribution
            age = int(np.random.normal(35, 12))
            age = max(18, min(80, age))
            
            customer = {
                'customer_id': customer_id,
                'personal_info': {
                    'name': fake.name(),
                    'age': age,
                    'gender': random.choice(['M', 'F']),
                    'email': fake.email(),
                    'phone': fake.phone_number(),
                    'address': {
                        'street': fake.street_address(),
                        'city': city,
                        'state': state,
                        'pincode': fake.postcode()
                    }
                },
                'shopping_preferences': {
                    'preferred_payment_methods': random.sample(
                        ['UPI', 'Credit Card', 'Debit Card', 'Net Banking', 'Cash'],
                        random.randint(1, 3)
                    ),
                    'preferred_categories': random.sample(
                        list(PRODUCT_CATEGORIES.keys()),
                        random.randint(1, 3)
                    ),
                    'preferred_shopping_times': random.choice(
                        ['Morning', 'Afternoon', 'Evening', 'Night']
                    )
                },
                'loyalty_info': {
                    'membership_tier': random.choice(
                        ['Bronze', 'Silver', 'Gold', 'Platinum']
                    ),
                    'points': random.randint(0, 10000),
                    'member_since': fake.date_between(
                        start_date='-5y', end_date='today').strftime('%Y-%m-%d')
                }
            }
            customers.append(customer)
        
        db.customers.insert_many(customers)
        return customers

    def generate_transactions(self, num_transactions: int = 5000) -> List[Dict]:
        """Generate transaction data with Indian context"""
        transactions = []
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        for _ in range(num_transactions):
            # Generate transaction date
            transaction_date = fake.date_time_between(
                start_date=start_date, end_date=end_date
            )
            
            # Check if date falls during any festival
            active_festivals = []
            festival_multiplier = 1.0
            for festival, details in INDIAN_FESTIVALS.items():
                if details['month'] == transaction_date.month:
                    active_festivals.append({
                        'name': festival,
                        'discount': details['discount']
                    })
                    festival_multiplier = max(festival_multiplier, 
                                           1 + details['popularity'])
            
            # Generate items in transaction
            num_items = int(random.triangular(1, 10, 3) * festival_multiplier)
            items = []
            total_amount = 0
            
            for _ in range(num_items):
                product_id = random.choice(self.product_ids)
                product = db.products.find_one({'product_id': product_id})
                
                quantity = random.randint(1, 5)
                base_price = product['pricing']['base_price']
                
                # Apply festival discounts if applicable
                discount = 0
                if active_festivals:
                    festival = random.choice(active_festivals)
                    discount = festival['discount']
                
                discounted_price = base_price * (1 - discount)
                item_total = discounted_price * quantity
                
                items.append({
                    'product_id': product_id,
                    'quantity': quantity,
                    'unit_price': base_price,
                    'discount': discount,
                    'final_price': discounted_price,
                    'total': item_total
                })
                
                total_amount += item_total
            
            transaction = {
                'transaction_id': str(uuid.uuid4()),
                'customer_id': random.choice(self.customer_ids),
                'store_id': random.choice(self.store_ids),
                'date': transaction_date,
                'items': items,
                'payment': {
                    'method': random.choice([
                        'UPI', 'Credit Card', 'Debit Card',
                        'Net Banking', 'Cash'
                    ]),
                    'total_amount': round(total_amount, 2),
                    'tax_amount': round(total_amount * 0.18, 2),
                    'final_amount': round(total_amount * 1.18, 2)
                },
                'festivals': active_festivals if active_festivals else None,
                'loyalty_points_earned': int(total_amount / 100),
                'rating': random.randint(1, 5) if random.random() < 0.3 else None
            }
            transactions.append(transaction)
        
        db.transactions.insert_many(transactions)
        return transactions

def main():
    seeder = IndianRetailSeeder()
    
    print("Generating stores...")
    seeder.generate_stores(50)
    
    print("Generating products...")
    seeder.generate_products(1000)
    
    print("Generating customers...")
    seeder.generate_customers(500)
    
    print("Generating transactions...")
    seeder.generate_transactions(5000)
    
    print("\nSeeding completed successfully!")
    print("\nCollection Statistics:")
    print(f"Stores: {db.stores.count_documents({})}")
    print(f"Products: {db.products.count_documents({})}")
    print(f"Customers: {db.customers.count_documents({})}")
    print(f"Transactions: {db.transactions.count_documents({})}")

if __name__ == "__main__":
    try:
        print("Starting Indian Retail Database Seeding...")
        main()
    except Exception as e:
        print(f"Error during seeding: {str(e)}")
        raise