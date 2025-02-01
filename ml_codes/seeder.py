import random
from datetime import datetime
import uuid
from typing import List, Dict
import numpy as np

from pymongo import MongoClient
from faker import Faker

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

def generate_realistic_coordinates(
    city: str, 
    area: str = None
) -> Dict[str, float]:
    """
    Generate realistic geographical coordinates for Indian cities
    
    Args:
        city (str): Name of the city
        area (str, optional): Specific area within the city
    
    Returns:
        Dict[str, float]: Coordinates with latitude and longitude
    """
        # Comprehensive coordinates for Indian cities and their areas
    CITY_COORDINATES = {
        'Mumbai': {
            'center': {'lat': 19.0760, 'lon': 72.8777},
            'areas': {
                'Andheri': {'lat': 19.1243, 'lon': 72.8344},
                'Bandra': {'lat': 19.0530, 'lon': 72.8243},
                'Colaba': {'lat': 18.9145, 'lon': 72.8295},
                'Powai': {'lat': 19.1284, 'lon': 72.9054},
                'Juhu': {'lat': 19.0970, 'lon': 72.8263}
            }
        },
        'Pune': {
            'center': {'lat': 18.5204, 'lon': 73.8567},
            'areas': {
                'Koregaon Park': {'lat': 18.5370, 'lon': 73.8897},
                'Kalyani Nagar': {'lat': 18.5464, 'lon': 73.9042},
                'Viman Nagar': {'lat': 18.5642, 'lon': 73.9266},
                'Kothrud': {'lat': 18.5173, 'lon': 73.8230}
            }
        },
        'Nagpur': {
            'center': {'lat': 21.1458, 'lon': 79.0882},
            'areas': {
                'Dharampeth': {'lat': 21.1415, 'lon': 79.0750},
                'Mahal': {'lat': 21.1385, 'lon': 79.0920},
                'Sitabuldi': {'lat': 21.1432, 'lon': 79.0886},
                'Ramdaspeth': {'lat': 21.1376, 'lon': 79.0830}
            }
        },
        'Thane': {
            'center': {'lat': 19.2183, 'lon': 72.9781},
            'areas': {
                'Vartak Nagar': {'lat': 19.2062, 'lon': 72.9702},
                'Kolshet': {'lat': 19.2294, 'lon': 72.9783},
                'Ghodbunder Road': {'lat': 19.2484, 'lon': 72.9702},
                'Majiwada': {'lat': 19.2014, 'lon': 72.9744}
            }
        },
        'Nashik': {
            'center': {'lat': 19.9975, 'lon': 73.7898},
            'areas': {
                'Nashik Road': {'lat': 20.0028, 'lon': 73.7916},
                'Gangapur Road': {'lat': 20.0175, 'lon': 73.7673},
                'Satpur': {'lat': 19.9906, 'lon': 73.7845},
                'Trimbak Road': {'lat': 20.0115, 'lon': 73.8469}
            }
        },
        'New Delhi': {
            'center': {'lat': 28.6139, 'lon': 77.2090},
            'areas': {
                'Connaught Place': {'lat': 28.6304, 'lon': 77.2177},
                'Hauz Khas': {'lat': 28.5546, 'lon': 77.2118},
                'Nehru Place': {'lat': 28.5479, 'lon': 77.2649},
                'Saket': {'lat': 28.5177, 'lon': 77.2002}
            }
        },
        'Noida': {
            'center': {'lat': 28.5355, 'lon': 77.3910},
            'areas': {
                'Sector 18': {'lat': 28.5272, 'lon': 77.3819},
                'Sector 62': {'lat': 28.6358, 'lon': 77.3772},
                'Sector 51': {'lat': 28.4034, 'lon': 77.0882},
                'Sector 137': {'lat': 28.5579, 'lon': 77.3732}
            }
        },
        'Gurgaon': {
            'center': {'lat': 28.4595, 'lon': 77.0266},
            'areas': {
                'DLF City': {'lat': 28.4700, 'lon': 77.0875},
                'Cyber City': {'lat': 28.4956, 'lon': 77.0898},
                'Sector 14': {'lat': 28.4778, 'lon': 77.0331},
                'Golf Course Road': {'lat': 28.4640, 'lon': 77.0832}
            }
        },
        'Faridabad': {
            'center': {'lat': 28.4089, 'lon': 77.3178},
            'areas': {
                'New Industrial Township': {'lat': 28.3958, 'lon': 77.3126},
                'Sector 14': {'lat': 28.4174, 'lon': 77.3265},
                'Mathura Road': {'lat': 28.4050, 'lon': 77.3250},
                'Ballabgarh': {'lat': 28.3997, 'lon': 77.3218}
            }
        },
        'Ghaziabad': {
            'center': {'lat': 28.6692, 'lon': 77.4538},
            'areas': {
                'Vaishali': {'lat': 28.6472, 'lon': 77.3533},
                'Indirapuram': {'lat': 28.6389, 'lon': 77.3769},
                'Raj Nagar': {'lat': 28.6847, 'lon': 77.4153},
                'Crossing Republik': {'lat': 28.6722, 'lon': 77.4550}
            }
        },
        'Bangalore': {
            'center': {'lat': 12.9716, 'lon': 77.5946},
            'areas': {
                'Koramangala': {'lat': 12.9279, 'lon': 77.6271},
                'Indiranagar': {'lat': 12.9784, 'lon': 77.6408},
                'Whitefield': {'lat': 12.9698, 'lon': 77.7500},
                'JP Nagar': {'lat': 12.9069, 'lon': 77.5825}
            }
        },
        'Mysore': {
            'center': {'lat': 12.2958, 'lon': 76.6394},
            'areas': {
                'Chamundipuram': {'lat': 12.2972, 'lon': 76.6394},
                'Saraswathipuram': {'lat': 12.3089, 'lon': 76.6273},
                'Jayalakshmipuram': {'lat': 12.3028, 'lon': 76.6343},
                'Kuvempunagar': {'lat': 12.2850, 'lon': 76.6504}
            }
        },
        'Hubli': {
            'center': {'lat': 15.3647, 'lon': 75.1240},
            'areas': {
                'Unkal': {'lat': 15.3679, 'lon': 75.1248},
                'Navanagar': {'lat': 15.3618, 'lon': 75.1320},
                'Gokul Road': {'lat': 15.3627, 'lon': 75.1170},
                'Vidyanagar': {'lat': 15.3699, 'lon': 75.1293}
            }
        },
        'Mangalore': {
            'center': {'lat': 12.8797, 'lon': 74.8813},
            'areas': {
                'Kadri': {'lat': 12.8921, 'lon': 74.8711},
                'Bunder': {'lat': 12.8660, 'lon': 74.8853},
                'Kankanady': {'lat': 12.8839, 'lon': 74.8869},
                'Bejai': {'lat': 12.8768, 'lon': 74.8762}
            }
        },
        'Belgaum': {
            'center': {'lat': 15.8497, 'lon': 74.4978},
            'areas': {
                'Khanapur Road': {'lat': 15.8531, 'lon': 74.4952},
                'Shahapur': {'lat': 15.8476, 'lon': 74.5043},
                'Tilakwadi': {'lat': 15.8412, 'lon': 74.4986},
                'Vadgaon': {'lat': 15.8466, 'lon': 74.4901}
            }
        },
        'Chennai': {
            'center': {'lat': 13.0827, 'lon': 80.2707},
            'areas': {
                'T Nagar': {'lat': 13.0389, 'lon': 80.2394},
                'Adyar': {'lat': 13.0069, 'lon': 80.2560},
                'Anna Nagar': {'lat': 13.0852, 'lon': 80.2100},
                'Velachery': {'lat': 12.9854, 'lon': 80.2224}
            }
        },
        'Coimbatore': {
            'center': {'lat': 11.0168, 'lon': 76.9558},
            'areas': {
                'RS Puram': {'lat': 11.0140, 'lon': 76.9624},
                'Gandhipuram': {'lat': 11.0173, 'lon': 76.9571},
                'Saibaba Colony': {'lat': 11.0168, 'lon': 76.9678},
                'Peelamedu': {'lat': 11.0246, 'lon': 76.9904}
            }
        },
        'Madurai': {
            'center': {'lat': 9.9252, 'lon': 78.1198},
            'areas': {
                'Tallakulam': {'lat': 9.9267, 'lon': 78.1170},
                'Gomathipuram': {'lat': 9.9329, 'lon': 78.1136},
                'KK Nagar': {'lat': 9.9179, 'lon': 78.1285},
                'Sellur': {'lat': 9.9224, 'lon': 78.1307}
            }
        },
        'Salem': {
            'center': {'lat': 11.6643, 'lon': 78.1460},
            'areas': {
                'Fairlands': {'lat': 11.6744, 'lon': 78.1453},
                'Suramangalam': {'lat': 11.6579, 'lon': 78.1534},
                'Alagapuram': {'lat': 11.6698, 'lon': 78.1376},
                'Hasthampatty': {'lat': 11.6512, 'lon': 78.1398}
            }
        },
        'Trichy': {
            'center': {'lat': 10.7900, 'lon': 78.7025},
            'areas': {
                'Srirangam': {'lat': 10.8231, 'lon': 78.6925},
                'Thillai Nagar': {'lat': 10.8122, 'lon': 78.6943},
                'K.K. Nagar': {'lat': 10.7992, 'lon': 78.7051},
                'Woraiyur': {'lat': 10.8280, 'lon': 78.6850}
            }
        },
        'Hyderabad': {
            'center': {'lat': 17.3850, 'lon': 78.4867},
            'areas': {
                'Banjara Hills': {'lat': 17.4126, 'lon': 78.4525},
                'Jubilee Hills': {'lat': 17.4289, 'lon': 78.4220},
                'HITEC City': {'lat': 17.4452, 'lon': 78.3771},
                'Gachibowli': {'lat': 17.4424, 'lon': 78.3484}
            }
        },
        'Warangal': {
            'center': {'lat': 17.9692, 'lon': 79.5955},
            'areas': {
                'Hanamkonda': {'lat': 17.9746, 'lon': 79.6009},
                'Kazipet': {'lat': 17.9968, 'lon': 79.5793},
                'Subedari': {'lat': 17.9782, 'lon': 79.5955},
                'Dharmaraopet': {'lat': 17.9629, 'lon': 79.6090}
            }
        },
        'Nizamabad': {
            'center': {'lat': 18.6733, 'lon': 78.1241},
            'areas': {
                'Sudha Nagar': {'lat': 18.6758, 'lon': 78.1201},
                'Urban': {'lat': 18.6701, 'lon': 78.1276},
                'Tilak Nagar': {'lat': 18.6779, 'lon': 78.1179},
                'Pochamma Colony': {'lat': 18.6647, 'lon': 78.1298}
            }
        },
        'Karimnagar': {
            'center': {'lat': 18.4393, 'lon': 79.1288},
            'areas': {
                'Rajeev Nagar': {'lat': 18.4429, 'lon': 79.1232},
                'Jagtial Road': {'lat': 18.4363, 'lon': 79.1356},
                'Bhagatsingh Nagar': {'lat': 18.4372, 'lon': 79.1201},
                'Vivekananda Nagar': {'lat': 18.4458, 'lon': 79.1176}
            }
        },
        'Khammam': {
            'center': {'lat': 17.2247, 'lon': 80.1460},
            'areas': {
                'Wyra Road': {'lat': 17.2276, 'lon': 80.1512},
                'Gandhi Chowk': {'lat': 17.2287, 'lon': 80.1403},
                'Sarada Chowk': {'lat': 17.2219, 'lon': 80.1489},
                'Madhava Reddy Nagar': {'lat': 17.2301, 'lon': 80.1376}
            }
        }
    }
    # Validate city
    if city not in CITY_COORDINATES:
        raise ValueError(f"Coordinates not available for {city}")
    
    # If specific area is provided and exists
    if area and area in CITY_COORDINATES[city]['areas']:
        area_coords = CITY_COORDINATES[city]['areas'][area]
        return {
            'latitude': round(
                area_coords['lat'] + random.uniform(-0.01, 0.01), 
                6
            ),
            'longitude': round(
                area_coords['lon'] + random.uniform(-0.01, 0.01), 
                6
            )
        }
    
    # If area not found or not provided, use city center with variation
    center_coords = CITY_COORDINATES[city]['center']
    return {
        'latitude': round(
            center_coords['lat'] + random.uniform(-0.05, 0.05), 
            6
        ),
        'longitude': round(
            center_coords['lon'] + random.uniform(-0.05, 0.05), 
            6
        )
    }

# City Areas with Detailed Information
CITY_AREAS = {
    'Mumbai': {
        'areas': ['Andheri', 'Bandra', 'Colaba', 'Powai', 'Juhu'],
        'roads': ['S.V. Road', 'Link Road', 'Western Express Highway'],
        'landmarks': ['Near Railway Station', 'Near Metro Station', 'Opposite Mall'],
        'pin_codes': ['400001', '400051', '400069']
    },
    'Pune': {
        'areas': ['Koregaon Park', 'Kalyani Nagar', 'Viman Nagar', 'Kothrud'],
        'roads': ['Karve Road', 'FC Road', 'MG Road'],
        'landmarks': ['Near IT Park', 'Near University', 'Opposite Garden'],
        'pin_codes': ['411001', '411006', '411014']
    },
    'Bangalore': {
        'areas': ['Koramangala', 'Indiranagar', 'Whitefield', 'JP Nagar'],
        'roads': ['MG Road', 'Brigade Road', 'Outer Ring Road'],
        'landmarks': ['Near Tech Park', 'Near Metro Station', 'Opposite Lake'],
        'pin_codes': ['560001', '560034', '560037']
    },
    'Chennai': {
        'areas': ['T Nagar', 'Adyar', 'Anna Nagar', 'Velachery'],
        'roads': ['Anna Salai', 'Mount Road', 'ECR'],
        'landmarks': ['Near Beach', 'Near Metro', 'Opposite Park'],
        'pin_codes': ['600001', '600020', '600040']
    },
    'Hyderabad': {
        'areas': ['Banjara Hills', 'Jubilee Hills', 'HITEC City', 'Gachibowli'],
        'roads': ['Tank Bund Road', 'Necklace Road', 'MG Road'],
        'landmarks': ['Near Hi-Tech City', 'Near Metro Station', 'Opposite Mall'],
        'pin_codes': ['500001', '500034', '500081']
    }
}

# Store Categories
STORE_CATEGORIES = [
    'Hypermarket', 
    'Supermarket', 
    'Premium Store', 
    'Express Store', 
    'Wholesale Club'
]

# Store Amenities
STORE_AMENITIES = [
    'Parking', 'AC', 'Home Delivery', 'Digital Payments', 'Loyalty Program',
    'Fresh Produce', 'Pharmacy', 'Customer Service Desk', 'Return Policy'
]

# Product Categories
PRODUCT_CATEGORIES = {
    'Groceries': [
        'Staples', 'Pulses', 'Rice', 'Atta & Flour', 
        'Masalas & Spices', 'Oil & Ghee', 'Dry Fruits'
    ],
    'Personal Care': [
        'Soap & Body Wash', 'Hair Care', 'Oral Care', 
        'Skin Care', 'Cosmetics', 'Deodorants'
    ],
    'Household': [
        'Cleaning Supplies', 'Laundry', 'Kitchen Tools', 
        'Storage', 'Home Decor', 'Furnishing'
    ],
    'Electronics': [
        'Mobile Accessories', 'Batteries', 'Small Appliances', 
        'Computer Accessories', 'Audio', 'Cables & Chargers'
    ],
    'Fashion': [
        'Men\'s Wear', 'Women\'s Wear', 'Kids\' Wear', 
        'Footwear', 'Accessories', 'Traditional Wear'
    ]
}

# Indian Festivals
INDIAN_FESTIVALS = {
    'Diwali': {'month': 11, 'discount': 0.25, 'popularity': 0.9},
    'Dussehra': {'month': 10, 'discount': 0.20, 'popularity': 0.8},
    'Holi': {'month': 3, 'discount': 0.15, 'popularity': 0.7},
    'Christmas': {'month': 12, 'discount': 0.15, 'popularity': 0.6}
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
        
        # Track generated IDs
        self.store_ids = []
        self.product_ids = []
        self.customer_ids = []

    def generate_stores(self, num_stores: int = 50) -> List[Dict]:
        """Generate store data with realistic Indian context"""
        stores = []
        
        def generate_city_specific_address(city: str, area: str) -> dict:
            """Generate address specific to a city"""
            city_data = CITY_AREAS.get(city, None)
            if not city_data:
                return {
                    'address': fake.street_address(),
                    'pincode': fake.postcode()
                }
                
            road = random.choice(city_data['roads'])
            landmark = random.choice(city_data['landmarks'])
            building_no = random.randint(1, 999)
            pincode = random.choice(city_data['pin_codes'])
            
            address = f"{building_no}, {area}, {road}, {landmark}, {city} - {pincode}"
            return {
                'address': address,
                'pincode': pincode
            }
        
        for _ in range(num_stores):
            state = random.choice(list(INDIAN_CITIES.keys()))
            city = random.choice(INDIAN_CITIES[state])
            store_id = str(uuid.uuid4())
            self.store_ids.append(store_id)
            
            # Get city-specific area and address
            if city in CITY_AREAS:
                area = random.choice(CITY_AREAS[city]['areas'])
                address_info = generate_city_specific_address(city, area)
                coordinates = generate_realistic_coordinates(city, area)
            else:
                area = f"Area {random.randint(1, 10)}"
                address_info = {
                    'address': fake.street_address(),
                    'pincode': fake.postcode()
                }
                coordinates = generate_realistic_coordinates(city)
            
            store = {
                'store_id': store_id,
                'name': f"IndianMart {area} {random.randint(1, 5)}",
                'category': random.choice(STORE_CATEGORIES),
                'location': {
                    'state': state,
                    'city': city,
                    'address': address_info['address'],
                    'pincode': address_info['pincode'],
                    'coordinates': coordinates
                },
                'contact': {
                    'phone': fake.phone_number(),
                    'email': fake.company_email(),
                    'manager_name': fake.name()
                },
                'amenities': random.sample(
                    STORE_AMENITIES, 
                    random.randint(3, len(STORE_AMENITIES))
                ),
                'ratings': {
                    'overall': round(random.uniform(3.5, 4.8), 1),
                    'service': round(random.uniform(3.5, 4.8), 1)
                },
                'opening_hours': {
                    'weekday': '9:00 AM - 10:00 PM',
                    'weekend': '9:00 AM - 11:00 PM'
                },
                'established_date': fake.date_between(
                    start_date='-10y', 
                    end_date='-1y'
                ).strftime('%Y-%m-%d')
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
            
            # Base price logic
            base_price_ranges = {
                'Groceries': (20, 500),
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
                    'discount_percentage': random.uniform(0, 0.3),
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
                    'gender': random.choice(['M', 'F', 'Other']),
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

def seed_database(
    num_stores: int = 50,
    num_products: int = 1000,
    num_customers: int = 500,
    num_transactions: int = 5000
):
    """
    Comprehensive database seeding function
    
    Args:
        num_stores (int): Number of stores to generate
        num_products (int): Number of products to generate
        num_customers (int): Number of customers to generate
        num_transactions (int): Number of transactions to generate
    """
    # Initialize seeder
    seeder = IndianRetailSeeder()
    
    print("Generating stores...")
    stores = seeder.generate_stores(num_stores)
    
    print("Generating products...")
    products = seeder.generate_products(num_products)
    
    print("Generating customers...")
    customers = seeder.generate_customers(num_customers)
    
    print("Generating transactions...")
    transactions = seeder.generate_transactions(num_transactions)
    
    # Print statistics
    print("\nSeeding Completed Successfully!")
    print("\nCollection Statistics:")
    print(f"Stores: {len(stores)}")
    print(f"Products: {len(products)}")
    print(f"Customers: {len(customers)}")
    print(f"Transactions: {len(transactions)}")
    
    # Optional: Return generated data for further processing
    return {
        'stores': stores,
        'products': products,
        'customers': customers,
        'transactions': transactions
    }

def main():
    try:
        # Seed the database
        seed_database()
    except Exception as e:
        print(f"Error during database seeding: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()