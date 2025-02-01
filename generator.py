import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm, beta, gamma

class SmartRetailDataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.current_date = datetime.now()
        
    def generate_customer_demographics(self, num_customers):
        """Generate realistic customer demographics with correlations"""
        
        # Age distribution - bell curve with peak around 35
        ages = norm.rvs(loc=35, scale=12, size=num_customers).astype(int)
        ages = np.clip(ages, 18, 80)  # Restrict to realistic range
        
        # Income distribution - gamma distribution for right skew
        base_income = gamma.rvs(a=2.0, loc=30000, scale=20000, size=num_customers)
        
        # Age-income correlation (higher age generally means higher income until retirement)
        age_factor = np.where(ages < 65, (ages - 18) / 47, (80 - ages) / 15)
        income = base_income * (1 + age_factor * 0.5)
        
        # Occupation probabilities based on age
        occupation_probs = {
            'Student': lambda age: 0.8 if age < 25 else 0.1 if age < 35 else 0.01,
            'Professional': lambda age: 0.1 if age < 25 else 0.6 if age < 60 else 0.1,
            'Self-employed': lambda age: 0.05 if age < 30 else 0.2,
            'Retired': lambda age: 0.8 if age >= 65 else 0.0,
            'Service': lambda age: 0.2,
            'Others': lambda age: 0.1
        }
        
        occupations = []
        for age in ages:
            probs = [prob(age) for prob in occupation_probs.values()]
            probs = np.array(probs) / sum(probs)  # Normalize
            occupations.append(np.random.choice(list(occupation_probs.keys()), p=probs))

        # Generate marital status based on age
        marital_status = [
            'Married' if (np.random.random() < (age - 20) / 60) else 'Single'
            for age in ages
        ]
        
        # Generate family size based on marital status
        family_size = [
            np.random.randint(1, 6) if status == 'Married' else 1
            for status in marital_status
        ]
        
        # Generate residential area based on income
        residential_area = [
            np.random.choice(
                ['Urban', 'Suburban', 'Rural'],
                p=[0.6, 0.3, 0.1] if inc > 60000 else [0.3, 0.4, 0.3]
            )
            for inc in income
        ]
        
        # Combine all demographics into a dictionary
        demographics = {
            'customer_id': [f'CUST{i:04d}' for i in range(num_customers)],
            'age': ages,
            'income': income,
            'occupation': occupations,
            'gender': np.random.choice(['M', 'F', 'Other'], num_customers, p=[0.48, 0.48, 0.04]),
            'marital_status': marital_status,
            'family_size': family_size,
            'residential_area': residential_area
        }
        
        return pd.DataFrame(demographics)
    
    def generate_outlet_data(self, num_outlets):
        """Generate retail outlet data with realistic patterns"""
        
        # Basic outlet information
        outlets = {
            'outlet_id': [f'OUT{i:03d}' for i in range(num_outlets)],
            'outlet_name': [f'Store {i+1}' for i in range(num_outlets)],
            'outlet_type': np.random.choice(
                ['Mall Outlet', 'Standalone Store', 'Street Shop'],
                num_outlets,
                p=[0.3, 0.5, 0.2]  # More standalone stores than others
            )
        }
        
        # Size based on type
        size_probs = {
            'Mall Outlet': [0.1, 0.3, 0.6],      # More likely to be large
            'Standalone Store': [0.2, 0.6, 0.2],  # More likely to be medium
            'Street Shop': [0.7, 0.2, 0.1]        # More likely to be small
        }
        
        outlets['outlet_size'] = [
            np.random.choice(
                ['Small', 'Medium', 'Large'],
                p=size_probs[otype]
            )
            for otype in outlets['outlet_type']
        ]
        
        # Generate other outlet attributes
        area_types = []
        parkings = []
        operational_hours = []
        latitudes = []
        longitudes = []
        
        for otype in outlets['outlet_type']:
            if otype == 'Mall Outlet':
                area_prob = [0.6, 0.3, 0.1]  # High income areas
                parking_prob = 0.95
                hours = np.random.choice([72, 84, 98], p=[0.1, 0.3, 0.6])
            elif otype == 'Standalone Store':
                area_prob = [0.3, 0.5, 0.2]  # Middle income areas
                parking_prob = 0.7
                hours = np.random.choice([72, 84, 98], p=[0.3, 0.5, 0.2])
            else:  # Street Shop
                area_prob = [0.1, 0.4, 0.5]  # Lower income areas
                parking_prob = 0.2
                hours = np.random.choice([72, 84, 98], p=[0.6, 0.3, 0.1])
                
            area_types.append(np.random.choice(
                ['High Income', 'Middle Income', 'Lower Income'],
                p=area_prob
            ))
            parkings.append(np.random.random() < parking_prob)
            operational_hours.append(hours)
            
            # Generate nearby coordinates (for demonstration)
            latitudes.append(np.random.uniform(28.0, 29.0))
            longitudes.append(np.random.uniform(76.0, 77.0))
        
        # Add generated attributes to outlets dictionary
        outlets.update({
            'latitude': latitudes,
            'longitude': longitudes,
            'parking_available': parkings,
            'operation_years': np.random.randint(1, 20, num_outlets),
            'competition_distance': np.random.uniform(0.5, 5.0, num_outlets),
            'area_type': area_types,
            'weekly_operational_hours': operational_hours
        })
        
        return pd.DataFrame(outlets)
    
    def generate_transactions(self, customers_df, outlets_df, num_transactions):
        """Generate transaction data with realistic patterns"""
        
        # Time patterns
        hours = np.random.choice(
            24,
            num_transactions,
            p=self._generate_hourly_distribution()
        )
        
        # Customer purchase patterns based on demographics
        customer_purchase_probs = self._calculate_customer_purchase_probabilities(customers_df)
        selected_customers = np.random.choice(
            customers_df['customer_id'],
            num_transactions,
            p=customer_purchase_probs
        )
        
        # Purchase amount based on income and store type
        base_amounts = []
        for cust_id in selected_customers:
            customer = customers_df[customers_df['customer_id'] == cust_id].iloc[0]
            income_factor = np.clip(customer['income'] / 50000, 0.5, 2.0)
            base_amounts.append(
                np.random.gamma(
                    shape=2.0,
                    scale=500 * income_factor
                )
            )
        
        # Generate transaction dates first
        transaction_dates = [
            self.current_date - timedelta(
                days=np.random.exponential(30),  # Recent transactions more likely
                hours=int(hour)
            )
            for hour in hours
        ]
        
        # Now create the transactions dictionary
        transactions = {
            'transaction_id': [f'TRX{i:06d}' for i in range(num_transactions)],
            'customer_id': selected_customers,
            'outlet_id': np.random.choice(outlets_df['outlet_id'], num_transactions),
            'transaction_date': transaction_dates,
            'purchase_amount': base_amounts,
            'num_items': np.random.negative_binomial(
                n=3,
                p=0.3,
                size=num_transactions
            ) + 1,  # At least 1 item
            'payment_method': np.random.choice(
                ['Credit Card', 'Debit Card', 'Cash', 'UPI', 'Wallet'],
                num_transactions,
                p=[0.3, 0.3, 0.2, 0.15, 0.05]
            ),
            'discount_applied': np.random.beta(2, 5, num_transactions),  # Most discounts will be smaller
            'weekend_flag': [
                1 if d.weekday() >= 5 else 0
                for d in transaction_dates
            ],
            'time_of_day': [
                'Morning' if 5 <= h < 12 else
                'Afternoon' if 12 <= h < 17 else
                'Evening' if 17 <= h < 21 else
                'Night'
                for h in hours
            ]
        }
        
        return pd.DataFrame(transactions)
    
    def _generate_hourly_distribution(self):
        """Generate realistic hourly distribution for store visits"""
        hours = np.zeros(24)
        
        # Morning rush (8-10 AM)
        hours[8:11] = norm.pdf(np.arange(3), loc=1, scale=0.5)
        
        # Lunch time (12-2 PM)
        hours[12:14] = norm.pdf(np.arange(2), loc=0, scale=0.5)
        
        # Evening rush (5-8 PM)
        hours[17:20] = norm.pdf(np.arange(3), loc=1, scale=0.5)
        
        # Normalize to probabilities
        return hours / hours.sum()
    
    def _calculate_customer_purchase_probabilities(self, customers_df):
        """Calculate purchase probability based on customer demographics"""
        
        # Income factor (higher income = more purchases)
        income_factor = np.clip(customers_df['income'] / 50000, 0.5, 2.0)
        
        # Age factor (middle age = more purchases)
        age_factor = 1 - np.abs(customers_df['age'] - 40) / 40
        
        # Family size factor (larger family = more purchases)
        family_factor = np.clip(customers_df['family_size'] / 3, 0.8, 1.5)
        
        # Combine factors
        purchase_prob = income_factor * age_factor * family_factor
        
        # Normalize to probabilities
        return purchase_prob / purchase_prob.sum()
    
    def generate_mumbai_locations(self):
        """Generate realistic Mumbai location data with tiers and populations"""
        
        # Mumbai locations with their characteristics
        mumbai_locations = {
            'location_id': [],
            'area_name': [],
            'tier': [],
            'population': [],
            'latitude': [],
            'longitude': [],
            'primary_demographic': [],
            'commercial_score': [],
            'residential_density': [],
            'public_transport_accessibility': []
        }
        
        # Define location data
        locations = [
            # South Mumbai (Tier 1)
            {
                'area': 'Colaba', 'tier': 1, 'pop_base': 130000,
                'lat': 18.9067, 'lon': 72.8147, 'demo': 'High Income',
                'comm_score': 0.9, 'res_density': 'High', 'transport': 0.95
            },
            {
                'area': 'Nariman Point', 'tier': 1, 'pop_base': 75000,
                'lat': 18.9256, 'lon': 72.8242, 'demo': 'High Income',
                'comm_score': 1.0, 'res_density': 'Medium', 'transport': 0.95
            },
            {
                'area': 'Fort', 'tier': 1, 'pop_base': 90000,
                'lat': 18.9345, 'lon': 72.8352, 'demo': 'High Income',
                'comm_score': 0.95, 'res_density': 'Medium', 'transport': 0.9
            },
            
            # Western Suburbs (Mix of Tiers)
            {
                'area': 'Bandra West', 'tier': 1, 'pop_base': 375000,
                'lat': 19.0596, 'lon': 72.8295, 'demo': 'High Income',
                'comm_score': 0.9, 'res_density': 'High', 'transport': 0.9
            },
            {
                'area': 'Andheri West', 'tier': 2, 'pop_base': 500000,
                'lat': 19.1136, 'lon': 72.8697, 'demo': 'Middle Income',
                'comm_score': 0.85, 'res_density': 'Very High', 'transport': 0.85
            },
            {
                'area': 'Juhu', 'tier': 1, 'pop_base': 150000,
                'lat': 19.1075, 'lon': 72.8263, 'demo': 'High Income',
                'comm_score': 0.8, 'res_density': 'Medium', 'transport': 0.75
            },
            
            # Central Mumbai (Mix of Tiers)
            {
                'area': 'Worli', 'tier': 1, 'pop_base': 200000,
                'lat': 19.0178, 'lon': 72.8478, 'demo': 'High Income',
                'comm_score': 0.9, 'res_density': 'High', 'transport': 0.85
            },
            {
                'area': 'Dadar', 'tier': 2, 'pop_base': 350000,
                'lat': 19.0178, 'lon': 72.8478, 'demo': 'Middle Income',
                'comm_score': 0.8, 'res_density': 'Very High', 'transport': 0.9
            },
            {
                'area': 'Parel', 'tier': 2, 'pop_base': 250000,
                'lat': 18.9977, 'lon': 72.8376, 'demo': 'Middle Income',
                'comm_score': 0.75, 'res_density': 'High', 'transport': 0.85
            },
            
            # Eastern Suburbs (Mostly Tier 2 and 3)
            {
                'area': 'Chembur', 'tier': 2, 'pop_base': 400000,
                'lat': 19.0522, 'lon': 72.9005, 'demo': 'Middle Income',
                'comm_score': 0.7, 'res_density': 'High', 'transport': 0.8
            },
            {
                'area': 'Ghatkopar', 'tier': 2, 'pop_base': 450000,
                'lat': 19.0858, 'lon': 72.9089, 'demo': 'Middle Income',
                'comm_score': 0.75, 'res_density': 'Very High', 'transport': 0.85
            },
            {
                'area': 'Kurla', 'tier': 3, 'pop_base': 500000,
                'lat': 19.0726, 'lon': 72.8845, 'demo': 'Lower Middle Income',
                'comm_score': 0.6, 'res_density': 'Very High', 'transport': 0.8
            },
            
            # Extended Western Suburbs (Tier 2 and 3)
            {
                'area': 'Goregaon', 'tier': 2, 'pop_base': 450000,
                'lat': 19.1663, 'lon': 72.8526, 'demo': 'Middle Income',
                'comm_score': 0.7, 'res_density': 'High', 'transport': 0.75
            },
            {
                'area': 'Malad', 'tier': 2, 'pop_base': 500000,
                'lat': 19.1873, 'lon': 72.8484, 'demo': 'Middle Income',
                'comm_score': 0.65, 'res_density': 'Very High', 'transport': 0.7
            },
            {
                'area': 'Kandivali', 'tier': 3, 'pop_base': 450000,
                'lat': 19.2037, 'lon': 72.8511, 'demo': 'Lower Middle Income',
                'comm_score': 0.6, 'res_density': 'High', 'transport': 0.65
            },
            
            # Navi Mumbai (Tier 2 and 3)
            {
                'area': 'Vashi', 'tier': 2, 'pop_base': 300000,
                'lat': 19.0745, 'lon': 72.9978, 'demo': 'Middle Income',
                'comm_score': 0.75, 'res_density': 'Medium', 'transport': 0.8
            },
            {
                'area': 'Nerul', 'tier': 2, 'pop_base': 280000,
                'lat': 19.0362, 'lon': 73.0184, 'demo': 'Middle Income',
                'comm_score': 0.7, 'res_density': 'Medium', 'transport': 0.75
            },
        ]
        
        # Add variation to population based on density and random factors
        for i, loc in enumerate(locations):
            # Generate location ID
            mumbai_locations['location_id'].append(f'LOC{i:03d}')
            mumbai_locations['area_name'].append(loc['area'])
            mumbai_locations['tier'].append(loc['tier'])
            
            # Add some random variation to population
            base_pop = loc['pop_base']
            variation = np.random.uniform(0.9, 1.1)
            mumbai_locations['population'].append(int(base_pop * variation))
            
            # Add coordinates
            mumbai_locations['latitude'].append(loc['lat'])
            mumbai_locations['longitude'].append(loc['lon'])
            
            # Add demographic and density information
            mumbai_locations['primary_demographic'].append(loc['demo'])
            mumbai_locations['commercial_score'].append(loc['comm_score'])
            mumbai_locations['residential_density'].append(loc['res_density'])
            mumbai_locations['public_transport_accessibility'].append(loc['transport'])
        
        # Convert to DataFrame
        locations_df = pd.DataFrame(mumbai_locations)
        
        # Add some derived metrics
        locations_df['population_density'] = locations_df.apply(
            lambda x: x['population'] / (1000 if x['residential_density'] == 'High' 
                                    else 2000 if x['residential_density'] == 'Medium'
                                    else 500),
            axis=1
        )
        
        # Sort by tier and population
        locations_df = locations_df.sort_values(['tier', 'population'], ascending=[True, False])
        
        return locations_df

class LuxuryRetailDataGenerator(SmartRetailDataGenerator):
    def __init__(self, seed=42):
        super().__init__(seed)
        self.store_categories = {
            'Luxury_Automotive': {
                'brands': ['BMW', 'Mercedes-Benz', 'Audi', 'Lexus', 'Porsche'],
                'avg_transaction': (5000000, 15000000),
                'target_areas': ['High Income'],
                'required_space': 'Large',
                'parking_required': True
            },
            'Mid_Range_Automotive': {
                'brands': ['Honda', 'Toyota', 'Hyundai', 'Kia', 'Volkswagen'],
                'avg_transaction': (1000000, 3000000),
                'target_areas': ['Middle Income', 'High Income'],
                'required_space': 'Medium',
                'parking_required': True
            },
            'Budget_Automotive': {
                'brands': ['Maruti Suzuki', 'Tata', 'Renault', 'Nissan'],
                'avg_transaction': (400000, 800000),
                'target_areas': ['Middle Income', 'Lower Middle Income'],
                'required_space': 'Medium',
                'parking_required': True
            },
            'Luxury_Fashion': {
                'brands': ['Gucci', 'Louis Vuitton', 'Prada', 'Hermes'],
                'avg_transaction': (50000, 200000),
                'target_areas': ['High Income'],
                'required_space': 'Medium',
                'parking_required': True
            },
            'Premium_Electronics': {
                'brands': ['Apple', 'Samsung Premium', 'Sony'],
                'avg_transaction': (50000, 150000),
                'target_areas': ['High Income', 'Middle Income'],
                'required_space': 'Medium',
                'parking_required': False
            },
            'Designer_Jewelry': {
                'brands': ['Tanishq Premium', 'Cartier', 'Tiffany'],
                'avg_transaction': (200000, 1000000),
                'target_areas': ['High Income'],
                'required_space': 'Small',
                'parking_required': True
            },
            'Premium_Lifestyle': {
                'brands': ['Lifestyle Premium', 'Shoppers Stop Premium'],
                'avg_transaction': (10000, 50000),
                'target_areas': ['High Income', 'Middle Income'],
                'required_space': 'Large',
                'parking_required': True
            },
            'General_Retail': {
                'brands': ['Local Brands', 'Regional Chains'],
                'avg_transaction': (1000, 10000),
                'target_areas': ['Middle Income', 'Lower Middle Income'],
                'required_space': 'Medium',
                'parking_required': False
            }
        }

    def generate_enhanced_outlet_data(self, num_outlets):
        """Generate retail outlet data with luxury categories"""
        
        outlets = {
            'outlet_id': [f'OUT{i:03d}' for i in range(num_outlets)],
            'outlet_name': [],
            'category': [],
            'brand': [],
            'target_demographic': [],
            'avg_transaction_value': [],
            'outlet_size': [],
            'parking_available': [],
            'area_type': [],
            'location_tier': [],
            'monthly_footfall': [],
            'conversion_rate': [],
            'customer_satisfaction': [],
            'luxury_index': []
        }
        
        for i in range(num_outlets):
            # Select category with appropriate probabilities
            category = np.random.choice(list(self.store_categories.keys()), p=[0.1, 0.15, 0.15, 0.1, 0.15, 0.1, 0.15, 0.1])
            category_info = self.store_categories[category]
            
            # Select brand
            brand = np.random.choice(category_info['brands'])
            
            # Generate store name
            outlet_name = f"{brand} - {np.random.choice(['Mall', 'Premium', 'Exclusive'])} Store"
            
            # Generate metrics based on category
            is_luxury = category.startswith('Luxury') or category.startswith('Premium')
            
            avg_transaction = np.random.uniform(*category_info['avg_transaction'])
            footfall = np.random.normal(
                1000 if is_luxury else 3000,
                200 if is_luxury else 500
            )
            conversion = np.random.beta(
                8 if is_luxury else 5,
                2 if is_luxury else 5
            )
            satisfaction = np.random.normal(
                4.5 if is_luxury else 4.0,
                0.3 if is_luxury else 0.5
            )
            
            # Calculate luxury index (0-100)
            luxury_base = 90 if category.startswith('Luxury') else \
                         70 if category.startswith('Premium') else \
                         50 if category.startswith('Mid') else 30
            luxury_index = np.clip(
                np.random.normal(luxury_base, 5),
                0, 100
            )
            
            # Add to outlets dictionary
            outlets['outlet_name'].append(outlet_name)
            outlets['category'].append(category)
            outlets['brand'].append(brand)
            outlets['target_demographic'].append(category_info['target_areas'][0])
            outlets['avg_transaction_value'].append(avg_transaction)
            outlets['outlet_size'].append(category_info['required_space'])
            outlets['parking_available'].append(category_info['parking_required'])
            outlets['area_type'].append(np.random.choice(category_info['target_areas']))
            outlets['location_tier'].append(1 if is_luxury else np.random.choice([2, 3]))
            outlets['monthly_footfall'].append(int(footfall))
            outlets['conversion_rate'].append(conversion)
            outlets['customer_satisfaction'].append(satisfaction)
            outlets['luxury_index'].append(luxury_index)
            
        return pd.DataFrame(outlets)

    def generate_enhanced_transactions(self, customers_df, outlets_df, num_transactions):
        """Generate transaction data with luxury purchase patterns"""
        
        # Get base transactions
        base_transactions = super().generate_transactions(customers_df, outlets_df, num_transactions)
        
        # Add luxury-specific attributes
        enhanced_transactions = base_transactions.copy()
        
        # Add product category
        enhanced_transactions['product_category'] = np.random.choice(
            ['Vehicles', 'Fashion', 'Electronics', 'Jewelry', 'Lifestyle'],
            num_transactions
        )
        
        # Add premium features
        enhanced_transactions['is_premium_purchase'] = np.random.choice(
            [True, False],
            num_transactions,
            p=[0.3, 0.7]
        )
        
        # Add customer feedback
        enhanced_transactions['customer_rating'] = np.random.normal(4.2, 0.5, num_transactions)
        enhanced_transactions['customer_rating'] = enhanced_transactions['customer_rating'].clip(1, 5)
        
        return enhanced_transactions

class IntegratedRetailDataGenerator(LuxuryRetailDataGenerator):
    def __init__(self, seed=42):
        super().__init__(seed)
        
    def generate_integrated_outlets(self, locations_df, num_outlets):
        """Generate outlets with proper location distribution"""
        
        outlets = {
            'outlet_id': [f'OUT{i:03d}' for i in range(num_outlets)],
            'outlet_name': [],
            'category': [],
            'brand': [],
            'location_id': [],
            'area_name': [],
            'latitude': [],
            'longitude': [],
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
        
        # Distribute outlets across locations based on commercial score and demographics
        for i in range(num_outlets):
            # Select category with appropriate probabilities
            category = np.random.choice(
                list(self.store_categories.keys()), 
                p=[0.1, 0.15, 0.15, 0.1, 0.15, 0.1, 0.15, 0.1]
            )
            category_info = self.store_categories[category]
            
            # Filter suitable locations based on category requirements
            suitable_locations = locations_df[
                locations_df['primary_demographic'].isin(category_info['target_areas'])
            ]
            
            if len(suitable_locations) > 0:
                # Weight locations by commercial score for selection
                weights = suitable_locations['commercial_score']
                location = suitable_locations.sample(n=1, weights=weights).iloc[0]
            else:
                location = locations_df.sample(n=1).iloc[0]
            
            # Select brand and generate store name
            brand = np.random.choice(category_info['brands'])
            outlet_name = f"{brand} - {location['area_name']}"
            
            # Generate metrics based on category and location
            is_luxury = category.startswith('Luxury') or category.startswith('Premium')
            
            # Adjust metrics based on location characteristics
            location_modifier = (location['commercial_score'] + 
                               location['public_transport_accessibility']) / 2
            
            avg_transaction = np.random.uniform(*category_info['avg_transaction'])
            footfall = int(np.random.normal(
                1000 if is_luxury else 3000,
                200 if is_luxury else 500
            ) * location_modifier)
            
            
            conversion = np.random.beta(
                8 if is_luxury else 5,
                2 if is_luxury else 5
            ) * location_modifier
            
            satisfaction = np.clip(
                np.random.normal(
                    4.5 if is_luxury else 4.0,
                    0.3 if is_luxury else 0.5
                ) * location_modifier,
                1, 5
            )
            
            # Calculate luxury index adjusted by location
            luxury_base = 90 if category.startswith('Luxury') else \
                         70 if category.startswith('Premium') else \
                         50 if category.startswith('Mid') else 30
            luxury_index = np.clip(
                np.random.normal(luxury_base, 5) * location_modifier,
                0, 100
            )
            
            # Calculate commercial viability score
            commercial_viability = np.clip(
                (location['commercial_score'] * 0.4 +
                 location['public_transport_accessibility'] * 0.3 +
                 (1 if location['primary_demographic'] in category_info['target_areas'] else 0.5) * 0.3) * 100,
                0, 100
            )
            
            # Calculate competition score
            nearby_similar = len(outlets['category']) - len([
                c for c, loc in zip(outlets['category'], outlets['location_id'])
                if c == category and loc == location['location_id']
            ])
            competition_score = np.clip(100 - (nearby_similar * 10), 0, 100)
            
            # Add to outlets dictionary
            outlets['outlet_name'].append(outlet_name)
            outlets['category'].append(category)
            outlets['brand'].append(brand)
            outlets['location_id'].append(location['location_id'])
            outlets['area_name'].append(location['area_name'])
            outlets['latitude'].append(location['latitude'])
            outlets['longitude'].append(location['longitude'])
            outlets['target_demographic'].append(category_info['target_areas'][0])
            outlets['avg_transaction_value'].append(avg_transaction)
            outlets['outlet_size'].append(category_info['required_space'])
            outlets['parking_available'].append(category_info['parking_required'])
            outlets['area_type'].append(location['primary_demographic'])
            outlets['location_tier'].append(location['tier'])
            outlets['monthly_footfall'].append(footfall)
            outlets['conversion_rate'].append(conversion)
            outlets['customer_satisfaction'].append(satisfaction)
            outlets['luxury_index'].append(luxury_index)
            outlets['commercial_viability'].append(commercial_viability)
            outlets['competition_score'].append(competition_score)
        
        return pd.DataFrame(outlets)

    def generate_integrated_transactions(self, customers_df, outlets_df, num_transactions, start_date=None):
        """Generate transactions with location-based patterns"""
        if start_date is None:
            start_date = self.current_date - timedelta(days=365)
            
        transactions = {
            'transaction_id': [f'TRX{i:06d}' for i in range(num_transactions)],
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
        
        # Generate dates with seasonal patterns
        dates = []
        for _ in range(num_transactions):
            # Add seasonal weighting
            month_weights = [0.7, 0.7, 0.8, 0.9, 1.0, 0.8,  # Jan-Jun
                           0.7, 0.9, 1.0, 1.1, 1.2, 1.1]    # Jul-Dec
            target_month = np.random.choice(12, p=np.array(month_weights)/sum(month_weights))
            
            # Generate a valid date for the target month
            year = start_date.year + np.random.randint(0, 2)  # Either current or next year
            month = target_month + 1  # Months are 1-based
            
            # Get the last day of the target month
            if month == 12:
                next_month = datetime(year + 1, 1, 1)
            else:
                next_month = datetime(year, month + 1, 1)
            last_day = (next_month - timedelta(days=1)).day
            
            # Generate a random day within the valid range for this month
            day = np.random.randint(1, last_day + 1)
            target_date = datetime(year, month, day)
            dates.append(target_date)
        
        # Sort dates chronologically
        dates.sort()
        
        for date in dates:
            # Select customer based on demographics
            customer_weights = self._calculate_customer_purchase_probabilities(customers_df)
            customer = customers_df.sample(n=1, weights=customer_weights).iloc[0]
            
            # Select outlet based on customer demographics and location
            suitable_outlets = outlets_df[
                outlets_df['target_demographic'] == 
                ('High Income' if customer['income'] > 100000 else
                 'Middle Income' if customer['income'] > 50000 else
                 'Lower Middle Income')
            ]
            
            if len(suitable_outlets) > 0:
                outlet = suitable_outlets.sample(n=1).iloc[0]
            else:
                outlet = outlets_df.sample(n=1).iloc[0]
            
            # Generate transaction details
            hour = np.random.choice(
                24,
                p=self._generate_hourly_distribution()
            )
            
            is_premium = outlet['category'].startswith(('Luxury', 'Premium'))
            base_amount = outlet['avg_transaction_value'] * np.random.normal(1, 0.2)
            
            # Adjust amount based on customer income
            income_factor = np.clip(customer['income'] / 50000, 0.5, 2.0)
            final_amount = base_amount * income_factor
            
            # Add to transactions
            transactions['customer_id'].append(customer['customer_id'])
            transactions['outlet_id'].append(outlet['outlet_id'])
            transactions['location_id'].append(outlet['location_id'])
            transactions['transaction_date'].append(date.replace(hour=hour))
            transactions['purchase_amount'].append(final_amount)
            transactions['product_category'].append(outlet['category'])
            transactions['is_premium_purchase'].append(is_premium)
            transactions['payment_method'].append(np.random.choice(
                ['Credit Card', 'Debit Card', 'Cash', 'UPI', 'Wallet'],
                p=[0.4 if is_premium else 0.3,
                   0.3 if is_premium else 0.3,
                   0.1 if is_premium else 0.2,
                   0.15 if is_premium else 0.15,
                   0.05 if is_premium else 0.05]
            ))
            transactions['discount_applied'].append(
                np.random.beta(2, 8) if is_premium else np.random.beta(2, 5)
            )
            transactions['customer_rating'].append(
                np.clip(np.random.normal(
                    outlet['customer_satisfaction'],
                    0.5
                ), 1, 5)
            )
            transactions['weekend_flag'].append(1 if date.weekday() >= 5 else 0)
            transactions['time_of_day'].append(
                'Morning' if 5 <= hour < 12 else
                'Afternoon' if 12 <= hour < 17 else
                'Evening' if 17 <= hour < 21 else
                'Night'
            )
            transactions['season'].append(
                'Spring' if date.month in [3,4,5] else
                'Summer' if date.month in [6,7,8] else
                'Fall' if date.month in [9,10,11] else
                'Winter'
            )
        
        return pd.DataFrame(transactions)
# Example usage
if __name__ == "__main__":
    generator = IntegratedRetailDataGenerator()
    
    print("Generating Mumbai location data...")
    locations = generator.generate_mumbai_locations()
    
    print("Generating customer data...")
    customers = generator.generate_customer_demographics(1000)
    
    print("Generating integrated outlet data...")
    outlets = generator.generate_integrated_outlets(locations, 50)
    
    print("Generating integrated transaction data...")
    transactions = generator.generate_integrated_transactions(customers, outlets, 10000)
    
    # Save to CSV files
    customers.to_csv('customers.csv', index=False)
    outlets.to_csv('integrated_outlets.csv', index=False)
    transactions.to_csv('integrated_transactions.csv', index=False)
    locations.to_csv('mumbai_locations.csv', index=False)
    
    print("\nData generation complete! Files saved.")
    
    # Print analytical summaries
    print("\nStore Distribution by Area:")
    print(outlets.groupby('area_name')['category'].value_counts().unstack().fillna(0))
    
    print("\nAverage Transaction Value by Location Tier:")
    print(transactions.merge(outlets[['outlet_id', 'location_tier']], on='outlet_id')
          .groupby('location_tier')['purchase_amount'].mean())
    
    print("\nTop 5 Locations by Commercial Viability:")
    print(outlets.groupby('area_name')['commercial_viability'].mean()
          .sort_values(ascending=False).head())
    
    print("\nCustomer Satisfaction by Store Category:")
    print(outlets.groupby('category')['customer_satisfaction'].mean().sort_values(ascending=False))
