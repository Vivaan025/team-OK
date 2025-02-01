from typing import List, Dict
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

class RevenueSuggestionService:
    def __init__(self, mongodb_uri='mongodb://localhost:27017/'):
        self.client = MongoClient(mongodb_uri)
        self.db = self.client['indian_retail']
        
    def analyze_store_performance(self, store_id: str) -> Dict:
        """Generate comprehensive revenue improvement suggestions for a store"""
        # Get store data
        store = self.db.stores.find_one({'store_id': store_id})
        if not store:
            raise ValueError(f"Store {store_id} not found")
            
        # Get store transactions
        transactions = list(self.db.transactions.find({'store_id': store_id}))
        if not transactions:
            raise ValueError(f"No transactions found for store {store_id}")
            
        suggestions = {
            'product_suggestions': self._analyze_product_opportunities(store_id, transactions),
            'pricing_suggestions': self._analyze_pricing_opportunities(store_id, transactions),
            'inventory_suggestions': self._analyze_inventory_opportunities(store_id, transactions),
            'timing_suggestions': self._analyze_timing_opportunities(store_id, transactions),
            'customer_suggestions': self._analyze_customer_opportunities(store_id, transactions)
        }
        
        return suggestions
    
    def _analyze_product_opportunities(self, store_id: str, transactions: List) -> List[Dict]:
        """Analyze product-related opportunities"""
        suggestions = []
        
        # Extract all product IDs from transactions
        product_ids = set()
        for txn in transactions:
            for item in txn['items']:
                product_ids.add(item['product_id'])
                
        # Get product details
        products = list(self.db.products.find({
            'product_id': {'$in': list(product_ids)}
        }))
        product_map = {p['product_id']: p for p in products}
        
        # Analyze product performance
        product_metrics = {}
        for txn in transactions:
            for item in txn['items']:
                prod_id = item['product_id']
                if prod_id not in product_metrics:
                    product_metrics[prod_id] = {
                        'total_quantity': 0,
                        'total_revenue': 0,
                        'transactions': 0,
                        'avg_discount': []
                    }
                
                metrics = product_metrics[prod_id]
                metrics['total_quantity'] += item['quantity']
                metrics['total_revenue'] += item['total']
                metrics['transactions'] += 1
                metrics['avg_discount'].append(item.get('discount', 0))
        
        # Calculate key metrics
        for prod_id, metrics in product_metrics.items():
            metrics['avg_discount'] = np.mean(metrics['avg_discount'])
            metrics['avg_price'] = metrics['total_revenue'] / metrics['total_quantity']
            
        # Identify high-potential products
        high_revenue_products = sorted(
            product_metrics.items(),
            key=lambda x: x[1]['total_revenue'],
            reverse=True
        )[:5]
        
        for prod_id, metrics in high_revenue_products:
            product = product_map[prod_id]
            suggestions.append({
                'type': 'product_focus',
                'product_id': prod_id,
                'product_name': product['name'],
                'category': product['category'],
                'metrics': metrics,
                'suggestion': (
                    f"Focus on promoting {product['name']} which has shown strong "
                    f"performance with ₹{metrics['total_revenue']:,.2f} in revenue. "
                    f"Consider bundle deals with complementary products."
                ),
                'priority': 'High'
            })
        
        # Identify underperforming products
        low_revenue_products = sorted(
            product_metrics.items(),
            key=lambda x: x[1]['total_revenue']
        )[:5]
        
        for prod_id, metrics in low_revenue_products:
            product = product_map[prod_id]
            if metrics['total_revenue'] > 0:  # Only consider products with some sales
                suggestions.append({
                    'type': 'product_improvement',
                    'product_id': prod_id,
                    'product_name': product['name'],
                    'category': product['category'],
                    'metrics': metrics,
                    'suggestion': (
                        f"Consider repositioning {product['name']} through better "
                        f"placement or promotional offers. Current revenue is low at "
                        f"₹{metrics['total_revenue']:,.2f}."
                    ),
                    'priority': 'Medium'
                })
        
        return suggestions
    
    def _analyze_pricing_opportunities(self, store_id: str, transactions: List) -> List[Dict]:
        """Analyze pricing-related opportunities"""
        suggestions = []
        
        # Analyze price elasticity
        product_price_data = {}
        for txn in transactions:
            for item in txn['items']:
                prod_id = item['product_id']
                if prod_id not in product_price_data:
                    product_price_data[prod_id] = []
                
                product_price_data[prod_id].append({
                    'price': item['final_price'],
                    'quantity': item['quantity'],
                    'discount': item.get('discount', 0)
                })
        
        for prod_id, price_data in product_price_data.items():
            df = pd.DataFrame(price_data)
            if len(df) > 10:  # Only analyze products with sufficient data
                # Calculate price elasticity
                avg_price = df['price'].mean()
                avg_quantity = df['quantity'].mean()
                
                high_price_qty = df[df['price'] > avg_price]['quantity'].mean()
                low_price_qty = df[df['price'] < avg_price]['quantity'].mean()
                
                if high_price_qty and low_price_qty:
                    elasticity = (high_price_qty - low_price_qty) / low_price_qty
                    
                    if elasticity < -0.5:  # Price sensitive product
                        product = self.db.products.find_one({'product_id': prod_id})
                        suggestions.append({
                            'type': 'pricing_strategy',
                            'product_id': prod_id,
                            'product_name': product['name'],
                            'metrics': {
                                'elasticity': elasticity,
                                'avg_price': avg_price
                            },
                            'suggestion': (
                                f"Consider strategic price reductions for {product['name']} "
                                f"as it shows high price sensitivity (elasticity: {elasticity:.2f}). "
                                f"A 10% price reduction might increase sales volume significantly."
                            ),
                            'priority': 'High'
                        })
        
        return suggestions
    
    def _analyze_inventory_opportunities(self, store_id: str, transactions: List) -> List[Dict]:
        """Analyze inventory-related opportunities"""
        suggestions = []
        
        # Calculate inventory turnover
        product_sales = {}
        for txn in transactions:
            for item in txn['items']:
                prod_id = item['product_id']
                if prod_id not in product_sales:
                    product_sales[prod_id] = {
                        'total_quantity': 0,
                        'sales_dates': set()
                    }
                
                product_sales[prod_id]['total_quantity'] += item['quantity']
                product_sales[prod_id]['sales_dates'].add(
                    pd.to_datetime(txn['date']).date()
                )
        
        for prod_id, sales in product_sales.items():
            product = self.db.products.find_one({'product_id': prod_id})
            
            # Calculate days between first and last sale
            if len(sales['sales_dates']) > 1:
                date_range = (max(sales['sales_dates']) - min(sales['sales_dates'])).days
                daily_sales = sales['total_quantity'] / max(date_range, 1)
                
                if daily_sales < 0.5:  # Less than 1 sale every 2 days
                    suggestions.append({
                        'type': 'inventory_optimization',
                        'product_id': prod_id,
                        'product_name': product['name'],
                        'metrics': {
                            'daily_sales': daily_sales,
                            'total_quantity': sales['total_quantity']
                        },
                        'suggestion': (
                            f"Reduce inventory levels for {product['name']} due to low "
                            f"turnover (avg {daily_sales:.1f} units/day). Consider "
                            f"reallocating shelf space to faster-moving products."
                        ),
                        'priority': 'Medium'
                    })
        
        return suggestions
    
    def _analyze_timing_opportunities(self, store_id: str, transactions: List) -> List[Dict]:
        """Analyze timing-related opportunities"""
        suggestions = []
        
        # Analyze sales patterns by hour and day
        df = pd.DataFrame([
            {
                'date': pd.to_datetime(t['date']),
                'amount': t['payment']['final_amount']
            }
            for t in transactions
        ])
        
        df['hour'] = df['date'].dt.hour
        df['day'] = df['date'].dt.day_name()
        
        # Analyze hourly patterns
        hourly_sales = df.groupby('hour')['amount'].agg(['sum', 'count'])
        peak_hours = hourly_sales.nlargest(3, 'sum').index
        slow_hours = hourly_sales.nsmallest(3, 'sum').index
        
        suggestions.append({
            'type': 'timing_optimization',
            'metrics': {
                'peak_hours': peak_hours.tolist(),
                'slow_hours': slow_hours.tolist()
            },
            'suggestion': (
                f"Peak sales hours are {', '.join(map(str, peak_hours))}. "
                f"Consider running promotions during slower hours "
                f"({', '.join(map(str, slow_hours))}) to increase foot traffic."
            ),
            'priority': 'Medium'
        })
        
        return suggestions
    
    def _analyze_customer_opportunities(self, store_id: str, transactions: List) -> List[Dict]:
        """Analyze customer-related opportunities"""
        suggestions = []
        
        # Analyze customer segments
        customer_metrics = {}
        for txn in transactions:
            cust_id = txn['customer_id']
            if cust_id not in customer_metrics:
                customer_metrics[cust_id] = {
                    'total_spent': 0,
                    'visit_count': 0,
                    'last_visit': None
                }
            
            metrics = customer_metrics[cust_id]
            metrics['total_spent'] += txn['payment']['final_amount']
            metrics['visit_count'] += 1
            
            visit_date = pd.to_datetime(txn['date'])
            if not metrics['last_visit'] or visit_date > metrics['last_visit']:
                metrics['last_visit'] = visit_date
        
        # Identify high-value customers
        high_value_customers = sorted(
            customer_metrics.items(),
            key=lambda x: x[1]['total_spent'],
            reverse=True
        )[:10]
        
        if high_value_customers:
            avg_spend = np.mean([m['total_spent'] for _, m in high_value_customers])
            suggestions.append({
                'type': 'customer_retention',
                'metrics': {
                    'top_customer_avg_spend': avg_spend,
                    'top_customer_count': len(high_value_customers)
                },
                'suggestion': (
                    f"Focus on retaining top {len(high_value_customers)} customers "
                    f"who spend an average of ₹{avg_spend:,.2f}. Consider implementing "
                    f"a VIP program with exclusive benefits."
                ),
                'priority': 'High'
            })
        
        # Identify churning customers
        current_date = max(cm['last_visit'] for cm in customer_metrics.values())
        churning_customers = [
            (cid, cm) for cid, cm in customer_metrics.items()
            if (current_date - cm['last_visit']).days > 30
            and cm['visit_count'] > 5
        ]
        
        if churning_customers:
            suggestions.append({
                'type': 'customer_reactivation',
                'metrics': {
                    'churning_customer_count': len(churning_customers)
                },
                'suggestion': (
                    f"Target {len(churning_customers)} potentially churning customers "
                    f"who haven't visited in over 30 days. Consider sending personalized "
                    f"offers to re-engage them."
                ),
                'priority': 'High'
            })
        
        return suggestions

def main():
    # Example usage
    service = RevenueSuggestionService()
    store_id = "example_store_id"  # Replace with actual store ID
    
    try:
        suggestions = service.analyze_store_performance(store_id)
        
        print("\nRevenue Improvement Suggestions:")
        for category, category_suggestions in suggestions.items():
            print(f"\n{category.replace('_', ' ').title()}:")
            for suggestion in category_suggestions:
                print(f"\nPriority: {suggestion['priority']}")
                print(suggestion['suggestion'])
                
    except Exception as e:
        print(f"Error generating suggestions: {str(e)}")

if __name__ == "__main__":
    main()