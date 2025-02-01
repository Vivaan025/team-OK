from typing import List, Dict
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from models.response_models import RevenueSuggestion, ProductMetrics, StoreSuggestions
from typing import Optional

class RevenueSuggestionService:
    def __init__(self, mongodb_uri='mongodb://localhost:27017/'):
        self.client = MongoClient(mongodb_uri)
        self.db = self.client['indian_retail']
        self.store_thresholds = {
            'urban': {
                'high_revenue': 100000,
                'medium_revenue': 50000,
                'min_transactions': 20,
                'stock_alert': 0.7,  # 70% of avg daily sales
                'peak_hour_multiplier': 1.5
            },
            'suburban': {
                'high_revenue': 75000,
                'medium_revenue': 35000,
                'min_transactions': 15,
                'stock_alert': 0.8,
                'peak_hour_multiplier': 1.3
            },
            'rural': {
                'high_revenue': 50000,
                'medium_revenue': 25000,
                'min_transactions': 10,
                'stock_alert': 0.9,
                'peak_hour_multiplier': 1.2
            }
        }

    
    async def analyze_product_opportunities(
        self,
        store_id: str,
        min_revenue: float = 0,
        category: Optional[str] = None
    ) -> List[RevenueSuggestion]:
        """Analyze product-specific revenue opportunities"""
        store = self.db.stores.find_one({'store_id': store_id})
        if not store:
            raise ValueError(f"Store {store_id} not found")

        # Get store transactions
        transactions = list(self.db.transactions.find({'store_id': store_id}))
        
        # Calculate metrics
        product_metrics = self._calculate_product_metrics(transactions)
        
        # Filter by category and revenue
        filtered_metrics = {
            k: v for k, v in product_metrics.items()
            if (not category or self._get_product_category(k) == category) and
               v.total_revenue >= min_revenue
        }
        
        # Generate suggestions
        suggestions = []
        for product_id, metrics in filtered_metrics.items():
            product = self.db.products.find_one({'product_id': product_id})
            potential_improvement = self._calculate_potential_improvement(metrics)
            
            suggestions.append(
                RevenueSuggestion(
                    type="product_optimization",
                    suggestion=self._generate_product_suggestion(product, metrics),
                    priority=self._determine_priority(potential_improvement),
                    metrics=metrics.dict(),
                    impact_estimate=potential_improvement,
                    implementation_difficulty="Medium",
                    timeframe="1-3 months"
                )
            )
        
        return sorted(suggestions, key=lambda x: x.impact_estimate or 0, reverse=True)
    
        
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


    def _calculate_product_metrics(self, transactions: List) -> Dict[str, ProductMetrics]:
        """Calculate detailed product metrics"""
        metrics = {}
        for txn in transactions:
            for item in txn['items']:
                prod_id = item['product_id']
                if prod_id not in metrics:
                    metrics[prod_id] = ProductMetrics(
                        total_quantity=0,
                        total_revenue=0,
                        transactions=0,
                        prices=[]
                    )
                
                m = metrics[prod_id]
                m.total_quantity += item['quantity']
                m.total_revenue += item['total']
                m.transactions += 1
                m.prices.append(item['final_price'])
        
        # Calculate additional metrics
        for m in metrics.values():
            m.avg_price = np.mean(m.prices)
            m.price_variance = np.var(m.prices)
            m.revenue_per_transaction = m.total_revenue / m.transactions
        
        return metrics

    def _calculate_potential_improvement(self, metrics: ProductMetrics) -> float:
        """Calculate potential revenue improvement based on metrics"""
        # Base improvement potential (20% of current revenue)
        base_potential = metrics.total_revenue * 0.2
        
        # Adjust based on various factors
        multiplier = 1.0
        
        # Low transaction count suggests growth potential
        if metrics.transactions < 10:
            multiplier += 0.3
        
        # High price variance suggests pricing optimization potential
        if metrics.price_variance and metrics.price_variance > 0.1:
            multiplier += 0.2
        
        # Low revenue per transaction suggests upselling potential
        if metrics.revenue_per_transaction < 100:
            multiplier += 0.2
            
        return base_potential * multiplier

    def _get_product_category(self, product_id: str) -> Optional[str]:
        """Get product category from database"""
        product = self.db.products.find_one({'product_id': product_id})
        return product.get('category') if product else None

    def _determine_priority(self, potential_improvement: float) -> str:
        """Determine priority based on potential improvement value"""
        if potential_improvement > 100000:  # > 1 lakh
            return "High"
        elif potential_improvement > 50000:  # > 50k
            return "Medium"
        else:
            return "Low"

    def _generate_product_suggestion(self, product: Dict, metrics: ProductMetrics) -> str:
        """Generate specific improvement suggestion for a product"""
        suggestions = []
        
        # Low transaction suggestions
        if metrics.transactions < 10:
            suggestions.append(
                "Increase visibility through better placement and targeted marketing."
            )
        
        # Price variance suggestions
        if metrics.price_variance and metrics.price_variance > 0.1:
            suggestions.append(
                "Standardize pricing strategy to optimize revenue. Current pricing shows high variability."
            )
        
        # Low revenue per transaction suggestions
        if metrics.revenue_per_transaction < 100:
            suggestions.append(
                "Create bundle offers with complementary products to increase transaction value."
            )
        
        # Low quantity suggestions
        if metrics.total_quantity < 50:
            suggestions.append(
                "Consider bulk purchase incentives to increase sales volume."
            )

        # Combine suggestions
        if suggestions:
            return f"For {product['name']}: " + " ".join(suggestions)
        else:
            return f"Monitor {product['name']} performance for optimization opportunities."

    async def analyze_product_opportunities(
        self,
        store_id: str,
        min_revenue: float = 0,
        category: Optional[str] = None
    ) -> List[RevenueSuggestion]:
        """Analyze product-specific revenue opportunities"""
        # Validate store
        store = self.db.stores.find_one({'store_id': store_id})
        if not store:
            raise ValueError(f"Store {store_id} not found")

        # Get transactions
        transactions = list(self.db.transactions.find({'store_id': store_id}))
        if not transactions:
            raise ValueError(f"No transactions found for store {store_id}")

        # Calculate metrics
        product_metrics = self._calculate_product_metrics(transactions)
        
        # Filter by category and revenue
        filtered_metrics = {}
        for prod_id, metrics in product_metrics.items():
            if metrics.total_revenue < min_revenue:
                continue
                
            if category:
                prod_category = self._get_product_category(prod_id)
                if prod_category != category:
                    continue
                    
            filtered_metrics[prod_id] = metrics

        # Generate suggestions
        suggestions = []
        for product_id, metrics in filtered_metrics.items():
            product = self.db.products.find_one({'product_id': product_id})
            if not product:
                continue
                
            potential_improvement = self._calculate_potential_improvement(metrics)
            
            # Convert metrics to dict for JSON serialization
            metrics_dict = {
                'total_quantity': metrics.total_quantity,
                'total_revenue': metrics.total_revenue,
                'transactions': metrics.transactions,
                'avg_price': metrics.avg_price,
                'price_variance': metrics.price_variance,
                'revenue_per_transaction': metrics.revenue_per_transaction
            }
            
            suggestion = RevenueSuggestion(
                type="product_optimization",
                suggestion=self._generate_product_suggestion(product, metrics),
                priority=self._determine_priority(potential_improvement),
                metrics=metrics_dict,
                impact_estimate=potential_improvement,
                implementation_difficulty="Medium",
                timeframe="1-3 months"
            )
            suggestions.append(suggestion)

        # Sort by impact estimate
        return sorted(suggestions, key=lambda x: x.impact_estimate or 0, reverse=True)

    def _calculate_product_metrics(self, transactions: List) -> Dict[str, ProductMetrics]:
        """Calculate detailed metrics for each product"""
        metrics = {}
        for txn in transactions:
            for item in txn['items']:
                prod_id = item['product_id']
                if prod_id not in metrics:
                    metrics[prod_id] = ProductMetrics(
                        total_quantity=0,
                        total_revenue=0,
                        transactions=0,
                        prices=[]
                    )
                
                m = metrics[prod_id]
                m.total_quantity += item['quantity']
                m.total_revenue += item['total']
                m.transactions += 1
                m.prices.append(item['final_price'])
        
        # Calculate derived metrics
        for m in metrics.values():
            if m.prices:
                m.avg_price = float(np.mean(m.prices))
                m.price_variance = float(np.var(m.prices))
            if m.transactions > 0:
                m.revenue_per_transaction = float(m.total_revenue / m.transactions)
        
        return metrics

    async def analyze_store_performance(self, store_id: str, days: int = 90) -> StoreSuggestions:
        """
        Generate comprehensive store performance analysis
        
        Args:
            store_id: ID of the store to analyze
            days: Number of days of history to analyze (default: 90)
        
        Returns:
            StoreSuggestions: Comprehensive analysis of store performance
        """
        # Validate store
        store = self.db.stores.find_one({'store_id': store_id})
        if not store:
            raise ValueError(f"Store {store_id} not found")
        
        # First get all transactions for the store
        transactions = list(self.db.transactions.find({'store_id': store_id}))
        
        if not transactions:
            raise ValueError(f"No transactions found for store {store_id}")
            
        # Convert dates and sort transactions
        for txn in transactions:
            txn['date'] = pd.to_datetime(txn['date'])
        
        # Sort by date
        transactions.sort(key=lambda x: x['date'])
        
        # If we have transactions, filter by date range if specified
        if days:
            latest_date = max(txn['date'] for txn in transactions)
            start_date = latest_date - timedelta(days=days)
            transactions = [
                txn for txn in transactions 
                if txn['date'] >= start_date
            ]
        
        if not transactions:
            raise ValueError(f"No transactions found for store {store_id} in the last {days} days")

        # Analyze product opportunities
        product_suggestions = await self.analyze_product_opportunities(store_id)

        # Analyze pricing opportunities
        pricing_suggestions = []
        price_metrics = self._calculate_product_metrics(transactions)
        for prod_id, metrics in price_metrics.items():
            if metrics.price_variance and metrics.price_variance > 0.1:
                product = self.db.products.find_one({'product_id': prod_id})
                if product:
                    pricing_suggestions.append(
                        RevenueSuggestion(
                            type='pricing_optimization',
                            suggestion=f"Optimize pricing for {product['name']} - high variance detected",
                            priority='High' if metrics.total_revenue > 50000 else 'Medium',
                            metrics={
                                'price_variance': metrics.price_variance,
                                'avg_price': metrics.avg_price,
                                'total_revenue': metrics.total_revenue
                            },
                            impact_estimate=metrics.total_revenue * 0.1,
                            implementation_difficulty='Medium',
                            timeframe='1 month'
                        )
                    )

        # Analyze inventory opportunities
        inventory_suggestions = []
        for prod_id, metrics in price_metrics.items():
            if metrics.total_quantity > 0:
                daily_sales = metrics.total_quantity / days
                if daily_sales < 0.5:  # Less than 1 sale every 2 days
                    product = self.db.products.find_one({'product_id': prod_id})
                    if product:
                        inventory_suggestions.append(
                            RevenueSuggestion(
                                type='inventory_optimization',
                                suggestion=f"Reduce inventory for {product['name']} - slow moving item",
                                priority='Medium',
                                metrics={
                                    'daily_sales': daily_sales,
                                    'total_quantity': metrics.total_quantity
                                },
                                impact_estimate=metrics.total_revenue * 0.05,
                                implementation_difficulty='Low',
                                timeframe='2 weeks'
                            )
                        )

        # Analyze timing patterns
        df = pd.DataFrame([{
            'date': pd.to_datetime(t['date']),
            'amount': t['payment']['final_amount']
        } for t in transactions])
        
        timing_suggestions = []
        if not df.empty:
            df['hour'] = df['date'].dt.hour
            hourly_sales = df.groupby('hour')['amount'].agg(['sum', 'count'])
            peak_hours = hourly_sales.nlargest(3, 'sum').index
            slow_hours = hourly_sales.nsmallest(3, 'sum').index
            
            timing_suggestions.append(
                RevenueSuggestion(
                    type='timing_optimization',
                    suggestion=(
                        f"Peak sales hours are {', '.join(map(str, peak_hours))}. "
                        f"Consider promotions during slower hours ({', '.join(map(str, slow_hours))})"
                    ),
                    priority='Medium',
                    metrics={
                        'peak_hours': peak_hours.tolist(),
                        'slow_hours': slow_hours.tolist()
                    },
                    impact_estimate=df['amount'].sum() * 0.05,
                    implementation_difficulty='Medium',
                    timeframe='1 month'
                )
            )

        # Analyze customer patterns
        customer_suggestions = []
        customer_data = {}
        for txn in transactions:
            cust_id = txn['customer_id']
            if cust_id not in customer_data:
                customer_data[cust_id] = {
                    'total_spent': 0,
                    'visits': 0,
                    'last_visit': None
                }
            
            data = customer_data[cust_id]
            data['total_spent'] += txn['payment']['final_amount']
            data['visits'] += 1
            visit_date = pd.to_datetime(txn['date'])
            if not data['last_visit'] or visit_date > data['last_visit']:
                data['last_visit'] = visit_date

        # Identify customer segments
        if customer_data:
            high_value_customers = sorted(
                customer_data.items(),
                key=lambda x: x[1]['total_spent'],
                reverse=True
            )[:10]
            
            avg_high_value_spend = np.mean([c[1]['total_spent'] for c in high_value_customers])
            customer_suggestions.append(
                RevenueSuggestion(
                    type='customer_retention',
                    suggestion=(
                        f"Focus on retaining top {len(high_value_customers)} customers "
                        f"who spend an average of ₹{avg_high_value_spend:,.2f}"
                    ),
                    priority='High',
                    metrics={
                        'high_value_customers': len(high_value_customers),
                        'avg_spend': avg_high_value_spend
                    },
                    impact_estimate=avg_high_value_spend * len(high_value_customers) * 0.2,
                    implementation_difficulty='Medium',
                    timeframe='3 months'
                )
            )

        # Return StoreSuggestions with analyzed data
        return StoreSuggestions(
            product_suggestions=product_suggestions,
            pricing_suggestions=pricing_suggestions,
            inventory_suggestions=inventory_suggestions,
            timing_suggestions=timing_suggestions,
            customer_suggestions=customer_suggestions
        )

    async def analyze_pricing_opportunities(
        self,
        store_id: str,
        min_revenue: float = 0,
        min_price_variance: float = 0.1
    ) -> List[RevenueSuggestion]:
        """
        Analyze pricing-related opportunities for products
        
        Args:
            store_id: ID of the store to analyze
            min_revenue: Minimum revenue threshold for products to analyze
            min_price_variance: Minimum price variance to flag for optimization
            
        Returns:
            List[RevenueSuggestion]: List of pricing-related suggestions
        """
        # Validate store
        store = self.db.stores.find_one({'store_id': store_id})
        if not store:
            raise ValueError(f"Store {store_id} not found")

        # Get store transactions
        transactions = list(self.db.transactions.find({'store_id': store_id}))
        if not transactions:
            return []

        # Analyze price elasticity and variance
        product_price_data = {}
        for txn in transactions:
            for item in txn['items']:
                prod_id = item['product_id']
                if prod_id not in product_price_data:
                    product_price_data[prod_id] = {
                        'prices': [],
                        'quantities': [],
                        'total_revenue': 0,
                        'dates': []
                    }
                
                data = product_price_data[prod_id]
                data['prices'].append(item['final_price'])
                data['quantities'].append(item['quantity'])
                data['total_revenue'] += item['final_price'] * item['quantity']
                data['dates'].append(pd.to_datetime(txn['date']))

        suggestions = []
        location_type = store.get('location_type', 'urban')
        thresholds = self.store_thresholds[location_type]

        for prod_id, data in product_price_data.items():
            if data['total_revenue'] < min_revenue:
                continue

            df = pd.DataFrame({
                'price': data['prices'],
                'quantity': data['quantities'],
                'date': data['dates']
            })

            if len(df) > 10:  # Only analyze products with sufficient data
                product = self.db.products.find_one({'product_id': prod_id})
                if not product:
                    continue

                # Calculate price statistics
                avg_price = df['price'].mean()
                price_variance = df['price'].var()
                
                # Calculate price elasticity
                high_price_qty = df[df['price'] > avg_price]['quantity'].mean() or 0
                low_price_qty = df[df['price'] < avg_price]['quantity'].mean() or 0
                
                metrics = {
                    'avg_price': float(avg_price),
                    'price_variance': float(price_variance),
                    'total_revenue': float(data['total_revenue']),
                    'transaction_count': len(df),
                    'date_range': (max(data['dates']) - min(data['dates'])).days
                }

                # Generate suggestions based on analysis
                if price_variance > min_price_variance:
                    potential_improvement = data['total_revenue'] * 0.1  # 10% improvement potential
                    
                    # Adjust potential improvement based on location
                    if location_type == 'urban':
                        potential_improvement *= 1.2  # 20% higher potential in urban areas
                    elif location_type == 'rural':
                        potential_improvement *= 0.8  # 20% lower potential in rural areas

                    suggestion_text = (
                        f"Optimize pricing strategy for {product['name']}. "
                        f"Current price variance is {price_variance:.2f} with "
                        f"average price of ₹{avg_price:.2f}. "
                    )

                    # Add elasticity analysis if available
                    if high_price_qty > 0 and low_price_qty > 0:
                        elasticity = (high_price_qty - low_price_qty) / low_price_qty
                        metrics['elasticity'] = float(elasticity)
                        
                        if elasticity < -0.5:  # Price sensitive product
                            suggestion_text += (
                                f"Product shows high price sensitivity "
                                f"(elasticity: {elasticity:.2f}). Consider targeted "
                                f"price reductions during off-peak hours."
                            )
                        else:
                            suggestion_text += (
                                f"Product shows low price sensitivity. "
                                f"Consider premium pricing during peak hours."
                            )

                    suggestions.append(
                        RevenueSuggestion(
                            type='pricing_optimization',
                            suggestion=suggestion_text,
                            priority=self._determine_priority(potential_improvement),
                            metrics=metrics,
                            impact_estimate=potential_improvement,
                            implementation_difficulty='Medium',
                            timeframe='1 month'
                        )
                    )

        # Sort suggestions by impact estimate
        return sorted(suggestions, key=lambda x: x.impact_estimate or 0, reverse=True)