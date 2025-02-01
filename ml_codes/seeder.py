from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timedelta
from models.response_models import RevenueSuggestion, ProductMetrics, StoreSuggestions

class RevenueSuggestionService:
    def __init__(self, mongodb_uri='mongodb://localhost:27017/'):
        self.client = MongoClient(mongodb_uri)
        self.db = self.client['indian_retail']
        
        # Category-specific configurations based on Indian retail context
        self.category_configs = {
            'Groceries': {
                'turnover_days': 7,  # Weekly turnover expected
                'margin_threshold': 0.15,  # 15% minimum margin
                'stock_threshold': 100,  # Minimum stock level
                'related_categories': ['Staples', 'Pulses', 'Rice', 'Atta & Flour', 
                                    'Masalas & Spices', 'Oil & Ghee', 'Dry Fruits'],
                'seasonal_factor': True,
                'festival_impact': 'high'
            },
            'Personal Care': {
                'turnover_days': 15,
                'margin_threshold': 0.25,
                'stock_threshold': 50,
                'related_categories': ['Soap & Body Wash', 'Hair Care', 'Oral Care', 
                                     'Skin Care', 'Cosmetics', 'Deodorants'],
                'seasonal_factor': False,
                'festival_impact': 'medium'
            },
            'Household': {
                'turnover_days': 30,
                'margin_threshold': 0.20,
                'stock_threshold': 75,
                'related_categories': ['Cleaning Supplies', 'Laundry', 'Kitchen Tools', 
                                     'Storage', 'Home Decor', 'Furnishing'],
                'seasonal_factor': True,
                'festival_impact': 'medium'
            },
            'Electronics': {
                'turnover_days': 45,
                'margin_threshold': 0.30,
                'stock_threshold': 25,
                'related_categories': ['Mobile Accessories', 'Batteries', 'Small Appliances', 
                                     'Computer Accessories', 'Audio', 'Cables & Chargers'],
                'seasonal_factor': False,
                'festival_impact': 'very_high'
            },
            'Fashion': {
                'turnover_days': 60,
                'margin_threshold': 0.40,
                'stock_threshold': 50,
                'related_categories': ["Men's Wear", "Women's Wear", "Kids' Wear", 
                                     'Footwear', 'Accessories', 'Traditional Wear'],
                'seasonal_factor': True,
                'festival_impact': 'high'
            }
        }

    async def analyze_product_opportunities(
        self,
        store_id: str,
        min_revenue: float = 0,
        category: Optional[str] = None
    ) -> List[RevenueSuggestion]:
        """Analyze product-specific revenue opportunities based on store and product categories"""
        store = self.db.stores.find_one({'store_id': store_id})
        if not store:
            raise ValueError(f"Store {store_id} not found")

        # Get store transactions for last 90 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        transactions = list(self.db.transactions.find({
            'store_id': store_id,
            'date': {'$gte': start_date, '$lte': end_date}
        }))

        # Calculate metrics with category awareness
        product_metrics = self._calculate_product_metrics(transactions)
        
        # Get category-specific products
        if category:
            product_metrics = {k: v for k, v in product_metrics.items() 
                             if self._get_product_category(k) == category}

        # Generate suggestions
        suggestions = []
        for product_id, metrics in product_metrics.items():
            if metrics.total_revenue < min_revenue:
                continue

            product = self.db.products.find_one({'product_id': product_id})
            if not product:
                continue

            product_category = product['category']
            category_config = self.category_configs.get(product_category, {})

            # Calculate potential improvement using category-specific factors
            potential_improvement = self._calculate_category_potential(
                metrics,
                product,
                category_config
            )

            suggestions.append(
                RevenueSuggestion(
                    type="product_optimization",
                    suggestion=self._generate_category_suggestion(
                        product,
                        metrics,
                        category_config
                    ),
                    priority=self._determine_priority(
                        potential_improvement,
                        product_category
                    ),
                    metrics=metrics.dict(),
                    impact_estimate=potential_improvement,
                    implementation_difficulty=self._assess_difficulty(
                        product,
                        category_config
                    ),
                    timeframe=self._estimate_timeframe(
                        product,
                        category_config
                    )
                )
            )

        return sorted(suggestions, key=lambda x: x.impact_estimate or 0, reverse=True)

    def _calculate_product_metrics(self, transactions: List) -> Dict[str, ProductMetrics]:
        """Calculate detailed product metrics with category context"""
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
                m.total_revenue += item['final_price'] * item['quantity']
                m.transactions += 1
                m.prices.append(item['final_price'])
        
        # Calculate derived metrics
        for prod_id, m in metrics.items():
            product = self.db.products.find_one({'product_id': prod_id})
            if product:
                m.avg_price = float(np.mean(m.prices))
                m.price_variance = float(np.var(m.prices))
                m.revenue_per_transaction = float(m.total_revenue / m.transactions)
                m.margin = (m.avg_price - product['pricing']['base_price']) / m.avg_price
                
                # Calculate category-specific metrics
                category_config = self.category_configs.get(product['category'], {})
                m.turnover_rate = m.total_quantity / (product['inventory']['total_stock'] or 1)
                m.stock_days = product['inventory']['total_stock'] / (m.total_quantity / 90 or 1)
                m.margin_gap = category_config.get('margin_threshold', 0.2) - m.margin
        
        return metrics

    def _calculate_category_potential(
        self,
        metrics: ProductMetrics,
        product: Dict,
        category_config: Dict
    ) -> float:
        """Calculate improvement potential using category-specific factors"""
        base_potential = metrics.total_revenue * 0.2
        multiplier = 1.0

        # Category-specific adjustments
        if metrics.stock_days > category_config.get('turnover_days', 30):
            multiplier += 0.3  # High potential in improving turnover

        if metrics.margin_gap > 0:
            multiplier += 0.2  # Margin improvement opportunity

        # Festival impact
        festival_impact = category_config.get('festival_impact', 'medium')
        if festival_impact == 'very_high':
            multiplier += 0.4
        elif festival_impact == 'high':
            multiplier += 0.3
        elif festival_impact == 'medium':
            multiplier += 0.2

        # Seasonal factor
        if category_config.get('seasonal_factor', False):
            current_month = datetime.now().month
            if self._is_peak_season(product['category'], current_month):
                multiplier += 0.3

        return base_potential * multiplier

    def _generate_category_suggestion(
        self,
        product: Dict,
        metrics: ProductMetrics,
        category_config: Dict
    ) -> str:
        """Generate category-specific improvement suggestions"""
        suggestions = []
        
        # Stock optimization suggestions
        if metrics.stock_days > category_config.get('turnover_days', 30):
            suggestions.append(
                f"Optimize stock levels for {product['name']}. "
                f"Current stock duration ({metrics.stock_days:.1f} days) exceeds "
                f"category target ({category_config.get('turnover_days')} days)."
            )

        # Margin improvement suggestions
        if metrics.margin_gap > 0:
            suggestions.append(
                f"Consider price optimization to achieve target margin of "
                f"{category_config.get('margin_threshold')*100:.1f}%. "
                f"Current margin is {metrics.margin*100:.1f}%."
            )

        # Seasonal suggestions
        if category_config.get('seasonal_factor', False):
            current_month = datetime.now().month
            if self._is_peak_season(product['category'], current_month):
                suggestions.append(
                    f"Capitalize on peak season for {product['subcategory']} "
                    f"with targeted promotions and optimal stock levels."
                )

        # Bundle suggestions
        if metrics.revenue_per_transaction < product['pricing']['base_price'] * 2:
            related_cats = category_config.get('related_categories', [])
            if related_cats:
                suggestions.append(
                    f"Create bundle offers with {', '.join(related_cats[:2])} "
                    f"products to increase transaction value."
                )

        return " ".join(suggestions) if suggestions else (
            f"Monitor {product['name']} performance for category-specific "
            f"optimization opportunities."
        )

    def _is_peak_season(self, category: str, month: int) -> bool:
        """Determine if current month is peak season for category"""
        peak_seasons = {
            'Fashion': [3, 4, 10, 11],  # Spring and Festival seasons
            'Electronics': [10, 11, 12],  # Festival and Year-end
            'Groceries': [10, 11, 12],   # Festival season
            'Household': [3, 4, 10, 11]  # Spring cleaning and Festival
        }
        return month in peak_seasons.get(category, [])

    def _determine_priority(self, potential_improvement: float, category: str) -> str:
        """Determine priority based on category-specific thresholds"""
        thresholds = {
            'Groceries': {'high': 50000, 'medium': 25000},
            'Personal Care': {'high': 75000, 'medium': 35000},
            'Household': {'high': 100000, 'medium': 50000},
            'Electronics': {'high': 150000, 'medium': 75000},
            'Fashion': {'high': 100000, 'medium': 50000}
        }
        
        category_threshold = thresholds.get(category, {'high': 100000, 'medium': 50000})
        
        if potential_improvement > category_threshold['high']:
            return "High"
        elif potential_improvement > category_threshold['medium']:
            return "Medium"
        return "Low"

    def _assess_difficulty(self, product: Dict, category_config: Dict) -> str:
        """Assess implementation difficulty based on category and product factors"""
        if product.get('inventory', {}).get('total_stock', 0) < category_config.get('stock_threshold', 50):
            return "High"  # Low stock makes changes harder
        if category_config.get('seasonal_factor', False):
            return "Medium"  # Seasonal products require careful timing
        return "Low"

    def _estimate_timeframe(self, product: Dict, category_config: Dict) -> str:
        """Estimate implementation timeframe based on category and product factors"""
        if category_config.get('seasonal_factor', False):
            return "1-3 months"  # Seasonal changes need planning
        if product.get('inventory', {}).get('total_stock', 0) > category_config.get('stock_threshold', 50):
            return "2-4 weeks"  # Good stock levels allow faster changes
        return "1-2 months"  # Default timeframe

    def _get_product_category(self, product_id: str) -> Optional[str]:
        """Get product category from database"""
        product = self.db.products.find_one({'product_id': product_id})
        return product.get('category') if product else None