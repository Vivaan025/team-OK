from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pymongo import MongoClient
from models.comparison import MetricComparison, StoreComparisonResponse, TransactionMetrics, ProductMetrics, CustomerMetrics, OperationalMetrics
from typing import Any

class StoreComparisonService:
    def __init__(self, mongodb_uri='mongodb://localhost:27017/'):
        self.client = MongoClient(mongodb_uri)
        self.db = self.client['indian_retail']

    async def compare_stores(
        self,
        store1_id: str,
        store2_id: str,
        days: int = 90
    ) -> StoreComparisonResponse:
        """Compare two stores across multiple metrics"""
        # Validate stores
        store1 = self.db.stores.find_one({'store_id': store1_id})
        store2 = self.db.stores.find_one({'store_id': store2_id})
        
        if not store1 or not store2:
            raise ValueError("One or both stores not found")

        # Get time period for analysis
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Get store transactions
        transactions1 = list(self.db.transactions.find({
            'store_id': store1_id,
            'date': {'$gte': start_date, '$lte': end_date}
        }))
        
        transactions2 = list(self.db.transactions.find({
            'store_id': store2_id,
            'date': {'$gte': start_date, '$lte': end_date}
        }))

        # Calculate all metrics
        revenue_comparison = self._compare_revenue_metrics(transactions1, transactions2)
        transaction_metrics = self._compare_transaction_metrics(transactions1, transactions2)
        product_metrics = self._compare_product_metrics(store1_id, store2_id, transactions1, transactions2)
        customer_metrics = self._compare_customer_metrics(store1_id, store2_id, transactions1, transactions2)
        operational_metrics = self._compare_operational_metrics(store1, store2)

        # Generate overall comparison
        overall_scores = self._calculate_overall_scores(
            revenue_comparison,
            transaction_metrics,
            product_metrics,
            customer_metrics
        )

        return StoreComparisonResponse(
            store1_id=store1_id,
            store2_id=store2_id,
            store1_name=store1['name'],
            store2_name=store2['name'],
            time_period=f"Last {days} days",
            
            revenue_comparison=revenue_comparison,
            profit_comparison=self._calculate_profit_comparison(transactions1, transactions2),
            
            transaction_metrics_1=transaction_metrics[0],
            transaction_metrics_2=transaction_metrics[1],
            transaction_comparison=self._create_metric_comparison(
                transaction_metrics[0].total_transactions,
                transaction_metrics[1].total_transactions,
                "transactions"
            ),
            
            product_metrics_1=product_metrics[0],
            product_metrics_2=product_metrics[1],
            product_comparison=self._compare_category_performance(
                product_metrics[0].category_performance,
                product_metrics[1].category_performance
            ),
            
            customer_metrics_1=customer_metrics[0],
            customer_metrics_2=customer_metrics[1],
            customer_comparison=self._create_customer_comparisons(
                customer_metrics[0],
                customer_metrics[1]
            ),
            
            operational_metrics_1=operational_metrics[0],
            operational_metrics_2=operational_metrics[1],
            unique_advantages_1=self._identify_unique_advantages(store1, store2),
            unique_advantages_2=self._identify_unique_advantages(store2, store1),
            
            overall_score_1=overall_scores[0],
            overall_score_2=overall_scores[1],
            key_insights=self._generate_key_insights(
                store1, store2,
                revenue_comparison,
                transaction_metrics,
                product_metrics,
                customer_metrics
            ),
            improvement_suggestions=self._generate_improvement_suggestions(
                store1, store2,
                revenue_comparison,
                transaction_metrics,
                product_metrics,
                customer_metrics
            )
        )

    def _create_metric_comparison(
        self,
        value1: float,
        value2: float,
        metric_type: str,
        context: Optional[Dict] = None
    ) -> MetricComparison:
        """
        Create a standardized comparison between two metric values.
        
        Args:
            value1: Value from first store
            value2: Value from second store
            metric_type: Type of metric being compared (e.g., 'transactions', 'revenue', 'customers')
            context: Optional additional context for insight generation
            
        Returns:
            MetricComparison: Standardized comparison with insights
        """
        # Handle division by zero and percentage calculation
        difference = value1 - value2
        percentage_difference = (
            ((value1 - value2) / value2 * 100) if value2 != 0
            else float('inf') if value1 > 0
            else float('-inf') if value1 < 0
            else 0
        )

        # Determine better performer
        better_performer = (
            "store1" if value1 > value2
            else "store2" if value2 > value1
            else "equal"
        )

        # Generate appropriate insight based on metric type
        insight = self._generate_metric_insight(
            value1,
            value2,
            metric_type,
            percentage_difference,
            context
        )

        return MetricComparison(
            store1_value=value1,
            store2_value=value2,
            difference=difference,
            percentage_difference=percentage_difference,
            better_performer=better_performer,
            insight=insight
        )

    def _generate_metric_insight(
        self,
        value1: float,
        value2: float,
        metric_type: str,
        percentage_difference: float,
        context: Optional[Dict] = None
    ) -> str:
        """Generate insight for specific metric comparison"""
        insights = []
        better_store = "first" if value1 > value2 else "second"
        
        # Format values based on metric type
        def format_value(value: float, metric_type: str) -> str:
            if metric_type == 'revenue':
                if value >= 10000000:  # 1 crore
                    return f"₹{value/10000000:.2f} crore"
                elif value >= 100000:  # 1 lakh
                    return f"₹{value/100000:.2f} lakh"
                return f"₹{value:,.2f}"
            elif metric_type in ['transactions', 'customers']:
                return f"{int(value):,}"
            elif metric_type in ['margin', 'rate', 'percentage']:
                return f"{value:.1f}%"
            return f"{value:,.2f}"

        # Base comparison insight
        if value1 == value2:
            insights.append(
                f"Both stores show identical performance with "
                f"{format_value(value1, metric_type)} {metric_type}"
            )
        else:
            insights.append(
                f"The {better_store} store performs better in {metric_type} with "
                f"{format_value(max(value1, value2), metric_type)} "
                f"({abs(percentage_difference):.1f}% difference)"
            )

        # Add metric-specific insights
        if metric_type == 'transactions':
            daily_diff = abs(value1 - value2) / 90  # Assuming 90 days
            if daily_diff >= 10:
                insights.append(
                    f"This represents a difference of {int(daily_diff)} "
                    f"transactions per day"
                )

        elif metric_type == 'revenue':
            if abs(percentage_difference) > 20:
                insights.append(
                    "This indicates a significant revenue performance gap"
                )
            elif abs(percentage_difference) > 10:
                insights.append(
                    "This shows a moderate revenue variation"
                )

        elif metric_type == 'customers':
            repeat_rate1 = context.get('repeat_rate1', 0) if context else 0
            repeat_rate2 = context.get('repeat_rate2', 0) if context else 0
            
            if abs(repeat_rate1 - repeat_rate2) > 0.1:  # 10% difference in repeat rates
                better_repeat = "first" if repeat_rate1 > repeat_rate2 else "second"
                insights.append(
                    f"The {better_repeat} store has a higher customer repeat rate "
                    f"({max(repeat_rate1, repeat_rate2):.1%} vs {min(repeat_rate1, repeat_rate2):.1%})"
                )

        elif metric_type == 'margin':
            if abs(percentage_difference) > 5:  # 5% difference in margins
                insights.append(
                    f"The {better_store} store maintains better profitability"
                )

        # Add trend insight if context provides it
        if context and 'trend' in context:
            trend = context['trend']
            insights.append(
                f"The trend shows {trend} performance over the analyzed period"
            )

        return " ".join(insights)
    
    def _compare_category_performance(
        self,
        categories1: Dict[str, float],
        categories2: Dict[str, float]
    ) -> Dict[str, MetricComparison]:
        """
        Compare performance of product categories between two stores.
        
        Args:
            categories1: Category performance metrics from first store
            categories2: Category performance metrics from second store
            
        Returns:
            Dict[str, MetricComparison]: Comparison metrics for each category
        """
        all_categories = set(categories1.keys()).union(categories2.keys())
        comparisons = {}
        
        for category in all_categories:
            value1 = categories1.get(category, 0)
            value2 = categories2.get(category, 0)
            
            # Create metric comparison for this category
            comparison = self._create_metric_comparison(
                value1,
                value2,
                "revenue",
                context={
                    'category': category,
                    'store1_has': category in categories1,
                    'store2_has': category in categories2
                }
            )
            
            # Add category-specific insights
            additional_insight = self._generate_category_specific_insight(
                category,
                value1,
                value2,
                category in categories1,
                category in categories2
            )
            
            if additional_insight:
                comparison.insight = f"{comparison.insight} {additional_insight}"
            
            comparisons[category] = comparison
        
        return comparisons

    def _generate_category_specific_insight(
        self,
        category: str,
        value1: float,
        value2: float,
        store1_has: bool,
        store2_has: bool
    ) -> str:
        """Generate category-specific insights"""
        insights = []
        
        # Handle cases where category is present in only one store
        if store1_has and not store2_has:
            insights.append(f"Category {category} is unique to the first store")
        elif store2_has and not store1_has:
            insights.append(f"Category {category} is unique to the second store")
        
        # Add category-specific analysis
        if store1_has and store2_has:
            difference = abs(value1 - value2)
            if difference > 100000:  # Significant difference (adjust threshold as needed)
                better_store = "first" if value1 > value2 else "second"
                insights.append(
                    f"The {better_store} store shows significantly stronger "
                    f"performance in this category"
                )
        
        return " ".join(insights)

    def _create_customer_comparisons(
        self,
        metrics1: CustomerMetrics,
        metrics2: CustomerMetrics
    ) -> Dict[str, MetricComparison]:
        """
        Create detailed comparisons of customer metrics.
        
        Args:
            metrics1: Customer metrics from first store
            metrics2: Customer metrics from second store
            
        Returns:
            Dict[str, MetricComparison]: Comparisons for different customer metrics
        """
        comparisons = {}
        
        # Compare total customers
        comparisons['total_customers'] = self._create_metric_comparison(
            metrics1.total_customers,
            metrics2.total_customers,
            'customers'
        )
        
        # Compare repeat customers
        comparisons['repeat_customers'] = self._create_metric_comparison(
            metrics1.repeat_customers,
            metrics2.repeat_customers,
            'customers',
            context={
                'repeat_rate1': metrics1.repeat_customers / metrics1.total_customers if metrics1.total_customers else 0,
                'repeat_rate2': metrics2.repeat_customers / metrics2.total_customers if metrics2.total_customers else 0
            }
        )
        
        # Compare customer lifetime value
        comparisons['customer_lifetime_value'] = self._create_metric_comparison(
            metrics1.avg_customer_lifetime_value,
            metrics2.avg_customer_lifetime_value,
            'revenue',
            context={'metric_name': 'Average Customer Lifetime Value'}
        )
        
        # Compare customer satisfaction if available
        if metrics1.customer_satisfaction is not None and metrics2.customer_satisfaction is not None:
            comparisons['satisfaction'] = self._create_metric_comparison(
                metrics1.customer_satisfaction,
                metrics2.customer_satisfaction,
                'rating',
                context={'metric_name': 'Customer Satisfaction'}
            )
        
        # Compare membership distribution
        comparisons['membership_distribution'] = self._compare_membership_distribution(
            metrics1.membership_distribution,
            metrics2.membership_distribution
        )
        
        return comparisons

    def _compare_membership_distribution(
        self,
        distribution1: Dict[str, int],
        distribution2: Dict[str, int]
    ) -> MetricComparison:
        """Compare membership tier distributions between stores"""
        # Calculate weighted scores based on tier values
        tier_weights = {
            'Bronze': 1,
            'Silver': 2,
            'Gold': 3,
            'Platinum': 4
        }
        
        def calculate_weighted_score(distribution: Dict[str, int]) -> float:
            total_members = sum(distribution.values())
            if total_members == 0:
                return 0
                
            weighted_sum = sum(
                count * tier_weights.get(tier, 0)
                for tier, count in distribution.items()
            )
            return weighted_sum / total_members
        
        score1 = calculate_weighted_score(distribution1)
        score2 = calculate_weighted_score(distribution2)
        
        # Create metric comparison
        comparison = self._create_metric_comparison(
            score1,
            score2,
            'score',
            context={
                'distribution1': distribution1,
                'distribution2': distribution2
            }
        )
        
        # Add distribution-specific insights
        total1 = sum(distribution1.values())
        total2 = sum(distribution2.values())
        
        if total1 > 0 and total2 > 0:
            # Find most common tier for each store
            top_tier1 = max(distribution1.items(), key=lambda x: x[1])[0]
            top_tier2 = max(distribution2.items(), key=lambda x: x[1])[0]
            
            additional_insight = (
                f"Most common membership tier is {top_tier1} for first store "
                f"and {top_tier2} for second store"
            )
            
            comparison.insight = f"{comparison.insight} {additional_insight}"
        
        return comparison

    def _compare_revenue_metrics(
        self,
        transactions1: List[Dict],
        transactions2: List[Dict]
    ) -> MetricComparison:
        """Compare revenue metrics between stores"""
        revenue1 = sum(t['payment']['final_amount'] for t in transactions1)
        revenue2 = sum(t['payment']['final_amount'] for t in transactions2)
        
        return MetricComparison(
            store1_value=revenue1,
            store2_value=revenue2,
            difference=revenue1 - revenue2,
            percentage_difference=((revenue1 - revenue2) / revenue2 * 100) if revenue2 else 0,
            better_performer="store1" if revenue1 > revenue2 else "store2",
            insight=self._generate_revenue_insight(revenue1, revenue2)
        )
    
    def _generate_revenue_insight(self, revenue1: float, revenue2: float) -> str:
        """
        Generate detailed insight comparing revenue performance between two stores.
        
        Args:
            revenue1 (float): Revenue of first store
            revenue2 (float): Revenue of second store
            
        Returns:
            str: Detailed insight about revenue comparison
        """
        # Calculate differences
        abs_difference = abs(revenue1 - revenue2)
        percentage_diff = (abs_difference / revenue2 * 100) if revenue2 else 0
        better_store = "first" if revenue1 > revenue2 else "second"
        
        # Generate base insight
        if revenue1 == revenue2:
            return "Both stores show identical revenue performance"
        
        # Format revenue values for readability
        def format_revenue(value: float) -> str:
            if value >= 10000000:  # 1 crore or more
                return f"₹{value/10000000:.2f} crore"
            elif value >= 100000:  # 1 lakh or more
                return f"₹{value/100000:.2f} lakh"
            else:
                return f"₹{value:,.2f}"
        
        # Build detailed insight
        insights = []
        
        # Add main comparison
        insights.append(
            f"The {better_store} store performs better with "
            f"{format_revenue(max(revenue1, revenue2))} in revenue "
            f"({percentage_diff:.1f}% higher)"
        )
        
        # Add magnitude context
        if percentage_diff > 50:
            insights.append(
                "This represents a significant performance gap that requires immediate attention"
            )
        elif percentage_diff > 25:
            insights.append(
                "This indicates a substantial difference in performance"
            )
        elif percentage_diff > 10:
            insights.append(
                "This shows a moderate variation in performance"
            )
        else:
            insights.append(
                "This indicates relatively comparable performance"
            )
        
        # Add specific insights based on difference magnitude
        if percentage_diff > 30:
            if better_store == "first":
                insights.append(
                    f"The first store's revenue of {format_revenue(revenue1)} substantially "
                    f"exceeds the second store's {format_revenue(revenue2)}"
                )
            else:
                insights.append(
                    f"The second store's revenue of {format_revenue(revenue2)} substantially "
                    f"exceeds the first store's {format_revenue(revenue1)}"
                )
        
        # Add daily revenue context
        daily_revenue1 = revenue1 / 90  # Assuming 90 days period
        daily_revenue2 = revenue2 / 90
        daily_diff = abs(daily_revenue1 - daily_revenue2)
        
        if daily_diff > 10000:  # More than 10k difference per day
            insights.append(
                f"On average, this amounts to a daily revenue difference of "
                f"{format_revenue(daily_diff)}"
            )
        
        return " ".join(insights)
    
    def _calculate_profit_comparison(
        self,
        transactions1: List[Dict],
        transactions2: List[Dict]
    ) -> MetricComparison:
        """
        Calculate and compare profit metrics between two stores.
        
        Args:
            transactions1: List of transactions from first store
            transactions2: List of transactions from second store
            
        Returns:
            MetricComparison: Detailed profit comparison metrics
        """
        def calculate_store_profit(transactions: List[Dict]) -> Tuple[float, Dict[str, float]]:
            """Calculate total profit and related metrics for a store"""
            total_revenue = 0
            total_cost = 0
            category_profits = {}
            
            for transaction in transactions:
                for item in transaction['items']:
                    # Get product details for cost calculation
                    product = self.db.products.find_one({'product_id': item['product_id']})
                    if not product:
                        continue
                        
                    # Calculate item revenue and cost
                    quantity = item['quantity']
                    revenue = item['final_price'] * quantity
                    cost = product['pricing']['base_price'] * quantity
                    
                    # Update totals
                    total_revenue += revenue
                    total_cost += cost
                    
                    # Track category-wise profits
                    category = product['category']
                    if category not in category_profits:
                        category_profits[category] = 0
                    category_profits[category] += revenue - cost
            
            total_profit = total_revenue - total_cost
            return total_profit, {
                'total_revenue': total_revenue,
                'total_cost': total_cost,
                'category_profits': category_profits,
                'profit_margin': (total_profit / total_revenue) if total_revenue > 0 else 0
            }
        
        # Calculate profits for both stores
        profit1, metrics1 = calculate_store_profit(transactions1)
        profit2, metrics2 = calculate_store_profit(transactions2)
        
        # Calculate difference and percentage
        profit_difference = profit1 - profit2
        percentage_difference = (
            (profit_difference / abs(profit2)) * 100 if profit2 != 0 
            else float('inf') if profit1 > 0 
            else float('-inf') if profit1 < 0 
            else 0
        )
        
        # Generate insight based on comparison
        insight = self._generate_profit_insight(
            profit1,
            profit2,
            metrics1,
            metrics2
        )
        
        return MetricComparison(
            store1_value=profit1,
            store2_value=profit2,
            difference=profit_difference,
            percentage_difference=percentage_difference,
            better_performer="store1" if profit1 > profit2 else "store2",
            insight=insight,
            additional_metrics={
                'store1': {
                    'profit_margin': metrics1['profit_margin'],
                    'category_profits': metrics1['category_profits']
                },
                'store2': {
                    'profit_margin': metrics2['profit_margin'],
                    'category_profits': metrics2['category_profits']
                }
            }
        )

    def _generate_profit_insight(
        self,
        profit1: float,
        profit2: float,
        metrics1: Dict[str, Any],
        metrics2: Dict[str, Any]
    ) -> str:
        """Generate detailed insight about profit comparison"""
        insights = []
        
        # Compare overall profits
        profit_diff = abs(profit1 - profit2)
        better_store = "first" if profit1 > profit2 else "second"
        
        # Format monetary values
        def format_amount(amount: float) -> str:
            if abs(amount) >= 10000000:  # 1 crore
                return f"₹{amount/10000000:.2f} crore"
            elif abs(amount) >= 100000:  # 1 lakh
                return f"₹{amount/100000:.2f} lakh"
            else:
                return f"₹{amount:,.2f}"
        
        # Base profit comparison
        insights.append(
            f"The {better_store} store shows better profitability with "
            f"{format_amount(max(profit1, profit2))} in profits"
        )
        
        # Margin comparison
        margin_diff = metrics1['profit_margin'] - metrics2['profit_margin']
        if abs(margin_diff) > 0.05:  # 5% difference in margin
            better_margin = "first" if margin_diff > 0 else "second"
            insights.append(
                f"The {better_margin} store maintains a higher profit margin "
                f"({max(metrics1['profit_margin'], metrics2['profit_margin']):.1%} vs "
                f"{min(metrics1['profit_margin'], metrics2['profit_margin']):.1%})"
            )
        
        # Category analysis
        categories1 = set(metrics1['category_profits'].keys())
        categories2 = set(metrics2['category_profits'].keys())
        common_categories = categories1.intersection(categories2)
        
        # Find best performing category for each store
        if common_categories:
            for store_num, metrics in [(1, metrics1), (2, metrics2)]:
                best_category = max(
                    common_categories,
                    key=lambda x: metrics['category_profits'].get(x, 0)
                )
                insights.append(
                    f"Store {store_num}'s most profitable category is {best_category} "
                    f"with {format_amount(metrics['category_profits'][best_category])} "
                    f"in profits"
                )
        
        # Cost efficiency
        cost_ratio1 = metrics1['total_cost'] / metrics1['total_revenue'] if metrics1['total_revenue'] > 0 else 0
        cost_ratio2 = metrics2['total_cost'] / metrics2['total_revenue'] if metrics2['total_revenue'] > 0 else 0
        
        if abs(cost_ratio1 - cost_ratio2) > 0.05:  # 5% difference in cost ratio
            better_cost = "first" if cost_ratio1 < cost_ratio2 else "second"
            insights.append(
                f"The {better_cost} store demonstrates better cost efficiency "
                f"with a cost-to-revenue ratio of "
                f"{min(cost_ratio1, cost_ratio2):.1%} vs {max(cost_ratio1, cost_ratio2):.1%}"
            )
        
        return " ".join(insights)

    def _compare_transaction_metrics(
        self,
        transactions1: List[Dict],
        transactions2: List[Dict]
    ) -> Tuple[TransactionMetrics, TransactionMetrics]:
        """Compare transaction metrics between stores"""
        def calculate_metrics(transactions: List[Dict]) -> TransactionMetrics:
            if not transactions:
                return TransactionMetrics(
                    total_transactions=0,
                    avg_transaction_value=0,
                    peak_hours=[],
                    popular_payment_methods=[],
                    customer_retention_rate=0
                )

            # Calculate key metrics
            total_amount = sum(t['payment']['final_amount'] for t in transactions)
            avg_value = total_amount / len(transactions)

            # Calculate peak hours
            hour_distribution = {}
            for t in transactions:
                hour = pd.to_datetime(t['date']).hour
                hour_distribution[hour] = hour_distribution.get(hour, 0) + 1
            peak_hours = sorted(
                hour_distribution.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]

            # Calculate payment method distribution
            payment_methods = {}
            for t in transactions:
                method = t['payment']['method']
                payment_methods[method] = payment_methods.get(method, 0) + 1
            popular_methods = sorted(
                payment_methods.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]

            # Calculate retention rate
            unique_customers = len(set(t['customer_id'] for t in transactions))
            repeat_customers = len([
                c for c, count in pd.value_counts(
                    [t['customer_id'] for t in transactions]
                ).items() if count > 1
            ])
            retention_rate = repeat_customers / unique_customers if unique_customers else 0

            return TransactionMetrics(
                total_transactions=len(transactions),
                avg_transaction_value=avg_value,
                peak_hours=[h[0] for h in peak_hours],
                popular_payment_methods=[m[0] for m in popular_methods],
                customer_retention_rate=retention_rate
            )

        return calculate_metrics(transactions1), calculate_metrics(transactions2)

    def _compare_product_metrics(
        self,
        store1_id: str,
        store2_id: str,
        transactions1: List[Dict],
        transactions2: List[Dict]
    ) -> Tuple[ProductMetrics, ProductMetrics]:
        """Compare product metrics between stores"""
        def calculate_metrics(store_id: str, transactions: List[Dict]) -> ProductMetrics:
            if not transactions:
                return ProductMetrics(
                    top_categories=[],
                    avg_margin=0,
                    stock_turnover_rate=0,
                    stockout_frequency=0,
                    category_performance={}
                )

            # Calculate category performance
            category_sales = {}
            for t in transactions:
                for item in t['items']:
                    product = self.db.products.find_one({'product_id': item['product_id']})
                    if product:
                        category = product['category']
                        if category not in category_sales:
                            category_sales[category] = 0
                        category_sales[category] += item['final_price'] * item['quantity']

            # Calculate margins
            margins = []
            for t in transactions:
                for item in t['items']:
                    product = self.db.products.find_one({'product_id': item['product_id']})
                    if product:
                        cost = product['pricing']['base_price']
                        price = item['final_price']
                        margin = (price - cost) / price
                        margins.append(margin)

            # Calculate stock turnover
            product_quantities = {}
            for t in transactions:
                for item in t['items']:
                    prod_id = item['product_id']
                    if prod_id not in product_quantities:
                        product_quantities[prod_id] = 0
                    product_quantities[prod_id] += item['quantity']

            turnover_rates = []
            for prod_id, quantity in product_quantities.items():
                product = self.db.products.find_one({'product_id': prod_id})
                if product:
                    stock = product['inventory']['total_stock']
                    if stock > 0:
                        turnover_rates.append(quantity / stock)

            return ProductMetrics(
                top_categories=sorted(
                    category_sales.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5],
                avg_margin=np.mean(margins) if margins else 0,
                stock_turnover_rate=np.mean(turnover_rates) if turnover_rates else 0,
                stockout_frequency=len([r for r in turnover_rates if r > 0.9]) / len(turnover_rates) if turnover_rates else 0,
                category_performance=category_sales
            )

        return calculate_metrics(store1_id, transactions1), calculate_metrics(store2_id, transactions2)

    def _compare_customer_metrics(
        self,
        store1_id: str,
        store2_id: str,
        transactions1: List[Dict],
        transactions2: List[Dict]
    ) -> Tuple[CustomerMetrics, CustomerMetrics]:
        """Compare customer metrics between stores"""
        def calculate_metrics(store_id: str, transactions: List[Dict]) -> CustomerMetrics:
            if not transactions:
                return CustomerMetrics(
                    total_customers=0,
                    repeat_customers=0,
                    avg_customer_lifetime_value=0,
                    membership_distribution={},
                    customer_satisfaction=None
                )

            # Calculate customer metrics
            customer_transactions = pd.DataFrame(transactions)
            unique_customers = customer_transactions['customer_id'].nunique()
            
            customer_frequency = customer_transactions['customer_id'].value_counts()
            repeat_customers = len(customer_frequency[customer_frequency > 1])

            # Calculate customer lifetime value
            customer_value = {}
            for t in transactions:
                cust_id = t['customer_id']
                if cust_id not in customer_value:
                    customer_value[cust_id] = 0
                customer_value[cust_id] += t['payment']['final_amount']

            avg_clv = np.mean(list(customer_value.values())) if customer_value else 0

            # Get membership distribution
            membership_dist = {}
            for cust_id in customer_transactions['customer_id'].unique():
                customer = self.db.customers.find_one({'customer_id': cust_id})
                if customer:
                    tier = customer['loyalty_info']['membership_tier']
                    membership_dist[tier] = membership_dist.get(tier, 0) + 1

            # Calculate satisfaction (if ratings exist)
            ratings = [t.get('rating') for t in transactions if t.get('rating')]
            satisfaction = np.mean(ratings) if ratings else None

            return CustomerMetrics(
                total_customers=unique_customers,
                repeat_customers=repeat_customers,
                avg_customer_lifetime_value=avg_clv,
                membership_distribution=membership_dist,
                customer_satisfaction=satisfaction
            )

        return calculate_metrics(store1_id, transactions1), calculate_metrics(store2_id, transactions2)

    def _compare_operational_metrics(
        self,
        store1: Dict,
        store2: Dict
    ) -> Tuple[OperationalMetrics, OperationalMetrics]:
        """Compare operational metrics between stores"""
        def extract_metrics(store: Dict) -> OperationalMetrics:
            return OperationalMetrics(
                amenities=store.get('amenities', []),
                operating_hours={
                    'weekday': store.get('opening_hours', {}).get('weekday', ''),
                    'weekend': store.get('opening_hours', {}).get('weekend', '')
                },
                staff_count=None,  # Not available in current data model
                area_size=None,    # Not available in current data model
                parking_available='Parking' in store.get('amenities', [])
            )

        return extract_metrics(store1), extract_metrics(store2)

    def _identify_unique_advantages(self, store1: Dict, store2: Dict) -> List[str]:
        """Identify unique advantages of store1 compared to store2"""
        advantages = []
        
        # Compare amenities
        unique_amenities = set(store1.get('amenities', [])) - set(store2.get('amenities', []))
        if unique_amenities:
            advantages.append(f"Unique amenities: {', '.join(unique_amenities)}")

        # Compare ratings
        if store1.get('ratings', {}).get('overall', 0) > store2.get('ratings', {}).get('overall', 0):
            advantages.append("Higher overall customer rating")

        # Compare location
        if 'location' in store1 and 'location' in store2:
            if store1['location'].get('city') in ['Mumbai', 'Delhi', 'Bangalore']:
                advantages.append("Located in major metropolitan area")

        return advantages

    def _calculate_overall_scores(
        self,
        revenue_comparison: MetricComparison,
        transaction_metrics: Tuple[TransactionMetrics, TransactionMetrics],
        product_metrics: Tuple[ProductMetrics, ProductMetrics],
        customer_metrics: Tuple[CustomerMetrics, CustomerMetrics]
    ) -> Tuple[float, float]:
        """Calculate overall performance scores for both stores"""
        def calculate_score(
            revenue: float,
            transactions: TransactionMetrics,
            products: ProductMetrics,
            customers: CustomerMetrics
        ) -> float:
            score = 0
            
            # Revenue score (40% weight)
            revenue_score = min(revenue / 1000000, 10) * 4
            
            # Transaction score (20% weight)
            transaction_score = (
                min(transactions.total_transactions / 1000, 10) * 1.5 +
                (transactions.customer_retention_rate * 100) * 0.5
            )
            
            # Product score (20% weight)
            product_score = (
                (products.avg_margin * 100) * 1.5 +
                min(products.stock_turnover_rate * 10, 5)
            )
            
            # Customer score (20% weight)
            customer_score = (
                min(customers.total_customers / 1000, 5) * 2 +
                (customers.customer_satisfaction or 0) * 2
            )
            
            return revenue_score + transaction_score + product_score + customer_score

        score1 = calculate_score(
            revenue_comparison.store1_value,
            transaction_metrics[0],
            product_metrics[0],
            customer_metrics[0]
        )
        
        score2 = calculate_score(
            revenue_comparison.store2_value,
            transaction_metrics[1],
            product_metrics[1],
            customer_metrics[1]
        )

        return score1, score2

    def _generate_key_insights(
        self,
        store1: Dict,
        store2: Dict,
        revenue_comparison: MetricComparison,
        transaction_metrics: Tuple[TransactionMetrics, TransactionMetrics],
        product_metrics: Tuple[ProductMetrics, ProductMetrics],
        customer_metrics: Tuple[CustomerMetrics, CustomerMetrics]
    ) -> List[str]:
        """Generate key insights from the comparison"""
        insights = []

        # Revenue insights
        if revenue_comparison.percentage_difference > 10:
            better_store = "first" if revenue_comparison.better_performer == "store1" else "second"
            insights.append(
                f"The {better_store} store outperforms in revenue by "
                f"{abs(revenue_comparison.percentage_difference):.1f}%"
            )

        # Transaction insights
        if (transaction_metrics[0].avg_transaction_value > 
            transaction_metrics[1].avg_transaction_value * 1.2):
            insights.append(
                f"First store has {((transaction_metrics[0].avg_transaction_value / transaction_metrics[1].avg_transaction_value) - 1) * 100:.1f}% "
                f"higher average transaction value"
            )

        # Customer insights
        retention_diff = (customer_metrics[0].customer_retention_rate - 
                        customer_metrics[1].customer_retention_rate)
        if abs(retention_diff) > 0.1:
            better_store = "first" if retention_diff > 0 else "second"
            insights.append(
                f"The {better_store} store has significantly better customer retention"
            )

        return insights
    
    def _compare_customer_metrics(
        self,
        store1_id: str,
        store2_id: str,
        transactions1: List[Dict],
        transactions2: List[Dict]
    ) -> Tuple[CustomerMetrics, CustomerMetrics]:
        """Compare customer metrics between stores"""
        def calculate_metrics(store_id: str, transactions: List[Dict]) -> CustomerMetrics:
            if not transactions:
                return CustomerMetrics(
                    total_customers=0,
                    repeat_customers=0,
                    avg_customer_lifetime_value=0,
                    membership_distribution={}
                )

            # Calculate customer metrics
            customer_transactions = pd.DataFrame(transactions)
            unique_customers = customer_transactions['customer_id'].nunique()
            
            customer_frequency = customer_transactions['customer_id'].value_counts()
            repeat_customers = len(customer_frequency[customer_frequency > 1])

            # Calculate customer lifetime value
            customer_value = {}
            for t in transactions:
                cust_id = t['customer_id']
                if cust_id not in customer_value:
                    customer_value[cust_id] = 0
                customer_value[cust_id] += t['payment']['final_amount']

            avg_clv = np.mean(list(customer_value.values())) if customer_value else 0

            # Get membership distribution
            membership_dist = {}
            for cust_id in customer_transactions['customer_id'].unique():
                customer = self.db.customers.find_one({'customer_id': cust_id})
                if customer:
                    tier = customer['loyalty_info']['membership_tier']
                    membership_dist[tier] = membership_dist.get(tier, 0) + 1

            # Calculate satisfaction (if ratings exist)
            ratings = [t.get('rating') for t in transactions if t.get('rating')]
            satisfaction = np.mean(ratings) if ratings else None

            return CustomerMetrics(
                total_customers=unique_customers,
                repeat_customers=repeat_customers,
                avg_customer_lifetime_value=avg_clv,
                membership_distribution=membership_dist,
                customer_satisfaction=satisfaction
            )

        return calculate_metrics(store1_id, transactions1), calculate_metrics(store2_id, transactions2)

    def _generate_improvement_suggestions(
        self,
        store1: Dict,
        store2: Dict,
        revenue_comparison: MetricComparison,
        transaction_metrics: Tuple[TransactionMetrics, TransactionMetrics],
        product_metrics: Tuple[ProductMetrics, ProductMetrics],
        customer_metrics: Tuple[CustomerMetrics, CustomerMetrics]
    ) -> Dict[str, List[str]]:
        """Generate improvement suggestions for both stores"""
        suggestions = {
            store1['store_id']: [],
            store2['store_id']: []
        }

        def generate_store_suggestions(
            store_id: str,
            metrics: Tuple[TransactionMetrics, ProductMetrics, CustomerMetrics],
            comparison_metrics: Tuple[TransactionMetrics, ProductMetrics, CustomerMetrics]
        ):
            store_suggestions = []
            
            # Transaction-based suggestions
            if metrics[0].avg_transaction_value < comparison_metrics[0].avg_transaction_value:
                store_suggestions.append(
                    "Consider implementing cross-selling strategies to increase "
                    "average transaction value"
                )

            # Product-based suggestions
            if metrics[1].stock_turnover_rate < comparison_metrics[1].stock_turnover_rate:
                store_suggestions.append(
                    "Optimize inventory management to improve stock turnover rate"
                )

            # Customer-based suggestions
            if metrics[2].customer_retention_rate < comparison_metrics[2].customer_retention_rate:
                store_suggestions.append(
                    "Implement customer loyalty programs to improve retention rate"
                )

            return store_suggestions

        # Generate suggestions for both stores
        suggestions[store1['store_id']] = generate_store_suggestions(
            store1['store_id'],
            (transaction_metrics[0], product_metrics[0], customer_metrics[0]),
            (transaction_metrics[1], product_metrics[1], customer_metrics[1])
        )

        suggestions[store2['store_id']] = generate_store_suggestions(
            store2['store_id'],
            (transaction_metrics[1], product_metrics[1], customer_metrics[1]),
            (transaction_metrics[0], product_metrics[0], customer_metrics[0])
        )

        return suggestions