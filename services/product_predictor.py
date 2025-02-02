from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pymongo import MongoClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from models.product import RevenuePredictionResponse, RevenueProjection
import traceback

class ProductRevenuePredictionService:
    def __init__(self, mongodb_uri='mongodb://localhost:27017/'):
        self.client = MongoClient(mongodb_uri)
        self.db = self.client['indian_retail']
        
        # Define Indian festivals and their impact
        self.festivals = {
            'Diwali': {'month': 11, 'impact': 1.5},
            'Dussehra': {'month': 10, 'impact': 1.3},
            'Holi': {'month': 3, 'impact': 1.2},
            'Christmas': {'month': 12, 'impact': 1.2}
        }
        
        # Define seasonal patterns for each category
        self.seasonal_patterns = {
            'Groceries': {
                'high_months': [10, 11, 12],  # Festival season
                'low_months': [1, 2],        # Post-holiday
                'base_margin': 0.15
            },
            'Fashion': {
                'high_months': [3, 4, 10, 11],  # Season change and festivals
                'low_months': [6, 7],          # Monsoon
                'base_margin': 0.40
            },
            'Electronics': {
                'high_months': [10, 11, 12],
                'low_months': [1, 2],
                'base_margin': 0.30
            }
        }

    async def _identify_festival_opportunities(
        self,
        predictions: List[RevenueProjection]
    ) -> List[Dict[str, Any]]:
        """Identify festival-related revenue opportunities from predictions"""
        opportunities = []
        current_month = datetime.now().month

        for prediction in predictions:
            try:
                product = self.db.products.find_one({'product_id': prediction.product_id})
                if not product:
                    continue

                category = product['category']
                
                # Identify upcoming festivals
                upcoming_festivals = []
                for festival, info in self.festivals.items():
                    months_until = (info['month'] - current_month) % 12
                    if 0 <= months_until <= 2:  # Within next 3 months
                        impact = info['impact']
                        if category == 'Fashion' and festival in ['Diwali', 'Dussehra']:
                            impact *= 1.2  # Higher impact for fashion during major festivals
                        
                        upcoming_festivals.append({
                            'name': festival,
                            'month': info['month'],
                            'months_until': months_until,
                            'impact': impact
                        })

                if upcoming_festivals:
                    # Calculate festival-specific projections
                    festival_revenue = prediction.projected_revenue * max(f['impact'] for f in upcoming_festivals)
                    revenue_increase = festival_revenue - prediction.current_revenue

                    festivals_str = ", ".join(f"{f['name']} ({f['months_until']} months)" for f in upcoming_festivals)
                    opportunities.append({
                        'product_id': prediction.product_id,
                        'product_name': prediction.product_name,
                        'category': category,
                        'opportunity_type': 'festival',
                        'upcoming_festivals': festivals_str,
                        'potential_revenue': float(festival_revenue),
                        'revenue_increase': float(revenue_increase),
                        'confidence_score': float(prediction.confidence_score),
                        'recommended_actions': [
                            f"Prepare inventory for {festivals_str}",
                            "Plan festival-specific promotions and displays",
                            f"Adjust pricing strategy for festival season",
                            "Consider festival-specific packaging or bundles"
                        ],
                        'optimal_stock': int(prediction.optimal_stock * 1.2) if prediction.optimal_stock else None,  # 20% extra for festivals
                        'optimal_price': prediction.optimal_price
                    })

            except Exception as e:
                print(f"Error identifying festival opportunity for {prediction.product_id}: {str(e)}")
                continue

        return sorted(
            opportunities,
            key=lambda x: x['revenue_increase'],
            reverse=True
        )
    
    def _generate_seasonal_recommendations(
        self,
        category: str,
        opportunity_type: str,
        upcoming_months: List[int],
        prediction: RevenueProjection
    ) -> List[str]:
        """Generate season-specific recommendations"""
        months_str = ', '.join([
            datetime.strptime(str(m), "%m").strftime("%B")
            for m in upcoming_months
        ])
        
        base_recommendations = []
        
        if opportunity_type == 'seasonal_increase':
            base_recommendations.extend([
                f"Prepare for increased demand in {months_str}",
                f"Adjust inventory levels for peak season",
                f"Consider seasonal pricing adjustments"
            ])
            
            # Category-specific recommendations
            if category == 'Fashion':
                base_recommendations.extend([
                    "Update product displays for seasonal collection",
                    "Plan seasonal fashion promotions"
                ])
            elif category == 'Electronics':
                base_recommendations.extend([
                    "Focus on seasonal electronics bundles",
                    "Prepare seasonal marketing campaigns"
                ])
            elif category == 'Groceries':
                base_recommendations.extend([
                    "Stock seasonal food items",
                    "Create seasonal product combinations"
                ])
                
        else:  # seasonal_decrease
            base_recommendations.extend([
                f"Optimize inventory for low season in {months_str}",
                "Plan promotional activities to maintain sales",
                "Consider inventory clearance strategies"
            ])
            
            # Category-specific recommendations
            if category == 'Fashion':
                base_recommendations.extend([
                    "Plan end-of-season sales",
                    "Focus on transitional fashion items"
                ])
            elif category == 'Electronics':
                base_recommendations.extend([
                    "Offer bundle deals to boost sales",
                    "Focus on evergreen electronics items"
                ])
            elif category == 'Groceries':
                base_recommendations.extend([
                    "Adjust ordering for seasonal demand drop",
                    "Promote non-seasonal alternatives"
                ])
        
        # Add stock recommendations if available
        if prediction.optimal_stock:
            base_recommendations.append(
                f"Target inventory level: {prediction.optimal_stock} units"
            )
            
        # Add price recommendations if available
        if prediction.optimal_price:
            base_recommendations.append(
                f"Recommended price point: ₹{prediction.optimal_price:,.2f}"
            )
            
        return base_recommendations

    async def _identify_seasonal_opportunities(
            self,
            predictions: List[RevenueProjection]
        ) -> List[Dict[str, Any]]:
            """Identify seasonal revenue opportunities from predictions"""
            opportunities = []
            current_month = datetime.now().month

            for prediction in predictions:
                try:
                    product = self.db.products.find_one({'product_id': prediction.product_id})
                    if not product:
                        continue

                    category = product['category']
                    if category not in self.seasonal_patterns:
                        continue

                    pattern = self.seasonal_patterns[category]
                    upcoming_months = [(current_month + i) % 12 or 12 for i in range(3)]  # Next 3 months
                    
                    # Calculate seasonal impact
                    base_revenue = prediction.current_revenue
                    seasonal_factors = [
                        1.2 if m in pattern['high_months'] else 
                        0.8 if m in pattern['low_months'] else 
                        1.0 
                        for m in upcoming_months
                    ]
                    max_seasonal_impact = max(seasonal_factors) - 1.0  # Convert to impact percentage
                    
                    # Check if there's significant seasonal opportunity
                    if abs(max_seasonal_impact) >= 0.1:  # 10% or more impact
                        seasonal_revenue = base_revenue * (1 + max_seasonal_impact)
                        revenue_change = seasonal_revenue - base_revenue
                        
                        opportunity_type = (
                            'seasonal_increase' if max_seasonal_impact > 0 
                            else 'seasonal_decrease'
                        )
                        
                        opportunities.append({
                            'product_id': prediction.product_id,
                            'product_name': prediction.product_name,
                            'category': category,
                            'opportunity_type': opportunity_type,
                            'seasonal_impact': float(max_seasonal_impact),
                            'base_revenue': float(base_revenue),
                            'projected_revenue': float(seasonal_revenue),
                            'revenue_change': float(revenue_change),
                            'confidence_score': float(prediction.confidence_score),
                            'upcoming_months': upcoming_months,
                            'recommended_actions': self._generate_seasonal_recommendations(
                                category,
                                opportunity_type,
                                upcoming_months,
                                prediction
                            ),
                            'optimal_stock': prediction.optimal_stock,
                            'optimal_price': prediction.optimal_price
                        })

                except Exception as e:
                    print(f"Error identifying seasonal opportunity for {prediction.product_id}: {str(e)}")
                    continue

            return sorted(
                opportunities,
                key=lambda x: abs(x['revenue_change']),
                reverse=True
            )

    async def predict_revenue_opportunities(
        self,
        store_id: str,
        prediction_months: int = 3
    ) -> RevenuePredictionResponse:
        """Generate revenue predictions and opportunities"""
        
        # Get historical data
        df = await self._get_historical_data(store_id)
        if len(df.index) == 0:
            raise ValueError(f"No historical data found for store {store_id}")

        # Get store and product info
        store = self.db.stores.find_one({'store_id': store_id})
        if not store:
            raise ValueError(f"Store {store_id} not found")

        # Calculate base predictions
        predictions = []
        for product_id in df['product_id'].unique():
            try:
                product = self.db.products.find_one({'product_id': product_id})
                if not product:
                    continue

                product_df = df[df['product_id'] == product_id].copy()
                if len(product_df.index) == 0:
                    continue

                prediction = await self._predict_product(
                    product,
                    product_df,
                    prediction_months
                )
                if prediction:
                    predictions.append(prediction)

            except Exception as e:
                print(f"Error predicting for product {product_id}: {str(e)}")
                continue

        # Sort predictions by growth potential
        predictions.sort(key=lambda x: x.growth_potential, reverse=True)

        # Generate opportunities
        seasonal_opps = await self._identify_seasonal_opportunities(predictions)
        festival_opps = await self._identify_festival_opportunities(predictions)

        return RevenuePredictionResponse(
            store_id=store_id,
            prediction_period=f"Next {prediction_months} months",
            predictions=predictions,
            total_projected_increase=sum(
                max(0, p.projected_revenue - p.current_revenue)
                for p in predictions
            ),
            top_categories=self._get_top_categories(predictions),
            seasonal_opportunities=seasonal_opps,
            festival_opportunities=festival_opps,
            confidence_level=np.mean([p.confidence_score for p in predictions])
        )

    async def _get_historical_data(self, store_id: str) -> pd.DataFrame:
        """Get historical transaction data with correct item processing"""
        try:
            # Get date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)

            # Get all transactions
            transactions = list(self.db.transactions.find({
                'store_id': store_id
            }))
            
            print(f"Found {len(transactions)} total transactions")

            # Transform to DataFrame
            data = []
            for txn in transactions:
                try:
                    # Handle date
                    date = pd.to_datetime(txn['date'])

                    # Process items with correct field access
                    for item in txn.get('items', []):
                        try:
                            # Extract item fields directly
                            if not isinstance(item, dict):
                                continue

                            data.append({
                                'date': date,
                                'product_id': item.get('product_id'),
                                'quantity': float(item.get('quantity', 0)),
                                'price': float(item.get('final_price', 0)),  # Use final_price directly
                                'unit_price': float(item.get('unit_price', 0)),
                                'discount': float(item.get('discount', 0)),
                                'revenue': float(item.get('total', 0)),  # Use total directly
                                'customer_id': txn.get('customer_id', 'unknown'),
                                'payment_method': txn.get('payment', {}).get('method'),
                                'has_rating': bool(txn.get('rating')),
                                'festival': txn.get('festivals')
                            })

                        except Exception as e:
                            print(f"Error processing item in transaction {txn.get('transaction_id')}: {str(e)}")
                            print(f"Problem item: {item}")
                            continue

                except Exception as e:
                    print(f"Error processing transaction: {str(e)}")
                    continue

            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Filter out rows with missing essential data
            df = df[
                df['product_id'].notna() & 
                (df['quantity'] > 0) & 
                (df['price'] > 0) & 
                (df['revenue'] > 0)
            ]

            # Add debugging information
            if len(df.index) > 0:
                print("\nDataFrame Summary:")
                print(f"Total rows: {len(df)}")
                print(f"Unique products: {df['product_id'].nunique()}")
                print(f"Date range: {df['date'].min()} to {df['date'].max()}")
                print("\nRevenue Statistics:")
                print(df['revenue'].describe())
                print("\nDiscount Statistics:")
                print(df['discount'].describe())
                print("\nPrice Range:")
                print(f"Min price: {df['price'].min():.2f}")
                print(f"Max price: {df['price'].max():.2f}")
                print("\nSample data:")
                print(df.head())
            else:
                print("No valid data processed into DataFrame")

            return df

        except Exception as e:
            print(f"Error in _get_historical_data: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    async def verify_transaction_processing(self, store_id: str) -> Dict:
        """Verify transaction processing with detailed item analysis"""
        try:
            # Get raw transactions
            transactions = list(self.db.transactions.find({
                'store_id': store_id
            }))
            
            diagnostics = {
                'total_transactions': len(transactions),
                'transactions_with_items': 0,
                'total_items': 0,
                'valid_items': 0,
                'invalid_items': 0,
                'revenue_stats': {
                    'total_revenue': 0,
                    'min_transaction': float('inf'),
                    'max_transaction': 0
                },
                'sample_items': []
            }
            
            for txn in transactions:
                items = txn.get('items', [])
                if items:
                    diagnostics['transactions_with_items'] += 1
                    diagnostics['total_items'] += len(items)
                    
                    # Analyze items
                    valid_items = 0
                    txn_revenue = 0
                    
                    for item in items:
                        try:
                            if all(k in item for k in ['product_id', 'quantity', 'final_price', 'total']):
                                valid_items += 1
                                txn_revenue += float(item['total'])
                                
                                # Add sample items (up to 5)
                                if len(diagnostics['sample_items']) < 5:
                                    diagnostics['sample_items'].append(item)
                        except:
                            continue
                    
                    diagnostics['valid_items'] += valid_items
                    
                    # Update revenue stats
                    if txn_revenue > 0:
                        diagnostics['revenue_stats']['total_revenue'] += txn_revenue
                        diagnostics['revenue_stats']['min_transaction'] = min(
                            diagnostics['revenue_stats']['min_transaction'],
                            txn_revenue
                        )
                        diagnostics['revenue_stats']['max_transaction'] = max(
                            diagnostics['revenue_stats']['max_transaction'],
                            txn_revenue
                        )
            
            diagnostics['invalid_items'] = diagnostics['total_items'] - diagnostics['valid_items']
            
            # Add sample valid and invalid transactions
            if transactions:
                diagnostics['sample_transaction'] = {
                    'transaction_id': transactions[0].get('transaction_id'),
                    'date': str(transactions[0].get('date')),
                    'items_count': len(transactions[0].get('items', [])),
                    'payment': transactions[0].get('payment'),
                    'sample_items': transactions[0].get('items', [])[:2]  # First 2 items
                }
            
            return diagnostics

        except Exception as e:
            return {
                'error': str(e),
                'total_transactions': 0
            }

    async def verify_transaction_processing(self, store_id: str) -> Dict:
        """Verify transaction processing and provide detailed diagnostics"""
        try:
            # Get raw transactions
            transactions = list(self.db.transactions.find({'store_id': store_id}))
            
            diagnostics = {
                'total_transactions': len(transactions),
                'items_found': 0,
                'items_processed': 0,
                'price_stats': {
                    'min_unit_price': float('inf'),
                    'max_unit_price': 0,
                    'min_total': float('inf'),
                    'max_total': 0
                },
                'sample_processing': []
            }
            
            # Process sample transaction
            if transactions:
                sample_txn = transactions[0]
                diagnostics['sample_transaction'] = {
                    'date': str(sample_txn.get('date')),
                    'total_items': len(sample_txn.get('items', [])),
                    'payment_info': sample_txn.get('payment')
                }
                
                # Process each item in sample transaction
                for item in sample_txn.get('items', []):
                    try:
                        # Get fields
                        product_id = item.get('product_id')
                        quantity = float(item.get('quantity', 0))
                        unit_price = float(item.get('unit_price', 0))
                        discount = float(item.get('discount', 0))
                        total = float(item.get('total', 0))

                        # Calculate values
                        final_price = unit_price * (1 - discount)
                        revenue = total if total > 0 else (final_price * quantity)

                        diagnostics['sample_processing'].append({
                            'product_id': product_id,
                            'original_item': item,
                            'processed_values': {
                                'quantity': quantity,
                                'unit_price': unit_price,
                                'final_price': final_price,
                                'revenue': revenue
                            }
                        })
                        
                        # Update price stats
                        diagnostics['price_stats']['min_unit_price'] = min(
                            diagnostics['price_stats']['min_unit_price'],
                            unit_price
                        )
                        diagnostics['price_stats']['max_unit_price'] = max(
                            diagnostics['price_stats']['max_unit_price'],
                            unit_price
                        )
                        diagnostics['price_stats']['min_total'] = min(
                            diagnostics['price_stats']['min_total'],
                            total
                        )
                        diagnostics['price_stats']['max_total'] = max(
                            diagnostics['price_stats']['max_total'],
                            total
                        )
                        
                    except Exception as e:
                        diagnostics['sample_processing'].append({
                            'error': str(e),
                            'item': item
                        })

            return diagnostics

        except Exception as e:
            return {
                'error': str(e),
                'total_transactions': 0
            }
    async def _predict_product(
        self,
        product: Dict,
        df: pd.DataFrame,
        prediction_months: int
    ) -> Optional[RevenueProjection]:
        """Generate prediction for a single product"""
        try:
            # Calculate current metrics
            current_metrics = self._calculate_current_metrics(df)
            
            # Calculate seasonality
            seasonal_factors = self._calculate_seasonal_factors(
                product['category'],
                df,
                prediction_months
            )
            
            # Calculate festival impact
            festival_impact = self._calculate_festival_impact(
                product['category'],
                prediction_months
            )
            
            # Calculate base projection
            base_projection = current_metrics['monthly_revenue'] * np.mean(seasonal_factors)
            
            # Apply festival impact
            projected_revenue = base_projection * (1 + festival_impact)
            
            # Calculate growth potential
            growth_factors = self._calculate_growth_factors(product, df)
            projected_revenue *= (1 + growth_factors['growth_potential'])
            
            # Calculate optimal price and stock
            optimal_price = self._calculate_optimal_price(
                product,
                current_metrics,
                seasonal_factors
            )
            
            optimal_stock = self._calculate_optimal_stock(
                current_metrics,
                seasonal_factors,
                festival_impact
            )

            return RevenueProjection(
                product_id=product['product_id'],
                product_name=product['name'],
                current_revenue=current_metrics['monthly_revenue'],
                projected_revenue=projected_revenue,
                growth_potential=growth_factors['growth_potential'],
                confidence_score=growth_factors['confidence'],
                contributing_factors=growth_factors['factors'],
                seasonal_index=float(np.mean(seasonal_factors)),
                optimal_price=optimal_price,
                optimal_stock=optimal_stock
            )

        except Exception as e:
            print(f"Error in product prediction: {str(e)}")
            return None

    def _calculate_current_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate current performance metrics"""
        if len(df.index) == 0:
            return {
                'monthly_revenue': 0.0,
                'avg_quantity': 0.0,
                'avg_price': 0.0
            }

        monthly = df.groupby(pd.Grouper(key='date', freq='M')).agg({
            'revenue': 'sum',
            'quantity': 'sum',
            'price': 'mean'
        })

        return {
            'monthly_revenue': float(monthly['revenue'].mean()),
            'avg_quantity': float(monthly['quantity'].mean()),
            'avg_price': float(monthly['price'].mean())
        }

    def _calculate_seasonal_factors(
        self,
        category: str,
        df: pd.DataFrame,
        prediction_months: int
    ) -> List[float]:
        """Calculate seasonal factors for prediction period"""
        if category not in self.seasonal_patterns:
            return [1.0] * prediction_months

        pattern = self.seasonal_patterns[category]
        current_month = datetime.now().month
        
        factors = []
        for i in range(prediction_months):
            month = (current_month + i) % 12 or 12
            if month in pattern['high_months']:
                factors.append(1.2)
            elif month in pattern['low_months']:
                factors.append(0.8)
            else:
                factors.append(1.0)

        return factors

    def _calculate_festival_impact(
        self,
        category: str,
        prediction_months: int
    ) -> float:
        """Calculate festival impact for prediction period"""
        current_month = datetime.now().month
        max_impact = 0.0
        
        for festival, info in self.festivals.items():
            festival_month = info['month']
            months_until = (festival_month - current_month) % 12
            
            if months_until < prediction_months:
                impact = info['impact']
                if category == 'Fashion' and festival in ['Diwali', 'Dussehra']:
                    impact *= 1.2  # Higher impact for fashion during major festivals
                max_impact = max(max_impact, impact - 1.0)  # Convert to growth factor

        return max_impact

    def _calculate_growth_factors(
        self,
        product: Dict,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate growth potential and contributing factors"""
        factors = []
        growth_potential = 0.0
        confidence = 0.7  # Base confidence
        
        # Price optimization potential
        if product['pricing']['current_price'] > product['pricing']['base_price'] * 1.5:
            growth_potential += 0.1
            factors.append("Price optimization opportunity")
            confidence += 0.05
            
        # Stock optimization
        stockout_rate = self._calculate_stockout_rate(df)
        if stockout_rate > 0.1:
            growth_potential += 0.15
            factors.append("Stock optimization needed")
            confidence -= 0.1
            
        # Customer rating impact
        if product.get('ratings', {}).get('average', 0) > 4.0:
            growth_potential += 0.1
            factors.append("High customer satisfaction")
            confidence += 0.1
            
        # Adjust confidence
        if len(df.index) < 30:  # Less than a month of data
            confidence -= 0.2
            
        return {
            'growth_potential': growth_potential,
            'confidence': min(max(confidence, 0.0), 1.0),
            'factors': factors
        }

    def _calculate_stockout_rate(self, df: pd.DataFrame) -> float:
        """Calculate stockout rate from historical data"""
        if len(df.index) == 0:
            return 0.0
            
        daily_sales = df.groupby(df['date'].dt.date)['quantity'].sum()
        zero_sales_days = (daily_sales == 0).sum()
        
        return float(zero_sales_days / len(daily_sales)) if len(daily_sales) > 0 else 0.0

    def _calculate_optimal_price(
        self,
        product: Dict,
        metrics: Dict[str, float],
        seasonal_factors: List[float]
    ) -> Optional[float]:
        """Calculate optimal price point"""
        if metrics['monthly_revenue'] == 0:
            return None
            
        base_price = product['pricing']['base_price']
        current_price = product['pricing']['current_price']
        category = product['category']
        
        if category not in self.seasonal_patterns:
            return current_price
            
        base_margin = self.seasonal_patterns[category]['base_margin']
        seasonal_adjustment = max(seasonal_factors) - 1.0
        
        optimal_margin = base_margin + (seasonal_adjustment * 0.5)
        optimal_price = base_price / (1 - optimal_margin)
        
        # Cap the price increase
        max_price = current_price * 1.5
        return min(optimal_price, max_price)

    def _calculate_optimal_stock(
        self,
        metrics: Dict[str, float],
        seasonal_factors: List[float],
        festival_impact: float
    ) -> Optional[int]:
        """Calculate optimal stock level"""
        if metrics['avg_quantity'] == 0:
            return None
            
        # Base stock for one month
        base_stock = metrics['avg_quantity'] * 1.5  # 1.5 months of stock
        
        # Apply seasonal and festival adjustments
        adjusted_stock = base_stock * max(seasonal_factors) * (1 + festival_impact)
        
        # Add safety stock (20% of adjusted stock)
        safety_stock = adjusted_stock * 0.2
        
        return int(adjusted_stock + safety_stock)

    def _get_top_categories(self, predictions: List[RevenueProjection]) -> List[str]:
        """Get top performing categories"""
        category_revenue = {}
        
        for prediction in predictions:
            product = self.db.products.find_one({'product_id': prediction.product_id})
            if product:
                category = product['category']
                if category not in category_revenue:
                    category_revenue[category] = 0
                category_revenue[category] += prediction.projected_revenue
                
        return [
            category for category, revenue in sorted(
                category_revenue.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ][:5]  # Top 5 categories