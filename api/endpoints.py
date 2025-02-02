# app/api/endpoints.py
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from models.request_models import PreferenceRequest
from models.response_models import (
    StoreAnalysis, PreferenceResponse, BasicMetrics,
    SalesPrediction, CustomerInsights, ProductInsights
)
from services.store_analyzer import MLStoreAnalyzer
from services.preference_predictor import RetailPreferencePredictor
from datetime import datetime, timedelta
from services.retail_intelligence import RetailIntelligenceService
from models.response_models import CustomerInsights, CustomerSegment, ProductRecommendation, StorePlacementRecommendation, SalesForecastResponse, RevenueSuggestion, StoreSuggestionsResponse
from models.request_models import StoreFeatures, PlacementRequest
from services.sales_predictor import SalesPredictorService
from services.revenue_suggestions import RevenueSuggestionService
from models.comparison import StoreComparisonResponse, MetricComparison
from services.store_comparison import StoreComparisonService
from typing import Dict
from services.product_predictor import ProductRevenuePredictionService
from models.product import RevenuePredictionResponse
import math

router = APIRouter()

# Dependencies
def get_analyzer():
    return MLStoreAnalyzer()

def get_predictor():
    return RetailPreferencePredictor()

@router.get("/stores", response_model=List[dict])
async def get_stores(analyzer: MLStoreAnalyzer = Depends(get_analyzer)):
    """Get list of all stores with their details"""
    try:
        # Retrieve all stores, excluding the MongoDB internal '_id' field
        stores = list(analyzer.db.stores.find({}, {'_id': 0}))
        return stores
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/transactions", response_model=List[dict])
async def get_transactions(analyzer: MLStoreAnalyzer = Depends(get_analyzer)):
    """Get list of all stores with their details"""
    try:
        # Retrieve all stores, excluding the MongoDB internal '_id' field
        stores = list(analyzer.db.transactions.find({}, {'_id': 0}))
        return stores
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/customers", response_model=List[dict])
async def get_customers(analyzer: MLStoreAnalyzer = Depends(get_analyzer)):
    """Get list of all stores with their details"""
    try:
        # Retrieve all stores, excluding the MongoDB internal '_id' field
        stores = list(analyzer.db.customers.find({}, {'_id': 0}))
        return stores
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/product/{product_id}", response_model=Optional[dict])
async def get_product(product_id: str, analyzer: MLStoreAnalyzer = Depends(get_analyzer)):
    """Get a single transaction by ID"""
    try:
        transaction = analyzer.db.products.find_one({"product_id": product_id}, {'_id': 0})
        if not transaction:
            raise HTTPException(status_code=404, detail="Transaction not found")
        return transaction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/store/{store_id}", response_model=StoreAnalysis)
async def analyze_store(
    store_id: str,
    analyzer: MLStoreAnalyzer = Depends(get_analyzer)
):
    """Get comprehensive analysis for a specific store"""
    try:
        return analyzer.analyze_store_performance(store_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/store/{store_id}/basic-metrics", response_model=BasicMetrics)
async def get_basic_metrics(
    store_id: str,
    analyzer: MLStoreAnalyzer = Depends(get_analyzer)
):
    """Get basic performance metrics for a store"""
    try:
        analysis = analyzer.analyze_store_performance(store_id)
        return analysis['basic_metrics']
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predict-preference/{customer_id}", response_model=PreferenceResponse)
async def predict_preference(
    customer_id: str,
    days: Optional[int] = Query(90, description="Number of days of transaction history to consider"),
    predictor: RetailPreferencePredictor = Depends(get_predictor)
):
    """Predict store preference for a customer based on their ID"""
    try:
        # Get customer information
        customer = predictor.db.customers.find_one({"customer_id": customer_id})
        if not customer:
            raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

        # Extract relevant customer info
        customer_info = {
            'age': customer['personal_info']['age'],
            'gender': customer['personal_info']['gender'],
            'city': customer['personal_info']['address']['city'],
            'state': customer['personal_info']['address']['state'],
            'membership_tier': customer['loyalty_info']['membership_tier']
        }

        print("Customer Info:", customer_info)

        # Use TransactionProcessor to get processed transactions
        # This will automatically handle validation, processing, and logging
        processed_transactions = predictor.transaction_processor.get_customer_transactions(
            customer_id, 
            days_back=days
        )

        # Make prediction
        predicted_category, best_store, confidence = predictor.predict(
            customer_info=customer_info,
            customer_id=customer_id,
            recent_transactions=processed_transactions
        )
        
        return PreferenceResponse(
            predicted_category=predicted_category,
            confidence=float(confidence),
            recommended_store=best_store
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/customer-transactions/{customer_id}")
async def get_customer_transactions(
    customer_id: str,
    days: Optional[int] = Query(90, description="Number of days of transaction history to fetch"),
    predictor: RetailPreferencePredictor = Depends(get_predictor)
):
    """Get recent transactions for a customer (helper endpoint for debugging)"""
    try:
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        transactions = list(predictor.db.transactions.find({
            'customer_id': customer_id,
            'date': {'$gte': cutoff_date}
        }).sort('date', -1))

        # Convert ObjectId to string for JSON serialization
        for txn in transactions:
            txn['_id'] = str(txn['_id'])

        return {
            'customer_id': customer_id,
            'transaction_count': len(transactions),
            'transactions': transactions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_intelligence_service():
    return RetailIntelligenceService()

@router.get(
    "/customer-segment/{customer_id}",
    response_model=CustomerSegment,
    summary="Get customer segment information"
)
async def get_customer_segment(
    customer_id: str,
    service: RetailIntelligenceService = Depends(get_intelligence_service)
):
    try:
        segment = await service.get_customer_segment(customer_id)
        if not segment:
            raise HTTPException(status_code=404, detail="Customer not found")
        return segment
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/predict-store-sales/{store_id}",
    response_model=SalesPrediction,
    summary="Predict store sales performance for a specific store"
)
async def predict_store_sales(
    store_id: str,
    service: RetailIntelligenceService = Depends(get_intelligence_service)
):
    try:
        # Get store data from database
        store = service.model.db.stores.find_one({"store_id": store_id})
        if not store:
            raise HTTPException(status_code=404, detail=f"Store {store_id} not found")

        # Get transaction data for the store
        transactions = list(service.model.db.transactions.find({"store_id": store_id}))
        
        # Prepare store features
        features = StoreFeatures(
            state=store['location']['state'],
            city=store['location']['city'],
            category=store['category'],
            latitude=store['location']['coordinates']['latitude'],
            longitude=store['location']['coordinates']['longitude'],
            transaction_count=len(transactions),
            avg_transaction_value=sum(t['payment']['final_amount'] for t in transactions) / len(transactions) if transactions else 0
        )
        
        # Get prediction
        return await service.predict_store_sales(features)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get(
    "/product-recommendations/{customer_id}",
    response_model=List[ProductRecommendation],
    summary="Get product recommendations for a customer"
)
async def get_product_recommendations(
    customer_id: str,
    top_n: int = 5,
    service: RetailIntelligenceService = Depends(get_intelligence_service)
):
    try:
        recommendations = await service.get_product_recommendations(customer_id, top_n)
        if not recommendations:
            raise HTTPException(
                status_code=404,
                detail="No recommendations found for this customer"
            )
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/store-placement-recommendations",
    response_model=List[StorePlacementRecommendation],
    summary="Get store placement recommendations"
)
async def get_store_placement_recommendations(
    request: PlacementRequest,
    service: RetailIntelligenceService = Depends(get_intelligence_service)
):
    try:
        return await service.get_store_placement_recommendations(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

def get_sales_predictor():
    return SalesPredictorService()

@router.get(
    "/forecast",
    response_model=SalesForecastResponse,
    summary="Get sales forecast for all stores"
)
async def get_sales_forecast(
    days: int = Query(30, description="Number of days to forecast"),
    service: SalesPredictorService = Depends(get_sales_predictor)
):
    """Get sales forecast for all stores"""
    try:
        return await service.get_sales_forecast(days=days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/forecast/{store_id}",
    response_model=SalesForecastResponse,
    summary="Get sales forecast for a specific store"
)
async def get_store_sales_forecast(
    store_id: str,
    days: int = Query(30, description="Number of days to forecast"),
    service: SalesPredictorService = Depends(get_sales_predictor)
):
    """Get sales forecast for a specific store"""
    try:
        return await service.get_sales_forecast(store_id=store_id, days=days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_suggestion_service():
    return RevenueSuggestionService()

@router.get(
    "/suggestions/{store_id}",
    response_model=StoreSuggestionsResponse,
    summary="Get revenue improvement suggestions for a store"
)
async def get_store_suggestions(
    store_id: str,
    days: int = Query(90, description="Analysis period in days"),
    service: RevenueSuggestionService = Depends(get_suggestion_service)
):
    """Get comprehensive revenue improvement suggestions for a specific store"""
    try:
        suggestions = await service.analyze_store_performance(store_id, days)
        return suggestions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/top-products/{store_id}",
    response_model=List[RevenueSuggestion],
    summary="Get product-specific revenue opportunities"
)
async def get_product_opportunities(
    store_id: str,
    min_revenue: float = Query(0, description="Minimum revenue threshold"),
    category: Optional[str] = None,
    service: RevenueSuggestionService = Depends(get_suggestion_service)
):
    """Get product-specific revenue opportunities for a store"""
    try:
        suggestions = await service.analyze_product_opportunities(
            store_id,
            min_revenue=min_revenue,
            category=category
        )
        return suggestions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/pricing-analysis/{store_id}",
    response_model=List[RevenueSuggestion],
    summary="Get pricing optimization suggestions"
)
async def get_pricing_suggestions(
    store_id: str,
    service: RevenueSuggestionService = Depends(get_suggestion_service)
):
    """Get pricing optimization suggestions for a store"""
    try:
        suggestions = await service.analyze_pricing_opportunities(store_id)
        return suggestions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

def get_comparison_service():
    return StoreComparisonService()

@router.get(
    "/compare/{store1_id}/{store2_id}",
    response_model=StoreComparisonResponse,
    summary="Compare two stores across multiple metrics"
)
async def compare_stores(
    store1_id: str,
    store2_id: str,
    days: int = Query(90, description="Number of days to analyze"),
    service: StoreComparisonService = Depends(get_comparison_service)
):
    """
    Compare two stores across multiple metrics
    """
    try:
        # Perform the store comparison
        comparison = await service.compare_stores(
            store1_id=store1_id,
            store2_id=store2_id,
            days=days
        )
        
        # Sanitize float values to prevent JSON serialization issues
        def sanitize_float(value):
            if isinstance(value, float):
                # Handle inf and -inf
                if math.isinf(value):
                    return 0.0
                # Limit to a reasonable range
                return max(min(value, 1e10), -1e10)
            return value
        
        def deep_sanitize(obj):
            if isinstance(obj, dict):
                return {k: deep_sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_sanitize(item) for item in obj]
            elif isinstance(obj, float):
                return sanitize_float(obj)
            return obj
        
        # Apply sanitization to the comparison object
        sanitized_comparison = deep_sanitize(comparison.dict())
        
        return StoreComparisonResponse(**sanitized_comparison)
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get(
    "/compare/metrics/{store1_id}/{store2_id}",
    response_model=Dict[str, MetricComparison],
    summary="Compare specific metrics between two stores"
)
async def compare_store_metrics(
    store1_id: str,
    store2_id: str,
    metrics: List[str] = Query(
        ['revenue', 'transactions', 'customers'],
        description="Metrics to compare"
    ),
    days: int = Query(90, description="Number of days to analyze"),
    service: StoreComparisonService = Depends(get_comparison_service)
):
    """
    Compare specific metrics between two stores.
    
    Parameters:
    - store1_id: ID of first store
    - store2_id: ID of second store
    - metrics: List of metrics to compare
    - days: Analysis period in days (default: 90)
    
    Available metrics:
    - revenue: Revenue comparison
    - transactions: Transaction metrics
    - customers: Customer metrics
    - products: Product performance
    - operations: Operational metrics
    """
    try:
        comparison = await service.compare_stores(
            store1_id=store1_id,
            store2_id=store2_id,
            days=days
        )
        
        # Filter requested metrics
        metric_comparisons = {}
        for metric in metrics:
            if metric == 'revenue':
                metric_comparisons['revenue'] = comparison.revenue_comparison
            elif metric == 'transactions':
                metric_comparisons['transactions'] = comparison.transaction_comparison
            elif metric == 'customers':
                metric_comparisons['customers'] = comparison.customer_comparison
                
        return metric_comparisons
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

def get_prediction_service():
    return ProductRevenuePredictionService()


@router.get(
    "/revenue-predictions/{store_id}",
    response_model=RevenuePredictionResponse,
    summary="Get revenue predictions for store products"
)
async def get_revenue_predictions(
    store_id: str,
    months: int = Query(3, description="Number of months to predict"),
    category: Optional[str] = None,
    service: ProductRevenuePredictionService = Depends(get_prediction_service)
):
    """
    Get detailed revenue predictions and opportunities for store products.
    
    Parameters:
    - store_id: Store identifier
    - months: Number of months to predict (default: 3)
    - category: Optional category filter
    
    Returns comprehensive prediction response including:
    - Product-specific predictions
    - Seasonal opportunities
    - Festival opportunities
    - Overall revenue projections
    """
    try:
        predictions = await service.predict_revenue_opportunities(
            store_id=store_id,
            prediction_months=months
        )
        
        # Apply category filter if specified
        if category:
            predictions.predictions = [
                p for p in predictions.predictions
                if service.db.products.find_one(
                    {'product_id': p.product_id}
                )['category'] == category
            ]
            
# Update total projected increase for filtered predictions
            predictions.total_projected_increase = sum(
                max(0, p.projected_revenue - p.current_revenue)
                for p in predictions.predictions
            )
            
            # Update top categories
            predictions.top_categories = [
                cat for cat in predictions.top_categories 
                if cat == category
            ]
            
            # Filter opportunities
            predictions.seasonal_opportunities = [
                opp for opp in predictions.seasonal_opportunities
                if service.db.products.find_one(
                    {'product_id': opp['product_id']}
                )['category'] == category
            ]
            
            predictions.festival_opportunities = [
                opp for opp in predictions.festival_opportunities
                if service.db.products.find_one(
                    {'product_id': opp['product_id']}
                )['category'] == category
            ]
        
        return predictions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/seasonal-predictions/{store_id}",
    response_model=List[Dict],
    summary="Get seasonal revenue opportunities"
)
async def get_seasonal_predictions(
    store_id: str,
    months: int = Query(3, description="Number of months to analyze"),
    min_impact: float = Query(0.2, description="Minimum seasonal impact"),
    service: ProductRevenuePredictionService = Depends(get_prediction_service)
):
    """
    Get seasonal revenue opportunities for the store.
    
    Parameters:
    - store_id: Store identifier
    - months: Number of months to analyze (default: 3)
    - min_impact: Minimum seasonal impact threshold (default: 0.2)
    
    Returns list of seasonal opportunities with recommendations.
    """
    try:
        predictions = await service.predict_revenue_opportunities(
            store_id=store_id,
            prediction_months=months
        )
        
        opportunities = [
            opp for opp in predictions.seasonal_opportunities
            if opp['seasonal_impact'] >= min_impact
        ]
        
        return sorted(
            opportunities,
            key=lambda x: x['potential_revenue'],
            reverse=True
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/festival-predictions/{store_id}",
    response_model=List[Dict],
    summary="Get festival-related revenue opportunities"
)
async def get_festival_predictions(
    store_id: str,
    months: int = Query(3, description="Number of months to analyze"),
    festival: Optional[str] = None,
    service: ProductRevenuePredictionService = Depends(get_prediction_service)
):
    """
    Get festival-related revenue opportunities for the store.
    
    Parameters:
    - store_id: Store identifier
    - months: Number of months to analyze (default: 3)
    - festival: Optional specific festival filter
    
    Returns list of festival opportunities with recommendations.
    """
    try:
        predictions = await service.predict_revenue_opportunities(
            store_id=store_id,
            prediction_months=months
        )
        
        opportunities = predictions.festival_opportunities
        if festival:
            opportunities = [
                opp for opp in opportunities
                if opp['festival'].lower() == festival.lower()
            ]
            
        return sorted(
            opportunities,
            key=lambda x: x['potential_revenue'],
            reverse=True
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/product-optimization/{store_id}/{product_id}",
    response_model=Dict,
    summary="Get product optimization recommendations"
)
async def get_product_optimization(
    store_id: str,
    product_id: str,
    service: ProductRevenuePredictionService = Depends(get_prediction_service)
):
    """
    Get detailed optimization recommendations for a specific product.
    
    Parameters:
    - store_id: Store identifier
    - product_id: Product identifier
    
    Returns optimization recommendations including:
    - Optimal price
    - Optimal stock levels
    - Seasonal adjustments
    - Festival opportunities
    """
    try:
        # Get historical data
        df = await service._get_historical_data(store_id)
        product_df = df[df['product_id'] == product_id].copy()
        
        if len(product_df.index) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for product {product_id}"
            )
            
        product = service.db.products.find_one({'product_id': product_id})
        if not product:
            raise HTTPException(
                status_code=404,
                detail=f"Product {product_id} not found"
            )
            
        # Calculate metrics
        current_metrics = service._calculate_current_metrics(product_df)
        seasonal_factors = service._calculate_seasonal_factors(
            product['category'],
            product_df,
            3  # Next 3 months
        )
        festival_impact = service._calculate_festival_impact(
            product['category'],
            3
        )
        
        # Get optimization recommendations
        optimal_price = service._calculate_optimal_price(
            product,
            current_metrics,
            seasonal_factors
        )
        
        optimal_stock = service._calculate_optimal_stock(
            current_metrics,
            seasonal_factors,
            festival_impact
        )
        
        return {
            'product_id': product_id,
            'product_name': product['name'],
            'category': product['category'],
            'current_metrics': current_metrics,
            'optimal_price': optimal_price,
            'optimal_stock': optimal_stock,
            'seasonal_factors': seasonal_factors,
            'festival_impact': festival_impact,
            'recommendations': [
                rec for rec in [
                    "Price optimization recommended" if optimal_price and optimal_price > product['pricing']['current_price'] else None,
                    "Stock level adjustment needed" if optimal_stock and abs(optimal_stock - product['inventory']['total_stock']) > 20 else None,
                    "Prepare for seasonal demand increase" if max(seasonal_factors) > 1.2 else None,
                    "Festival opportunity identified" if festival_impact > 0.1 else None
                ] if rec is not None
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))