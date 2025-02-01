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
from models.response_models import CustomerInsights, CustomerSegment, ProductRecommendation, StorePlacementRecommendation, SalesForecastResponse
from models.request_models import StoreFeatures, PlacementRequest
from services.sales_predictor import SalesPredictorService

router = APIRouter()

# Dependencies
def get_analyzer():
    return MLStoreAnalyzer()

def get_predictor():
    return RetailPreferencePredictor()

@router.get("/stores", response_model=List[str])
async def get_stores(analyzer: MLStoreAnalyzer = Depends(get_analyzer)):
    """Get list of all store IDs"""
    try:
        stores = list(analyzer.db.stores.find({}, {'store_id': 1}))
        return [store['store_id'] for store in stores]
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

        # Get recent transactions
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        recent_transactions = list(predictor.db.transactions.find({
            'customer_id': customer_id,
            'date': {'$gte': cutoff_date}
        }).sort('date', -1))

        if not recent_transactions:
            print(f"No recent transactions found for customer {customer_id}")

        # Make prediction
        predicted_category, best_store, confidence = predictor.predict(
            customer_info,
            recent_transactions
        )
        
        return PreferenceResponse(
            predicted_category=predicted_category,
            confidence=float(confidence),
            recommended_store=best_store
        )
    except HTTPException:
        raise
    except Exception as e:
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
