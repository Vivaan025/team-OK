# app/models/response_models.py
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime

class BasicMetrics(BaseModel):
    total_sales: float
    total_transactions: int
    avg_transaction_value: float
    current_month_sales: float
    last_month_sales: float
    mom_growth: float

class SalesPrediction(BaseModel):
    dates: List[datetime]
    predictions: List[float]
    confidence_score: float
    feature_importance: Dict[str, float]

class CustomerInsights(BaseModel):
    total_customers: int
    avg_customer_spend: float
    loyal_customers: int

class ProductInsights(BaseModel):
    total_items_sold: int
    avg_item_price: float
    top_products: Dict[str, int]

class StoreAnalysis(BaseModel):
    basic_metrics: BasicMetrics
    sales_prediction: SalesPrediction
    customer_insights: CustomerInsights
    product_insights: ProductInsights

class StoreRecommendation(BaseModel):
    name: str
    address: str
    city: str
    pincode: str
    ratings: Optional[Dict[str, float]]
    amenities: Optional[List[str]]

class PreferenceResponse(BaseModel):
    predicted_category: str
    confidence: float
    recommended_store: StoreRecommendation


class CustomerSegment(BaseModel):
    customer_id: str
    segment: int
    features: Dict[str, float]

class SalesPrediction(BaseModel):
    predicted_revenue: float
    confidence_score: float

class ProductRecommendation(BaseModel):
    product_id: str
    name: str
    category: str
    confidence_score: float

class StorePlacementRecommendation(BaseModel):
    latitude: float
    longitude: float
    estimated_revenue: float
    nearby_cities: List[str]

class SalesForecast(BaseModel):
    date: datetime
    forecasted_sales: float
    lower_ci: float
    upper_ci: float

class SalesForecastResponse(BaseModel):
    forecasts: List[SalesForecast]
    summary: Dict[str, float]