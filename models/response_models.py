# app/models/response_models.py
from pydantic import BaseModel, validator
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
    sales: float
    lower_ci: Optional[float] = None
    upper_ci: Optional[float] = None
    is_forecast: bool = True

    @validator('lower_ci', 'upper_ci', pre=True)
    def handle_none_confidence_intervals(cls, v, values):
        if v is None:
            # If we have forecasted_sales, calculate CI based on standard deviation
            if 'sales' in values:
                forecast = values['sales']
                # Using a default confidence interval of ±20% if none provided
                margin = forecast * 0.2
                return forecast - margin if v is values.get('lower_ci') else forecast + margin
            return 0.0  # fallback default
        return v

    @validator('upper_ci')
    def upper_ci_must_be_greater(cls, v, values):
        if v is not None and 'lower_ci' in values and values['lower_ci'] is not None:
            if v <= values['lower_ci']:
                # If upper_ci is less than lower_ci, adjust it to be 20% above forecasted_sales
                forecast = values.get('forecasted_sales', 0)
                return forecast * 1.2
        return v

class SalesForecastResponse(BaseModel):
    forecasts: List[SalesForecast]
    summary: Dict[str, float] = {}


class RevenueSuggestion(BaseModel):
    type: str
    suggestion: str
    priority: str
    metrics: Dict
    impact_estimate: Optional[float]
    implementation_difficulty: str
    timeframe: str

class RevenueSuggestionResponse(BaseModel):
    type: str
    suggestion: str
    priority: str
    metrics: dict
    impact_estimate: float
    implementation_difficulty: str
    timeframe: str

class StoreSuggestionsResponse(BaseModel):
    product_suggestions: List[RevenueSuggestionResponse]
    pricing_suggestions: List[RevenueSuggestionResponse]
    inventory_suggestions: List[RevenueSuggestionResponse]
    timing_suggestions: List[RevenueSuggestionResponse]
    customer_suggestions: List[RevenueSuggestionResponse]

class ProductMetrics(BaseModel):
    total_quantity: int
    total_revenue: float
    transactions: int
    prices: List[float]
    avg_price: Optional[float] = None
    price_variance: Optional[float] = None
    revenue_per_transaction: Optional[float] = None

class StoreSuggestions(BaseModel):
    product_suggestions: List[RevenueSuggestion]
    pricing_suggestions: List[RevenueSuggestion]
    inventory_suggestions: List[RevenueSuggestion]
    timing_suggestions: List[RevenueSuggestion]
    customer_suggestions: List[RevenueSuggestion]

