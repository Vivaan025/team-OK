from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
from typing import Any

class ProductFeatures(BaseModel):
    """Features used for product revenue prediction"""
    product_id: str
    category: str
    subcategory: str
    base_price: float
    current_price: float
    current_stock: int
    avg_monthly_sales: float
    seasonal_multipliers: Dict[int, float]  # Month -> multiplier
    festival_impact: float  # Impact score for festive seasons
    margin: float
    days_since_launch: int
    stockout_frequency: float
    customer_rating: Optional[float]
    competitor_price_ratio: Optional[float]

class SeasonalPattern(BaseModel):
    """Seasonal patterns for product categories"""
    category: str
    monthly_factors: Dict[int, float]  # Month -> factor
    festival_boost: Dict[str, float]  # Festival -> boost factor
    weather_impact: Dict[str, float]  # Weather condition -> impact factor

class RevenueProjection(BaseModel):
    """Revenue projection for a product"""
    product_id: str
    product_name: str
    current_revenue: float
    projected_revenue: float
    growth_potential: float
    confidence_score: float
    contributing_factors: List[str]
    seasonal_index: float
    optimal_price: Optional[float]
    optimal_stock: Optional[int]

class RevenuePredictionResponse(BaseModel):
    """Complete response for revenue prediction"""
    store_id: str
    prediction_period: str
    predictions: List[RevenueProjection]
    total_projected_increase: float
    top_categories: List[str]
    seasonal_opportunities: List[Dict[str, Any]]
    festival_opportunities: List[Dict[str, Any]]
    confidence_level: float

class InventoryOptimization(BaseModel):
    """Inventory optimization recommendations"""
    product_id: str
    optimal_stock_level: int
    reorder_point: int
    safety_stock: int
    seasonal_adjustment: float
    demand_forecast: float