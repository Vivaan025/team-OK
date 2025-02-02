from pydantic import BaseModel, validator
from typing import List, Dict, Optional, Union, Any, Tuple
from datetime import datetime

class MetricComparison(BaseModel):
    """Comparison of a single metric between two stores"""
    store1_value: float
    store2_value: float
    difference: float
    percentage_difference: float
    better_performer: str  # store_id of better performing store or "equal"
    insight: str
    additional_metrics: Optional[Dict[str, Any]] = None

class TransactionMetrics(BaseModel):
    """Transaction-related metrics"""
    total_transactions: int
    avg_transaction_value: float
    peak_hours: List[int]
    popular_payment_methods: List[str]
    customer_retention_rate: float

class ProductMetrics(BaseModel):
    """Product-related metrics"""
    top_categories: List[Tuple[str, float]]  # (category_name, sales_value)
    avg_margin: float
    stock_turnover_rate: float
    stockout_frequency: float
    category_performance: Dict[str, float]

class CustomerMetrics(BaseModel):
    """Customer-related metrics"""
    total_customers: int
    repeat_customers: int
    avg_customer_lifetime_value: float
    membership_distribution: Dict[str, int]
    customer_satisfaction: Optional[float] = None
    
    @property
    def retention_rate(self) -> float:
        """Calculate retention rate from customer metrics"""
        return self.repeat_customers / self.total_customers if self.total_customers > 0 else 0

    @property
    def customer_retention_rate(self) -> float:
        """Alias for retention_rate to maintain compatibility"""
        return self.retention_rate

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True

class OperationalMetrics(BaseModel):
    """Operational metrics"""
    amenities: List[str]
    operating_hours: Dict[str, str]
    staff_count: Optional[int]
    area_size: Optional[float]
    parking_available: bool

class StoreComparisonResponse(BaseModel):
    """Complete store comparison response"""
    store1_id: str
    store2_id: str
    store1_name: str
    store2_name: str
    time_period: str  # e.g., "Last 90 days"
    
    # Revenue metrics
    revenue_comparison: MetricComparison
    profit_comparison: MetricComparison
    
    # Transaction metrics
    transaction_metrics_1: TransactionMetrics
    transaction_metrics_2: TransactionMetrics
    transaction_comparison: MetricComparison
    
    # Product metrics
    product_metrics_1: ProductMetrics
    product_metrics_2: ProductMetrics
    product_comparison: Dict[str, MetricComparison]
    
    # Customer metrics
    customer_metrics_1: CustomerMetrics
    customer_metrics_2: CustomerMetrics
    customer_comparison: Dict[str, MetricComparison]
    
    # Operational comparison
    operational_metrics_1: OperationalMetrics
    operational_metrics_2: OperationalMetrics
    unique_advantages_1: List[str]
    unique_advantages_2: List[str]
    
    # Overall comparison
    overall_score_1: float
    overall_score_2: float
    key_insights: List[str]
    improvement_suggestions: Dict[str, List[str]]