# app/models/request_models.py
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime

class CustomerInfo(BaseModel):
    age: int
    gender: str
    city: str
    state: str
    membership_tier: str

class TransactionItem(BaseModel):
    product_id: str
    quantity: int
    unit_price: float
    discount: float
    final_price: float
    total: float

class PaymentInfo(BaseModel):
    method: str
    total_amount: float
    tax_amount: float
    final_amount: float

class Festival(BaseModel):
    name: str
    discount: float

class Transaction(BaseModel):
    date: datetime
    store_id: str
    items: List[TransactionItem]
    payment: PaymentInfo
    festivals: Optional[List[Festival]]
    loyalty_points_earned: int
    rating: Optional[int]

class PreferenceRequest(BaseModel):
    customer_info: CustomerInfo
    recent_transactions: Optional[List[Transaction]]

class StoreFeatures(BaseModel):
    state: str
    city: str
    category: str
    latitude: float
    longitude: float
    transaction_count: int
    avg_transaction_value: float

class PlacementRequest(BaseModel):
    num_recommendations: Optional[int] = 5