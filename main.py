# app/main.py
from fastapi import FastAPI, HTTPException
from api.endpoints import router as api_router

app = FastAPI(
    title="Retail Analytics API",
    description="API for store analysis and customer preference prediction",
    version="1.0.0"
)

app.include_router(api_router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Retail Analytics API is running"}