from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    mongodb_uri: str = "mongodb://localhost:27017/"
    database_name: str = "indian_retail"
    model_path: str = "retail_optimized_model.txt"
    preprocessing_path: str = "retail_optimized_model_preprocessing.joblib"
    model_path_1: str = "retail_optimized_model_1.txt"
    preprocessing_path_1: str = "retail_optimized_model_preprocessing_1.joblib"

    class Config:
        env_file = ".env"

settings = Settings()