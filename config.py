from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Redis/Celery
    redis_url: str = "redis://redis:6379/0"
    
    # Storage paths
    data_dir: str = "/app/data"
    uploads_dir: str = "uploads"
    results_dir: str = "results"
    
    # Cleanup
    max_age_days: int = 7
    cleanup_interval_hours: int = 24
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Processing
    default_mode: str = "gray"
    default_backend: str = "auto"
    
    class Config:
        env_file = ".env"
        env_prefix = "SCANARIO_"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
