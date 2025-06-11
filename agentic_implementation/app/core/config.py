from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "Agentic Web Content Analysis"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    
    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/agentic_db")
    
    # Wikipedia Configuration
    WIKIPEDIA_LANGUAGE: str = "en"
    WIKIPEDIA_MAX_RESULTS: int = 5
    
    # Content Processing
    MAX_CONTENT_LENGTH: int = 10000
    SUMMARY_MAX_LENGTH: int = 500
    
    class Config:
        case_sensitive = True

settings = Settings() 