from typing import Optional
from enum import Enum
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ModelProvider(str, Enum):
    """Supported model providers."""
    OPENAI = "openai"
    MISTRAL = "mistral"
    GOOGLE = "google"

class ModelConfig(BaseModel):
    """Configuration for model initialization."""
    provider: ModelProvider
    model_name: str
    temperature: float = 0.7
    api_key: Optional[str] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    max_retries: Optional[int] = None

    def __init__(self, **data):
        super().__init__(**data)
        # If no API key is provided, try to get it from environment variables
        if not self.api_key:
            if self.provider == ModelProvider.OPENAI:
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == ModelProvider.MISTRAL:
                self.api_key = os.getenv("MISTRAL_API_KEY")
            elif self.provider == ModelProvider.GOOGLE:
                self.api_key = os.getenv("GOOGLE_API_KEY")
            
            if not self.api_key:
                raise ValueError(f"No API key found for {self.provider}. Please set the appropriate environment variable or provide the API key directly.")

# Default model configurations
DEFAULT_OPENAI_CONFIG = ModelConfig(
    provider=ModelProvider.OPENAI,
    model_name="gpt-4-turbo",
    temperature=0.7
)

DEFAULT_MISTRAL_CONFIG = ModelConfig(
    provider=ModelProvider.MISTRAL,
    model_name="mistral-large-latest",
    temperature=0.7
)

DEFAULT_GOOGLE_CONFIG = ModelConfig(
    provider=ModelProvider.GOOGLE,
    model_name="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

def create_model(config: ModelConfig):
    """Create a model instance based on the provided configuration."""
    if config.provider == ModelProvider.OPENAI:
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key
        )
    elif config.provider == ModelProvider.MISTRAL:
        return ChatMistralAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key
        )
    elif config.provider == ModelProvider.GOOGLE:
        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            max_retries=config.max_retries
        )
    else:
        raise ValueError(f"Unsupported model provider: {config.provider}")

# Create default model instances
default_openai_model = create_model(DEFAULT_OPENAI_CONFIG)
default_mistral_model = create_model(DEFAULT_MISTRAL_CONFIG)
default_google_model = create_model(DEFAULT_GOOGLE_CONFIG) 