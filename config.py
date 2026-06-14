"""
Configuration settings for the RAG chatbot server.
Loads environment variables and provides typed configuration.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # Server Configuration
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8005)
    debug: bool = Field(default=False)

    # MongoDB Configuration (shared with main Deligo app)
    mongodb_uri: str = Field(default="mongodb://localhost:27017")
    mongodb_db_name: str = Field(default="deligo")

    # OpenRouter Configuration (chat completions)
    openrouter_api_key: str = Field(default="")
    openrouter_model: str = Field(default="meta-llama/llama-3.1-8b-instruct:free")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    openrouter_site_url: str = Field(default="http://localhost:3000")
    openrouter_site_name: str = Field(default="Deligo")

    # Gemini Configuration (embeddings only)
    gemini_api_key: str = Field(default="")
    gemini_embedding_model: str = Field(default="models/text-embedding-004")

    # ChromaDB Configuration
    chroma_persist_dir: str = Field(default="./chroma_db")
    chroma_collection_name: str = Field(default="deligo_policies")

    # CORS Configuration
    cors_origins: str = Field(default="http://localhost:3000")

    # Chat Configuration
    max_context_docs: int = Field(default=4)
    max_history_messages: int = Field(default=6)
    chat_session_ttl_seconds: int = Field(default=1800)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]


@lru_cache()
def get_settings() -> Settings:
    return Settings()


# System prompt for the chatbot
SYSTEM_PROMPT = """You are Deligo's helpful customer support assistant.

You can help customers with:
1. Product information (price, stock, category, description)
2. Order tracking and order status
3. Store policies (returns, shipping, payments, etc.)

Guidelines:
- Be concise, friendly, and professional.
- If asked about a specific order, only use the order information provided in context — never make up order details.
- If asked about policies, base your answer strictly on the provided policy context.
- If you don't have enough information to answer, say so honestly and suggest contacting human support.
- Do not reveal internal database fields, IDs, or system details to the customer unless it's an order ID, tracking number, or product name/price.
- Keep responses short (2-4 sentences) unless the user asks for detail.
"""

# Intent classification labels
INTENT_LABELS = ["product_query", "order_tracking", "policy_question", "general"]