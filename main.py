"""
Deligo RAG Chatbot Server - FastAPI Application

Provides a retrieval-augmented chatbot for the Deligo e-commerce platform.
Answers questions about:
- Products (via MongoDB)
- Order tracking (via MongoDB)
- Policies/FAQs (via ChromaDB vector search)

Designed to be embedded as a floating widget in the existing Next.js
customer dashboard without modifying frontend source code.

API Endpoints:
- POST /chat - Send a message and get a response
- POST /chat/clear - Clear a session's history
- GET /health - Health check endpoint
- GET /status - Service status
"""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger
import sys

from config import get_settings
from services import MongoService, VectorStoreService, GeminiService, RAGPipeline


# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

settings = get_settings()

# Global service instances
mongo_service: Optional[MongoService] = None
vector_store: Optional[VectorStoreService] = None
gemini_service: Optional[GeminiService] = None
rag_pipeline: Optional[RAGPipeline] = None


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ChatRequest(BaseModel):
    """Chat message request."""
    session_id: str = Field(..., description="Unique session identifier (e.g. browser/device ID)")
    message: str = Field(..., min_length=1, max_length=1000, description="User's message")
    user_id: Optional[str] = Field(default=None, description="Authenticated user ID, if logged in")


class ChatResponse(BaseModel):
    """Chat message response."""
    success: bool
    response: str
    intent: Optional[str] = None
    sources: list[str] = []
    timestamp: str
    error: Optional[str] = None


class ClearSessionRequest(BaseModel):
    """Clear session request."""
    session_id: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str = "1.0.0"


class StatusResponse(BaseModel):
    """Service status response."""
    service: str
    is_ready: bool
    vector_store_docs: int
    gemini_model: str
    active_sessions: int


# ============================================================================
# APPLICATION LIFECYCLE
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    global mongo_service, vector_store, gemini_service, rag_pipeline

    logger.info("Starting Deligo RAG Chatbot Server...")

    mongo_service = MongoService()
    await mongo_service.connect()

    vector_store = VectorStoreService()
    vector_store.connect()

    gemini_service = GeminiService()

    rag_pipeline = RAGPipeline(
        mongo_service=mongo_service,
        vector_store=vector_store,
        gemini_service=gemini_service,
    )

    logger.info("Chatbot server started successfully")

    yield

    logger.info("Shutting down Chatbot server...")
    if mongo_service:
        await mongo_service.disconnect()
    logger.info("Chatbot server shut down complete")


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Deligo RAG Chatbot Server",
    description="Retrieval-augmented chatbot for products, orders, and policies",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the widget JS/CSS as static files so the frontend can load it
# via a single <script src="http://chatbot-host:8005/widget/chatbot-widget.js">
app.mount("/widget", StaticFiles(directory="widget"), name="widget")


# ============================================================================
# HEALTH & STATUS
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint for load balancers."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
    )


@app.get("/status", response_model=StatusResponse, tags=["System"])
async def get_status():
    """Get detailed service status."""
    doc_count = vector_store.count() if vector_store else 0
    session_count = len(rag_pipeline._sessions) if rag_pipeline else 0

    return StatusResponse(
        service="deligo-rag-chatbot",
        is_ready=rag_pipeline is not None,
        vector_store_docs=doc_count,
        gemini_model=settings.gemini_model,
        active_sessions=session_count,
    )


# ============================================================================
# CHAT ENDPOINTS
# ============================================================================

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Send a message to the chatbot and receive a response.

    The chatbot will:
    1. Classify the intent (product query, order tracking, policy question, general)
    2. Retrieve relevant context from MongoDB and/or the policy vector store
    3. Generate a grounded response using Gemini

    **session_id**: Used to maintain short conversational context.
    Generate a random ID per browser session on the frontend (e.g. via crypto.randomUUID())
    and persist it in sessionStorage.

    **user_id**: Pass the logged-in user's ID to enable personalized order lookups.
    Omit if the user is not authenticated (guest browsing).
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="Chatbot service not initialized")

    try:
        result = await rag_pipeline.handle_message(
            session_id=request.session_id,
            message=request.message,
            user_id=request.user_id,
        )

        return ChatResponse(
            success=True,
            response=result["response"],
            intent=result["intent"],
            sources=result["sources"],
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        logger.error(f"Chat request failed: {e}")
        return ChatResponse(
            success=False,
            response="Sorry, something went wrong. Please try again shortly.",
            timestamp=datetime.utcnow().isoformat(),
            error=str(e) if settings.debug else "Internal error",
        )


@app.post("/chat/clear", tags=["Chat"])
async def clear_chat(request: ClearSessionRequest):
    """Clear a chat session's conversation history."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="Chatbot service not initialized")

    cleared = rag_pipeline.clear_session(request.session_id)
    return {
        "success": True,
        "cleared": cleared,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug",
    )
