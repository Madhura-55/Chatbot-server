"""
RAG Pipeline Service

Orchestrates the full request flow:
1. Classify user intent (via OpenRouter)
2. Retrieve relevant context from MongoDB (products/orders) and/or
   ChromaDB (policy documents) based on intent
3. Generate a grounded response via OpenRouter
4. Maintain short-lived in-memory chat session history

Note: embeddings are generated via Gemini (EmbeddingService) since
OpenRouter does not provide an embeddings endpoint.
"""

import re
import time
from typing import Optional
from loguru import logger

from config import get_settings
from .mongo_service import MongoService
from .vector_store import VectorStoreService
from .embedding_service import EmbeddingService
from .openrouter_service import OpenRouterService


# Matches a 24-character hex string (MongoDB ObjectId)
OBJECT_ID_PATTERN = re.compile(r"\b[a-fA-F0-9]{24}\b")


class RAGPipeline:
    """Coordinates retrieval-augmented generation for the chatbot."""

    def __init__(
        self,
        mongo_service: MongoService,
        vector_store: VectorStoreService,
        embedding_service: EmbeddingService,
        llm_service: OpenRouterService,
    ):
        self.settings = get_settings()
        self.mongo = mongo_service
        self.vector_store = vector_store
        self.embeddings = embedding_service
        self.llm = llm_service

        # In-memory session store: session_id -> {"history": [...], "last_active": ts}
        self._sessions: dict[str, dict] = {}

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================

    def _get_session(self, session_id: str) -> dict:
        """Get or create a chat session, pruning expired ones."""
        now = time.time()
        ttl = self.settings.chat_session_ttl_seconds

        # Prune expired sessions
        expired = [sid for sid, s in self._sessions.items() if now - s["last_active"] > ttl]
        for sid in expired:
            del self._sessions[sid]

        if session_id not in self._sessions:
            self._sessions[session_id] = {"history": [], "last_active": now}

        self._sessions[session_id]["last_active"] = now
        return self._sessions[session_id]

    def _append_history(self, session: dict, role: str, text: str) -> None:
        session["history"].append({"role": role, "text": text})
        max_messages = self.settings.max_history_messages
        if len(session["history"]) > max_messages:
            session["history"] = session["history"][-max_messages:]

    def clear_session(self, session_id: str) -> bool:
        """Clear a session's history. Returns True if it existed."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    async def handle_message(
        self,
        session_id: str,
        message: str,
        user_id: Optional[str] = None,
    ) -> dict:
        """
        Process a user message end-to-end and return a chat response.

        Args:
            session_id: Unique identifier for the chat session (e.g. browser session)
            message: The user's message text
            user_id: Optional authenticated user ID (for order lookups)

        Returns:
            dict with 'response', 'intent', and 'sources'
        """
        session = self._get_session(session_id)

        intent = self.llm.classify_intent(message)
        logger.info(f"Session {session_id}: intent='{intent}' message='{message[:80]}'")

        context_text = ""
        sources: list[str] = []

        if intent == "product_query":
            context_text, sources = await self._build_product_context(message)
        elif intent == "order_tracking":
            context_text, sources = await self._build_order_context(message, user_id)
        elif intent == "policy_question":
            context_text, sources = self._build_policy_context(message)
        else:
            # For general chat, still try a light policy lookup in case it's relevant
            context_text, sources = self._build_policy_context(message, n_results=2)

        response_text = self.llm.generate_response(
            user_message=message,
            context=context_text,
            chat_history=session["history"],
        )

        self._append_history(session, "user", message)
        self._append_history(session, "model", response_text)

        return {
            "response": response_text,
            "intent": intent,
            "sources": sources,
        }

    # =========================================================================
    # CONTEXT BUILDERS
    # =========================================================================

    async def _build_product_context(self, message: str) -> tuple[str, list[str]]:
        """Build context from MongoDB product search results."""
        products = await self.mongo.search_products(message, limit=5)

        if not products:
            return "No matching products were found in the catalog.", []

        lines = ["Matching products from the catalog:"]
        sources = []
        for p in products:
            lines.append(
                f"- {p['name']} | Price: ₹{p['discount_price']} "
                f"(orig ₹{p['price']}) | Stock: {p['stock']} | "
                f"Rating: {p['rating']} | Product ID: {p['product_id']}"
            )
            sources.append(f"product:{p['product_id']}")

        return "\n".join(lines), sources

    async def _build_order_context(
        self, message: str, user_id: Optional[str]
    ) -> tuple[str, list[str]]:
        """Build context from MongoDB order data."""
        # Try to extract an explicit order ID (24-char hex) from the message
        match = OBJECT_ID_PATTERN.search(message)

        order = None
        if match:
            order = await self.mongo.get_order_by_id(match.group(0), user_id=user_id)

        if order:
            return self._format_order_for_context([order]), [f"order:{order['order_id']}"]

        # No explicit order ID found - fall back to user's recent orders (if authenticated)
        if user_id:
            recent_orders = await self.mongo.get_recent_orders(user_id, limit=3)
            if recent_orders:
                sources = [f"order:{o['order_id']}" for o in recent_orders]
                return self._format_order_for_context(recent_orders), sources

        return (
            "No order information could be found. The user may need to provide "
            "their order ID, or may not be logged in.",
            [],
        )

    def _format_order_for_context(self, orders: list[dict]) -> str:
        lines = ["Order information:"]
        for o in orders:
            items_str = ", ".join(
                f"{i['name']} (x{i['quantity']})" for i in o["items"]
            ) or "N/A"
            lines.append(
                f"- Order ID: {o['order_id']} | Status: {o['status']} | "
                f"Total: ₹{o['total_amount']} | Items: {items_str} | "
                f"Tracking: {o.get('tracking_number') or 'N/A'} | "
                f"Placed: {o.get('created_at') or 'N/A'} | "
                f"Last updated: {o.get('updated_at') or 'N/A'}"
            )
        return "\n".join(lines)

    def _build_policy_context(self, message: str, n_results: Optional[int] = None) -> tuple[str, list[str]]:
        """Build context from ChromaDB policy/documentation search."""
        n = n_results or self.settings.max_context_docs

        try:
            query_embedding = self.embeddings.embed_text(message, task_type="retrieval_query")
            results = self.vector_store.query(query_embedding, n_results=n)
        except Exception as e:
            logger.error(f"Policy retrieval failed: {e}")
            return "", []

        if not results:
            return "", []

        lines = ["Relevant policy/documentation excerpts:"]
        sources = []
        for r in results:
            title = r["metadata"].get("title", "Policy Document")
            lines.append(f"[{title}]\n{r['document']}")
            sources.append(f"policy:{title}")

        return "\n\n".join(lines), sources