"""
Gemini Service

Wraps Google's Gemini API for:
- Generating embeddings (for ChromaDB ingestion and queries)
- Generating chat completions (RAG-augmented responses)
- Lightweight intent classification

Uses the free tier of Gemini (gemini-1.5-flash + text-embedding-004).
"""

from typing import Optional
import google.generativeai as genai
from loguru import logger

from config import get_settings, SYSTEM_PROMPT, INTENT_LABELS


class GeminiService:
    """Service for interacting with Google's Gemini API."""

    def __init__(self):
        self.settings = get_settings()
        genai.configure(api_key=self.settings.gemini_api_key)

        self._chat_model = genai.GenerativeModel(
            model_name=self.settings.gemini_model,
            system_instruction=SYSTEM_PROMPT,
        )

        # Lightweight model for fast intent classification
        self._intent_model = genai.GenerativeModel(
            model_name=self.settings.gemini_model,
        )

    # =========================================================================
    # EMBEDDINGS
    # =========================================================================

    def embed_text(self, text: str, task_type: str = "retrieval_document") -> list[float]:
        """
        Generate an embedding vector for a single piece of text.

        task_type options: 'retrieval_document', 'retrieval_query'
        """
        result = genai.embed_content(
            model=self.settings.gemini_embedding_model,
            content=text,
            task_type=task_type,
        )
        return result["embedding"]

    def embed_batch(self, texts: list[str], task_type: str = "retrieval_document") -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_text(text, task_type=task_type))
        return embeddings

    # =========================================================================
    # INTENT CLASSIFICATION
    # =========================================================================

    def classify_intent(self, message: str) -> str:
        """
        Classify the user's message into one of INTENT_LABELS.

        Used to decide which data sources (MongoDB products/orders,
        ChromaDB policies, or none) should be queried for context.
        """
        prompt = f"""Classify the following customer message into exactly ONE of these categories:
- product_query: asking about a product's price, availability, description, or recommendations
- order_tracking: asking about order status, tracking, delivery, or a specific order
- policy_question: asking about returns, refunds, shipping policy, payments, account, or app usage
- general: greetings, small talk, or anything that doesn't fit the above

Respond with ONLY the category name, nothing else.

Message: "{message}"
Category:"""

        try:
            response = self._intent_model.generate_content(prompt)
            label = response.text.strip().lower()

            for valid_label in INTENT_LABELS:
                if valid_label in label:
                    return valid_label

            return "general"
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return "general"

    # =========================================================================
    # CHAT COMPLETION
    # =========================================================================

    def generate_response(
        self,
        user_message: str,
        context: str,
        chat_history: Optional[list[dict]] = None,
    ) -> str:
        """
        Generate a RAG-augmented chat response.

        Args:
            user_message: The latest user message.
            context: Retrieved context (product/order/policy data) as text.
            chat_history: Previous turns as [{"role": "user"|"model", "text": str}]
        """
        history = []
        for turn in (chat_history or []):
            history.append({
                "role": turn["role"],
                "parts": [turn["text"]],
            })

        prompt = f"""Context information:
{context if context.strip() else "No additional context was found for this query."}

Customer message: {user_message}

Using the context above (if relevant), respond to the customer's message following your guidelines."""

        try:
            chat = self._chat_model.start_chat(history=history)
            response = chat.send_message(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return (
                "Sorry, I'm having trouble processing your request right now. "
                "Please try again in a moment, or contact our support team for help."
            )
