"""
OpenRouter Service

Wraps OpenRouter's OpenAI-compatible chat completions API for:
- Generating RAG-augmented chat responses
- Lightweight intent classification

Uses the official `openai` SDK pointed at OpenRouter's base URL.
Free-tier models (suffixed ':free') are available, e.g.
'meta-llama/llama-3.1-8b-instruct:free'.
"""

from typing import Optional
from openai import OpenAI
from loguru import logger

from config import get_settings, SYSTEM_PROMPT, INTENT_LABELS


class OpenRouterService:
    """Service for interacting with OpenRouter's chat completion API."""

    def __init__(self):
        self.settings = get_settings()

        self._client = OpenAI(
            base_url=self.settings.openrouter_base_url,
            api_key=self.settings.openrouter_api_key,
        )

        # Extra headers recommended by OpenRouter for attribution/rankings
        self._extra_headers = {
            "HTTP-Referer": self.settings.openrouter_site_url,
            "X-Title": self.settings.openrouter_site_name,
        }

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
            response = self._client.chat.completions.create(
                model=self.settings.openrouter_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=20,
                extra_headers=self._extra_headers,
            )
            label = response.choices[0].message.content.strip().lower()

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
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for turn in (chat_history or []):
            role = "assistant" if turn["role"] == "model" else "user"
            messages.append({"role": role, "content": turn["text"]})

        prompt = f"""Context information:
{context if context.strip() else "No additional context was found for this query."}

Customer message: {user_message}

Using the context above (if relevant), respond to the customer's message following your guidelines."""

        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self.settings.openrouter_model,
                messages=messages,
                temperature=0.4,
                max_tokens=400,
                extra_headers=self._extra_headers,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenRouter generation failed: {e}")
            return (
                "Sorry, I'm having trouble processing your request right now. "
                "Please try again in a moment, or contact our support team for help."
            )