"""
MongoDB Service

Provides read access to product and order data from the shared
Deligo MongoDB database for the chatbot's RAG pipeline.
"""

from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from loguru import logger
from bson import ObjectId
from bson.errors import InvalidId

from config import get_settings


class MongoService:
    """Async MongoDB service for product and order lookups."""

    def __init__(self):
        self.settings = get_settings()
        self._client: Optional[AsyncIOMotorClient] = None
        self._db: Optional[AsyncIOMotorDatabase] = None

    async def connect(self) -> None:
        if self._db is None:
            self._client = AsyncIOMotorClient(self.settings.mongodb_uri)
            self._db = self._client[self.settings.mongodb_db_name]
            logger.info(f"Connected to MongoDB: {self.settings.mongodb_db_name}")

    async def disconnect(self) -> None:
        if self._client:
            self._client.close()
            logger.info("MongoDB connection closed")

    @property
    def db(self) -> AsyncIOMotorDatabase:
        if self._db is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._db

    # =========================================================================
    # PRODUCT QUERIES
    # =========================================================================

    async def search_products(self, query: str, limit: int = 5) -> list[dict]:
        """
        Search products by name, description, category, or tags
        using a case-insensitive text match.
        """
        await self.connect()

        regex = {"$regex": query, "$options": "i"}
        mongo_query = {
            "status": {"$in": ["active", "published"]},
            "$or": [
                {"name": regex},
                {"description": regex},
                {"tags": regex},
            ],
        }

        products = []
        cursor = self.db.products.find(mongo_query).limit(limit)
        async for doc in cursor:
            products.append(self._format_product(doc))

        logger.debug(f"Product search '{query}' returned {len(products)} results")
        return products

    async def get_product_by_id(self, product_id: str) -> Optional[dict]:
        """Fetch a single product by its ObjectId string."""
        await self.connect()

        try:
            obj_id = ObjectId(product_id)
        except InvalidId:
            return None

        doc = await self.db.products.find_one({"_id": obj_id})
        return self._format_product(doc) if doc else None

    def _format_product(self, doc: dict) -> dict:
        """Format a raw product document for chatbot context."""
        return {
            "product_id": str(doc["_id"]),
            "name": doc.get("name", ""),
            "description": doc.get("description", ""),
            "price": float(doc.get("price", 0)),
            "discount_price": float(doc.get("discountPrice", doc.get("price", 0))),
            "stock": int(doc.get("stock", 0)),
            "category": doc.get("category", ""),
            "rating": float(doc.get("averageRating", doc.get("rating", 0))),
            "tags": doc.get("tags", []),
        }

    # =========================================================================
    # ORDER QUERIES
    # =========================================================================

    async def get_order_by_id(self, order_id: str, user_id: Optional[str] = None) -> Optional[dict]:
        """
        Fetch an order by its ObjectId string.

        If user_id is provided, the order must belong to that user
        (prevents customers from looking up other users' orders).
        """
        await self.connect()

        try:
            obj_id = ObjectId(order_id)
        except InvalidId:
            return None

        query: dict = {"_id": obj_id}
        if user_id:
            try:
                query["userId"] = ObjectId(user_id)
            except InvalidId:
                query["userId"] = user_id

        doc = await self.db.orders.find_one(query)
        return self._format_order(doc) if doc else None

    async def get_recent_orders(self, user_id: str, limit: int = 5) -> list[dict]:
        """Fetch a user's most recent orders, sorted by creation date."""
        await self.connect()

        try:
            obj_id = ObjectId(user_id)
        except InvalidId:
            return []

        orders = []
        cursor = (
            self.db.orders.find({"userId": obj_id})
            .sort("createdAt", -1)
            .limit(limit)
        )
        async for doc in cursor:
            orders.append(self._format_order(doc))

        return orders

    async def get_order_by_tracking_number(
        self, tracking_number: str, user_id: Optional[str] = None
    ) -> Optional[dict]:
        """Fetch an order by its tracking/shipment number."""
        await self.connect()

        query: dict = {"trackingNumber": tracking_number}
        if user_id:
            try:
                query["userId"] = ObjectId(user_id)
            except InvalidId:
                query["userId"] = user_id

        doc = await self.db.orders.find_one(query)
        return self._format_order(doc) if doc else None

    def _format_order(self, doc: dict) -> dict:
        """Format a raw order document for chatbot context."""
        items = []
        for item in doc.get("items", []):
            items.append({
                "product_id": str(item.get("productId") or item.get("product") or ""),
                "name": item.get("name", ""),
                "quantity": item.get("quantity", 1),
                "price": float(item.get("price", 0)),
            })

        created_at = doc.get("createdAt")
        updated_at = doc.get("updatedAt")

        return {
            "order_id": str(doc["_id"]),
            "status": doc.get("status", "unknown"),
            "items": items,
            "total_amount": float(doc.get("totalAmount", doc.get("total", 0))),
            "tracking_number": doc.get("trackingNumber"),
            "estimated_delivery": doc.get("estimatedDelivery"),
            "shipping_address": doc.get("shippingAddress", {}),
            "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else created_at,
            "updated_at": updated_at.isoformat() if hasattr(updated_at, "isoformat") else updated_at,
        }

    # =========================================================================
    # USER QUERIES
    # =========================================================================

    async def get_user_by_id(self, user_id: str) -> Optional[dict]:
        """Fetch basic user info (for personalizing chatbot greetings)."""
        await self.connect()

        try:
            obj_id = ObjectId(user_id)
        except InvalidId:
            return None

        doc = await self.db.users.find_one({"_id": obj_id})
        if not doc:
            return None

        return {
            "user_id": str(doc["_id"]),
            "name": doc.get("name", ""),
            "email": doc.get("email", ""),
        }
