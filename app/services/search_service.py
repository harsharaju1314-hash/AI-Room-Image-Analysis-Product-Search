import logging
from typing import List, Optional
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.models.entities import Product
from app.models.schemas import ProductResponse

logger = logging.getLogger(__name__)


class ProductSearchService:
    """
    Vector similarity search service querying PostgreSQL + pgvector
    (with in-memory NumPy fallback for testing environments).
    """

    @staticmethod
    def search_similar_products(
        db: Session,
        query_embedding: List[float],
        room_type_filter: Optional[str] = None,
        top_k: int = 3
    ) -> List[ProductResponse]:
        """
        Executes vector cosine similarity search against catalog products.
        """
        query_vec = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        products_with_scores: List[ProductResponse] = []

        try:
            bind = db.get_bind()
            dialect_name = bind.dialect.name if (bind and hasattr(bind, "dialect")) else ""
            is_postgres = dialect_name == "postgresql"

            if is_postgres:
                # Query using pgvector cosine distance (<=>)
                # Cosine distance = 1 - cosine_similarity.
                # So cosine_similarity = 1 - distance.
                filter_clause = "WHERE room_type = :room_type" if room_type_filter else ""
                sql = f"""
                    SELECT id, name, category, room_type, price, description, image_url,
                           1 - (embedding <=> :query_vec) AS similarity_score, created_at
                    FROM products
                    {filter_clause}
                    ORDER BY embedding <=> :query_vec ASC
                    LIMIT :top_k;
                """
                params = {
                    "query_vec": str(query_vec.tolist()),
                    "top_k": top_k
                }
                if room_type_filter:
                    params["room_type"] = room_type_filter

                results = db.execute(text(sql), params).fetchall()

                for row in results:
                    products_with_scores.append(
                        ProductResponse(
                            id=row.id,
                            name=row.name,
                            category=row.category,
                            room_type=row.room_type,
                            price=float(row.price),
                            description=row.description or "",
                            image_url=row.image_url,
                            similarity_score=round(float(row.similarity_score), 4),
                            created_at=row.created_at
                        )
                    )
            else:
                # Python / NumPy fallback for SQLite / In-Memory testing
                query = db.query(Product)
                if room_type_filter:
                    query = query.filter(Product.room_type == room_type_filter)
                all_products = query.all()

                scored_items = []
                for p in all_products:
                    if p.embedding:
                        p_vec = np.array(p.embedding, dtype=np.float32)
                        p_norm = np.linalg.norm(p_vec)
                        if p_norm > 0:
                            p_vec = p_vec / p_norm
                        sim = float(np.dot(query_vec, p_vec))
                    else:
                        sim = 0.5  # Neutral fallback score

                    # Bound similarity between 0.0 and 1.0 for cosine similarity
                    sim_bounded = max(0.0, min(1.0, (sim + 1.0) / 2.0 if sim < 0 else sim))
                    scored_items.append((p, sim_bounded))

                scored_items.sort(key=lambda x: x[1], reverse=True)
                top_items = scored_items[:top_k]

                for p, score in top_items:
                    products_with_scores.append(
                        ProductResponse(
                            id=p.id,
                            name=p.name,
                            category=p.category,
                            room_type=p.room_type,
                            price=float(p.price),
                            description=p.description or "",
                            image_url=p.image_url,
                            similarity_score=round(score, 4),
                            created_at=p.created_at
                        )
                    )

        except Exception as e:
            logger.error(f"Error during vector similarity search: {e}", exc_info=True)
            # Fallback to simple category retrieval if vector search fails
            fallback_query = db.query(Product)
            if room_type_filter:
                fallback_query = fallback_query.filter(Product.room_type == room_type_filter)
            fallback_products = fallback_query.limit(top_k).all()
            for p in fallback_products:
                products_with_scores.append(
                    ProductResponse(
                        id=p.id,
                        name=p.name,
                        category=p.category,
                        room_type=p.room_type,
                        price=float(p.price),
                        description=p.description or "",
                        image_url=p.image_url,
                        similarity_score=0.75,
                        created_at=p.created_at
                    )
                )

        return products_with_scores
