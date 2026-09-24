import pytest
import numpy as np
from app.models.entities import Product
from app.services.search_service import ProductSearchService


def test_vector_similarity_ranking(db_session):
    # Base query vector
    query_vec = [1.0] + [0.0] * 511

    # Product A: very close to query vector
    vec_a = [0.99] + [0.0] * 511
    # Product B: orthogonal to query vector
    vec_b = [0.0] * 511 + [1.0]

    p_a = Product(
        name="Close Match Sofa",
        category="sofa",
        room_type="living_room",
        price=500.0,
        embedding=vec_a
    )
    p_b = Product(
        name="Distant Item",
        category="table",
        room_type="living_room",
        price=200.0,
        embedding=vec_b
    )
    db_session.add_all([p_a, p_b])
    db_session.commit()

    results = ProductSearchService.search_similar_products(
        db=db_session,
        query_embedding=query_vec,
        room_type_filter="living_room",
        top_k=2
    )

    assert len(results) > 0
    # Top result should be p_a
    assert results[0].name == "Close Match Sofa"
    assert results[0].similarity_score > results[1].similarity_score


def test_vector_similarity_room_filter(db_session):
    query_vec = [0.5] * 512

    results = ProductSearchService.search_similar_products(
        db=db_session,
        query_embedding=query_vec,
        room_type_filter="bedroom",
        top_k=5
    )

    for item in results:
        assert item.room_type == "bedroom"
