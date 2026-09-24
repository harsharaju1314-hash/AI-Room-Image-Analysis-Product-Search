import os
import sys
import numpy as np
from PIL import Image, ImageDraw

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.session import SessionLocal, init_db
from app.models.entities import Product
from app.services.vision_model import get_vision_service

SAMPLE_PRODUCTS = [
    # Living Room Products
    {
        "name": "Modern 3-Seater Fabric Sofa",
        "category": "sofa",
        "room_type": "living_room",
        "price": 899.00,
        "description": "Contemporary minimalist charcoal grey sofa with high-density foam cushions.",
        "image_url": "/static/products/modern_sofa.jpg",
        "color_hint": (60, 60, 65)
    },
    {
        "name": "Scandinavian Oak Coffee Table",
        "category": "table",
        "room_type": "living_room",
        "price": 249.50,
        "description": "Solid natural oak round coffee table with tapered legs.",
        "image_url": "/static/products/oak_coffee_table.jpg",
        "color_hint": (180, 140, 95)
    },
    {
        "name": "Mid-Century Velvet Accent Armchair",
        "category": "armchair",
        "room_type": "living_room",
        "price": 420.00,
        "description": "Plush emerald green velvet armchair with brushed gold metal accents.",
        "image_url": "/static/products/velvet_armchair.jpg",
        "color_hint": (20, 80, 50)
    },
    {
        "name": "Arched Brass Floor Lamp",
        "category": "lighting",
        "room_type": "living_room",
        "price": 175.00,
        "description": "Overhead brass reading floor lamp with marble base.",
        "image_url": "/static/products/brass_floor_lamp.jpg",
        "color_hint": (210, 180, 70)
    },

    # Bedroom Products
    {
        "name": "Upholstered Queen Platform Bed",
        "category": "bed",
        "room_type": "bedroom",
        "price": 650.00,
        "description": "Beige woven fabric headboard with solid wooden slats support.",
        "image_url": "/static/products/queen_platform_bed.jpg",
        "color_hint": (200, 190, 175)
    },
    {
        "name": "Minimalist 2-Drawer Nightstand",
        "category": "nightstand",
        "room_type": "bedroom",
        "price": 129.99,
        "description": "Matte white bedside table with soft-closing drawer glides.",
        "image_url": "/static/products/white_nightstand.jpg",
        "color_hint": (230, 230, 235)
    },
    {
        "name": "Organic Washed Cotton Duvet Set",
        "category": "bedding",
        "room_type": "bedroom",
        "price": 110.00,
        "description": "Breathable 100% organic percale cotton duvet cover in dusty rose.",
        "image_url": "/static/products/duvet_set.jpg",
        "color_hint": (215, 170, 170)
    },
    {
        "name": "6-Drawer Wood Dresser",
        "category": "storage",
        "room_type": "bedroom",
        "price": 540.00,
        "description": "Walnut finish vertical dresser with clean cut handles.",
        "image_url": "/static/products/walnut_dresser.jpg",
        "color_hint": (110, 75, 50)
    },

    # Kitchen Products
    {
        "name": "Solid Walnut Kitchen Island Stool",
        "category": "seating",
        "room_type": "kitchen",
        "price": 145.00,
        "description": "Counter-height ergonomic wooden stool with curved saddle seat.",
        "image_url": "/static/products/island_stool.jpg",
        "color_hint": (130, 85, 45)
    },
    {
        "name": "Stainless Steel Kitchen Pendant Light",
        "category": "lighting",
        "room_type": "kitchen",
        "price": 89.00,
        "description": "Industrial style dome pendant lighting fixture in brushed nickel.",
        "image_url": "/static/products/kitchen_pendant.jpg",
        "color_hint": (160, 165, 170)
    },
    {
        "name": "Marble-Top Dining & Prep Cart",
        "category": "cart",
        "room_type": "kitchen",
        "price": 310.00,
        "description": "White Carrara marble utility cart with caster wheels and wine rack.",
        "image_url": "/static/products/marble_cart.jpg",
        "color_hint": (240, 240, 240)
    },
    {
        "name": "Matte Black Pull-Down Kitchen Faucet",
        "category": "fixture",
        "room_type": "kitchen",
        "price": 185.00,
        "description": "Single-handle high-arc pull down spray faucet with ceramic cartridge.",
        "image_url": "/static/products/black_faucet.jpg",
        "color_hint": (40, 40, 40)
    },

    # Bathroom Products
    {
        "name": "Floating Single Bathroom Vanity (36-inch)",
        "category": "vanity",
        "room_type": "bathroom",
        "price": 720.00,
        "description": "Wall-mounted natural teak vanity with integrated ceramic sink.",
        "image_url": "/static/products/teak_vanity.jpg",
        "color_hint": (175, 125, 75)
    },
    {
        "name": "LED Backlit Anti-Fog Bathroom Mirror",
        "category": "mirror",
        "room_type": "bathroom",
        "price": 195.00,
        "description": "Round 32-inch dimmable LED lighted vanity mirror.",
        "image_url": "/static/products/led_mirror.jpg",
        "color_hint": (220, 235, 245)
    },
    {
        "name": "Freestanding Acrylic Soaking Bathtub",
        "category": "bathtub",
        "room_type": "bathroom",
        "price": 1150.00,
        "description": "Contemporary 60-inch oval deep soaking tub with polished chrome overflow.",
        "image_url": "/static/products/soaking_tub.jpg",
        "color_hint": (250, 250, 255)
    },
    {
        "name": "Matte Black Shower Fixture System",
        "category": "fixture",
        "room_type": "bathroom",
        "price": 280.00,
        "description": "Rainfall showerhead system with handheld sprayer and pressure balance valve.",
        "image_url": "/static/products/shower_system.jpg",
        "color_hint": (30, 30, 35)
    }
]


def create_synthetic_product_image(color_hint: tuple, name: str) -> Image.Image:
    """
    Creates a clean 224x224 synthetic product visual to extract realistic ResNet embeddings.
    """
    img = Image.new("RGB", (224, 224), color=color_hint)
    draw = ImageDraw.Draw(img)
    # Add contrasting interior structure pattern
    draw.rectangle([40, 40, 184, 184], outline=(255, 255, 255), width=3)
    draw.ellipse([70, 70, 154, 154], fill=(color_hint[0] // 2, color_hint[1] // 2, color_hint[2] // 2))
    return img


def seed_database():
    """
    Populates the database with initial catalog products and their 512-dim visual embeddings.
    """
    print("Initializing database tables...")
    init_db()

    db = SessionLocal()
    vision_service = get_vision_service()

    try:
        existing_count = db.query(Product).count()
        if existing_count > 0:
            print(f"Database already contains {existing_count} products. Skipping duplicate seed.")
            return

        print(f"Seeding {len(SAMPLE_PRODUCTS)} catalog products with PyTorch ResNet-18 embeddings...")

        for item in SAMPLE_PRODUCTS:
            prod_img = create_synthetic_product_image(item["color_hint"], item["name"])
            embedding, _, _, _ = vision_service.process_image(prod_img)

            product = Product(
                name=item["name"],
                category=item["category"],
                room_type=item["room_type"],
                price=item["price"],
                description=item["description"],
                image_url=item["image_url"],
                embedding=embedding
            )
            db.add(product)

        db.commit()
        print(f"Successfully seeded {len(SAMPLE_PRODUCTS)} products.")
    except Exception as e:
        db.rollback()
        print(f"Error seeding database: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
