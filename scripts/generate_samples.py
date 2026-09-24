import os
from PIL import Image, ImageDraw

def generate_sample_images():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sample_images")
    os.makedirs(out_dir, exist_ok=True)

    samples = [
        ("living_room.jpg", (210, 195, 175), "Living Room - Sofa & Coffee Table", (100, 70, 50)),
        ("bedroom.jpg", (190, 195, 210), "Bedroom - Bed & Lamp", (70, 80, 110)),
        ("kitchen.jpg", (220, 220, 225), "Kitchen - Island & Faucet", (60, 60, 60)),
        ("bathroom.jpg", (200, 225, 230), "Bathroom - Vanity & Tub", (40, 100, 120))
    ]

    for filename, bg_color, title, element_color in samples:
        img = Image.new("RGB", (640, 480), color=bg_color)
        draw = ImageDraw.Draw(img)

        # Draw room-like walls and floor
        draw.line([(0, 360), (640, 360)], fill=(120, 100, 80), width=4)
        draw.rectangle([0, 360, 640, 480], fill=(160, 130, 100)) # Floor

        # Draw furniture elements
        draw.rectangle([120, 220, 520, 360], fill=element_color, outline=(255, 255, 255), width=2)
        draw.rectangle([200, 140, 440, 220], outline=(80, 80, 80), width=2) # Window / Picture

        path = os.path.join(out_dir, filename)
        img.save(path, "JPEG", quality=90)
        print(f"Generated sample image: {path}")

if __name__ == "__main__":
    generate_sample_images()
