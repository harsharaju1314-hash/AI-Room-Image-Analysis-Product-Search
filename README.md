# AI Room Image Analysis & Product Search

A lightweight REST service for interior room classification, visual complexity analysis, and embedding-based product similarity search using TensorFlow / Keras, OpenCV, scikit-image, FastAPI, and PostgreSQL with pgvector.

---

## Overview

In home design, e-commerce, and interior space planning, customers frequently have inspiration photos of rooms but lack an efficient way to identify the room type and discover matching furniture or fixtures.

This project implements a complete computer vision and vector search pipeline:
1. Ingests and sanitizes user-uploaded interior space photos.
2. Extracts visual quality and complexity metrics (Shannon entropy and GLCM contrast).
3. Classifies room type (`living_room`, `bedroom`, `kitchen`, `bathroom`) and generates a 512-dimensional visual embedding using a pretrained **TensorFlow MobileNetV2** model.
4. Searches a home-design product catalog using vector cosine similarity powered by **PostgreSQL and pgvector**.
5. Returns structured JSON analysis results and recommended products through a **FastAPI** service.

---

## Problem Statement

Manual catalog tagging and keyword-based search fail when users search using interior imagery. Visual search bridges this gap by comparing high-level visual representations directly against product inventories.

This service solves three key technical challenges:
* **Automated Room Understanding**: Recognizing interior scenes without manual metadata input.
* **Vector Similarity Matching**: Querying catalog items based on feature proximity in embedding space.
* **Defensive Ingestion**: Enforcing file validation, size limits, and header verification before processing.

---

## Key Features

* **Multi-Stage Image Ingestion**: Safe file validation using Pillow header verification (`Image.verify()`), size bounding, and format whitelisting.
* **Visual Metric Extraction**: Computes image dimensions, Shannon entropy (visual detail density), and Gray-Level Co-occurrence Matrix (GLCM) contrast using scikit-image and OpenCV.
* **TensorFlow Feature Extraction**: Uses a pretrained MobileNetV2 backbone to extract L2-normalized 512-dimensional visual embeddings.
* **Room Scene Classification**: Evaluates class probabilities across four core interior categories: Living Room, Bedroom, Kitchen, and Bathroom.
* **Vector Cosine Similarity Search**: Executes nearest-neighbor queries against PostgreSQL with pgvector (using the `<=>` cosine distance operator) with room-type filtering.
* **Automated Test Suite**: 18 unit and integration tests covering security, model inference, image processing, and API contracts.
* **Continuous Integration**: Azure DevOps CI pipeline automating dependency installation, linting, and test execution.

---

## Tech Stack

| Technology | Purpose |
| :--- | :--- |
| **Python 3.12** | Core application programming language |
| **TensorFlow & Keras** | Pretrained MobileNetV2 backbone for embedding extraction and scene classification |
| **OpenCV (`cv2`)** | Image matrix resizing (224x224), array formatting, and color space conversion |
| **Pillow (`PIL`)** | Stream decoding, format verification, and header validation |
| **scikit-image** | Computation of Shannon entropy and GLCM texture contrast metrics |
| **FastAPI** | High-performance asynchronous REST API framework |
| **PostgreSQL & pgvector** | Relational storage for analysis records and vector similarity search for embeddings |
| **SQLAlchemy** | ORM for database schema management and query construction |
| **pytest & httpx** | Unit and API integration test suite |
| **Azure DevOps** | CI pipeline for automated testing and validation |
| **Git** | Source code version control |

---

## System Architecture / Workflow

```
User / Client
      │
      ▼  (POST /analyze with image)
┌───────────────────────────────────────────────────────────┐
│ FastAPI Application (app/main.py)                         │
│  ├─ Security Validation (Extension, Size, PIL Verify)     │
│  ├─ Image Processor (OpenCV resize, scikit-image stats)   │
│  └─ TensorFlow MobileNetV2 (512-dim embedding & Room Class)│
└────────────┬─────────────────────────────────┬────────────┘
             │                                 │
             ▼                                 ▼
┌───────────────────────────┐   ┌────────────────────────────┐
│ PostgreSQL (pgvector)     │   │ Structured JSON Response   │
│  ├─ room_analyses table   │   │  ├─ Room type & score      │
│  ├─ products catalog      │   │  ├─ Image metrics          │
│  └─ Cosine Search (<=>)   │   │  └─ Similar products       │
└───────────────────────────┘   └────────────────────────────┘
```

### Execution Flow:
1. **Request Ingestion**: Client submits an image file to `POST /analyze`.
2. **Security & Validation**: File size is capped (5MB max), extension is verified, and Pillow validates binary image integrity.
3. **Preprocessing & Quality Metrics**: Image is converted to RGB, resized via OpenCV, and analyzed for entropy and GLCM contrast via scikit-image.
4. **Model Inference**: TensorFlow MobileNetV2 runs inference; feature extractor extracts the 512-dim L2-normalized embedding, while classification heads determine the room type.
5. **Vector Search**: PostgreSQL executes a pgvector cosine distance query (`<=>`) to find top-k matching products.
6. **Persistence & Response**: Analysis metadata is stored in the database, and the JSON payload is returned to the client.

---

## Project Structure

```
AI-Room-Image-Analysis-Product-Search/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application entrypoint & lifecycle
│   ├── api/
│   │   ├── __init__.py
│   │   └── endpoints.py         # REST route handlers (/analyze, /health, etc.)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py            # Pydantic Settings & environment configuration
│   │   └── security.py          # File upload validation & size guards
│   ├── db/
│   │   ├── __init__.py
│   │   └── session.py           # Database engine & session management
│   ├── models/
│   │   ├── __init__.py
│   │   ├── entities.py          # SQLAlchemy ORM models (Product, RoomAnalysis)
│   │   └── schemas.py           # Pydantic request & response schemas
│   └── services/
│       ├── __init__.py
│       ├── image_processor.py   # Pillow, OpenCV, scikit-image processing
│       ├── search_service.py    # Vector cosine similarity search
│       └── vision_model.py      # TensorFlow MobileNetV2 feature extractor
├── data/
│   └── sample_images/           # Sample room images for testing
├── scripts/
│   ├── generate_samples.py      # Generates synthetic test room imagery
│   └── seed_catalog.py          # Seeds catalog products with visual embeddings
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Pytest fixtures & in-memory test DB
│   ├── test_api.py              # End-to-end API & security tests
│   ├── test_image_processor.py  # Image preprocessing & metric unit tests
│   ├── test_vector_search.py    # Vector similarity ranking unit tests
│   └── test_vision.py           # TensorFlow model & embedding shape tests
├── azure-pipelines.yml          # Azure DevOps CI workflow
├── requirements.txt             # Pinned project dependencies
├── INTERVIEW_PREP.md            # Technical interview Q&A reference
├── .env.example                 # Environment configuration template
└── README.md
```

---

## Machine Learning & Computer Vision Details

* **Dataset / Seed Catalog**: Contains curated home design catalog items across 4 interior rooms (Living Room, Bedroom, Kitchen, Bathroom) with product descriptions, prices, and visual vectors.
* **Image Preprocessing**:
  * Input resized to $(224 \times 224)$ via OpenCV bilinear interpolation.
  * Normalized with standard MobileNetV2 preprocessing to `[-1.0, 1.0]`.
* **Feature Extraction**:
  * Backbone: TensorFlow `tf.keras.applications.MobileNetV2(weights="imagenet")`.
  * Intercepts `global_average_pooling2d` representation and projects to a 512-dimensional vector.
  * Feature vectors are L2-normalized: $\|v\|_2 = 1.0$.
* **Visual Metrics**:
  * **Shannon Entropy**: Measures scene detail density / visual information content.
  * **GLCM Contrast**: Measures texture definition and lighting variance.

---

## API Endpoints

| Method | Endpoint | Purpose |
| :--- | :--- | :--- |
| `GET` | `/health` | Returns service status, model availability, and DB connection state |
| `POST` | `/analyze` | Ingests room image, extracts embedding, and returns room type with matching products |
| `GET` | `/analysis/{id}` | Retrieves past room analysis record and matched products by ID |
| `GET` | `/products/{id}` | Retrieves specific catalog product details |
| `GET` | `/docs` | Interactive OpenAPI / Swagger documentation |

---

## Database Schema

The database uses PostgreSQL with the `pgvector` extension.

### `products` Table
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | `INTEGER PRIMARY KEY` | Auto-incrementing product identifier |
| `name` | `VARCHAR(255)` | Product name |
| `category` | `VARCHAR(100)` | Product category (sofa, bed, table, etc.) |
| `room_type` | `VARCHAR(50)` | Associated room category |
| `price` | `FLOAT` | Product price |
| `description` | `TEXT` | Detailed item description |
| `image_url` | `VARCHAR(500)` | Image asset path |
| `embedding` | `vector(512)` | 512-dimensional visual embedding vector |
| `created_at` | `TIMESTAMP` | Record creation timestamp |

### `room_analyses` Table
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | `INTEGER PRIMARY KEY` | Auto-incrementing analysis identifier |
| `filename` | `VARCHAR(255)` | Original uploaded image filename |
| `room_type` | `VARCHAR(50)` | Predicted room category |
| `confidence` | `FLOAT` | Classification confidence score |
| `class_probabilities` | `JSON` | Probability breakdown across room categories |
| `image_metrics` | `JSON` | Entropy, contrast, and resolution metadata |
| `matched_product_ids` | `JSON` | Array of matched product IDs |
| `embedding` | `vector(512)` | Query image visual embedding vector |
| `created_at` | `TIMESTAMP` | Record creation timestamp |

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/harsharaju1314-hash/AI-Room-Image-Analysis-Product-Search.git
cd AI-Room-Image-Analysis-Product-Search
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
Copy the environment template:
```bash
cp .env.example .env
```

---

## Running the Application

### 1. Seed Catalog Data
```bash
python scripts/seed_catalog.py
```

### 2. Start the FastAPI Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
* Interactive API Documentation: `http://localhost:8000/docs`
* Health Check: `http://localhost:8000/health`

---

## Testing

Run the automated test suite with pytest:
```bash
pytest -v
```

To run with coverage:
```bash
pytest --cov=app --cov-report=term-missing
```

### Test Coverage Summary:
* **API Endpoints**: Valid image upload, invalid extension rejection, oversized file blocking (>5MB), corrupted binary detection, record retrieval by ID.
* **Vision Service**: 512-dim embedding dimensionality, L2 normalization ($\|v\|_2 = 1.0$), and category probability bounds.
* **Image Processing**: Safe Pillow loading, OpenCV resizing, Shannon entropy, and GLCM contrast calculation.
* **Vector Search**: Cosine similarity ordering, room-type filtering, and top-k retrieval bounds.

---

## Azure DevOps CI Pipeline

The repository includes a pipeline definition (`azure-pipelines.yml`) that runs on every pull request and push to `main`:
1. Sets up Python 3.12 environment.
2. Installs requirements and test tooling (`flake8`, `pytest-cov`).
3. Executes static code linting.
4. Executes the full `pytest` suite and publishes test results.

---

## Limitations

* **Zero-Shot Category Mapping**: Uses mapped ImageNet activations rather than fine-tuning on a dedicated dataset like Places365.
* **Whole-Image Embedding**: Encodes the entire room scene rather than localizing and segmenting individual items.
* **Catalog Size**: Built and tested with a curated sample catalog for demonstration.

---

## Future Improvements

* **Object Localization / Segmentation**: Integrate YOLO or Mask R-CNN to detect and crop individual furniture items before vector search.
* **Fine-Tuned Domain Backbone**: Fine-tune a vision transformer (ViT) or CLIP model on domain-specific interior design datasets.
* **HNSW Vector Indexing**: Add HNSW indexing in pgvector for sub-millisecond retrieval on large-scale catalogs.
* **Cloud Storage**: Integrate Azure Blob Storage / AWS S3 for uploaded image persistence.

---

## Key Takeaways & What I Learned

* Built an end-to-end computer vision pipeline connecting image preprocessing, deep feature extraction with TensorFlow/Keras, and vector similarity search.
* Implemented intermediate feature extraction using TensorFlow Keras functional sub-models.
* Structured relational metadata and high-dimensional vector embeddings within PostgreSQL using pgvector.
* Designed defensive API validation for multipart image uploads, guarding against oversized payloads and malformed binaries.
* Implemented modular unit and integration testing with transactional SQLite fixtures and automated CI execution in Azure DevOps.

---

## Author

**Harsha Raju**  
*Aspiring AI/ML Engineer*  
GitHub: [@harsharaju1314-hash](https://github.com/harsharaju1314-hash)
