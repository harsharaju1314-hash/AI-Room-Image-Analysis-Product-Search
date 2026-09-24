# Interview Preparation Guide: AI Room Image Analysis & Product Search

This guide contains clear, defensible, and realistic answers to key interview questions about the architecture, implementation choices, and trade-offs of this project.

---

### 1. 60-Second Project Explanation
> **"Can you walk me through your project in 60 seconds?"**

*"I built **AI Room Image Analysis & Product Search**, a focused REST service designed for home design and interior retail workflows. When a user uploads a photo of an interior space—like a living room or bedroom—the system runs a multi-stage pipeline:*
1. *Validates and sanitizes the uploaded file to ensure format and security integrity.*
2. *Computes visual complexity and contrast metrics using **scikit-image** and **OpenCV**.*
3. *Extracts a 512-dimensional visual embedding and classifies the room type using a pretrained **TensorFlow / Keras MobileNetV2** model.*
4. *Performs vector cosine similarity search against a home-design catalog stored in **PostgreSQL with pgvector** to retrieve the most visually and contextually relevant furniture and fixtures.*
5. *Persists the analysis audit record and returns structured JSON through **FastAPI**.*

*The project includes an automated test suite across security, computer vision, and vector search, integrated into an **Azure DevOps CI** pipeline."*

---

### 2. Why TensorFlow & Keras?
> **"Why did you choose TensorFlow / Keras for the vision model?"**

* **Lightweight & Fast MobileNetV2 Architecture**: TensorFlow Keras provides built-in, highly optimized mobile backbones (`MobileNetV2`) that offer low latency on CPU while maintaining strong feature extraction capabilities.
* **Functional Sub-Model Extraction**: Keras makes intermediate feature extraction straightforward using functional APIs (`Model(inputs=..., outputs=gap_layer.output)`), allowing clean interception of feature bottleneck representations.
* **Standardized Preprocessing**: Built-in preprocessing functions (`tf.keras.applications.mobilenet_v2.preprocess_input`) ensure exact consistency with training-time data scaling.
* **Production Serving Options**: TensorFlow models can easily be serialized into SavedModel/ONNX or served using TensorFlow Serving / TFLite.

---

### 3. Why FastAPI?
> **"Why did you select FastAPI instead of Flask or Django?"**

* **Asynchronous I/O Support**: FastAPI natively supports `async/await`, which is ideal for I/O-bound tasks like reading multipart image uploads and database queries.
* **Automatic Data Validation & Type Safety**: Through Pydantic models, request parameters and response schemas are strictly validated and serialized at runtime.
* **Interactive OpenAPI Docs**: Generates `/docs` (Swagger UI) automatically, making API exploration and testing immediate.
* **Lightweight & High Performance**: FastAPI has lower overhead than monolithic frameworks like Django while providing richer features than raw Flask.

---

### 4. Why PostgreSQL?
> **"Why PostgreSQL for storing analysis records and catalog data?"**

* **ACID Compliance & Reliability**: Guarantees transactional consistency for catalog records and analysis logs.
* **Single Unified Data Store**: Avoids the operational overhead of running a separate vector database alongside a relational database. PostgreSQL holds both structured relational metadata (prices, room types, categories) and visual embeddings.
* **Structured Filtering + Vector Querying**: Allows combined SQL queries that filter by metadata (e.g. `WHERE room_type = 'living_room'`) while computing vector distance in a single pass.

---

### 5. Why pgvector?
> **"Why use pgvector instead of a standalone vector database or in-memory vector index?"**

* **Zero Infrastructure Sprawl**: For small-to-midscale catalogs, pgvector embeds vector indexing directly into PostgreSQL without requiring external vector cluster management.
* **Native Distance Operators**: Provides cosine distance (`<=>`), L2 distance (`<->`), and inner product (`<#>`) directly inside SQL statements.
* **Index Support**: Supports IVFFlat and HNSW indexing for approximate nearest neighbors (ANN) as the dataset grows.
* **Cost Efficiency**: Ideal for early-stage and portfolio architectures where operational simplicity is paramount.

---

### 6. Why use OpenCV, Pillow, and scikit-image together?
> **"Why are three different image processing libraries used in the project?"**

Each library handles a specific, complementary responsibility in the pipeline:
* **Pillow (`PIL`)**: Serves as the safe file ingestion layer. It inspects raw byte streams, parses file headers with `Image.verify()`, and guarantees clean RGB formatting.
* **OpenCV (`cv2`)**: Handles matrix-level resizing, channel conversions, and high-performance array transformations into formats expected by CNN backbones.
* **scikit-image (`skimage`)**: Provides specialized scientific image analysis. In this project, it extracts **Shannon Entropy** (`shannon_entropy` to gauge visual clutter/detail) and **GLCM Contrast** (`graycomatrix` & `graycoprops` to quantify texture and lighting definition), enriching the room analysis with interpretable visual quality metrics.

---

### 7. Why use a Pretrained Model?
> **"Why use a pretrained MobileNetV2 instead of training a CNN from scratch?"**

* **Transfer Learning Efficacy**: Training a deep convolutional network from scratch requires hundreds of thousands of labeled interior images and significant compute.
* **Generalizable Low-Level Features**: Pretrained ImageNet weights already contain robust representations for edges, textures, shapes, and object compositions.
* **Lightweight Footprint**: MobileNetV2 provides an ultra-lightweight parameter footprint (~3.5M parameters) and low latency (~15–30ms on CPU), making it suitable for responsive API endpoints.

---

### 8. How would you scale the system?
> **"If traffic increased to thousands of queries per minute, how would you scale this architecture?"**

1. **Decouple Heavy ML Inference**:
   * Offload the TensorFlow inference step to an asynchronous worker pool (e.g., Celery or Redis Queue) or a dedicated model serving runtime (Triton Inference Server / TensorFlow Serving).
2. **Horizontal API Scaling**:
   * Run stateless FastAPI replicas behind an Application Load Balancer / Nginx.
3. **Database & Vector Optimization**:
   * Add **HNSW (Hierarchical Navigable Small World)** indexing on the pgvector column for sub-millisecond approximate nearest neighbor searches.
   * Introduce read-replicas for catalog browsing.
4. **Caching Layer**:
   * Cache frequent product recommendation vectors and repeated image hashes in Redis.
5. **Blob Storage**:
   * Store uploaded room images in cloud object storage (Azure Blob Storage / AWS S3) with CDN delivery, saving only URIs in the database.

---

### 9. How do you secure image uploads?
> **"What security measures are implemented for user uploads?"**

* **File Extension Whitelisting**: Restricts uploads strictly to `.jpg`, `.jpeg`, `.png`, and `.webp`.
* **Payload Size Limits**: Enforces a strict 5MB limit (`MAX_UPLOAD_SIZE_MB`) returning HTTP 413 for oversized payloads to prevent Denial of Service (DoS) memory exhaustion.
* **Header & Magic Byte Verification**: Uses Pillow's `Image.verify()` on the binary stream to ensure the uploaded payload is a genuine, uncorrupted image and not an executable disguised with an image extension.
* **In-Memory Buffer Processing**: Image processing operates in isolated memory buffers (`io.BytesIO`) without executing or saving untrusted binary files directly to the host filesystem.
* **Sanitized Error Handling**: Catches internal exceptions and returns clean, uniform HTTP error messages without leaking internal tracebacks or system paths.

---

### 10. How does the Azure DevOps pipeline work?
> **"Explain the structure of your CI pipeline."**

The Azure DevOps CI pipeline is defined in `azure-pipelines.yml`:
1. **Trigger**: Triggers automatically on commits/merges to `main` and `master`.
2. **Environment**: Provisions a clean Ubuntu hosted runner with Python 3.12.
3. **Dependency Installation**: Upgrades `pip` and installs pinned dependencies from `requirements.txt`.
4. **Code Quality Linting**: Executes `flake8` to catch syntax errors, undefined variables, and structural flaws.
5. **Automated Testing & Coverage**: Runs the `pytest` test suite with code coverage tracking.
6. **Test Publishing**: Publishes structured test results via `PublishTestResults@2` for visibility in the Azure DevOps build dashboard.

---

### 11. Unit vs. Integration Testing in this project
> **"How are tests structured across unit and integration levels?"**

* **Unit Tests (`test_vision.py`, `test_image_processor.py`)**:
  * Test individual functions in isolation.
  * Verify that the image processor correctly resizes, normalizes, and calculates entropy.
  * Verify that the TensorFlow model outputs a valid 512-dimensional vector with L2 norm $\approx 1.0$ and valid class probabilities.
* **Integration Tests (`test_api.py`, `test_vector_search.py`)**:
  * Test end-to-end request/response cycles using FastAPI `TestClient`.
  * Verify multi-step interactions: Upload $\rightarrow$ Validation $\rightarrow$ Metric Computation $\rightarrow$ Feature Extraction $\rightarrow$ Database Storage $\rightarrow$ Vector Search $\rightarrow$ JSON Response.
  * Test edge cases: oversized payloads, corrupt files, invalid extensions, and non-existent IDs.

---

### 12. What are the limitations of the project?
> **"What are the current limitations of your implementation?"**

* **Zero-shot Category Mapping**: The room classifier maps ImageNet interior features rather than being fine-tuned on a dedicated dataset (such as MIT Places365 or ADE20K).
* **Single Dominant Room Classification**: Analyzes the image as a single scene rather than segmenting multiple objects/furniture pieces individually (e.g. using YOLO or Mask R-CNN).
* **Small Sample Catalog**: The vector catalog contains a curated demonstration set of home design products rather than an enterprise-scale inventory.
* **CPU Inference**: The current setup is optimized for CPU; high-throughput deployments would benefit from batching and GPU acceleration.
