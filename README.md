# Gutbiome Fingerprinting Platform

An advanced, unsupervised machine learning and microservice platform designed to discover microbial communities, generate individual microbiome fingerprints, calculate diversity metrics, and build population similarity networks from stool sample abundance data.

The project features an end-to-end automated stack: an analytics engine pipelines raw high-dimensional biological data into a structured relational database, which is exposed via a high-performance REST API for real-time inference and client consumption.

---

## 🚀 Project Status

* **[x] Module 1: Machine Learning & Analytics Pipeline** (Completed)
* **[x] Module 2: Relational Database Integration** (Completed)
* **[x] Module 3: Backend API Microservice Development** (Completed)
* **[ ] Module 4: Interactive Frontend UI Dashboard** (In Progress)

---

## 🏗️ System Architecture & Data Flow

The platform is designed around a decoupled, three-tier data engineering architecture:

* **1. Analytics Engine:** Processes raw abundance matrices, trains the champion unsupervised pipeline, scales metrics, and serializes standalone ML estimators (`.pkl`/`.joblib`).
* **2. Database Layer:** A structured relational database schema powered by SQLAlchemy ORM. The data ingestion engine cleans, validates, and seeds patient samples, UMAP embeddings, cluster metadata distributions, and topological network matrices.
* **3. Backend Service:** An asynchronous FastAPI engine loads serialization artifacts into memory at startup to serve real-time vector alignment, CLR transformation, and instantaneous cluster coordinate prediction.

---

## 📊 Pipeline Architecture & ML Strategy

### 1. Exploratory Data Analysis (EDA)

Generates comprehensive distribution profiles and analysis charts saved directly to `docs/figures/`:

* `dataset_overview.png`: High-level metrics on features and metadata distributions.
* `missing_values.png`: Data integrity mapping.
* `top_microbes.png` & `most_variable_features.png`: Abundance spectrum profiles.
* `pca_variance.png`: Dimensionality analysis.
* `umap_preview.png`: Core geometric structure visualization.

### 2. Preprocessing & Normalization

* Filters and partitions clinical metadata from microbial abundance features (identified by the `k__` prefix).
* Applies a Centered Log-Ratio (CLR) transformation with a strict pseudocount offset to handle compositional zero-inflated arrays stably.

### 3. Unsupervised Clustering Engine

Compares three distinct competitive pipelines to automatically isolate structural communities:

* **Pipeline A:** Principal Component Analysis (PCA) $\rightarrow$ K-Means
* **Pipeline B:** Principal Component Analysis (PCA) $\rightarrow$ Gaussian Mixture Model (GMM)
* **Pipeline C:** Uniform Manifold Approximation and Projection (UMAP) $\rightarrow$ HDBSCAN *(Selected Champion)*

### 4. Multi-Tier Bio-Fingerprinting & Explainability

* **Alpha Diversity:** Calculates the *Shannon Diversity Index*, categorizing profiles into *Low*, *Medium*, or *High* diversity tiers.
* **Novelty Scoring:** Normalizes Euclidean vector distance bounds from cluster centroids to accurately flag *Common*, *Moderately Unique*, or *Highly Unique* microbiome profiles.
* **Similarity Engine:** Builds a topological network graph using `NetworkX` based on Cosine Similarity metrics to uncover profile proximities.
* **Cluster Explainability:** Evaluates surrogate Random Forest models over generated cluster assignments to calculate feature importance ranks and isolate defining signature microbes.

---

## 💾 Database & Backend Infrastructure

### Relational Database Design

The schema ensures strict data integrity across one-to-one and one-to-many boundaries using SQLAlchemy constraints:

* **`samples`**: Core clinical records (Subject ID, Age, Country, Gender, Disease status, BMI, Bodysite).
* **`embeddings`**: Static coordinate matrices storing UMAP coordinates (`umap_x`, `umap_y`) and PCA data.
* **`fingerprints`**: Computed alpha-diversity indices, novelty tiers, and dominant taxonomic subsets mapped directly to samples.
* **`cluster_summaries`**: Pre-aggregated population cohorts storing text narratives, demographic profiles, age variations, and top driving biological signatures.

### API Endpoint Registry

* **GET `/health**`: Verifies server operational status, database health, and loaded ML model states.
* **GET `/api/v1/samples**`: Paginated sample arrays with dynamic filtering on country, gender, cluster, and disease.
* **GET `/api/v1/samples/{sample_id}**`: Full transactional detail for an isolated sample, nested with its fingerprint data.
* **GET `/api/v1/clusters**`: Comprehensive collection of discovered clusters along with their driving cohort metadata.
* **GET `/api/v1/network**`: Extracts the raw similarity nodes and edge connections cached for graph visualization.
* **POST `/api/v1/fingerprint/predict**`: Real-time ML inference pipeline. Accepts dynamic abundance vectors, triggers inline CLR scaling, runs UMAP projection, and matches point centroids.

---

## 📂 Project Structure

```text
gutbiome-fingerprinting/
│
├── backend/                    # FastAPI Microservice Application
│   ├── config.py               # Central environment configurations and asset directory paths
│   ├── database.py             # SQLAlchemy configuration, pool session lifecycle, and engine hook
│   ├── main.py                 # FastAPI initialization, lifespans, and endpoint routing definitions
│   ├── models.py               # Declarative SQLAlchemy ORM database models
│   ├── schemas.py              # Strict Pydantic v2 validation models for request/response bodies
│   └── seed.py                 # Core database parsing and ingestion engine
│
├── data/                       # (Ignored by Git)
│   ├── raw/                    # Input datasets (abundance_stoolsubset.csv)
│   ├── processed/              # Normalized matrices and embeddings
│   └── results/                # CSV/JSON outputs (fingerprints, networks)
│
├── docs/                       # Project documentation
│   └── figures/                # (Ignored by Git) Generated EDA plots
│
├── ml/                         # Modular Machine Learning Pipeline
│   ├── config.py               # Hyperparameters, random seeds, and thresholds
│   ├── eda.py                  # Profiling and visual graphing logic
│   ├── preprocessing.py        # Feature isolation and transformation
│   ├── dimensionality_reduction.py # PCA and UMAP wrappers
│   ├── clustering.py           # Clustering pipeline initializations
│   ├── evaluation.py           # Model diagnostics and selection
│   ├── fingerprint.py          # Diversity, novelty, and signature compilation
│   ├── similarity.py           # NetworkX graph and similarity engines
│   ├── explainability.py       # Random Forest cluster feature extraction
│   └── train.py                # Pipeline training orchestration execution script
│
├── models/                     # (Ignored by Git) Pickled joblib models and pipeline configurations
├── gutbiome.db                 # (Ignored by Git) Native SQLite relational database
├── .gitignore                  # Keeps environments and heavy assets out of revision control
├── README.md                   # System documentation
└── requirements.txt            # Unified platform dependency manifests

```

---

## 🛠️ Local Setup & Execution

### 1. Prerequisites

Ensure you have Python 3.10+ installed locally.

### 2. Environment Configuration

Clone the repository, initialize a local virtual environment, and install dependencies:

```bash
# Initialize Virtual Environment
python -m venv venv

# Activate Environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate Environment (Mac/Linux)
source venv/bin/activate

# Install Core Framework Libraries
pip install -r requirements.txt

```

### 3. Pipeline Ingestion & Seeding

To process the machine learning modeling metrics and seed your local relational tables, run the automation tools sequentially from the project root directory:

```bash
# Step A: Run the Unsupervised Analytics Engine to generate ML models
python -m ml.train

# Step B: Run the Data Ingestion script to construct tables and seed raw records
python -m backend.seed

```

### 4. Booting the Application Server

Fire up the local backend API web microservice instance via Uvicorn:

```bash
uvicorn backend.main:app --reload

```

Once initialized, the interactive Swagger documentation interface can be accessed directly at **`http://127.0.0.1:8000/docs`**.

---

## 🔮 Planned Enhancements

* **Frontend UI Dashboard (Streamlit & Plotly):** Construct an interactive single-page web dashboard to visually map patient cohorts across UMAP matrices, provide complete clinical breakdowns of individual biomes, and build an inline form simulator to superimpose live user abundance predictions as glowing reference landmarks on global population maps.