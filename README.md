# RealtyAI - AI-Powered Real Estate Assistant

A Retrieval-Augmented Generation (RAG) chatbot that collects publicly available real estate information from **DarGlobal** and **Wasalt**, indexes it locally, and answers natural-language property queries using **NVIDIA Nemotron 3 Ultra** via OpenRouter.

## Architecture Overview

```
Scrape → Process → Index → Retrieve → Generate → Display → Dockerise → Deploy
```

### Technology Stack

| Layer | Technology |
|-------|------------|
| Scraping | Requests, BeautifulSoup, Playwright (fallback) |
| Data Processing | Pandas, Pydantic, JSONL |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (local) |
| Vector Store | ChromaDB |
| LLM | Nemotron 3 Ultra via OpenRouter (free tier) |
| Frontend | Streamlit |
| Deployment | Docker → Hugging Face Spaces |

## Quick Start

### Prerequisites

- Python 3.11+
- OpenRouter API key (free tier available at [openrouter.ai](https://openrouter.ai))

### Installation

```bash
# Clone and setup
git clone <repo>
cd realty-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browser (only if JS rendering needed)
playwright install chromium

# Configure environment
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### Phase 1: Data Collection (Reconnaissance First)

```bash
# 1. Reconnaissance - discover actual website structure
python scripts/recon.py --source both --max-pages 5

# 2. Check reports/darglobal_recon.json and reports/wasalt_recon.json
#    Update scraper selectors based on findings

# 3. Scrape data
python scripts/scrape.py --source both --output dataset/raw

# 4. Validate and normalize
python scripts/validate.py --source both
```

### Phase 2: Build Vector Index (Coming in Phase 2)

```bash
# Process data and generate embeddings
python scripts/process_data.py

# Build ChromaDB index
python scripts/build_index.py
```

### Phase 3: Run Locally

```bash
# Start the Streamlit app
streamlit run app/main.py
```

Open http://localhost:7860 in your browser.

### Docker Deployment

```bash
# Build image
docker build -t realty-ai .

# Run with Docker Compose (includes data volumes)
docker-compose up -d

# Or run scrapers via compose profiles
docker-compose --profile scraping run scraper
docker-compose --profile scraping run validator
```



## Project Structure

```
realty-ai/
├── app/
│   ├── main.py              # Streamlit entry point (Phase 4)
│   ├── config.py            # Pydantic settings
│   ├── scraper/             # Web scrapers
│   │   ├── base_scraper.py  # Abstract base with retry/pool
│   │   ├── darglobal.py     # DarGlobal scraper
│   │   └── wasalt.py        # Wasalt scraper
│   ├── data/
│   │   ├── schema.py        # PropertyRecord (Pydantic)
│   │   ├── processor.py     # Validation pipeline
│   │   ├── jsonld.py        # JSON-LD extraction
│   │   └── ingestion.py     # ChromaDB ingestion (Phase 2)
│   ├── rag/
│   │   ├── embeddings.py    # Local embeddings (Phase 2)
│   │   ├── vector_store.py  # ChromaDB wrapper (Phase 2)
│   │   ├── retriever.py     # Hybrid retrieval (Phase 3)
│   │   └── pipeline.py      # RAG pipeline (Phase 3)
│   ├── llm/
│   │   ├── openrouter.py    # OpenRouter client (Phase 3)
│   │   └── prompts.py       # System prompts (Phase 3)
│   └── ui/
│       ├── chat.py          # Chat interface (Phase 4)
│       ├── sidebar.py       # Filters (Phase 4)
│       └── property_card.py # Property display (Phase 4)
├── dataset/
│   ├── raw/                 # Raw JSONL from scrapers
│   ├── processed/           # Validated PropertyRecords
│   ├── rejected/            # Failed validation records
│   └── chroma_db/           # Vector index (Phase 2)
├── scripts/
│   ├── recon.py             # Website reconnaissance
│   ├── scrape.py            # Run scrapers
│   ├── validate.py          # Validate raw → PropertyRecord
│   ├── process_data.py      # Normalize & embed (Phase 2)
│   └── build_index.py       # Build ChromaDB (Phase 2)
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Data Schema

```python
class PropertyRecord(BaseModel):
    source: str                    # "DarGlobal" or "Wasalt"
    title: str
    property_type: Optional[str]   # "Apartment", "Villa", "Project"
    transaction_type: Optional[str] # "Sale", "Rent"
    country: Optional[str]
    city: Optional[str]
    district: Optional[str]
    price: Optional[float]
    currency: Optional[str]        # "SAR", "USD", "AED"
    bedrooms: Optional[int]
    bathrooms: Optional[int]
    area: Optional[float]          # sqm
    description: Optional[str]
    amenities: List[str]
    status: Optional[str]          # "Available", "Under Construction"
    source_url: str                # Original listing URL
    image_url: Optional[str]
```

## Example Queries

Once deployed, the chatbot supports queries like:

1. "Find luxury residential projects from DarGlobal"
2. "Show me three-bedroom apartments in Riyadh"
3. "Find properties under SAR 1 million"
4. "Compare DarGlobal and Wasalt offerings"
5. "Tell me about a specific property and its amenities"
6. "Show me properties in a particular location with their source links"

Follow-up questions work: *"Show me cheaper alternatives"*

## Responsible Scraping

- Respects `robots.txt` and crawl delays
- Uses conservative rate limiting (1.5s + 0.5s jitter)
- Only accesses publicly available pages
- Does not circumvent access controls
- Identifies as `RealtyAI/1.0` with contact URL

