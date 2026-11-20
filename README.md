# RealtyAI — AI-Powered Real Estate Discovery Assistant

RealtyAI is a Retrieval-Augmented Generation (RAG) application for discovering and asking questions about real-estate development projects. It combines locally generated sentence embeddings, a persistent ChromaDB vector index, hybrid retrieval, and an LLM served through OpenRouter in a Streamlit chat interface.

**Current application data:** the deployed knowledge base contains 29 verified DarGlobal projects. Wasalt scraping is present in the repository, but Wasalt records are not currently part of the deployed RAG dataset.

## Live application

The application is deployed on **Streamlit Community Cloud**.

> Add the public Streamlit app URL here if you want visitors to open the live demo directly.

## Features

- Natural-language questions about indexed DarGlobal projects.
- Semantic and hybrid retrieval over a local ChromaDB index.
- Local embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- Country, city, community, and source filters in the sidebar.
- Source cards with project details, original source links, and available images.
- OpenRouter-powered answer generation, with a configured optional fallback model.
- Retrieval-only operation when an OpenRouter API key is not configured.
- Persistent local vector storage; the index can be rebuilt from the RAG-ready JSONL dataset.

## Architecture

```text
DarGlobal project data (JSONL)
          |
          v
Validation and normalisation
          |
          v
RAG-ready project records
          |
          v
Document construction
          |
          v
Sentence Transformers embeddings
          |
          v
ChromaDB persistent vector index
          |
          v
Hybrid retrieval + metadata filters
          |
          v
Context construction
          |
          v
OpenRouter LLM (optional)
          |
          v
Streamlit chat UI + source references
```

## Technology stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Scraping utilities | Requests, BeautifulSoup, lxml, Playwright |
| Data validation | Pydantic |
| Dataset format | JSON Lines (JSONL) |
| Embeddings | Sentence Transformers — `all-MiniLM-L6-v2` |
| Vector database | ChromaDB |
| Retrieval | Hybrid retrieval with metadata filters |
| LLM integration | OpenRouter using the OpenAI-compatible client |
| User interface | Streamlit |
| Containerisation | Docker and Docker Compose |
| Tests | pytest |

## Data coverage and limitations

The current RAG-ready dataset is `dataset/processed/darglobal_rag_ready.jsonl` and contains **29 verified DarGlobal project records**.

The dataset is strongest for project names, descriptions, source URLs, and geographic discovery. Field coverage is not uniform:

- Project title, URL, and description are present for all 29 records.
- Country information is present for most records; city and image coverage is partial.
- Community information is available for some records.
- Price, bedroom count, and area are not populated in the current RAG-ready dataset.

Accordingly, RealtyAI should not be treated as a live property inventory or pricing service. It cannot reliably answer questions requiring uncollected fields such as current availability, exact prices, bedroom counts, floor areas, or a complete amenities list. It should state when requested information is not available in the indexed sources rather than infer or invent it.

Wasalt scraper code is retained for future ingestion work, but **Wasalt is not currently an indexed or searchable data source**. Cross-source DarGlobal/Wasalt comparisons are therefore not supported by the current dataset.

## Quick start

### Prerequisites

- Python 3.11 or newer
- Git
- An OpenRouter API key for LLM-generated answers (optional for retrieval-only mode)

### 1. Clone the repository

```bash
git clone https://github.com/ShreyanshShakya/RealtyAI.git
cd RealtyAI
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Playwright is included for the optional scraping utilities. The Streamlit application does not need a Playwright browser to query the already prepared RAG dataset.

### 4. Configure environment variables

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set your OpenRouter key:

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
OPENROUTER_FALLBACK_MODEL=openrouter/free
```

Keep API keys private. Do not commit `.env` or place secrets in source code. The fallback model is configurable; it is not a guarantee that a free model will always be available.

### 5. Run the application

```bash
streamlit run app/main.py
```

Open **http://localhost:8501**.

On first startup, the application loads the embedding model and initializes or rebuilds the ChromaDB index from the configured RAG-ready dataset when required. Initial startup can take longer while the embedding model is downloaded or the index is built.

If you do not configure `OPENROUTER_API_KEY`, the app can still provide retrieval/search results, but it cannot generate LLM answers.

## Rebuild the vector index manually

The RAG-ready dataset is already included in the repository. To explicitly rebuild the local ChromaDB index:

```bash
python scripts/build_index.py --source darglobal --rebuild
```

Useful options:

```bash
python scripts/build_index.py --help
```

The default index directory is `dataset/chroma_db`. This is generated runtime data and should not be committed. If the index is deleted, it can be recreated from the RAG-ready JSONL file.

## Optional data collection and validation

The repository also contains scraping and validation utilities. These are separate from running the deployed chat application.

Reconnaissance:

```bash
python scripts/recon.py --help
```

Scraping:

```bash
python scripts/scrape.py --help
```

Validation:

```bash
python scripts/validate.py --help
```

Dataset validation:

```bash
python scripts/validate_dataset.py --help
```

The scripts expose source options for DarGlobal, Wasalt, or both where supported. However, the existence of a scraper does not mean a source has usable records in the current application dataset. Website structure, access restrictions, and data quality can change; inspect the generated reports and validate records before indexing them. Do not bypass CAPTCHAs, bot protections, authentication, or other access controls.

## Docker

Build and start the Streamlit service:

```bash
docker compose build
docker compose up -d
```

The container exposes Streamlit on **http://localhost:7860**.

Set `OPENROUTER_API_KEY` in the shell or an environment file before starting the service if you want LLM-generated answers. Docker Compose mounts the dataset, reports, and logs directories from the host.

To stop the service:

```bash
docker compose down
```

The scraper and validator are optional Compose services under the `scraping` profile. They are not required to run the application:

```bash
docker compose --profile scraping run --rm scraper
docker compose --profile scraping run --rm validator
```

## Deploy on Streamlit Community Cloud

1. Push the repository to GitHub.
2. In Streamlit Community Cloud, create an app from this repository and select the `master` branch.
3. Set the main file path to `app/main.py`.
4. Configure the required secrets in the app's **Secrets** settings. For example:

   ```toml
   OPENROUTER_API_KEY = "your_openrouter_api_key"
   OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
   OPENROUTER_FALLBACK_MODEL = "openrouter/free"
   ```

5. Deploy and review the application logs if startup or model initialization fails.

Do not commit Streamlit secrets or API keys to the repository. The first deployment or a fresh runtime may need to download the embedding model and construct the vector index.

## Repository structure

```text
RealtyAI/
├── app/
│   ├── main.py                 # Streamlit entry point
│   ├── config.py               # Environment and application settings
│   ├── types.py                # Shared scraper data types
│   ├── data/
│   │   ├── schema.py           # Validated data schemas
│   │   ├── processor.py        # Data processing and normalisation
│   │   └── jsonld.py           # JSON-LD extraction helpers
│   ├── scraper/
│   │   ├── base_scraper.py     # Shared scraper functionality
│   │   ├── darglobal.py        # DarGlobal scraper
│   │   └── wasalt.py           # Wasalt scraper
│   ├── rag/
│   │   ├── document_builder.py # Build documents from project records
│   │   ├── embeddings.py       # Local embedding model
│   │   ├── vector_store.py     # ChromaDB persistence and indexing
│   │   ├── retriever.py        # Hybrid retrieval and filters
│   │   ├── context_builder.py  # Retrieved context construction
│   │   └── pipeline.py         # End-to-end RAG orchestration
│   ├── llm/
│   │   └── openrouter.py       # OpenRouter client and prompts
│   ├── ui/
│   │   └── chat.py             # Streamlit chat interface
│   └── utils/                  # Shared utilities
├── dataset/
│   ├── raw/                    # Raw scraper output
│   ├── processed/
│   │   ├── darglobal_validated.jsonl
│   │   └── darglobal_rag_ready.jsonl
│   ├── rejected/               # Rejected records
│   ├── excluded/               # Records excluded from RAG ingestion
│   └── chroma_db/              # Generated local vector index (not committed)
├── scripts/
│   ├── recon.py                # Website reconnaissance
│   ├── scrape.py               # Scraper entry point
│   ├── validate.py             # Validation and processing entry point
│   ├── clean_darglobal.py      # DarGlobal dataset cleaning
│   ├── validate_dataset.py     # Dataset quality checks
│   ├── build_index.py          # ChromaDB index builder
│   ├── evaluate_rag.py         # RAG evaluation utility
│   └── production_integration_test.py
├── tests/                      # Automated tests
├── reports/                    # Scraping and diagnostic reports
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Example questions

Examples aligned with the fields currently available in the dataset:

- "Tell me about The Astera."
- "Show DarGlobal projects in Oman."
- "Which projects are associated with this city?"
- "What information is available about AIDA?"
- "Show projects with source links."
- "Which projects have published price information?"

Answers depend on the records retrieved from the current dataset. If a field is missing from those records, RealtyAI should say so.

## Responsible data collection

- Respect each source website's published terms and `robots.txt` directives.
- Use conservative request delays and retries.
- Collect only publicly accessible information.
- Do not circumvent authentication, CAPTCHAs, bot protections, or other access controls.
- Preserve source URLs so users can inspect the original project information.
- Validate and review scraped records before adding them to the RAG index.

## License

No license is specified in this repository. Unless a license is added, the code is not granted an explicit open-source reuse license.
