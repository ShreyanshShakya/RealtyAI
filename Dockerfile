# RealtyAI container image (Streamlit on port 7860)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Playwright dependencies are retained for the optional scraping profile.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget gnupg ca-certificates fonts-liberation \
    libasound2 libatk-bridge2.0-0 libatk1.0-0 libatspi2.0-0 \
    libcups2 libdbus-1-3 libdrm2 libgbm1 libgtk-3-0 libnspr4 \
    libnss3 libwayland-client0 libxcomposite1 libxdamage1 \
    libxfixes3 libxkbcommon0 libxrandr2 xdg-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt \
    && playwright install chromium \
    && playwright install-deps chromium

COPY . .

RUN mkdir -p dataset/raw dataset/processed dataset/rejected dataset/chroma_db logs reports

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/_stcore/health', timeout=5)" || exit 1

# The Streamlit entry point owns RAG pipeline initialization.
CMD ["streamlit", "run", "app/main.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
