# RealtyAI Dockerfile for Hugging Face Spaces
FROM python:3.11-slim

# Install system dependencies for Playwright, ChromaDB, and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    gnupg \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libwayland-client0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright Chromium
RUN playwright install chromium
RUN playwright install-deps chromium

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p dataset/raw dataset/processed dataset/rejected dataset/chroma_db logs reports

# Expose Streamlit port (HF Spaces uses 7860)
EXPOSE 7860

# Health check - use Python for more reliable check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/_stcore/health', timeout=5)" || exit 1

# Run Streamlit with initialization
CMD ["python", "-c", "from app.rag.pipeline import RAGPipeline; p = RAGPipeline(); p.initialize(); print('Index initialized'); import streamlit.web.cli as cli; cli.main(['run', 'app/main.py', '--server.port=7860', '--server.address=0.0.0.0', '--server.headless=true'])"]