# Lightweight Python base
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (leverage Docker layer caching)
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source
COPY 05_GenAI/langgraph_agent ./05_GenAI/langgraph_agent

# Expose Streamlit port
EXPOSE 8501

# Default command: run the Streamlit UI
CMD ["streamlit", "run", "05_GenAI/langgraph_agent/ui_streamlit.py", "--server.port", "8501", "--server.address", "0.0.0.0"]

