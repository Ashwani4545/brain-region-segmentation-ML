# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Build — install all heavy dependencies
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /install

# System libraries required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python deps into a prefix directory so we can copy selectively
RUN pip install --no-cache-dir --prefix=/deps -r requirements.txt && \
    pip install --no-cache-dir --prefix=/deps \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --prefix=/deps gunicorn==21.2.0 whitenoise==6.7.0

# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime image
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Runtime system libraries for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /deps /usr/local

WORKDIR /app

# Copy the full project
COPY . .

# Set working directory inside webapp for Django management commands
WORKDIR /app/webapp

# Collect static files (needs DJANGO_SETTINGS_MODULE)
ENV DJANGO_SETTINGS_MODULE=webapp.settings \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN python manage.py collectstatic --noinput

# Create writable media directories
RUN mkdir -p media/uploads media/results

# HF Spaces Docker apps MUST listen on port 7860
EXPOSE 7860

# Use a non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Run migrations at runtime to connect to external databases, then start Gunicorn
CMD python manage.py migrate --noinput && \
    gunicorn webapp.wsgi:application \
    --bind 0.0.0.0:7860 \
    --workers 1 \
    --threads 2 \
    --timeout 180 \
    --log-level info
