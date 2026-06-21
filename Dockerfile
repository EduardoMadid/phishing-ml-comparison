# ---------- STAGE 1: builder ----------
FROM python:3.13-slim AS builder

WORKDIR /app

# Instala dependências num diretório isolado
COPY requirements-api.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt

# ---------- STAGE 2: runtime ----------
FROM python:3.13-slim

WORKDIR /app

# Copia as libs já instaladas do builder (sem o lixo de build)
COPY --from=builder /install /usr/local

# Usuário não-root (segurança!)
RUN useradd --create-home appuser
USER appuser

# Copia só o necessário pra rodar
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]