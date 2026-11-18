# Imagem base leve com Python 3.12
FROM python:3.12-slim

# Evita geração de .pyc, garante logs em tempo real e configura Matplotlib em modo headless
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

# Instala FFmpeg (necessário para exportar animações em .mp4 com libx264)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instala dependências Python primeiro para aproveitar o cache de camadas do Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia os módulos do projeto
COPY config.py model.py analysis_linear.py simulation.py plots.py animation.py main.py README.md ./

# Garante que o diretório de saídas exista (pode ser sobrescrito por volume montado)
RUN mkdir -p /app/outputs

ENTRYPOINT ["python", "main.py"]
