FROM python:3.12-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Mettre à jour pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copier requirements.txt
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code d'entraînement
COPY train/ .

# Commande de démarrage
CMD ["python", "main.py"]