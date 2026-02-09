FROM python:3.12-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Mettre à jour pip et installer setuptools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copier requirements.txt
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code (y compris templates/)
COPY . .

# Exposer le port
EXPOSE 5000

# Volume pour le modèle
VOLUME ["/models"]

# Commande de démarrage
CMD ["python", "app/app.py"]
