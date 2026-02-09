# Dockerfile pour l'API de prédiction
FROM python:3.12-slim

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier requirements depuis la racine
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'API
COPY app/app.py .

# Exposer le port Flask
EXPOSE 5000

# Volume pour le modèle
VOLUME ["/models"]

# Commande pour lancer l'API
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
