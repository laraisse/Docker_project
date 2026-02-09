# Dockerfile pour l'entraînement du modèle CNN
FROM python:3.12-slim

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier requirements depuis la racine
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY train/main.py .

# Exposer les volumes pour les données et modèles
VOLUME ["/data", "/models"]

# Commande par défaut
CMD ["python", "main.py"]