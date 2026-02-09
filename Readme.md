# Projet CNN - Classification d'Images CIFAR-10

## 📋 Description du Projet

Ce projet implémente un système complet de classification d'images utilisant un réseau de neurones convolutionnel (CNN) sur le dataset CIFAR-10. Le projet est entièrement conteneurisé avec Docker pour assurer la reproductibilité et faciliter le déploiement.

### Problématique
Classifier automatiquement des images en 10 catégories différentes : avions, automobiles, oiseaux, chats, cerfs, chiens, grenouilles, chevaux, navires et camions.

### Objectifs
- Entraîner un modèle CNN performant sur CIFAR-10
- Conteneuriser l'entraînement et le déploiement avec Docker
- Déployer une API REST pour faire des prédictions en temps réel

## 🏗️ Architecture du Projet

```
project/
├── app/                      # Module API
│   ├── Dockerfile           # Docker pour l'API
│   └── app.py               # API Flask
│
├── train/                    # Module d'entraînement
│   ├── Dockerfile           # Docker pour training
│   └── main.py              # Script d'entraînement
│
├── data/                     # Dataset (créé automatiquement)
├── models/                   # Modèles entraînés
├── docker-compose.yml        # Orchestration des services
├── requirements.txt          # Dépendances Python
└── README.md                # Ce fichier
```

## 🧠 Modèle CNN

### Architecture
Le modèle SimpleCNN comprend :
- **3 blocs convolutionnels** avec BatchNormalization et MaxPooling
  - Conv1: 3→32 canaux
  - Conv2: 32→64 canaux
  - Conv3: 64→128 canaux
- **2 couches fully connected** avec Dropout (0.5)
- **Fonction d'activation**: ReLU
- **Sortie**: 10 classes (softmax)

### Hyperparamètres
- Batch size: 64
- Epochs: 10
- Learning rate: 0.001
- Optimizer: Adam
- Loss: CrossEntropyLoss

## 📊 Dataset - CIFAR-10

- **Source**: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Images**: 60,000 images couleur 32x32
- **Classes**: 10 catégories
- **Train/Test**: 50,000 / 10,000 images
- **Téléchargement**: Automatique via torchvision

### Prétraitement
- Normalisation avec mean et std de CIFAR-10
- Augmentation de données (training):
  - RandomHorizontalFlip
  - RandomCrop avec padding

## 🚀 Installation et Exécution

### Prérequis
- Docker
- Docker Compose
- (Optionnel) GPU avec CUDA pour accélération

### Étape 1: Cloner le projet
```bash
git clone <votre-repo>
cd cnn-cifar10-project
```

### Étape 2: Créer la structure
```bash
# Créer les dossiers nécessaires
mkdir -p data models
```

### Étape 3: Construire les images Docker
```bash
docker-compose build
```

### Étape 4: Entraîner le modèle
```bash
# Lancer l'entraînement avec docker-compose
docker-compose run train

# OU directement avec Docker
docker build -t cnn-training ./train
docker run -v $(pwd)/data:/data -v $(pwd)/models:/models cnn-training
```

### Étape 5: Lancer l'API de prédiction
```bash
# Démarrer l'API
docker-compose up api

# L'API sera accessible sur http://localhost:5000
```

## 🔌 Utilisation de l'API

### Vérifier le statut
```bash
curl http://localhost:5000/health
```

### Faire une prédiction
```bash
curl -X POST -F "image=@image.jpg" http://localhost:5000/predict
```

### Exemple avec Python
```python
import requests

url = "http://localhost:5000/predict"
files = {'image': open('test_image.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

### Exemple de réponse
```json
{
  "success": true,
  "prediction": "cat",
  "confidence": 0.89,
  "top3_predictions": [
    {"class": "cat", "confidence": 0.89},
    {"class": "dog", "confidence": 0.08},
    {"class": "deer", "confidence": 0.02}
  ]
}
```

## 📈 Performances Attendues

Avec cette architecture simple :
- **Accuracy sur test set**: ~70-75%
- **Temps d'entraînement (CPU)**: ~20-30 min pour 10 epochs
- **Temps d'entraînement (GPU)**: ~3-5 min pour 10 epochs

## 🐳 Docker - Détails Techniques

### Structure des Volumes
- `./data:/data` - Persistance du dataset CIFAR-10
- `./models:/models` - Sauvegarde des modèles entraînés

### Réseau
- Network bridge `ml_network` pour la communication inter-conteneurs

### Bonnes Pratiques Respectées
✅ Images légères (python:3.9-slim)
✅ Cache des layers optimisé
✅ Volumes pour la persistance des données
✅ Variables d'environnement pour la configuration
✅ Séparation des préoccupations (train/api)
✅ Pas de données sensibles dans les images

## 🔄 Dimension MLOps

### Reproductibilité
- Versions figées des dépendances (requirements.txt)
- Environnement Docker isolé et reproductible
- Seed aléatoire fixe possible pour reproduire les résultats

### Versioning
- Modèles sauvegardés avec métadonnées (epoch, accuracy)
- Structure modulaire facilitant le versioning

### CI/CD Ready
- Tests automatisables
- Déploiement simplifié via Docker
- Scalabilité horizontale possible

## 🧪 Tests

### Test manuel de l'API
```bash
# Télécharger une image test
wget https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/200px-Cat03.jpg -O cat.jpg

# Tester la prédiction
curl -X POST -F "image=@cat.jpg" http://localhost:5000/predict
```

### Test avec script Python
```python
import requests
import json

def test_api():
    # Test health endpoint
    health = requests.get('http://localhost:5000/health')
    print("Health check:", health.json())
    
    # Test prediction
    with open('cat.jpg', 'rb') as f:
        files = {'image': f}
        response = requests.post('http://localhost:5000/predict', files=files)
        print("Prediction:", json.dumps(response.json(), indent=2))

if __name__ == '__main__':
    test_api()
```

## 📝 Comparaison Local vs Docker

### Exécution Locale
```bash
# Installer les dépendances
pip install -r requirements.txt

# Entraîner
cd train
python main.py

# Lancer l'API
cd ../app
python app.py
```

### Exécution Docker
```bash
# Tout en une commande
docker-compose up
```

**Avantages Docker:**
- ✅ Environnement isolé et reproductible
- ✅ Pas de conflit de dépendances
- ✅ Déploiement simplifié
- ✅ Portabilité garantie

## 🚀 Améliorations Possibles

### Modèle
- Utiliser ResNet ou VGG pré-entraînés (Transfer Learning)
- Implémenter le learning rate scheduling
- Ajouter plus d'augmentation de données

### Infrastructure
- Support GPU dans Docker (nvidia-docker)
- Monitoring avec Prometheus/Grafana
- Logging centralisé

### API
- FastAPI au lieu de Flask (plus performant)
- Authentification JWT
- Rate limiting
- Batch predictions
- WebSocket pour streaming

### MLOps
- Intégration MLflow pour le tracking
- Tests automatiques (pytest)
- CI/CD avec GitHub Actions
- Versioning des datasets (DVC)

## 🛠️ Dépannage

### Problème: Le modèle ne se charge pas dans l'API
**Solution**: Assurez-vous d'avoir entraîné le modèle avant de lancer l'API
```bash
docker-compose run train
docker-compose up api
```

### Problème: Erreur de permissions sur les volumes
**Solution**: Vérifier les permissions des dossiers
```bash
chmod -R 777 data models
```

### Problème: Port 5000 déjà utilisé
**Solution**: Modifier le port dans docker-compose.yml
```yaml
ports:
  - "5001:5000"  # Utiliser le port 5001
```

## 👥 Auteurs

- [Votre nom]
- [Nom du binôme]

## 📄 Licence

Projet académique - 3A-SDD 2025-2026
Technologies IA: Conteneurisation et déploiement