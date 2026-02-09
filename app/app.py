from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from flask_cors import CORS
from torchvision import transforms
from PIL import Image
import io


app = Flask(__name__)
CORS(app)
# Classes CIFAR-10
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


# Architecture CNN (même que main.py)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# Chargement du modèle
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN(num_classes=10).to(DEVICE)

# Charger les poids du modèle entraîné
try:
    checkpoint = torch.load('../models/best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded successfully with accuracy: {checkpoint['accuracy']:.2f}%")
except Exception as e:
    print(f"Warning: Could not load model - {e}")

# Transformation pour les images de test
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'device': str(DEVICE),
        'model_loaded': True
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Vérifier si une image est envoyée
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']

        # Lire et prétraiter l'image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Appliquer les transformations
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        # Faire la prédiction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        # Préparer la réponse
        predicted_class = CLASSES[predicted.item()]
        confidence_score = confidence.item()

        # Top 3 prédictions
        top3_prob, top3_indices = torch.topk(probabilities, 3)
        top_3_predictions = [
            {
                'class': CLASSES[idx.item()],
                'confidence': prob.item()
            }
            for prob, idx in zip(top3_prob[0], top3_indices[0])
        ]

        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence_score,
            'top_3_predictions': top_3_predictions
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)