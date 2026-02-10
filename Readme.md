# 🖼️ CIFAR-10 Image Classifier

A complete deep learning project for classifying images into 10 categories using a Convolutional Neural Network (CNN). This project features automated training with Docker and a beautiful web interface for real-time image classification.

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technologies](#technologies)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [API Documentation](#api-documentation)
- [Screenshots](#screenshots)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a state-of-the-art CNN model to classify images from the CIFAR-10 dataset into one of 10 categories:

- ✈️ **Airplane**
- 🚗 **Automobile**
- 🐦 **Bird**
- 🐱 **Cat**
- 🦌 **Deer**
- 🐕 **Dog**
- 🐸 **Frog**
- 🐴 **Horse**
- 🚢 **Ship**
- 🚚 **Truck**

The project is containerized using Docker for easy deployment and includes both a training pipeline and a production-ready API with an interactive web interface.

## ✨ Features

### Training Pipeline
- 🔄 **Automated Training**: Complete training pipeline with Docker orchestration
- 📊 **Data Augmentation**: Random flips, rotations, and affine transformations
- 📈 **Performance Tracking**: Real-time monitoring with progress bars
- 💾 **Model Checkpointing**: Automatic saving of best models
- ⏱️ **Early Stopping**: Prevents overfitting with patience-based stopping
- 📉 **Learning Rate Scheduling**: Adaptive learning rate adjustment

### Web Application
- 🎨 **Modern UI**: Beautiful, gradient-styled interface
- 📤 **Drag & Drop**: Easy image upload with drag-and-drop support
- 🔍 **Live Preview**: Instant image preview before classification
- 🏆 **Top-3 Predictions**: Shows confidence scores for top 3 predictions
- 📊 **Confidence Visualization**: Progress bars for probability display
- 📱 **Responsive Design**: Works on desktop and mobile devices

### API
- 🚀 **RESTful API**: Clean JSON-based prediction endpoint
- ❤️ **Health Check**: Monitor service status
- 🔒 **CORS Enabled**: Ready for cross-origin requests
- ⚡ **Fast Inference**: Optimized for quick predictions

## 🏗️ Architecture

### CNN Model Architecture

```
Input (32x32x3)
    ↓
[Conv Block 1] → 32 filters, 3x3
    ├─ Conv2d + BatchNorm + ReLU
    ├─ Conv2d + BatchNorm + ReLU
    ├─ MaxPool2d (2x2)
    └─ Dropout2d (0.25)
    ↓
[Conv Block 2] → 64 filters, 3x3
    ├─ Conv2d + BatchNorm + ReLU
    ├─ Conv2d + BatchNorm + ReLU
    ├─ MaxPool2d (2x2)
    └─ Dropout2d (0.25)
    ↓
[Conv Block 3] → 128 filters, 3x3
    ├─ Conv2d + BatchNorm + ReLU
    ├─ Conv2d + BatchNorm + ReLU
    ├─ MaxPool2d (2x2)
    └─ Dropout2d (0.25)
    ↓
[Fully Connected Layers]
    ├─ FC (2048 → 256) + BatchNorm + ReLU + Dropout (0.5)
    ├─ FC (256 → 128) + BatchNorm + ReLU + Dropout (0.5)
    └─ FC (128 → 10)
    ↓
Output (10 classes)
```

**Key Features:**
- Batch Normalization for stable training
- Dropout layers to prevent overfitting
- Multiple convolutional blocks for feature extraction
- ~1.2M trainable parameters

## 🛠️ Technologies

- **Deep Learning**: PyTorch 2.0+
- **Web Framework**: Flask 3.0
- **Computer Vision**: torchvision, PIL
- **Containerization**: Docker, Docker Compose
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Data**: CIFAR-10 dataset (60,000 images)

## 📁 Project Structure

```
├── app/
│   ├── data/ 
│   ├── models/                     # Saved model checkpoints
│   ├──  best_model.pth
│   ├── Dockerfile              # API service Docker configuration
│   ├── app.py                  # Flask application & prediction API
│   └── templates/
│      └── index.html          # Web interface
├── train/
│   ├── Dockerfile              # Training service Docker configuration
│   └── main.py                      
├── docker-compose.yml          # Multi-service orchestration
├── README.md
└── requirements.txt
```

## 🚀 Installation

### Prerequisites

- Docker (version 20.0+)
- Docker Compose (version 2.0+)
- 4GB+ RAM recommended
- 2GB+ free disk space

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/cifar10-classifier.git
cd cifar10-classifier
```

2. **Build and run with Docker Compose**
```bash
docker-compose up --build
```

This will:
- Build both training and API containers
- Download the CIFAR-10 dataset automatically
- Train the model (takes 30-60 minutes depending on hardware)
- Start the API service on port 5000

3. **Access the application**

Open your browser and navigate to:
```
http://localhost:5000
```

## 💻 Usage

### Using the Web Interface

1. Open `http://localhost:5000` in your browser
2. Click the upload area or drag & drop an image
3. Click "🚀 Classify Image" button
4. View the prediction results with confidence scores

### Using the API

#### Health Check
```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "device": "cpu",
  "model_loaded": true
}
```

#### Predict Image
```bash
curl -X POST -F "image=@path/to/your/image.jpg" \
  http://localhost:5000/predict
```

Response:
```json
{
  "success": true,
  "predicted_class": "cat",
  "confidence": 0.8734,
  "top_3_predictions": [
    {
      "class": "cat",
      "confidence": 0.8734
    },
    {
      "class": "dog",
      "confidence": 0.0892
    },
    {
      "class": "bird",
      "confidence": 0.0234
    }
  ]
}
```

### Training Only

To only train the model without starting the API:

```bash
docker-compose up train
```

### Custom Training

You can modify training parameters in `train/main.py`:

```python
classifier.train(
    epochs=40,           # Number of epochs
    learning_rate=0.001, # Initial learning rate
    patience=5           # Early stopping patience
)
```

## 📊 Model Details

### Training Configuration

- **Optimizer**: Adam
- **Learning Rate**: 0.001 (with ReduceLROnPlateau scheduling)
- **Batch Size**: 64
- **Data Split**: 90% train, 10% validation
- **Epochs**: 40 (with early stopping)
- **Early Stopping Patience**: 5 epochs

### Data Augmentation

The training uses several augmentation techniques:
- Random horizontal flips
- Random rotation (±15°)
- Random affine transformations
- Random resized crops
- Normalization: mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)

### Regularization Techniques

- **Dropout**: 0.25 in convolutional layers, 0.5 in FC layers
- **Batch Normalization**: After each convolutional and FC layer
- **Weight Decay**: Through Adam optimizer
- **Data Augmentation**: Multiple transformations

## 📈 Performance

### Expected Results

- **Validation Accuracy**: ~85-90%
- **Training Time**: 30-60 minutes (CPU)
- **Inference Time**: <100ms per image
- **Model Size**: ~5MB

### Performance Optimization

The application includes several optimizations:
- Multi-worker data loading
- Persistent workers for faster epoch iteration
- GPU support (auto-detected if available)
- Efficient batch processing

## 🔧 Development

### Running in Development Mode

1. **API Development** (with live reload):
```bash
docker-compose up api
```

The Flask app runs in debug mode with template auto-reload.

2. **Local Development** (without Docker):

Install dependencies:
```bash
pip install -r requirements.txt
```

Train the model:
```bash
cd train
python main.py
```

Run the API:
```bash
cd app
python app.py
```

### Customization

#### Modify the Model

Edit the `CNNModel` class in `train/main.py` or `app/app.py` to experiment with different architectures.

#### Update the UI

The web interface is in `app/templates/index.html`. It's a single-file application with embedded CSS and JavaScript for easy customization.

## 🐛 Troubleshooting

### Common Issues

**Problem**: Model file not found
```
Solution: Make sure training completed successfully. Check ./models/best_model.pth exists
```

**Problem**: Out of memory during training
```
Solution: Reduce batch_size in train/main.py (try 32 or 16)
```

**Problem**: Docker build fails
```
Solution: Ensure Docker has sufficient resources allocated (4GB+ RAM)
```

**Problem**: Port 5000 already in use
```
Solution: Change the port in docker-compose.yml:
  ports:
    - "8000:5000"  # Now accessible on port 8000
```

## 🤝 Contributing

Contributions are welcome! Here are some ways you can help:

1. 🐛 Report bugs
2. 💡 Suggest new features
3. 📝 Improve documentation
4. 🔧 Submit pull requests

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **CIFAR-10 Dataset**: Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
- **PyTorch Team**: For the excellent deep learning framework
- **Flask Team**: For the lightweight web framework

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with ❤️ and PyTorch**

⭐ Star this repository if you found it helpful!