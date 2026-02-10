import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os


class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()

        # First Convolutional Block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)

        # Second Convolutional Block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)

        # Third Convolutional Block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.25)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 128)
        self.bn8 = nn.BatchNorm1d(128)
        self.dropout5 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # First Block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Second Block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Third Block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully Connected Layers
        x = F.relu(self.bn7(self.fc1(x)))
        x = self.dropout4(x)

        x = F.relu(self.bn8(self.fc2(x)))
        x = self.dropout5(x)

        x = self.fc3(x)

        return x


class CIFAR10Classifier:
    def __init__(self, num_classes=10, device=None):
        self.num_classes = num_classes
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        print(f"Using device: {self.device}")

    def load_data(self, batch_size=256, validation_split=0.1, use_augmentation=True):
        print("Loading CIFAR-10 dataset...")

        # Define transforms
        if use_augmentation:
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomResizedCrop(32, scale=(0.9, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Load datasets
        train_dataset = torchvision.datasets.CIFAR10(
            root='/app/data', train=True, download=True, transform=train_transform
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root='/app/data', train=False, download=True, transform=test_transform
        )

        # Split training into train and validation
        train_size = int((1 - validation_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        # Create data loaders
        num_cores = os.cpu_count()
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_cores - 1,  # Use most cores for data loading
            pin_memory=False,  # Not needed for CPU
            persistent_workers=True  # Keep workers alive between epochs
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

        return self

    def build_model(self):
        print("Building CNN model...")
        self.model = CNNModel(num_classes=self.num_classes).to(self.device)
        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        return self

    def train_epoch(self, criterion, optimizer):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / total,
                'acc': 100. * correct / total
            })

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def validate_epoch(self, criterion):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc='Validation'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def train(self, epochs=70, learning_rate=0.001, patience=7):
        print("Starting training...")

        os.makedirs('/app/models', exist_ok=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience // 2
        )

        best_val_acc = 0.0
        epochs_no_improve = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 60)

            # Train
            train_loss, train_acc = self.train_epoch(criterion, optimizer)

            # Validate
            val_loss, val_acc = self.validate_epoch(criterion)

            # Update learning rate
            scheduler.step(val_loss)

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # Print epoch summary
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                # Save best model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'accuracy': val_acc * 100,  # Add this - the app expects it!
                    'history': self.history
                }, '/app/models/best_model.pth')
                print(f"✓ New best model saved! (Val Acc: {val_acc * 100:.2f}%)")
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epoch(s)")

            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                # Load best model
                checkpoint = torch.load('/app/models/best_model.pth')
                self.model.load_state_dict(checkpoint['model_state_dict'])
                break

        print("\nTraining completed!")
        return self

    def save_model(self, filepath='/app/models/best_model.pth'):
        best_acc = max(self.history['val_acc']) * 100 if self.history['val_acc'] else 0.0

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'accuracy': best_acc,  # App needs this!
            'history': self.history
        }, filepath)
        print(f"Model saved to {filepath}")
        return self

    def load_model(self, filepath='/app/models/best_model.pth'):
        checkpoint = torch.load(filepath, map_location=self.device)

        if self.model is None:
            self.build_model()

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint['history']

        print(f"Model loaded from {filepath}")
        return self


# Example usage
if __name__ == "__main__":
    # Create classifier instance
    classifier = CIFAR10Classifier()

    # Load and preprocess data
    classifier.load_data(batch_size=64, validation_split=0.1, use_augmentation=True)

    # Build model
    classifier.build_model()

    # Train the model
    classifier.train(epochs=100, learning_rate=0.001, patience=10)

    # Save model
    classifier.save_model()