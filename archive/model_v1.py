import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

class PlantDiseaseDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class PlantDiseaseDetector:
    def __init__(self, model_name='efficientnet-b0', num_classes=38):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
        self.model = self.model.to(self.device)
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def prepare_data(self, data_dir, batch_size=32):
        """
        Prepare DataLoaders from PlantVillage dataset directory
        """
        # Get all image paths and labels
        image_paths = []
        labels = []
        class_names = []
        
        for class_idx, class_name in enumerate(os.listdir(data_dir)):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                class_names.append(class_name)
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        image_paths.append(os.path.join(class_dir, img_name))
                        labels.append(class_idx)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            image_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Create datasets
        train_dataset = PlantDiseaseDataset(X_train, y_train, self.train_transform)
        test_dataset = PlantDiseaseDataset(X_test, y_test, self.test_transform)
        
        # Create dataloaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size)
        self.class_names = class_names
        
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of testing samples: {len(test_dataset)}")
        return self.train_loader, self.test_loader
    
    def train(self, num_epochs=10, learning_rate=0.001):
        """
        Train the model
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3, verbose=True
        )
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for inputs, labels in tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            train_loss = train_loss / len(self.train_loader)
            train_acc = 100. * correct / total
            
            # Validation phase
            val_loss, val_acc = self.evaluate()
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            print('-' * 50)
        
        return history
    
    def evaluate(self):
        """
        Evaluate the model on test set
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return test_loss / len(self.test_loader), 100. * correct / total
    
    def predict(self, image_path):
        """
        Make prediction on a single image
        """
        self.model.eval()
        image = Image.open(image_path).convert('RGB')
        image = self.test_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            
        return {
            'predicted_class': self.class_names[prediction.item()],
            'confidence': confidence.item()
        }
    
    def save_model(self, path):
        """Save the model"""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """Load a saved model"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))