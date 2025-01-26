from model import PlantDiseaseDetector

# Create instance of our model
detector = PlantDiseaseDetector(num_classes=2)  # Change 3 to number of your disease classes

# Prepare the data
train_loader, test_loader = detector.prepare_data(
    data_dir='data',  # Path to your data folder
    batch_size=32
)

# Train the model
history = detector.train(num_epochs=10)

# Save the trained model
detector.save_model('plant_disease_model.pth')