from model import PlantDiseaseDetector

# Create model instance
detector = PlantDiseaseDetector(num_classes=2)  # Same number as training

# Load the trained model
detector.load_model('plant_disease_model.pth')

# Prepare the data (you need to provide the correct path to your data)
data_dir = 'data'  # Path to the data folder used during training
detector.prepare_data(data_dir)

# Test on a single image
result = detector.predict('test_image.jpg')

# Output the result
print(f"Predicted Disease: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2f}")
