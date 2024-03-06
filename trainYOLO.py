from ultralytics import YOLO

data = None
epochs = None
imgsz = 640

# Create a new YOLO model from scratch
modelScratch = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
modelPretrained = YOLO('yolov8n.pt')

# Training the model from scratch
# Train the model using the custom dataset for 300 epochs
resultScratch = modelScratch.train(data=data, epochs=epochs, imgsz=imgsz, plots=True, task=detect)

# Evaluate the model's performance on the validation set
resultSVal = modelScratch.val()

# Training the model from scratch
# Train the model using the custom dataset for 300 epochs
resultPretrained = modelPretrained.train(data=data, epochs=epochs, imgsz=imgsz, plots=True, task=detect)

# Evaluate the model's performance on the validation set
resultPVal = modelPretrained.val()

# Perform object detection on an image using the model
# results = model('https://ultralytics.com/images/bus.jpg')

# Export the model to ONNX format
# success = model.export(format='onnx')
