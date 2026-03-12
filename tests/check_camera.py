import cv2
import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import cast

# Ensure the project root is on sys.path so the `src` package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import LeNet5

# 1. Load your trained model
model = LeNet5(num_classes=10)
model.load_state_dict(torch.load('results/lenet5_3_0.001.pth', map_location='cpu'))
model.eval()

# 2. Define the transformation (Must match training!)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    # Add normalization if you used it during training!
])

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 3. Open Camera
cap = cv2.VideoCapture(0)

print("Press SPACE to predict, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    cv2.imshow('Camera', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # Spacebar
        # Preprocess the frame
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor: torch.Tensor = cast(torch.Tensor, transform(img)).unsqueeze(0) # Add batch dimension
        
        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            idx = int(predicted.item())
            print(f"Prediction: {classes[idx]}")
            
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()