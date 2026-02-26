import torch
from PIL import Image
from torchvision import transforms
import os
from models.age_model import get_age_model

device = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(PROJECT_ROOT, "models", "best_age_model.pth")

# =====================
# LOAD CHECKPOINT
# =====================
checkpoint = torch.load(model_path, map_location=device)

model = get_age_model(num_classes=3).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# Load class mapping dynamically
class_to_idx = checkpoint["class_to_idx"]
idx_to_class = {v: k for k, v in class_to_idx.items()}

print("Loaded class mapping:", idx_to_class)

# =====================
# TRANSFORM
# =====================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# =====================
# TEST IMAGE
# =====================
image_path = os.path.join(PROJECT_ROOT, "test_image5S.jpg")

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")

img = Image.open(image_path).convert("RGB")
x = transform(img).unsqueeze(0).to(device)

# =====================
# PREDICTION
# =====================
with torch.no_grad():
    outputs = model(x)
    probs = torch.softmax(outputs, dim=1)
    pred = probs.argmax(1).item()

pred_label = idx_to_class[pred]

print("Prediction:", pred_label.capitalize())
print("Confidence:", round(probs[0][pred].item() * 100, 2), "%")
