import torch
from models.age_model import get_age_model

# Paths
MODEL_PTH = "models/best_age_model.pth"
ONNX_PATH = "models/age_model.onnx"

# Load model
device = "cpu"   # ONNX export on CPU
model = get_age_model(num_classes=3)
model.load_state_dict(torch.load(MODEL_PTH, map_location=device))
model.eval()

# Dummy input (batch_size=1, RGB, 224x224)
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=11
)

print("✅ ONNX model exported:", ONNX_PATH)
