import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from modelClass import MobileNetV2
import json

# ---------- load classes ----------
with open("classes.json") as f:
    class_names = json.load(f)

# ---------- load model ----------
dropout = 0.14510733657567784
neurons_per_hidden_layer = [72, 72, 64]

device = torch.device("cpu")

model = MobileNetV2(
    neurons_per_hidden_layer,
    dropout,
    num_classes=34
)

model.load_state_dict(torch.load("modelWeights_backend.pth", map_location=device))
model.to(device)
model.eval()

# ---------- transform ----------
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def _make_crops(image):
    """
    Safe test-time crops:
    - 3 scales
    - 5 positions each
    - no extreme zoom that can cut off disease patterns
    """
    width, height = image.size
    scales = [0.75, 0.9, 1.0]  # safer than 0.5
    crops = []

    for scale in scales:
        size = int(min(width, height) * scale)
        size = max(1, min(size, min(width, height)))

        left_c = (width - size) // 2
        top_c = (height - size) // 2

        # center
        crops.append(image.crop((left_c, top_c, left_c + size, top_c + size)))

        # corners
        crops.append(image.crop((0, 0, size, size)))  # top-left
        crops.append(image.crop((width - size, 0, width, size)))  # top-right
        crops.append(image.crop((0, height - size, size, height)))  # bottom-left
        crops.append(image.crop((width - size, height - size, width, height)))  # bottom-right

    return crops

def predict(image):
    image = image.convert("RGB")

    crops = _make_crops(image)
    logits_list = []

    with torch.inference_mode():
        for crop in crops:
            # original crop
            inp = val_transform(crop).unsqueeze(0).to(device)
            logits = model(inp)
            logits_list.append(logits)

            # horizontal flip
            flipped = TF.hflip(crop)
            inp_flip = val_transform(flipped).unsqueeze(0).to(device)
            logits_flip = model(inp_flip)
            logits_list.append(logits_flip)

    # average logits, then softmax
    avg_logits = torch.mean(torch.stack(logits_list), dim=0)
    probs = torch.softmax(avg_logits, dim=1)

    top5_probs, top5_idx = torch.topk(probs, 5)

    print("\nFINAL TOP 5:")
    for i in range(5):
        idx = top5_idx[0][i].item()
        prob = top5_probs[0][i].item()
        print(class_names[idx], ":", prob)

    pred_idx = top5_idx[0][0].item()
    confidence = top5_probs[0][0].item()

    return class_names[pred_idx], confidence