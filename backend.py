import json
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from modelClass import MobileNetV2

# ---------- load classes ----------
with open("classes.json", "r") as f:
    class_names = json.load(f)

# ---------- load model ----------
dropout = 0.14510733657567784
neurons_per_hidden_layer = [72, 72, 64]

device = torch.device("cpu")

model = MobileNetV2(
    neurons_per_hidden_layer=neurons_per_hidden_layer,
    dropout=dropout,
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

# ---------- crop maker ----------
def _make_crops(image):
    """
    Safe test-time crops:
    - 3 scales
    - 5 positions each
    - center crop gets higher weight
    """
    width, height = image.size
    scales = [0.75, 0.9, 1.0]
    crops = []

    for scale in scales:
        size = int(min(width, height) * scale)
        size = max(1, min(size, min(width, height)))

        left_c = (width - size) // 2
        top_c = (height - size) // 2

        # center crop gets more weight
        center_crop = image.crop((left_c, top_c, left_c + size, top_c + size))
        crops.append((center_crop, 2.0))

        # corners
        crops.append((image.crop((0, 0, size, size)), 1.0))                          # top-left
        crops.append((image.crop((width - size, 0, width, size)), 1.0))              # top-right
        crops.append((image.crop((0, height - size, size, height)), 1.0))             # bottom-left
        crops.append((image.crop((width - size, height - size, width, height)), 1.0)) # bottom-right

    return crops

def predict(image, confidence_threshold=0.4):
    """
    Returns:
        (label, confidence)

    If confidence is below threshold:
        ("Uncertain (Retake Image)", confidence)
    """
    image = image.convert("RGB")

    crops = _make_crops(image)
    logits_list = []
    weights = []

    with torch.inference_mode():
        for crop, crop_weight in crops:
            # original crop
            inp = val_transform(crop).unsqueeze(0).to(device)
            logits = model(inp)
            logits_list.append(logits)
            weights.append(crop_weight)

            # horizontal flip
            flipped = TF.hflip(crop)
            inp_flip = val_transform(flipped).unsqueeze(0).to(device)
            logits_flip = model(inp_flip)
            logits_list.append(logits_flip)
            weights.append(crop_weight)

    # weighted average of logits, then softmax
    stacked_logits = torch.cat(logits_list, dim=0)  # shape: [N, num_classes]
    weights_t = torch.tensor(weights, dtype=stacked_logits.dtype, device=stacked_logits.device).unsqueeze(1)  # [N, 1]

    avg_logits = (stacked_logits * weights_t).sum(dim=0, keepdim=True) / weights_t.sum()
    probs = torch.softmax(avg_logits, dim=1)

    topk = min(5, probs.shape[1])
    top5_probs, top5_idx = torch.topk(probs, topk)

    print("\nFINAL TOP 5:")
    for i in range(topk):
        idx = top5_idx[0][i].item()
        prob = top5_probs[0][i].item()
        print(f"{class_names[idx]}: {prob:.4f}")

    pred_idx = top5_idx[0][0].item()
    confidence = top5_probs[0][0].item()



    return class_names[pred_idx], confidence