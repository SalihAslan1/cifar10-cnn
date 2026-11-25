from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "results" / "best.pt"
IMAGE_SIZE = 32
CLASSES = (
    "Uçak", "Otomobil", "Kuş", "Kedi", "Geyik",
    "Köpek", "Kurbağa", "At", "Gemi", "Kamyon",
)


class CifarCNN(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.as_tensor(std).reshape(1, 3, 1, 1))
        layers = []
        in_channels = 3
        for channels in (64, 128, 256):
            layers.extend([
                nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            in_channels = channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(256, len(CLASSES)),
        )

    def forward(self, images):
        pixels = images.permute(0, 3, 1, 2).float() / 255
        return self.classifier(self.features((pixels - self.mean) / self.std))


def load_model(path=MODEL_PATH, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(checkpoint["classes"]) != CLASSES:
        raise ValueError("Model sınıfları uygulamayla uyuşmuyor.")
    state = checkpoint["model"]
    model = CifarCNN(state["mean"], state["std"])
    model.load_state_dict(state)
    return model.to(device=device, memory_format=torch.channels_last).eval()


def prepare_image(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    return ImageOps.fit(image, (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)


@torch.inference_mode()
def predict(model, image):
    image = prepare_image(image)
    pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    inputs = pixels.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3).to(model.mean.device)
    probabilities = model(inputs).softmax(dim=1)[0].tolist()
    return dict(zip(CLASSES, probabilities))
