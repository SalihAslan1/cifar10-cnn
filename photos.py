import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps

from model import CLASSES, ROOT, load_model, predict


def evaluate_photos(manifest, output):
    with manifest.open(encoding="utf-8-sig", newline="") as csv_file:
        items = list(csv.DictReader(csv_file))
    if len(items) != 5 or not all(item.get("image") and item.get("label") for item in items):
        raise ValueError("CSV 'image,label' sütunlarında beş fotoğraf ve gerçek sınıf bulunmalı.")

    photos = [(manifest.parent / item["image"], item["label"].strip()) for item in items]
    fingerprints = {hashlib.sha256(path.read_bytes()).hexdigest() for path, _ in photos}
    if len(fingerprints) != 5:
        raise ValueError("Beş farklı fotoğraf kullanılmalı.")
    if any(label not in CLASSES for _, label in photos):
        raise ValueError(f"Gerçek sınıflar şu etiketlerden biri olmalı: {', '.join(CLASSES)}.")

    torch.set_num_threads(min(4, torch.get_num_threads()))
    checkpoint = output / "best.pt"
    model = load_model(checkpoint)
    photo_dir = output / "phone"
    photo_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (path, label) in enumerate(photos, 1):
        with Image.open(path) as image:
            scores = predict(model, image)
            preview = ImageOps.exif_transpose(image).convert("RGB")
            preview.thumbnail((640, 640))
            preview_name = f"{i:02}.jpg"
            preview.save(photo_dir / preview_name, quality=90)
        prediction = max(scores, key=scores.get)
        rows.append({
            "photo": preview_name, "source_name": path.name,
            "ground_truth": label, "prediction": prediction,
            "probability": scores[prediction], "correct": prediction == label,
        })

    result = {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "photos": rows}
    (photo_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"Telefon fotoğrafları: {sum(row['correct'] for row in rows)}/5 doğru.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Beş telefon fotoğrafını değerlendir")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    arguments = parser.parse_args()
    evaluate_photos(arguments.manifest, arguments.output)
