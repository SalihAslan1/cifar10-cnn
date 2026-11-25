import argparse
import json
import platform
from pathlib import Path
from statistics import median, quantiles
from time import perf_counter_ns

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data import read_dataset
from model import CLASSES, ROOT, load_model, predict


@torch.inference_mode()
def test_predictions(model, images, labels, device):
    chunks = []
    loss = torch.zeros((), device=device)
    for start in range(0, len(images), 256):
        batch = images[start:start + 256].to(device)
        targets = labels[start:start + 256].to(device)
        logits = model(batch)
        loss += F.cross_entropy(logits, targets, reduction="sum")
        chunks.append(logits.argmax(1).cpu())
    return torch.cat(chunks), loss.item() / len(images)


def classification_metrics(labels, predictions):
    classes = len(CLASSES)
    matrix = torch.bincount(labels * classes + predictions, minlength=classes ** 2)
    matrix = matrix.reshape(classes, classes).numpy()
    correct = matrix.diagonal()
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    precision = correct / np.maximum(predicted, 1)
    recall = correct / np.maximum(support, 1)
    f1 = 2 * correct / np.maximum(support + predicted, 1)
    return {
        "accuracy": float(correct.sum() / matrix.sum()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": [
            {"class": name, "support": int(support[i]), "precision": float(precision[i]),
             "recall": float(recall[i]), "f1": float(f1[i])}
            for i, name in enumerate(CLASSES)
        ],
        "confusion_matrix": matrix.tolist(),
    }


def cpu_name():
    if platform.system() == "Windows":
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor() or platform.machine()


def benchmark(model, images):
    generator = torch.Generator().manual_seed(42)
    sample_indices = torch.randperm(len(images), generator=generator)[:100]
    samples = [Image.fromarray(images[i].numpy()) for i in sample_indices]
    for sample in samples[:20]:
        predict(model, sample)
    timings = []
    for sample in samples:
        started = perf_counter_ns()
        predict(model, sample)
        timings.append((perf_counter_ns() - started) / 1_000_000)
    return {
        "device": "CPU",
        "cpu": cpu_name(),
        "threads": torch.get_num_threads(),
        "precision": "FP32",
        "batch_size": 1,
        "warmup_runs": 20,
        "samples": len(samples),
        "median_ms": median(timings),
        "p95_ms": quantiles(timings, n=100, method="inclusive")[94],
    }


def plot_training(history, output):
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
    series = (
        ("loss", "Kayıp", 1),
        ("accuracy", "Doğruluk (%)", 100),
    )
    for ax, (key, ylabel, scale) in zip(axes, series):
        ax.plot(epochs, [scale * row[f"train_{key}"] for row in history],
                color="#306a9c", label="Eğitim")
        ax.plot(epochs, [scale * row[f"validation_{key}"] for row in history],
                color="#bb5736", label="Doğrulama")
        ax.set(xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
    figure.savefig(output / "training_curves.png", dpi=180)
    plt.close(figure)


def plot_confusion(values, output):
    matrix = np.array(values)
    figure, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    heatmap = ax.imshow(matrix, cmap="Blues")
    for row, col in np.ndindex(matrix.shape):
        count = matrix[row, col]
        ax.text(col, row, str(count), ha="center", va="center",
                color="white" if count > matrix.max() / 2 else "#15283a", fontsize=8)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set(xlabel="Tahmin edilen sınıf", ylabel="Gerçek sınıf")
    figure.colorbar(heatmap, ax=ax, label="Görüntü sayısı")
    figure.savefig(output / "confusion_matrix.png", dpi=180)
    plt.close(figure)


def evaluate(output):
    output = output.resolve()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = output / "best.pt"
    training = json.loads((output / "training.json").read_text(encoding="utf-8"))
    model = load_model(checkpoint, device)
    images, labels = read_dataset(train=False)
    predictions, loss = test_predictions(model, images, labels, device)
    summary = classification_metrics(labels, predictions)
    summary.update({
        "test_count": len(labels),
        "test_loss": loss,
        "evaluation_device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "evaluation_precision": "FP32",
        "inference": benchmark(load_model(checkpoint), images),
    })
    (output / "evaluation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    plot_training(training["history"], output)
    plot_confusion(summary["confusion_matrix"], output)
    print(f"Test n={len(labels)} | Accuracy {summary['accuracy']:.2%} | "
          f"Macro F1 {summary['macro_f1']:.2%} | "
          f"CPU median {summary['inference']['median_ms']:.3f} ms", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 resmi test kümesini değerlendir")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    evaluate(parser.parse_args().output)
