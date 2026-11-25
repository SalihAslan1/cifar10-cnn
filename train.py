import argparse
import json
import os
import platform
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from data import channel_statistics, read_dataset, stratified_split
from model import CLASSES, IMAGE_SIZE, ROOT, CifarCNN


def augment(images):
    images = images.permute(0, 3, 1, 2)
    padded = F.pad(images, (4, 4, 4, 4)).permute(0, 2, 3, 1)
    n = len(images)
    device = images.device
    coords = torch.arange(IMAGE_SIZE, device=device)
    offsets = torch.randint(0, 9, (n, 2), device=device)
    flip = torch.rand(n, 1, device=device) < 0.5
    rows = coords + offsets[:, 0:1]
    cols = torch.where(flip, IMAGE_SIZE - 1 - coords, coords) + offsets[:, 1:2]
    return padded[torch.arange(n, device=device)[:, None, None], rows[:, :, None], cols[:, None, :]]


def run_epoch(model, data, labels, batch_size, criterion, optimizer=None, amp=False):
    training = optimizer is not None
    model.train(training)
    order = torch.randperm(len(data), device=data.device) if training else None
    loss_sum = torch.zeros((), device=data.device)
    correct = torch.zeros((), device=data.device, dtype=torch.int64)
    with torch.set_grad_enabled(training):
        for start in range(0, len(data), batch_size):
            stop = start + batch_size
            if training:
                indices = order[start:stop]
                images, targets = augment(data[indices]), labels[indices]
                optimizer.zero_grad(set_to_none=True)
            else:
                images, targets = data[start:stop], labels[start:stop]
            with torch.autocast(data.device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.detach() * len(targets)
            correct += (logits.detach().argmax(1) == targets).sum()
    return loss_sum.item() / len(data), correct.item() / len(data)


def atomic_torch_save(value, path):
    temporary = path.with_suffix(".pt.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def describe_environment(device, amp):
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "torchvision": torchvision.__version__,
        "cpu": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "device_type": device.type,
        "cuda": torch.version.cuda if device.type == "cuda" else None,
        "precision": "BF16 autocast / FP32 weights" if amp else "FP32",
        "cpu_threads": torch.get_num_threads(),
    }


def train(args):
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA bu PyTorch ortamında kullanılamıyor.")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    environment = describe_environment(device, amp)
    config = {
        "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
        "learning_rate": 0.001, "min_learning_rate": 0.00001, "weight_decay": 0.0005,
        "label_smoothing": 0.05,
    }
    last_path = args.output / "last.pt"
    args.output.mkdir(parents=True, exist_ok=True)
    if last_path.exists() and not args.resume:
        raise SystemExit("Bu dizinde bir eğitim var. --resume kullanın veya farklı --output verin.")

    images, labels = read_dataset(train=True, download=True)
    train_idx, val_idx = stratified_split(labels, args.seed)
    train_images = images[train_idx]
    mean, std = channel_statistics(train_images)
    train_images, train_labels = train_images.to(device), labels[train_idx].to(device)
    val_images, val_labels = images[val_idx].to(device), labels[val_idx].to(device)
    del images, labels

    model = CifarCNN(mean, std).to(device=device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=config["min_learning_rate"],
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    history, best, start_epoch = [], None, 1
    if args.resume:
        previous = torch.load(last_path, map_location="cpu", weights_only=True)
        if previous["config"] != config or previous["environment"] != environment:
            raise SystemExit("Kaldığınız yerden devam etmek için aynı eğitim ayarları ve ortamı gerekli.")
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        history, best = previous["history"], previous["best"]
        start_epoch = history[-1]["epoch"] + 1
        torch.set_rng_state(previous["cpu_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(previous["cuda_rng"], device=device)

    print(f"{environment['device']} | train={len(train_images)} val={len(val_images)} "
          f"| {environment['precision']}", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        started = perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = run_epoch(
            model, train_images, train_labels, args.batch_size, criterion, optimizer, amp,
        )
        val_loss, val_acc = run_epoch(
            model, val_images, val_labels, args.batch_size, criterion, amp=amp,
        )
        scheduler.step()
        row = {
            "epoch": epoch, "lr": lr, "train_loss": train_loss, "train_accuracy": train_acc,
            "validation_loss": val_loss, "validation_accuracy": val_acc,
            "seconds": perf_counter() - started,
        }
        history.append(row)
        if best is None or val_acc > best["validation_accuracy"]:
            best = {
                "epoch": epoch, "validation_accuracy": val_acc, "classes": CLASSES,
                "model": {name: value.detach().to("cpu", copy=True)
                          for name, value in model.state_dict().items()},
            }
        atomic_torch_save({
            "config": config, "environment": environment, "history": history, "best": best,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }, last_path)
        print(f"{epoch:02}/{args.epochs} | train {train_loss:.4f} {train_acc:.2%} "
              f"| val {val_loss:.4f} {val_acc:.2%} | {row['seconds']:.1f} s", flush=True)

    atomic_torch_save(best, args.output / "best.pt")
    summary = {
        "config": config, "environment": environment, "history": history,
        "best_epoch": best["epoch"], "best_validation_accuracy": best["validation_accuracy"],
        "training_seconds": sum(row["seconds"] for row in history),
        "train_count": len(train_images), "validation_count": len(val_images),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "mean": mean.tolist(), "std": std.tolist(),
    }
    (args.output / "training.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {args.output / 'best.pt'} | best epoch: {best['epoch']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 CNN eğitimi")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.epochs < 1 or arguments.batch_size < 1:
        parser.error("Epoch ve batch size pozitif olmalı.")
    train(arguments)
