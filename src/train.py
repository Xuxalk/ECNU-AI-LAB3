import argparse
import csv
import json
import random
import os
import time
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from PIL import Image

from src.cutmix import cutmix_batch, cutmix_loss
from src.models import build_model


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class IndexedCIFAR10(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image = Image.fromarray(self.dataset.data[real_idx])
        label = self.dataset.targets[real_idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def build_dataloaders(config, device):
    norm = config["normalization"]

    normalize = transforms.Normalize(
        norm["mean"], norm["std"]
    )

    augmentation = config.get("augmentation", "none")

    train_transforms = []

    if augmentation == "crop_flip":
        train_transforms.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5)
        ])
    elif augmentation == "crop_flip_color":
        train_transforms.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            )
        ])
    elif augmentation == "crop_flip_cutout":
        train_transforms.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5)
        ])
    elif augmentation != "none":
        raise ValueError(f"Unknown augmentation: {augmentation}")

    train_transforms.extend([
        transforms.ToTensor(),
        normalize
    ])

    if augmentation == "crop_flip_cutout":
        size = config.get("cutout_size", 8)
        probability = config.get("cutout_probability", 0.5)
        area = (size / 32) ** 2

        train_transforms.append(
            transforms.RandomErasing(
                p=probability,
                scale=(area, area),
                ratio=(1.0, 1.0),
                value=0.0
            )
        )

    train_transform = transforms.Compose(train_transforms)

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    base = CIFAR10(
        root=config["data_root"],
        train=True,
        download=False
    )

    train_indices = json.loads(
        Path(config["train_indices"]).read_text()
    )
    val_indices = json.loads(
        Path(config["val_indices"]).read_text()
    )

    train_set = IndexedCIFAR10(
        base, train_indices, train_transform
    )
    val_set = IndexedCIFAR10(
        base, val_indices, val_transform
    )

    generator = torch.Generator().manual_seed(config["seed"])

    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        generator=generator
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )

    return train_loader, val_loader, base.classes


def train_one_epoch(model, loader, criterion, optimizer, device, cutmix_config=None):
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if cutmix_config is not None:
            (
                images,
                labels_a,
                labels_b,
                lam,
                cutmix_applied,
                _,
                _,
            ) = cutmix_batch(
                images,
                labels,
                alpha=cutmix_config["alpha"],
                probability=cutmix_config["probability"],
            )
        else:
            labels_a = labels
            labels_b = None
            lam = 1.0
            cutmix_applied = False

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        if cutmix_applied:
            loss = cutmix_loss(criterion, outputs, labels_a, labels_b, lam)
        else:
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        n = labels.size(0)
        loss_sum += loss.item() * n
        predictions = outputs.argmax(1)
        if cutmix_applied:
            correct_a = (predictions == labels_a).sum().item()
            correct_b = (predictions == labels_b).sum().item()
            correct += lam * correct_a + (1.0 - lam) * correct_b
        else:
            correct += (predictions == labels).sum().item()
        total += n

    return loss_sum / total, 100 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        n = labels.size(0)
        loss_sum += loss.item() * n
        correct += (outputs.argmax(1) == labels).sum().item()
        total += n

    return loss_sum / total, 100 * correct / total


def main(config_path, resume=False):
    config = json.loads(Path(config_path).read_text())

    set_seed(config["seed"])
    device = get_device()

    train_loader, val_loader, classes = build_dataloaders(
        config, device
    )

    model = build_model(config, num_classes=len(classes)).to(device)

    parameter_count = sum(
        p.numel() for p in model.parameters()
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    scheduler = build_scheduler(config, optimizer)

    name = config["experiment_name"]
    run_dir = Path("experiments") / name
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    history_path = run_dir / "history.csv"
    best_path = checkpoint_dir / f"{name}_best.pth"
    last_path = checkpoint_dir / f"{name}_last.pth"

    saved_config = run_dir / "config.json"
    if saved_config.exists():
        existing = json.loads(saved_config.read_text())
        if existing != config:
            raise ValueError(
                f"Experiment {name} already has a different config. "
                "Choose a new experiment_name."
            )
        if not resume:
            raise FileExistsError(
                f"Experiment {name} already exists. Use --resume to continue "
                "from its last checkpoint or choose a new experiment_name."
            )
    else:
        saved_config.write_text(json.dumps(config, indent=4))

    best_accuracy = -1
    best_epoch = 0
    start_time = time.perf_counter()

    completed_epochs = 0
    elapsed_before_resume = 0.0

    if resume:
        if not last_path.exists():
            raise FileNotFoundError(f"No resumable checkpoint: {last_path}")
        checkpoint = torch.load(
            last_path, map_location="cpu", weights_only=True
        )
        if checkpoint.get("config") != config:
            raise ValueError("Checkpoint config does not match requested config")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None:
            if checkpoint.get("scheduler_state_dict") is None:
                raise ValueError(
                    "Checkpoint has no scheduler state; scheduled training "
                    "cannot resume safely."
                )
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        completed_epochs = int(checkpoint["epoch"])
        elapsed_before_resume = float(
            checkpoint.get("elapsed_training_seconds", 0.0)
        )
        restore_rng_state(checkpoint, train_loader, device)
        best_accuracy = float(checkpoint.get("best_val_accuracy", -1))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        if best_path.exists():
            best_checkpoint = torch.load(
                best_path, map_location="cpu", weights_only=True
            )
            best_accuracy = float(best_checkpoint["val_accuracy"])
            best_epoch = int(best_checkpoint["epoch"])
        if history_path.exists():
            with history_path.open(newline="") as file:
                rows = list(csv.DictReader(file))
            retained = [
                row for row in rows if int(row["epoch"]) <= completed_epochs
            ]
            if len(retained) != len(rows):
                temporary_history = history_path.with_suffix(".csv.tmp")
                with temporary_history.open("w", newline="") as file:
                    writer = csv.DictWriter(
                        file, fieldnames=retained[0].keys() if retained else [
                            "epoch", "train_loss", "train_accuracy",
                            "val_loss", "val_accuracy", "learning_rate",
                            "epoch_seconds",
                        ]
                    )
                    writer.writeheader()
                    writer.writerows(retained)
                os.replace(temporary_history, history_path)
        print(f"Resuming after epoch {completed_epochs}", flush=True)

    print("=" * 55)
    print("Experiment:", name)
    print("Device:", device)
    print("Parameters:", f"{parameter_count:,}")
    print("Training samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))
    print("=" * 55)

    for epoch in range(completed_epochs + 1, config["epochs"] + 1):
        epoch_start = time.perf_counter()
        learning_rate = float(optimizer.param_groups[0]["lr"])

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            cutmix_config=config.get("cutmix"),
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.perf_counter() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": learning_rate,
            "epoch_seconds": epoch_time
        }

        exists = history_path.exists()
        with history_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not exists:
                writer.writeheader()
            writer.writerow(row)

        print(
            f"Epoch [{epoch}/{config['epochs']}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.2f}% "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc:.2f}% "
            f"LR: {learning_rate:.8f} "
            f"Time: {epoch_time:.1f}s",
            flush=True
        )

        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": cpu_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "rng_state": capture_rng_state(train_loader, device),
            "best_epoch": best_epoch,
            "best_val_accuracy": best_accuracy,
            "elapsed_training_seconds": (
                elapsed_before_resume + time.perf_counter() - start_time
            ),
            "classes": classes,
            "config": config,
        }

        improved = val_acc > best_accuracy
        if improved:
            best_accuracy = val_acc
            best_epoch = epoch
            checkpoint_state["best_epoch"] = best_epoch
            checkpoint_state["best_val_accuracy"] = best_accuracy
            checkpoint_state["val_accuracy"] = val_acc
            save_checkpoint(checkpoint_state, best_path)

            print("  Best checkpoint saved", flush=True)

        # Save after every completed epoch so an interrupted run can resume.
        checkpoint_state["best_epoch"] = best_epoch
        checkpoint_state["best_val_accuracy"] = best_accuracy
        save_checkpoint(checkpoint_state, last_path)

    elapsed = elapsed_before_resume + time.perf_counter() - start_time

    summary = {
        "experiment": name,
        "device": str(device),
        "parameters": parameter_count,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy,
        "training_seconds": elapsed
    }
    with history_path.open(newline="") as file:
        completed_history = list(csv.DictReader(file))
    if completed_history and "learning_rate" in completed_history[-1]:
        summary["last_epoch_learning_rate"] = float(
            completed_history[-1]["learning_rate"]
        )
    if scheduler is not None:
        summary["scheduler"] = config["scheduler"]
        summary["learning_rate_after_final_step"] = float(
            optimizer.param_groups[0]["lr"]
        )

    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=4)
    )

    print("\nTRAINING COMPLETE")
    print("Best epoch:", best_epoch)
    print("Best validation accuracy:", f"{best_accuracy:.2f}%")
    print("Total time:", f"{elapsed / 60:.2f} minutes")


def cpu_state_dict(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def build_scheduler(config, optimizer):
    scheduler_name = config.get("scheduler", "none")
    if scheduler_name == "none":
        return None
    if scheduler_name == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["t_max"],
            eta_min=config["eta_min"],
        )
    raise ValueError(f"Unknown scheduler: {scheduler_name}")


def capture_rng_state(loader, device):
    state = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "loader": loader.generator.get_state(),
    }
    if device.type == "mps":
        state["torch_mps"] = torch.mps.get_rng_state(device)
    elif device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(checkpoint, loader, device):
    state = checkpoint.get("rng_state")
    if state is None:
        # Support last checkpoints written before RNG states were grouped.
        torch.set_rng_state(checkpoint["torch_rng_state"])
        loader.generator.set_state(checkpoint["loader_rng_state"])
        return

    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    loader.generator.set_state(state["loader"])
    if device.type == "mps" and "torch_mps" in state:
        torch.mps.set_rng_state(state["torch_mps"], device)
    elif device.type == "cuda" and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(checkpoint, path):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume this experiment from its last completed epoch.",
    )
    args = parser.parse_args()
    main(args.config, resume=args.resume)
