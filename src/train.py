"""Training loop for ChestX-ray14 PyTorch model."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Callable
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    device: torch.device,
) -> float:
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable,
    device: torch.device,
) -> float:
    """Validate model performance on validation set."""
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    device: torch.device,
    epochs: int = 10,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
) -> tuple[nn.Module, list, list]:
    """Train and validate model for multiple epochs."""
    model.to(device)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if scheduler:
            scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return model, train_losses, val_losses
