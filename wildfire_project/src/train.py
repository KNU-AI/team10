import torch
from torch import nn, optim
from .config import DEVICE, EPOCHS, LEARNING_RATE
from .utils import get_dataloaders
from .model import get_model

def train():
    print("[TRAIN] Preparing training...")
    model = get_model()
    train_loader, val_loader, _, class_names = get_dataloaders()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        print(f"\n[Epoch {epoch + 1}/{EPOCHS}]")
        model.train()
        total_loss, correct = 0, 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

            if i % 10 == 0:
                print(f"[TRAIN] Batch {i}/{len(train_loader)} - Loss: {loss.item():.4f}")

        acc = correct / len(train_loader.dataset)
        print(f"[TRAIN] Epoch {epoch+1} Complete - Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")

    print("[TRAIN] Training finished.")
    return model
