from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from .config import TRAIN_DIR, VAL_DIR, TEST_DIR, BATCH_SIZE


def get_dataloaders():
    print("[UTILS] Loading datasets...")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)

    print(f"[UTILS] Classes found: {train_dataset.classes}")
    print(f"[UTILS] Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.classes

