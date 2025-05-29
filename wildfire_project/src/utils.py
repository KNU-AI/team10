from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from .config import TRAIN_DIR, VAL_DIR, TEST_DIR, BATCH_SIZE
import numpy as np

def get_dataloaders(val_ratio=0.2):
    print("[UTILS] Loading dataset with train/val split...")

    # [1] Train transform: 데이터 증강 포함
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 밝기와 대비를 랜덤하게 변화시킴
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 크기 랜덤하게 조절
        transforms.RandomVerticalFlip(),  # 수직 뒤집기 추가
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # [2] Val/Test transform: 증강 X
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # 전체 데이터셋 불러오기
    full_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    class_names = full_dataset.classes
    indices = np.arange(len(full_dataset))
    labels = [label for _, label in full_dataset.samples]

    # train/val 데이터셋 분할
    train_idx, val_idx = train_test_split(
        indices, test_size=val_ratio, stratify=labels, random_state=42
    )

    # Subset으로 train/val 데이터셋 생성
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    # 테스트셋은 독립적으로 로드
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_test_transform)

    print(f"[UTILS] Classes: {class_names}")
    print(f"[UTILS] Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # DataLoader 설정
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, class_names
