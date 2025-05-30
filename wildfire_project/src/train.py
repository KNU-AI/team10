import torch
from torch import nn, optim
from .config import DEVICE, EPOCHS, LEARNING_RATE, FOCAL_GAMMA, FOCAL_ALPHA
from .utils import get_dataloaders
from .model import get_model

# ===== Focal Loss 정의 =====
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # list or tensor of weights for each class
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)  # shape: [batch_size]
        pt = torch.exp(-ce_loss)  # pt is the predicted probability of the correct class
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, list):
                alpha_tensor = torch.tensor(self.alpha).to(inputs.device)
            else:
                alpha_tensor = self.alpha.to(inputs.device)
            at = alpha_tensor[targets]
            focal_loss *= at

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ===== Training 함수 =====
def train():
    print("[TRAIN] Preparing training...")
    model = get_model()
    train_loader, val_loader, _, class_names = get_dataloaders()

    # 클래스 불균형 대응 focal loss 사용
    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
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
