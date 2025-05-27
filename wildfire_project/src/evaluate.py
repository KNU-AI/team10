import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from .config import DEVICE
from .utils import get_dataloaders

def evaluate(model):
    print("[EVAL] Starting evaluation...")
    _, _, test_loader, class_names = get_dataloaders()
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()

            y_true.extend(labels.numpy())
            y_pred.extend(preds)

            if i % 5 == 0:
                print(f"[EVAL] Processed batch {i}/{len(test_loader)}")

    print("\n[EVAL] Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("[EVAL] Displaying confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
