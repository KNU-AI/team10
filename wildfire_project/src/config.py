import os
import torch # pip install torch torchvision

print("[CONFIG] Loading configuration...")

# 프로젝트 루트
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 데이터 경로
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

# 하이퍼파라미터
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_CLASSES = 3

# 장비
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[CONFIG] Device set to: {DEVICE}")
print(f"[CONFIG] Project root: {PROJECT_ROOT}")