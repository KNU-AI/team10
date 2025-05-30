import os
import torch

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

# Focal Loss 설정 (클래스별 alpha 비율 맞추기)
# fire: 6772, normal: 950, smoke: 5867 -> 비율 역수로 class별 가중치 줄 수 있음
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = [3.0, 1.0, 2.0]  # fire=3.0 가장중요, normal=1.0 기본값 유지, smoke=2.0 어느정도 중요

# 장비
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[CONFIG] Device set to: {DEVICE}")
print(f"[CONFIG] Project root: {PROJECT_ROOT}")
