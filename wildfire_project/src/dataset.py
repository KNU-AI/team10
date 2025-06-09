import os
import shutil
import random
from glob import glob

# ======== 기본 설정 ========
random.seed(42)

# 절대 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # wildfire_project
BASE_RAW_DIR = os.path.join(PROJECT_ROOT, "raw_data")
BASE_DST_DIR = os.path.join(PROJECT_ROOT, "data")
CLASS_MAP = {0: "smoke", 1: "fire"}

def create_dir_if_not_exists(path):
    os.makedirs(path, exist_ok=True)

def copy_images(file_list, dst_folder):
    create_dir_if_not_exists(dst_folder)
    for file_path in file_list:
        shutil.copy(file_path, dst_folder)

# ======== Forest Fire Dataset 처리 ========
def process_forest_fire_dataset():
    print(">> Forest Fire Dataset 처리 중...")

    # 원본 경로
    train_fire_src = os.path.join(BASE_RAW_DIR, "forest-fire-dataset", "Forest Fire Dataset", "Training", "fire")
    train_normal_src = os.path.join(BASE_RAW_DIR, "forest-fire-dataset", "Forest Fire Dataset", "Training", "nofire")
    test_src = os.path.join(BASE_RAW_DIR, "forest-fire-dataset", "Forest Fire Dataset", "Testing")

    fire_files = glob(os.path.join(train_fire_src, "*.jpg"))
    fire_files = fire_files[:len(fire_files) // 5]
    normal_files = glob(os.path.join(train_normal_src, "*.jpg"))
    test_files = glob(os.path.join(test_src, "*.jpg"))

    # 라벨링: test 데이터셋은 파일명으로 구분
    test_normal_files = [f for f in test_files if "nofire" in os.path.basename(f).lower()]
    test_fire_files = [f for f in test_files if "fire" in os.path.basename(f).lower() and f not in test_normal_files]

    # Split train/val
    def split_data(file_list, split_ratio=0.8):
        random.shuffle(file_list)
        split_idx = int(len(file_list) * split_ratio)
        return file_list[:split_idx], file_list[split_idx:]

    fire_train, fire_val = split_data(fire_files)
    normal_train, normal_val = split_data(normal_files)

    # 복사
    copy_images(fire_train, os.path.join(BASE_DST_DIR, "train", "fire"))
    copy_images(fire_val, os.path.join(BASE_DST_DIR, "val", "fire"))
    copy_images(test_fire_files, os.path.join(BASE_DST_DIR, "test", "fire"))

    copy_images(normal_train, os.path.join(BASE_DST_DIR, "train", "normal"))
    copy_images(normal_val, os.path.join(BASE_DST_DIR, "val", "normal"))
    copy_images(test_normal_files, os.path.join(BASE_DST_DIR, "test", "normal"))

    print(f"[fire] 총 이미지: {len(fire_files)} | Train: {len(fire_train)} | Val: {len(fire_val)}")
    print(f"[normal] 총 이미지: {len(normal_files)} | Train: {len(normal_train)} | Val: {len(normal_val)}")
    print(f"[test fire]: {len(test_fire_files)} | [test normal]: {len(test_normal_files)}")

# ======== Smoke-Fire YOLO Dataset 처리 ========
def process_smoke_fire_yolo():
    print("\n>> Smoke-Fire YOLO Dataset 처리 중...")

    yolo_base = os.path.join(BASE_RAW_DIR, "smoke-fire-detection-yolo", "data")
    stats = {"train": {"fire": 0, "smoke": 0}, "val": {"fire": 0, "smoke": 0}, "test": {"fire": 0, "smoke": 0}}

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(yolo_base, split, "images")
        label_dir = os.path.join(yolo_base, split, "labels")

        img_paths = glob(os.path.join(img_dir, "*.jpg"))

        for img_path in img_paths:
            label_path = os.path.join(label_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
            if not os.path.exists(label_path):
                continue

            with open(label_path, "r") as f:
                labels = f.readlines()

            label_classes = [int(line.strip().split()[0]) for line in labels]
            if not label_classes:
                continue

            # fire(1)가 하나라도 있으면 fire로 분류, 아니면 smoke
            class_id = 1 if 1 in label_classes else 0
            class_name = CLASS_MAP.get(class_id)

            if class_name:
                dst_folder = os.path.join(BASE_DST_DIR, split, class_name)
                copy_images([img_path], dst_folder)
                stats[split][class_name] += 1

    for split in ["train", "val", "test"]:
        print(f"[{split}] fire: {stats[split]['fire']} | smoke: {stats[split]['smoke']}")

# ======== 결과 요약 ========
def print_dataset_summary():
    print("\n>> 최종 데이터셋 파일 개수:")
    total_counts = {"fire": 0, "normal": 0, "smoke": 0}
    for split in ["train", "val", "test"]:
        print(f"\n[ {split.upper()} ]")
        for cls in ["fire", "normal", "smoke"]:
            folder = os.path.join(BASE_DST_DIR, split, cls)
            if os.path.exists(folder):
                count = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
                total_counts[cls] += count
                print(f"{cls:7}: {count}")
    print("\n[ 총합 ]")
    for cls, cnt in total_counts.items():
        print(f"{cls:7}: {cnt}")

# ======== 메인 실행 ========
def main():
    print("== 데이터셋 전처리 시작 ==")
    process_forest_fire_dataset()
    process_smoke_fire_yolo()
    print_dataset_summary()

if __name__ == "__main__":
    main()
