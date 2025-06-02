# Flask 메인 서버
# 수정1
import sys
import os

# app.py 위치 기준 상위 폴더(wildfire_project) 모듈 탐색 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
from flask import Flask, render_template, send_file
from predictor import predict_image

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(BASE_DIR, "data", "test")
CLASS_FOLDERS = ["fire", "normal", "smoke"]

@app.route("/")
def index():
    # 전체 test 이미지 경로 및 클래스 리스트 생성
    all_images = []
    for cls in CLASS_FOLDERS:
        cls_dir = os.path.join(TEST_DIR, cls)
        if os.path.exists(cls_dir):
            imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".png"))]
            for img in imgs:
                all_images.append((cls, img))

    # 전체에서 12장 랜덤 샘플링
    sampled_images = random.sample(all_images, min(12, len(all_images)))

    selected_images = []
    for cls, img in sampled_images:
        abs_path = os.path.join(TEST_DIR, cls, img)
        pred_cls, confidence = predict_image(abs_path)
        selected_images.append({
            "filename": img,
            "true_cls": cls,
            "pred": pred_cls,
            "conf": f"{confidence*100:.1f}%",
            "cls_folder": cls
        })

    return render_template("index.html", images=selected_images)

@app.route("/test_images/<cls>/<filename>")
def serve_test_image(cls, filename):
    img_path = os.path.join(TEST_DIR, cls, filename)
    if os.path.exists(img_path):
        return send_file(img_path)
    else:
        return "Image not found", 404

if __name__ == "__main__":
    app.run(debug=True)
