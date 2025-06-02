import os
import torch
from PIL import Image
from torchvision import transforms
from src.model import get_model
from src.config import DEVICE


class WildfireClassifier:
    def __init__(self):
        self.device = DEVICE
        self.model = get_model()

        # 프로젝트 루트 기준 절대 경로 지정
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # web/에서 두 단계 위 = wildfire_project
        model_path = os.path.join(BASE_DIR, "model", "model.pth")

        print(f"[PREDICTOR] Loading model from: {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # 이미지 전처리 파이프라인
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

        # 클래스 이름 리스트는 모델.py나 config.py가 아닌 여기서 직접 관리해도 됨
        class_names = ['fire', 'normal', 'smoke']
        pred_class = class_names[pred_idx.item()]
        confidence = conf.item()

        return pred_class, confidence


wildfire_classifier = WildfireClassifier()


def predict_image(image_path):
    return wildfire_classifier.predict(image_path)
