# models/acne_classifier.py
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np


def remove_pores_advanced(img):
    """Xử lý giống training"""
    smooth = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(smooth, cv2.MORPH_CLOSE, kernel, iterations=1)
    final = cv2.GaussianBlur(closed, (3, 3), 0)
    return final


def check_acne_redness(img):
    """
    Kiểm tra màu đỏ VIÊM của mụn (không phải màu da tự nhiên)

    Returns:
        float: Acne redness score (0-1)
    """
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ✅ CHỈ DETECT MÀU ĐỎ VIÊM (không phải màu da)
    # Mụn viêm: Hue 0-20 (đỏ tươi), Saturation cao, Value cao
    lower_acne_red = np.array([0, 100, 100])  # S và V cao hơn
    upper_acne_red = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower_acne_red, upper_acne_red)

    # Calculate percentage
    red_percentage = np.sum(mask > 0) / mask.size

    # ✅ CHỈ TÍNH KHI CÓ VÙNG ĐỎ TẬP TRUNG (không phải rải rác)
    # Find contours để xem có vùng đỏ tập trung không
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Lấy vùng đỏ lớn nhất
        largest_contour = max(contours, key=cv2.contourArea)
        largest_area = cv2.contourArea(largest_contour)
        total_area = img.shape[0] * img.shape[1]

        # Nếu vùng đỏ lớn nhất < 1% → không phải mụn
        if largest_area / total_area < 0.01:
            return 0.0

    return red_percentage


def check_texture_variance(img):
    """Kiểm tra độ biến thiên texture"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    normalized_variance = min(variance / 100.0, 1.0)
    return normalized_variance


def check_spot_presence(img):
    """
    Kiểm tra có điểm nhô lên (bump) không
    Sử dụng edge detection để tìm vùng nổi

    Returns:
        float: Spot score (0-1)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian blur để giảm noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Tìm contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0

    # Lọc contours theo kích thước (mụn thường có kích thước nhất định)
    min_area = 20  # pixels
    max_area = 500  # pixels

    acne_like_contours = [
        cnt for cnt in contours
        if min_area <= cv2.contourArea(cnt) <= max_area
    ]

    # Score dựa trên số lượng contours giống mụn
    spot_score = min(len(acne_like_contours) / 5.0, 1.0)  # Normalize to 0-1

    return spot_score


class AcneClassifier:
    def __init__(self, model_path: str, device: str = "cpu", strict_mode: bool = True):
        self.device = torch.device(device)
        self.strict_mode = strict_mode

        print(f"🔧 Loading model from: {model_path}")
        print(f"   Device: {self.device}")
        print(f"   Strict mode: {'ENABLED ✅' if strict_mode else 'DISABLED'}")

        self.model = models.mobilenet_v2(pretrained=False)

        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print("✅ Model loaded successfully!")

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict_with_strict_criteria(self, image):
        """
        Predict với tiêu chí GẮT GẠO + TIN CNN NHIỀU HƠN
        """
        # Preprocess
        image_processed = remove_pores_advanced(image)

        # 1. CNN Prediction
        image_rgb = cv2.cvtColor(image_processed, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            cnn_probability = torch.sigmoid(output).item()

        # ✅ CNN THRESHOLD GẮT GẠO
        cnn_has_acne = cnn_probability < 0.3
        cnn_confidence = abs(cnn_probability - 0.5) * 2

        if not self.strict_mode:
            return cnn_has_acne, cnn_confidence, {
                'cnn_probability': cnn_probability,
                'cnn_decision': cnn_has_acne
            }

        # 2. Additional Checks
        redness_score = check_acne_redness(image)  # ← Sửa đổi
        texture_score = check_texture_variance(image)
        spot_score = check_spot_presence(image)  # ← Thêm mới

        # 3. ✅ ENSEMBLE DECISION - TIN CNN NHIỀU HƠN
        # Chiến lược: CNN là tiêu chí BẮT BUỘC

        if not cnn_has_acne:
            # ✅ NẾU CNN NÓI KHÔNG → TIN CNN LUÔN
            final_has_acne = False
            final_confidence = 1.0 - cnn_confidence

            details = {
                'cnn_probability': cnn_probability,
                'cnn_decision': cnn_has_acne,
                'redness_score': redness_score,
                'texture_score': texture_score,
                'spot_score': spot_score,
                'decision_reason': 'CNN says NO → Trusted',
                'final_decision': final_has_acne
            }

            return final_has_acne, final_confidence, details

        # ✅ NẾU CNN NÓI CÓ → KIỂM TRA THÊM
        additional_criteria = 0
        total_additional = 3

        # Criterion 1: Has acne-like redness (THRESHOLD CAO HƠN)
        if redness_score > 0.15:  # Tăng từ 0.05 lên 0.15
            additional_criteria += 1

        # Criterion 2: Has texture variation
        if texture_score > 0.5:  # Tăng từ 0.3 lên 0.5
            additional_criteria += 1

        # Criterion 3: Has visible spots
        if spot_score > 0.3:
            additional_criteria += 1

        # ✅ CẦN ÍT NHẤT 2/3 TIÊU CHÍ PHỤ (ngoài CNN)
        final_has_acne = additional_criteria >= 2
        final_confidence = (additional_criteria / total_additional) * cnn_confidence

        details = {
            'cnn_probability': cnn_probability,
            'cnn_decision': cnn_has_acne,
            'redness_score': redness_score,
            'texture_score': texture_score,
            'spot_score': spot_score,
            'additional_criteria_met': f"{additional_criteria}/{total_additional}",
            'decision_reason': f'CNN says YES + {additional_criteria}/3 additional',
            'final_decision': final_has_acne
        }

        return final_has_acne, final_confidence, details

    def predict(self, image):
        has_acne, _, _ = self.predict_with_strict_criteria(image)
        return has_acne

    def predict_with_confidence(self, image):
        has_acne, confidence, details = self.predict_with_strict_criteria(image)

        # In ra chi tiết
        print(f"   CNN prob: {details['cnn_probability']:.4f}")
        print(f"   CNN says: {'ACNE' if details['cnn_decision'] else 'CLEAN'}")
        print(f"   Redness: {details['redness_score']:.3f}")
        print(f"   Texture: {details['texture_score']:.3f}")
        print(f"   Spots: {details['spot_score']:.3f}")
        print(f"   Decision: {details['decision_reason']}")

        return has_acne, confidence

    def get_raw_probability(self, image):
        image_processed = remove_pores_advanced(image)
        image_rgb = cv2.cvtColor(image_processed, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probability = torch.sigmoid(output).item()

        return probability