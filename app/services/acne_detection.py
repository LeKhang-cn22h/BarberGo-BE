# services/acne_detection.py
from typing import Dict
import numpy as np
import cv2
from app.core.config import MODEL_PATH, FACE_REGIONS, DEVICE
from app.models.face_mesh import FaceMeshDetector
from app.models.acne_classifier import AcneClassifier


class AcneDetectionService:
    """Service phát hiện mụn (với pore removal)"""

    def __init__(self):
        # Load model đã train với pore removal
        self.classifier = AcneClassifier(MODEL_PATH, device=DEVICE)
        self.face_detector = FaceMeshDetector()

    def detect_acne_in_region(self, region_img):
        """
        Phát hiện mụn trong vùng da

        KHÔNG CẦN preprocessing ở đây vì classifier đã xử lý bên trong
        """
        if region_img is None or region_img.size == 0:
            return {
                'has_acne': False,
                'confidence': 0.0,
                'severity': 'none'
            }

        # ✅ Classifier tự xử lý pore removal
        has_acne, confidence = self.classifier.predict_with_confidence(region_img)

        # Phân loại severity
        if not has_acne:
            severity = 'none'
        elif confidence < 0.65:
            severity = 'mild'
        elif confidence < 0.8:
            severity = 'moderate'
        else:
            severity = 'severe'

        return {
            'has_acne': bool(has_acne),
            'confidence': float(confidence),
            'severity': severity
        }

    def process_image(self, img) -> Dict:
        """Xử lý ảnh chính diện"""
        print("🔍 Detecting face regions...")

        # Extract face regions
        all_regions = self.face_detector.extract_face_regions(img, FACE_REGIONS)

        if not all_regions:
            print("   ⚠️  No face detected")
            return {}

        print(f"   ✓ Found {len(all_regions)} regions")

        # Detect acne for each region
        results = {}
        for region_name, region_data in all_regions.items():
            acne_result = self.detect_acne_in_region(region_data['image'])
            results[region_name] = acne_result

            status = "CÓ MỤN" if acne_result['has_acne'] else "SẠCH"
            conf = acne_result['confidence']
            severity = acne_result['severity']
            print(f"      • {region_name}: {status} ({severity}) - conf: {conf:.3f}")

        return results

    def get_summary(self, results: Dict) -> Dict:
        """Tổng hợp kết quả"""
        if not results:
            return {
                'total_regions': 0,
                'acne_regions': 0,
                'clear_regions': 0,
                'severity_count': {},
                'overall_severity': 'none',
                'average_confidence': 0.0,
                'regions_detail': {}
            }

        acne_regions = sum(1 for r in results.values() if r['has_acne'])
        clear_regions = len(results) - acne_regions

        severity_count = {}
        total_confidence = 0

        for region_data in results.values():
            severity = region_data['severity']
            severity_count[severity] = severity_count.get(severity, 0) + 1
            total_confidence += region_data['confidence']

        avg_confidence = total_confidence / len(results)

        if acne_regions == 0:
            overall_severity = 'none'
        elif acne_regions <= 2:
            overall_severity = 'mild'
        elif acne_regions <= 5:
            overall_severity = 'moderate'
        else:
            overall_severity = 'severe'

        return {
            'total_regions': len(results),
            'acne_regions': acne_regions,
            'clear_regions': clear_regions,
            'severity_count': severity_count,
            'overall_severity': overall_severity,
            'average_confidence': float(avg_confidence),
            'regions_detail': results
        }