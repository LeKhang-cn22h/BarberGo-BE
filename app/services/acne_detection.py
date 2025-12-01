from ultralytics import YOLO
from typing import Dict, List
import numpy as np
from app.core.config import MODEL_PATH, ACNE_LABELS, FACE_REGIONS
from app.models.face_mesh import FaceMeshDetector


class AcneDetectionService:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.face_detector = FaceMeshDetector()

    def detect_acne_in_region(self, region_img):
        """Phát hiện mụn trong một vùng da"""
        results = self.model(region_img, verbose=False)
        detections = results[0].boxes.data.cpu().numpy()

        region_result = {label: 0 for label in ACNE_LABELS}
        acne_details = []

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            label = ACNE_LABELS[int(cls)]
            region_result[label] += 1

            acne_details.append({
                'type': label,
                'confidence': float(conf),
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })

        return {
            'count': region_result,
            'total': sum(region_result.values()),
            'details': acne_details
        }

    def process_single_image(self, img):
        """Xử lý một ảnh"""
        regions = self.face_detector.extract_face_regions(img, FACE_REGIONS)

        if not regions:
            return None

        results = {}
        for region_name, region_data in regions.items():
            acne_result = self.detect_acne_in_region(region_data['image'])
            results[region_name] = acne_result

        return results

    def aggregate_results(self, all_results: Dict) -> Dict:
        """Tổng hợp kết quả từ 3 góc"""
        summary = {}

        for region_name in FACE_REGIONS.keys():
            total_count = 0
            merged_count = {label: 0 for label in ACNE_LABELS}

            for position in ['left', 'front', 'right']:
                if position in all_results and region_name in all_results[position]:
                    region_data = all_results[position][region_name]
                    total_count += region_data['total']
                    for label, count in region_data['count'].items():
                        merged_count[label] += count

            summary[region_name] = {
                'total': total_count,
                'count': merged_count
            }

        return summary
