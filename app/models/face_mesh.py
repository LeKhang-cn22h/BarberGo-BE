import mediapipe as mp
import numpy as np
import cv2


class FaceMeshDetector:
    def __init__(self):
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def get_region_bbox(self, landmarks, region_indices, img_shape):
        """Tính bounding box cho một vùng da dựa trên landmarks"""
        h, w = img_shape[:2]
        points = []

        for idx in region_indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                x = int(lm.x * w)
                y = int(lm.y * h)
                points.append([x, y])

        if not points:
            return None

        points = np.array(points)
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)

        # Mở rộng bbox
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        return (x_min, y_min, x_max, y_max)

    def extract_face_regions(self, img, face_regions_config):
        """Trích xuất các vùng da từ ảnh"""
        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        regions = {}

        for region_name, region_indices in face_regions_config.items():
            bbox = self.get_region_bbox(landmarks, region_indices, img.shape)
            if bbox:
                x_min, y_min, x_max, y_max = bbox
                region_img = img[y_min:y_max, x_min:x_max]

                if region_img.size > 0:
                    regions[region_name] = {
                        'image': region_img,
                        'bbox': bbox
                    }

        return regions
