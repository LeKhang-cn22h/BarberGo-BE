from typing import Dict, List


class AdviceGenerator:
    @staticmethod
    def generate_advice(summary: Dict) -> List[Dict]:
        """Tạo lời khuyên dựa trên phân tích mụn"""
        advice = []

        # Phân tích trán
        forehead_total = summary.get('forehead', {}).get('total', 0)
        if forehead_total > 3:
            severity = 'moderate' if forehead_total < 7 else 'severe'
            advice.append({
                'zone': 'Trán',
                'severity': severity,
                'acne_count': forehead_total,
                'tips': [
                    'Có thể do stress, vấn đề tiêu hóa hoặc mất cân bằng hormone',
                    'Rửa mặt 2 lần/ngày với sữa rửa mặt dịu nhẹ',
                    'Tránh để tóc che trán quá lâu',
                    'Uống đủ 2 lít nước/ngày, ngủ đủ 7-8 tiếng',
                    'Giảm thực phẩm nhiều đường và tinh bột'
                ]
            })

        # Phân tích thái dương
        temple_total = summary.get('temple_left', {}).get('total', 0) + \
                       summary.get('temple_right', {}).get('total', 0)
        if temple_total > 2:
            advice.append({
                'zone': 'Thái dương',
                'severity': 'moderate',
                'acne_count': temple_total,
                'tips': [
                    'Có thể liên quan đến chức năng gan/mật',
                    'Giảm thức ăn chiên rán, dầu mỡ',
                    'Vệ sinh gọng kính, tai nghe thường xuyên',
                    'Tránh chạm tay vào vùng này'
                ]
            })

        # Phân tích mũi
        nose_total = summary.get('nose', {}).get('total', 0)
        if nose_total > 2:
            advice.append({
                'zone': 'Mũi',
                'severity': 'mild',
                'acne_count': nose_total,
                'tips': [
                    'Vùng chữ T dễ tiết dầu nhất',
                    'Dùng sản phẩm kiểm soát dầu (BHA/Salicylic Acid)',
                    'Dùng giấy thấm dầu khi cần',
                    'Tuyệt đối không nặn mụn'
                ]
            })

        # Phân tích má
        cheek_total = summary.get('cheek_left', {}).get('total', 0) + \
                      summary.get('cheek_right', {}).get('total', 0)
        if cheek_total > 4:
            severity = 'moderate' if cheek_total < 8 else 'severe'
            advice.append({
                'zone': 'Má',
                'severity': severity,
                'acne_count': cheek_total,
                'tips': [
                    'Có thể do điện thoại, gối hoặc dị ứng',
                    'Lau màn hình điện thoại bằng cồn hàng ngày',
                    'Thay vỏ gối 2-3 lần/tuần',
                    'Kiểm tra các sản phẩm makeup/skincare có gây dị ứng',
                    'Hạn chế dùng tay chống má'
                ]
            })

        # Phân tích cằm
        chin_total = summary.get('chin', {}).get('total', 0)
        if chin_total > 3:
            advice.append({
                'zone': 'Cằm',
                'severity': 'moderate',
                'acne_count': chin_total,
                'tips': [
                    'Thường do mất cân bằng hormone',
                    'Với nữ: theo dõi chu kỳ kinh nguyệt',
                    'Giảm đường tinh luyện và sữa trong chế độ ăn',
                    'Tăng cường rau xanh, omega-3',
                    'Nếu kéo dài >3 tháng, nên gặp bác sĩ da liễu'
                ]
            })

        # Da khỏe mạnh
        if not advice:
            advice.append({
                'zone': 'Tổng quan',
                'severity': 'healthy',
                'acne_count': 0,
                'tips': [
                    'Da của bạn trong tình trạng tốt!',
                    'Duy trì thói quen chăm sóc da hiện tại',
                    'Vệ sinh sạch sẽ, chế độ ăn cân bằng',
                    'Sử dụng kem chống nắng hàng ngày'
                ]
            })

        return advice
