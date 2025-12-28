# services/advice_generator.py
from typing import Dict, List


class AdviceGenerator:
    """Tạo lời khuyên dựa trên kết quả phân loại mụn (Binary Classification)"""

    @staticmethod
    def generate_advice(summary: Dict) -> List[Dict]:
        """
        Tạo lời khuyên dựa trên vùng có mụn

        Args:
            summary: {
                region_name: {
                    'has_acne': bool,
                    'confidence': float,
                    'severity': str  # none/mild/moderate/severe
                }
            }

        Returns:
            List[Dict]: Danh sách lời khuyên
        """
        advice = []

        # Phân tích trán
        forehead_data = summary.get('forehead', {})
        if forehead_data.get('has_acne', False):
            severity = forehead_data.get('severity', 'mild')
            advice.append({
                'zone': 'Trán',
                'severity': severity,
                'confidence': forehead_data.get('confidence', 0.0),
                'tips': [
                    'Có thể do stress, vấn đề tiêu hóa hoặc mất cân bằng hormone',
                    'Rửa mặt 2 lần/ngày với sữa rửa mặt dịu nhẹ',
                    'Tránh để tóc che trán quá lâu',
                    'Uống đủ 2 lít nước/ngày, ngủ đủ 7-8 tiếng',
                    'Giảm thực phẩm nhiều đường và tinh bột'
                ]
            })

        # Phân tích mũi
        nose_data = summary.get('nose', {})
        if nose_data.get('has_acne', False):
            severity = nose_data.get('severity', 'mild')
            advice.append({
                'zone': 'Mũi',
                'severity': severity,
                'confidence': nose_data.get('confidence', 0.0),
                'tips': [
                    'Vùng chữ T dễ tiết dầu nhất',
                    'Dùng sản phẩm kiểm soát dầu (BHA/Salicylic Acid)',
                    'Dùng giấy thấm dầu 2-3 lần/ngày',
                    'Có thể có mụn ẩn, mụn đầu đen',
                    'Tuyệt đối không nặn mụn'
                ]
            })

        # Phân tích má
        cheek_left_data = summary.get('cheek_left', {})
        cheek_right_data = summary.get('cheek_right', {})

        cheek_left = cheek_left_data.get('has_acne', False)
        cheek_right = cheek_right_data.get('has_acne', False)

        if cheek_left or cheek_right:
            # Lấy severity cao nhất giữa 2 má
            severity_left = cheek_left_data.get('severity', 'none')
            severity_right = cheek_right_data.get('severity', 'none')

            severity_order = {'none': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
            max_severity = max(severity_left, severity_right,
                               key=lambda x: severity_order.get(x, 0))

            avg_confidence = (
                                     cheek_left_data.get('confidence', 0.0) +
                                     cheek_right_data.get('confidence', 0.0)
                             ) / 2

            advice.append({
                'zone': 'Má',
                'severity': max_severity,
                'confidence': avg_confidence,
                'tips': [
                    'Có thể do điện thoại, gối hoặc dị ứng',
                    'Lau màn hình điện thoại bằng cồn hàng ngày',
                    'Thay vỏ gối 2-3 lần/tuần',
                    'Kiểm tra các sản phẩm makeup/skincare có gây dị ứng',
                    'Hạn chế dùng tay chống má'
                ]
            })

        # Phân tích cằm
        chin_data = summary.get('chin', {})
        if chin_data.get('has_acne', False):
            severity = chin_data.get('severity', 'mild')
            advice.append({
                'zone': 'Cằm',
                'severity': severity,
                'confidence': chin_data.get('confidence', 0.0),
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
                'confidence': 1.0,
                'tips': [
                    'Da của bạn trong tình trạng tốt!',
                    'Duy trì thói quen chăm sóc da hiện tại',
                    'Vệ sinh sạch sẽ, chế độ ăn cân bằng',
                    'Sử dụng kem chống nắng SPF 30+ hàng ngày'
                ]
            })

        return advice

    @staticmethod
    def get_overall_summary(advice: List[Dict], summary_data: Dict) -> Dict:
        """
        Tổng hợp đánh giá chung dựa trên advice và summary

        Args:
            advice: Output từ generate_advice()
            summary_data: Output từ acne_service.get_summary()

        Returns:
            dict: {
                'severity': str,           # healthy/mild/moderate/severe
                'recommendation': str,     # Lời khuyên tổng quát
                'need_doctor': bool,       # Có cần gặp bác sĩ không
                'affected_regions': int    # Số vùng bị ảnh hưởng
            }
        """
        # Nếu không có mụn
        if not advice or advice[0].get('severity') == 'healthy':
            return {
                'severity': 'healthy',
                'recommendation': 'Da của bạn khá sạch! Tiếp tục chăm sóc như hiện tại.',
                'need_doctor': False,
                'affected_regions': 0
            }

        # Lấy thông tin từ summary_data
        overall_severity = summary_data.get('overall_severity', 'mild')
        acne_regions = summary_data.get('acne_regions', 0)

        # Tạo recommendation dựa trên severity
        if overall_severity == 'severe':
            recommendation = '🚨 Da có nhiều mụn nghiêm trọng. Nên gặp bác sĩ da liễu để được tư vấn điều trị chuyên sâu.'
            need_doctor = True

        elif overall_severity == 'moderate':
            recommendation = '⚠️ Da có mụn ở mức trung bình. Nên sử dụng sản phẩm trị mụn phù hợp và theo dõi tình trạng trong 2-3 tuần.'
            need_doctor = False

        elif overall_severity == 'mild':
            recommendation = 'Da có mụn nhẹ. Duy trì vệ sinh da mặt và chế độ ăn uống lành mạnh, tránh stress.'
            need_doctor = False

        else:  # healthy
            recommendation = 'Da của bạn khá sạch! Tiếp tục chăm sóc như hiện tại.'
            need_doctor = False

        return {
            'severity': overall_severity,
            'recommendation': recommendation,
            'need_doctor': need_doctor,
            'affected_regions': acne_regions
        }