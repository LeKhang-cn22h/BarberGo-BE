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



# # services/advice_generator.py
# from typing import Dict, List
#
#
# class AdviceGenerator:
#     """Tạo lời khuyên dựa trên kết quả phân loại mụn"""
#
#     @staticmethod
#     def generate_advice(summary: Dict) -> List[Dict]:
#         """
#         Tạo lời khuyên dựa trên vùng có mụn
#
#         Args:
#             summary: {
#                 region_name: {
#                     'has_acne': bool,
#                     'confidence': float
#                 }
#             }
#
#         Returns:
#             List[Dict]: Danh sách lời khuyên
#         """
#         advice = []
#
#         # Phân tích trán
#         if summary.get('forehead', {}).get('has_acne', False):
#             advice.append({
#                 'zone': 'Trán',
#                 'severity': 'detected',
#                 'tips': [
#                     'Có thể do stress, vấn đề tiêu hóa hoặc mất cân bằng hormone',
#                     'Rửa mặt 2 lần/ngày với sữa rửa mặt dịu nhẹ',
#                     'Tránh để tóc che trán quá lâu',
#                     'Uống đủ 2 lít nước/ngày, ngủ đủ 7-8 tiếng',
#                     'Giảm thực phẩm nhiều đường và tinh bột'
#                 ]
#             })
#
#
#
#         # Phân tích mũi
#         if summary.get('nose', {}).get('has_acne', False):
#             advice.append({
#                 'zone': 'Mũi',
#                 'severity': 'detected',
#                 'tips': [
#                     'Vùng chữ T dễ tiết dầu nhất',
#                     'Dùng sản phẩm kiểm soát dầu (BHA/Salicylic Acid)',
#                     'Có mụn ẩn, mụn đầu đen',
#                     'Tuyệt đối không nặn mụn'
#                 ]
#             })
#
#         # Phân tích má
#         cheek_left = summary.get('cheek_left', {}).get('has_acne', False)
#         cheek_right = summary.get('cheek_right', {}).get('has_acne', False)
#
#         if cheek_left or cheek_right:
#             advice.append({
#                 'zone': 'Má',
#                 'severity': 'detected',
#                 'tips': [
#                     'Có thể do điện thoại, gối hoặc dị ứng',
#                     'Lau màn hình điện thoại bằng cồn hàng ngày',
#                     'Thay vỏ gối 2-3 lần/tuần',
#                     'Kiểm tra các sản phẩm makeup/skincare có gây dị ứng',
#                     'Hạn chế dùng tay chống má'
#                 ]
#             })
#
#         # Phân tích cằm
#         if summary.get('chin', {}).get('has_acne', False):
#             advice.append({
#                 'zone': 'Cằm',
#                 'severity': 'detected',
#                 'tips': [
#                     'Thường do mất cân bằng hormone',
#                     'Với nữ: theo dõi chu kỳ kinh nguyệt',
#                     'Giảm đường tinh luyện và sữa trong chế độ ăn',
#                     'Tăng cường rau xanh, omega-3',
#                     'Nếu kéo dài >3 tháng, nên gặp bác sĩ da liễu'
#                 ]
#             })
#
#         # Da khỏe mạnh
#         if not advice:
#             advice.append({
#                 'zone': 'Tổng quan',
#                 'severity': 'healthy',
#                 'tips': [
#                     'Da của bạn trong tình trạng tốt!',
#                     'Duy trì thói quen chăm sóc da hiện tại',
#                     'Vệ sinh sạch sẽ, chế độ ăn cân bằng',
#                     'Sử dụng kem chống nắng hàng ngày'
#                 ]
#             })
#
#         return advice
# # # services/advice_generator.py
# # from typing import Dict, List
# #
# #
# # class AdviceGenerator:
# #     """Tạo lời khuyên dựa trên kết quả phân loại mụn"""
# #
# #     # Database lời khuyên cho từng loại mụn
# #     ACNE_TYPE_ADVICE = {
# #         'blackheads': {
# #             'name': 'Mụn đầu đen',
# #             'description': 'Lỗ chân lông bị tắc bởi dầu và tế bào chết, oxy hóa thành màu đen',
# #             'tips': [
# #                 'Dùng BHA (Salicylic Acid 2%) để làm sạch sâu lỗ chân lông',
# #                 'Tẩy tế bào chết 2-3 lần/tuần',
# #                 'Dùng mặt nạ đất sét (clay mask) 1-2 lần/tuần',
# #                 'Rửa mặt với oil cleanser để hòa tan bã nhờn',
# #                 'TUYỆT ĐỐI không nặn tay → dễ nhiễm trùng'
# #             ],
# #             'products': [
# #                 'Paula\'s Choice 2% BHA Liquid',
# #                 'COSRX BHA Blackhead Power Liquid',
# #                 'Innisfree Volcanic Clay Mask'
# #             ]
# #         },
# #
# #         'whiteheads': {
# #             'name': 'Mụn đầu trắng',
# #             'description': 'Lỗ chân lông bị tắc kín, dầu và vi khuẩn bị mắc kẹt bên trong',
# #             'tips': [
# #                 'Dùng AHA (Glycolic Acid) để loại bỏ tế bào chết',
# #                 'Dùng retinol để tăng tốc tái tạo da',
# #                 'Đắp khăn ấm trước khi rửa mặt để mở lỗ chân lông',
# #                 'Dùng sản phẩm có Niacinamide để kiểm soát dầu',
# #                 'Nếu cần nặn: khử trùng kim, tay và vùng da trước'
# #             ],
# #             'products': [
# #                 'The Ordinary Glycolic Acid 7% Toning Solution',
# #                 'CeraVe Resurfacing Retinol Serum',
# #                 'Paula\'s Choice 10% Niacinamide Booster'
# #             ]
# #         },
# #
# #         'papules': {
# #             'name': 'Mụn sẩn',
# #             'description': 'Mụn đỏ, sưng nhưng chưa có mủ, do viêm nhiễm nhẹ',
# #             'tips': [
# #                 'Dùng Benzoyl Peroxide 2.5-5% để diệt khuẩn',
# #                 'Chườm đá để giảm sưng và đỏ',
# #                 'KHÔNG NẶN vì chưa có đầu mụn → dễ tổn thương da',
# #                 'Dùng kem chống viêm có Centella Asiatica',
# #                 'Tránh makeup vùng mụn sẩn'
# #             ],
# #             'products': [
# #                 'La Roche-Posay Effaclar Duo+ (5.5% BP)',
# #                 'COSRX Centella Blemish Cream',
# #                 'Mario Badescu Drying Lotion'
# #             ]
# #         },
# #
# #         'pustules': {
# #             'name': 'Mụn mủ',
# #             'description': 'Mụn có đầu mủ trắng/vàng, viêm nhiễm trung bình',
# #             'tips': [
# #                 'Dùng Benzoyl Peroxide hoặc Tea Tree Oil',
# #                 'Đắp miếng hút mụn hydrocolloid (sticker) qua đêm',
# #                 'Nếu mụn đã chín: khử trùng và nặn nhẹ, sau đó bôi kháng sinh',
# #                 'Uống nhiều nước, tránh thức khuya',
# #                 'Nếu mụn lan rộng: gặp bác sĩ để được kê kháng sinh uống'
# #             ],
# #             'products': [
# #                 'COSRX Acne Pimple Master Patch',
# #                 'Neutrogena On-The-Spot Acne Treatment',
# #                 'Tea Tree Oil (The Body Shop, Thursday Plantation)'
# #             ]
# #         },
# #
# #         'nodules': {
# #             'name': 'Mụn cục',
# #             'description': 'Mụn cứng, đau, nằm sâu dưới da, khó điều trị',
# #             'tips': [
# #                 '⚠️ NGHIÊM TRỌNG - Nên gặp bác sĩ da liễu ngay',
# #                 'Chườm ấm để tăng lưu thông máu',
# #                 'Uống thuốc kháng sinh theo chỉ định bác sĩ',
# #                 'TUYỆT ĐỐI không nặn → để lại scar sâu',
# #                 'Có thể cần tiêm steroid trực tiếp vào mụn'
# #             ],
# #             'products': [
# #                 'Kháng sinh đường uống (Doxycycline, Minocycline)',
# #                 'Tretinoin/Adapalene theo đơn bác sĩ',
# #                 'Liệu pháp ánh sáng (LED therapy)'
# #             ]
# #         },
# #
# #         'cysts': {
# #             'name': 'Mụn bọc',
# #             'description': 'Mụn lớn, đỏ tấy, đau nhức, chứa mủ sâu bên trong',
# #             'tips': [
# #                 '🚨 RẤT NGHIÊM TRỌNG - GẶP BÁC SĨ NGAY LẬP TỨC',
# #                 'Có thể cần Isotretinoin (Accutane) - thuốc mạnh nhất',
# #                 'Chườm đá để giảm đau',
# #                 'KHÔNG BAO GIỜ tự nặn → nguy cơ nhiễm trùng máu',
# #                 'Có thể cần phẫu thuật nhỏ để dẫn lưu'
# #             ],
# #             'products': [
# #                 'Isotretinoin (chỉ theo đơn bác sĩ)',
# #                 'Kháng sinh mạnh đường uống',
# #                 'Corticosteroid tiêm'
# #             ]
# #         }
# #     }
# #
# #     # Lời khuyên theo vùng da
# #     ZONE_SPECIFIC_ADVICE = {
# #         'forehead': {
# #             'cause': 'Stress, vấn đề tiêu hóa, mất cân bằng hormone',
# #             'tips': [
# #                 'Tránh để tóc che trán quá lâu',
# #                 'Giảm thực phẩm nhiều đường và tinh bột',
# #                 'Ngủ đủ 7-8 tiếng/ngày',
# #                 'Uống đủ 2 lít nước/ngày'
# #             ]
# #         },
# #         'nose': {
# #             'cause': 'Vùng chữ T tiết dầu nhiều nhất',
# #             'tips': [
# #                 'Dùng giấy thấm dầu 2-3 lần/ngày',
# #                 'Rửa mặt với sản phẩm kiểm soát dầu',
# #                 'Không chạm tay vào mũi'
# #             ]
# #         },
# #         'cheek_left': {
# #             'cause': 'Điện thoại, gối, dị ứng makeup',
# #             'tips': [
# #                 'Lau màn hình điện thoại bằng cồn hàng ngày',
# #                 'Thay vỏ gối 2-3 lần/tuần',
# #                 'Kiểm tra sản phẩm makeup có gây dị ứng'
# #             ]
# #         },
# #         'cheek_right': {
# #             'cause': 'Điện thoại, gối, dị ứng makeup',
# #             'tips': [
# #                 'Lau màn hình điện thoại bằng cồn hàng ngày',
# #                 'Thay vỏ gối 2-3 lần/tuần',
# #                 'Hạn chế dùng tay chống má'
# #             ]
# #         },
# #         'chin': {
# #             'cause': 'Mất cân bằng hormone (đặc biệt ở nữ)',
# #             'tips': [
# #                 'Theo dõi chu kỳ kinh nguyệt (nếu là nữ)',
# #                 'Giảm đường tinh luyện và sữa',
# #                 'Tăng cường rau xanh, omega-3',
# #                 'Gặp bác sĩ nếu kéo dài >3 tháng'
# #             ]
# #         }
# #     }
# #
# #     @staticmethod
# #     def generate_advice(results: Dict) -> List[Dict]:
# #         """
# #         Tạo lời khuyên dựa trên kết quả phân loại mụn
# #
# #         Args:
# #             results: {
# #                 region_name: {
# #                     'acne_type': str,
# #                     'confidence': float,
# #                     'top_3': [...]
# #                 }
# #             }
# #
# #         Returns:
# #             List[Dict]: Danh sách lời khuyên chi tiết
# #         """
# #         advice = []
# #
# #         for region_name, region_data in results.items():
# #             acne_type = region_data.get('acne_type')
# #             confidence = region_data.get('confidence', 0.0)
# #
# #             # Lấy thông tin loại mụn
# #             acne_info = AdviceGenerator.ACNE_TYPE_ADVICE.get(acne_type, {})
# #
# #             # Lấy thông tin vùng da
# #             zone_info = AdviceGenerator.ZONE_SPECIFIC_ADVICE.get(region_name, {})
# #
# #             # Tạo lời khuyên
# #             advice_item = {
# #                 'region': region_name,
# #                 'acne_type': acne_type,
# #                 'acne_name': acne_info.get('name', 'Không xác định'),
# #                 'confidence': confidence,
# #                 'description': acne_info.get('description', ''),
# #                 'zone_cause': zone_info.get('cause', ''),
# #                 'treatment_tips': acne_info.get('tips', []),
# #                 'zone_tips': zone_info.get('tips', []),
# #                 'recommended_products': acne_info.get('products', []),
# #                 'severity': AdviceGenerator._get_severity(acne_type)
# #             }
# #
# #             advice.append(advice_item)
# #
# #         # Nếu không có mụn
# #         if not advice:
# #             advice.append({
# #                 'region': 'all',
# #                 'acne_type': 'none',
# #                 'acne_name': 'Da khỏe mạnh',
# #                 'confidence': 1.0,
# #                 'description': 'Da của bạn trong tình trạng tốt!',
# #                 'treatment_tips': [
# #                     'Duy trì thói quen chăm sóc da hiện tại',
# #                     'Vệ sinh da sạch sẽ, chế độ ăn cân bằng',
# #                     'Sử dụng kem chống nắng SPF 30+ hàng ngày',
# #                     'Uống đủ nước và ngủ đủ giấc'
# #                 ],
# #                 'severity': 'healthy'
# #             })
# #
# #         return advice
# #
# #     @staticmethod
# #     def _get_severity(acne_type: str) -> str:
# #         """Xác định mức độ nghiêm trọng"""
# #         severity_map = {
# #             'blackheads': 'mild',  # Nhẹ
# #             'whiteheads': 'mild',  # Nhẹ
# #             'papules': 'moderate',  # Trung bình
# #             'pustules': 'moderate',  # Trung bình
# #             'nodules': 'severe',  # Nặng
# #             'cysts': 'very_severe'  # Rất nặng
# #         }
# #         return severity_map.get(acne_type, 'unknown')
# #
# #     @staticmethod
# #     def get_overall_summary(advice: List[Dict]) -> Dict:
# #         """
# #         Tổng hợp mức độ nghiêm trọng và khuyến nghị chung
# #
# #         Args:
# #             advice: Output từ generate_advice()
# #
# #         Returns:
# #             dict: Tổng quan tình trạng da
# #         """
# #         if not advice or advice[0].get('acne_type') == 'none':
# #             return {
# #                 'overall_severity': 'healthy',
# #                 'recommendation': 'Da khỏe mạnh, duy trì chăm sóc hiện tại',
# #                 'need_doctor': False
# #             }
# #
# #         # Đếm mức độ nghiêm trọng
# #         severity_count = {
# #             'mild': 0,
# #             'moderate': 0,
# #             'severe': 0,
# #             'very_severe': 0
# #         }
# #
# #         for item in advice:
# #             severity = item.get('severity', 'unknown')
# #             if severity in severity_count:
# #                 severity_count[severity] += 1
# #
# #         # Xác định mức độ tổng thể
# #         if severity_count['very_severe'] > 0 or severity_count['severe'] > 0:
# #             overall = 'severe'
# #             recommendation = '🚨 Nên gặp bác sĩ da liễu để được tư vấn chuyên sâu'
# #             need_doctor = True
# #         elif severity_count['moderate'] > 2:
# #             overall = 'moderate'
# #             recommendation = '⚠️ Nên cải thiện thói quen chăm sóc da, theo dõi thêm 2-3 tuần'
# #             need_doctor = False
# #         else:
# #             overall = 'mild'
# #             recommendation = 'Tình trạng nhẹ, có thể tự chăm sóc tại nhà'
# #             need_doctor = False
# #
# #         return {
# #             'overall_severity': overall,
# #             'severity_breakdown': severity_count,
# #             'recommendation': recommendation,
# #             'need_doctor': need_doctor
# #         }