# services/advice_generator.py
import json
import os
from typing import Dict, List, Optional
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

class AIProvider(Enum):
    """Các nhà cung cấp API AI"""
    GROQ = "groq"
    GEMINI = "gemini"
    CLAUDE = "claude"
    OPENAI = "openai"


class AdviceGenerator:
    """Tạo lời khuyên chăm sóc da dựa trên y khoa"""

    # Kiến thức y học chuẩn về mụn
    MEDICAL_KNOWLEDGE = {
        'forehead': {
            'causes': [
                'Rối loạn nội tiết tố (hormone)',
                'Stress kéo dài',
                'Vấn đề tiêu hóa',
                'Thói quen vệ sinh kém (tóc dính dầu che trán)',
                'Sử dụng sản phẩm tạo kiểu tóc bít lỗ chân lông'
            ],
            'ingredients': ['Salicylic Acid 0.5-2%', 'Benzoyl Peroxide 2.5-5%', 'Niacinamide 4-5%', 'Retinoids']
        },
        'nose': {
            'causes': [
                'Vùng chữ T tiết bã nhờn nhiều nhất',
                'Lỗ chân lông to, dễ bị bít tắc',
                'Mụn đầu đen (comedones)',
                'Da dầu, thiếu cân bằng độ ẩm'
            ],
            'ingredients': ['BHA/Salicylic Acid 2%', 'AHA/Glycolic Acid 5-10%', 'Niacinamide', 'Clay Mask']
        },
        'cheek': {
            'causes': [
                'Tiếp xúc với bề mặt không sạch (điện thoại, gối, tay)',
                'Dị ứng sản phẩm makeup/skincare',
                'Vấn đề hô hấp (hút thuốc, ô nhiễm)',
                'Rối loạn hormone (PCOS ở nữ)',
                'Vi khuẩn từ khẩu trang'
            ],
            'ingredients': ['Azelaic Acid 10-20%', 'Niacinamide 5%', 'Centella Asiatica', 'Tea Tree Oil']
        },
        'chin': {
            'causes': [
                'Mụn nội tiết (hormonal acne)',
                'Chu kỳ kinh nguyệt ở nữ giới',
                'Hội chứng buồng trứng đa nang (PCOS)',
                'Chế độ ăn nhiều đường, sữa',
                'Stress mạn tính'
            ],
            'ingredients': ['Adapalene 0.1%', 'Azelaic Acid', 'Niacinamide', 'Retinol', 'Spironolactone (theo toa)']
        }
    }

    # Sản phẩm có thật tại Việt Nam
    VN_PRODUCTS = {
        'cleanser': {
            'budget': ['Simple Refreshing Facial Wash (~150k)', 'Hada Labo Gokujyun Cleanser (~180k)',
                       'Senka Perfect Whip (~120k)'],
            'mid': ['La Roche-Posay Effaclar (~350k)', 'Cetaphil Gentle Skin Cleanser (~280k)',
                    'CeraVe Foaming Cleanser (~350k)'],
            'high': ['Vichy Normaderm (~450k)', 'Bioderma Sebium Gel (~420k)']
        },
        'toner': {
            'budget': ['Klairs Supple Preparation Toner (~280k)', 'Some By Mi AHA BHA PHA Toner (~250k)'],
            'mid': ['Paula\'s Choice 2% BHA Liquid (~650k)', 'The Ordinary Glycolic Acid 7% (~220k)'],
            'high': ['SK-II Facial Treatment Essence (~2.5tr)', 'Drunk Elephant T.L.C. Framboos (~1.8tr)']
        },
        'treatment': {
            'budget': ['Acnes Sealing Gel (~50k)', 'Bepanthol Acnes (~120k)', 'The Ordinary Niacinamide 10% (~180k)'],
            'mid': ['La Roche-Posay Effaclar Duo+ (~450k)', 'Paula\'s Choice Clear BHA (~750k)',
                    'Cocoon Tea Tree Oil (~150k)'],
            'high': ['Differin Gel 0.1% Adapalene (~800k)', 'Paula\'s Choice 1% Retinol (~950k)']
        },
        'moisturizer': {
            'budget': ['Cetaphil Daily Hydrating Lotion (~250k)', 'Simple Hydrating Light Moisturizer (~160k)'],
            'mid': ['CeraVe PM Facial Moisturizing Lotion (~350k)', 'La Roche-Posay Effaclar Mat (~420k)'],
            'high': ['Vichy Normaderm Phytosolution (~550k)',
                     'Clinique Dramatically Different Moisturizing Gel (~850k)']
        },
        'sunscreen': {
            'budget': ['Sunplay Skin Aqua (~150k)', 'Bioré UV Aqua Rich (~180k)', 'Anessa Perfect UV (~250k)'],
            'mid': ['La Roche-Posay Anthelios (~450k)', 'Eucerin Oil Control (~380k)',
                    'Skin1004 Madagascar Centella (~280k)'],
            'high': ['EltaMD UV Clear (~950k)', 'Shiseido Anessa Perfect UV (~600k)']
        }
    }

    def __init__(self):
        """
        Args:
            api_key: Gemini API key. Nếu None, lấy từ GEMINI_API_KEY
        """
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Cần có GEMINI_API_KEY")

        import google.generativeai as genai
        genai.configure(api_key=self.api_key)

        # Cấu hình safety để không bị chặn nội dung y tế
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        self.model = genai.GenerativeModel(
            'gemini-1.5-flash',
            safety_settings=safety_settings
        )

    def generate_advice(
            self,
            summary: Dict,
            user_info: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Tạo lời khuyên dựa trên cơ sở y khoa

        Args:
            summary: Kết quả phân tích mụn
            user_info: Thông tin người dùng
        """
        affected_regions = self._extract_affected_regions(summary)

        if not affected_regions:
            return self._generate_healthy_skin_advice()

        prompt = self._build_medical_prompt(affected_regions, user_info)

        try:
            response = self.model.generate_content(prompt)
            advice_data = self._parse_response(response.text, affected_regions)
            return advice_data

        except Exception as e:
            print(f"Gemini API Error: {e}")
            # Fallback về kiến thức y khoa có sẵn
            return self._generate_medical_fallback(affected_regions)

    def _extract_affected_regions(self, summary: Dict) -> List[Dict]:
        """Trích xuất vùng bị mụn"""
        affected = []
        translations = {
            'forehead': 'Trán',
            'nose': 'Mũi',
            'cheek_left': 'Má trái',
            'cheek_right': 'Má phải',
            'chin': 'Cằm'
        }

        for region, data in summary.items():
            if data.get('has_acne', False):
                affected.append({
                    'zone': translations.get(region, region),
                    'zone_en': region if region in ['forehead', 'nose', 'chin'] else 'cheek',
                    'severity': data.get('severity', 'mild'),
                    'confidence': data.get('confidence', 0.0)
                })

        return affected

    def _build_medical_prompt(
            self,
            affected_regions: List[Dict],
            user_info: Optional[Dict]
    ) -> str:
        """Tạo prompt y khoa chi tiết với kiến thức có sẵn"""

        # Thu thập kiến thức y học cho các vùng bị ảnh hưởng
        medical_context = []
        for region in affected_regions:
            zone_en = region['zone_en']
            if zone_en in self.MEDICAL_KNOWLEDGE:
                knowledge = self.MEDICAL_KNOWLEDGE[zone_en]
                medical_context.append(f"""
Vùng {region['zone']} - Mức độ: {region['severity']}
Nguyên nhân y khoa đã được chứng minh:
{chr(10).join(f'  • {cause}' for cause in knowledge['causes'])}
Thành phần điều trị hiệu quả:
{chr(10).join(f'  • {ing}' for ing in knowledge['ingredients'])}
""")

        user_context = ""
        if user_info:
            user_context = f"""
Thông tin bệnh nhân:
- Tuổi: {user_info.get('age', 'Không rõ')}
- Giới tính: {user_info.get('gender', 'Không rõ')}
- Loại da: {user_info.get('skin_type', 'Không rõ')}
- Tiền sử: {user_info.get('history', 'Không có')}
"""

        # Danh sách sản phẩm có thật
        product_list = f"""
SẢN PHẨM CÓ THẬT TẠI VIỆT NAM (CHỈ GỢI Ý TỪ DANH SÁCH NÀY):

1. Sữa rửa mặt:
   Budget: {', '.join(self.VN_PRODUCTS['cleanser']['budget'])}
   Mid: {', '.join(self.VN_PRODUCTS['cleanser']['mid'])}
   High: {', '.join(self.VN_PRODUCTS['cleanser']['high'])}

2. Toner/Exfoliant:
   Budget: {', '.join(self.VN_PRODUCTS['toner']['budget'])}
   Mid: {', '.join(self.VN_PRODUCTS['toner']['mid'])}

3. Serum/Treatment:
   Budget: {', '.join(self.VN_PRODUCTS['treatment']['budget'])}
   Mid: {', '.join(self.VN_PRODUCTS['treatment']['mid'])}
   High: {', '.join(self.VN_PRODUCTS['treatment']['high'])}

4. Kem dưỡng ẩm:
   Budget: {', '.join(self.VN_PRODUCTS['moisturizer']['budget'])}
   Mid: {', '.join(self.VN_PRODUCTS['moisturizer']['mid'])}

5. Kem chống nắng:
   Budget: {', '.join(self.VN_PRODUCTS['sunscreen']['budget'])}
   Mid: {', '.join(self.VN_PRODUCTS['sunscreen']['mid'])}
"""

        prompt = f"""Bạn là bác sĩ da liễu với 15 năm kinh nghiệm. Phân tích dựa HOÀN TOÀN trên y khoa.

{user_context}

CƠ SỞ Y HỌC:
{chr(10).join(medical_context)}

{product_list}

YÊU CẦU BẮT BUỘC:
1. Chỉ phân tích dựa trên nguyên nhân y khoa đã nêu ở trên
2. CHỈ gợi ý sản phẩm có trong danh sách trên (bao gồm cả giá)
3. KHÔNG tự bịa tên sản phẩm
4. KHÔNG nói chung chung kiểu "dùng sản phẩm có BHA" - phải nêu TÊN CỤ THỂ
5. Nếu mụn severe, BẮT BUỘC khuyên gặp bác sĩ da liễu

Trả về JSON (KHÔNG dùng ```json):
{{
  "advice": [
    {{
      "zone": "Tên vùng",
      "severity": "mild/moderate/severe",
      "medical_causes": ["Nguyên nhân 1", "Nguyên nhân 2"],
      "lifestyle_changes": [
        "Thay đổi cụ thể 1 (ví dụ: Lau điện thoại bằng cồn 70% mỗi ngày)",
        "Thay đổi cụ thể 2"
      ],
      "skincare_routine": {{
        "morning": [
          "Bước 1: [Tên sản phẩm cụ thể + giá]",
          "Bước 2: [Tên sản phẩm cụ thể + giá]"
        ],
        "evening": [
          "Bước 1: [Tên sản phẩm cụ thể + giá]",
          "Bước 2: [Tên sản phẩm cụ thể + giá]"
        ]
      }},
      "active_ingredients": ["Salicylic Acid 2%", "Niacinamide 5%"],
      "expected_timeline": "Kết quả sau X tuần",
      "red_flags": ["Dấu hiệu cần gặp bác sĩ ngay"]
    }}
  ]
}}

VÍ DỤ ĐÚNG:
"morning": ["Rửa mặt: La Roche-Posay Effaclar (~350k)", "Chống nắng: Bioré UV Aqua Rich (~180k)"]

VÍ DỤ SAI (KHÔNG LÀM NHƯ THẾ NÀY):
"morning": ["Dùng sữa rửa mặt có BHA", "Thoa kem chống nắng"]

BẮT ĐẦU PHÂN TÍCH:"""

        return prompt

    def _parse_response(
            self,
            response_text: str,
            affected_regions: List[Dict]
    ) -> List[Dict]:
        """Parse response từ Gemini"""
        try:
            clean_text = response_text.strip()

            # Loại bỏ markdown
            if "```json" in clean_text:
                clean_text = clean_text.split("```json")[1].split("```")[0]
            elif "```" in clean_text:
                clean_text = clean_text.split("```")[1].split("```")[0]

            clean_text = clean_text.strip()

            data = json.loads(clean_text)
            advice_list = data.get('advice', [])

            # Thêm confidence
            for advice in advice_list:
                zone = advice.get('zone', '')
                for region in affected_regions:
                    if region['zone'] == zone:
                        advice['confidence'] = region['confidence']
                        break

            return advice_list

        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}")
            print(f"Response: {response_text[:500]}")
            return self._generate_medical_fallback(affected_regions)

    def _generate_medical_fallback(
            self,
            affected_regions: List[Dict]
    ) -> List[Dict]:
        """Fallback dựa trên kiến thức y khoa có sẵn"""
        fallback = []

        for region in affected_regions:
            zone_en = region['zone_en']
            severity = region['severity']

            knowledge = self.MEDICAL_KNOWLEDGE.get(zone_en, {})

            # Chọn sản phẩm phù hợp
            if severity in ['mild', 'moderate']:
                products = {
                    'cleanser': self.VN_PRODUCTS['cleanser']['budget'][0],
                    'treatment': self.VN_PRODUCTS['treatment']['budget'][0],
                    'moisturizer': self.VN_PRODUCTS['moisturizer']['budget'][0],
                    'sunscreen': self.VN_PRODUCTS['sunscreen']['budget'][0]
                }
            else:
                products = {
                    'cleanser': self.VN_PRODUCTS['cleanser']['mid'][0],
                    'treatment': self.VN_PRODUCTS['treatment']['high'][0],
                    'moisturizer': self.VN_PRODUCTS['moisturizer']['mid'][0],
                    'sunscreen': self.VN_PRODUCTS['sunscreen']['mid'][0]
                }

            advice_item = {
                'zone': region['zone'],
                'severity': severity,
                'confidence': region['confidence'],
                'medical_causes': knowledge.get('causes', ['Nguyên nhân chưa xác định']),
                'lifestyle_changes': [
                    'Rửa mặt 2 lần/ngày với nước ấm',
                    'Uống đủ 2 lít nước/ngày',
                    'Ngủ đủ 7-8 tiếng/đêm',
                    'Giảm đường, sữa trong chế độ ăn'
                ],
                'skincare_routine': {
                    'morning': [
                        f"Rửa mặt: {products['cleanser']}",
                        f"Kem dưỡng: {products['moisturizer']}",
                        f"Chống nắng: {products['sunscreen']}"
                    ],
                    'evening': [
                        f"Rửa mặt: {products['cleanser']}",
                        f"Điều trị: {products['treatment']}",
                        f"Kem dưỡng: {products['moisturizer']}"
                    ]
                },
                'active_ingredients': knowledge.get('ingredients', []),
                'expected_timeline': '4-6 tuần để thấy cải thiện rõ rệt' if severity == 'mild' else '6-12 tuần với điều trị đúng cách',
                'red_flags': [
                    'Mụn không giảm sau 6 tuần điều trị',
                    'Mụn sưng to, đau, có mủ',
                    'Để lại scar/thâm nặng',
                    'Kèm triệu chứng toàn thân (sốt, mệt mỏi)'
                ] if severity == 'severe' else []
            }

            fallback.append(advice_item)

        return fallback

    def _generate_healthy_skin_advice(self) -> List[Dict]:
        """Lời khuyên cho da khỏe mạnh"""
        return [{
            'zone': 'Tổng quan',
            'severity': 'healthy',
            'confidence': 1.0,
            'medical_causes': [],
            'lifestyle_changes': [
                'Duy trì thói quen hiện tại',
                'Vệ sinh da 2 lần/ngày',
                'Chống nắng SPF 30+ hàng ngày',
                'Uống đủ 2L nước/ngày'
            ],
            'skincare_routine': {
                'morning': [
                    f"Rửa mặt: {self.VN_PRODUCTS['cleanser']['budget'][0]}",
                    f"Kem dưỡng: {self.VN_PRODUCTS['moisturizer']['budget'][0]}",
                    f"Chống nắng: {self.VN_PRODUCTS['sunscreen']['budget'][0]}"
                ],
                'evening': [
                    f"Rửa mặt: {self.VN_PRODUCTS['cleanser']['budget'][0]}",
                    f"Kem dưỡng: {self.VN_PRODUCTS['moisturizer']['budget'][0]}"
                ]
            },
            'active_ingredients': [],
            'expected_timeline': 'Da đang khỏe mạnh',
            'red_flags': []
        }]

    def get_overall_summary(
            self,
            advice: List[Dict],
            summary_data: Dict
    ) -> Dict:
        """Tổng quan y khoa"""
        if not advice or advice[0].get('severity') == 'healthy':
            return {
                'severity': 'healthy',
                'recommendation': ' Da khỏe mạnh! Tiếp tục duy trì thói quen chăm sóc.',
                'need_doctor': False,
                'affected_regions': 0,
                'medical_note': 'Không cần can thiệp y tế'
            }

        overall_severity = summary_data.get('overall_severity', 'mild')
        acne_regions = summary_data.get('acne_regions', 0)

        medical_recs = {
            'severe': {
                'recommendation': ' MỨC NẶNG - CẦN GẶP BÁC SĨ DA LIỄU\n\nĐây là mức độ cần điều trị y khoa chuyên sâu. Bác sĩ có thể kê toa: Isotretinoin, kháng sinh uống, hoặc liệu pháp hormone.',
                'need_doctor': True,
                'medical_note': 'Cần toa thuốc từ bác sĩ da liễu'
            },
            'moderate': {
                'recommendation': ' MỨC TRUNG BÌNH\n\nThử điều trị tại nhà 4-6 tuần. Nếu không cải thiện → gặp bác sĩ để xem xét thuốc kê toa (Adapalene, Azelaic Acid).',
                'need_doctor': False,
                'medical_note': 'Theo dõi 4-6 tuần, nếu không đỡ thì gặp bác sĩ'
            },
            'mild': {
                'recommendation': ' MỨC NHẸ\n\nCó thể tự điều trị tại nhà với sản phẩm OTC (over-the-counter). Kết quả sau 4-6 tuần.',
                'need_doctor': False,
                'medical_note': 'Có thể tự điều trị'
            }
        }

        severity_info = medical_recs.get(overall_severity, medical_recs['mild'])

        return {
            'severity': overall_severity,
            'recommendation': severity_info['recommendation'],
            'need_doctor': severity_info['need_doctor'],
            'affected_regions': acne_regions,
            'medical_note': severity_info['medical_note']
        }


# # services/advice_generator.py
# from typing import Dict, List
#
#
# class AdviceGenerator:
#     """Tạo lời khuyên dựa trên kết quả phân loại mụn (Binary Classification)"""
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
#                     'confidence': float,
#                     'severity': str  # none/mild/moderate/severe
#                 }
#             }
#
#         Returns:
#             List[Dict]: Danh sách lời khuyên
#         """
#         advice = []
#
#         # Phân tích trán
#         forehead_data = summary.get('forehead', {})
#         if forehead_data.get('has_acne', False):
#             severity = forehead_data.get('severity', 'mild')
#             advice.append({
#                 'zone': 'Trán',
#                 'severity': severity,
#                 'confidence': forehead_data.get('confidence', 0.0),
#                 'tips': [
#                     'Có thể do stress, vấn đề tiêu hóa hoặc mất cân bằng hormone',
#                     'Rửa mặt 2 lần/ngày với sữa rửa mặt dịu nhẹ',
#                     'Tránh để tóc che trán quá lâu',
#                     'Uống đủ 2 lít nước/ngày, ngủ đủ 7-8 tiếng',
#                     'Giảm thực phẩm nhiều đường và tinh bột'
#                 ]
#             })
#
#         # Phân tích mũi
#         nose_data = summary.get('nose', {})
#         if nose_data.get('has_acne', False):
#             severity = nose_data.get('severity', 'mild')
#             advice.append({
#                 'zone': 'Mũi',
#                 'severity': severity,
#                 'confidence': nose_data.get('confidence', 0.0),
#                 'tips': [
#                     'Vùng chữ T dễ tiết dầu nhất',
#                     'Dùng sản phẩm kiểm soát dầu (BHA/Salicylic Acid)',
#                     'Dùng giấy thấm dầu 2-3 lần/ngày',
#                     'Có thể có mụn ẩn, mụn đầu đen',
#                     'Tuyệt đối không nặn mụn'
#                 ]
#             })
#
#         # Phân tích má
#         cheek_left_data = summary.get('cheek_left', {})
#         cheek_right_data = summary.get('cheek_right', {})
#
#         cheek_left = cheek_left_data.get('has_acne', False)
#         cheek_right = cheek_right_data.get('has_acne', False)
#
#         if cheek_left or cheek_right:
#             # Lấy severity cao nhất giữa 2 má
#             severity_left = cheek_left_data.get('severity', 'none')
#             severity_right = cheek_right_data.get('severity', 'none')
#
#             severity_order = {'none': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
#             max_severity = max(severity_left, severity_right,
#                                key=lambda x: severity_order.get(x, 0))
#
#             avg_confidence = (
#                                      cheek_left_data.get('confidence', 0.0) +
#                                      cheek_right_data.get('confidence', 0.0)
#                              ) / 2
#
#             advice.append({
#                 'zone': 'Má',
#                 'severity': max_severity,
#                 'confidence': avg_confidence,
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
#         chin_data = summary.get('chin', {})
#         if chin_data.get('has_acne', False):
#             severity = chin_data.get('severity', 'mild')
#             advice.append({
#                 'zone': 'Cằm',
#                 'severity': severity,
#                 'confidence': chin_data.get('confidence', 0.0),
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
#                 'confidence': 1.0,
#                 'tips': [
#                     'Da của bạn trong tình trạng tốt!',
#                     'Duy trì thói quen chăm sóc da hiện tại',
#                     'Vệ sinh sạch sẽ, chế độ ăn cân bằng',
#                     'Sử dụng kem chống nắng SPF 30+ hàng ngày'
#                 ]
#             })
#
#         return advice
#
#     @staticmethod
#     def get_overall_summary(advice: List[Dict], summary_data: Dict) -> Dict:
#         """
#         Tổng hợp đánh giá chung dựa trên advice và summary
#
#         Args:
#             advice: Output từ generate_advice()
#             summary_data: Output từ acne_service.get_summary()
#
#         Returns:
#             dict: {
#                 'severity': str,           # healthy/mild/moderate/severe
#                 'recommendation': str,     # Lời khuyên tổng quát
#                 'need_doctor': bool,       # Có cần gặp bác sĩ không
#                 'affected_regions': int    # Số vùng bị ảnh hưởng
#             }
#         """
#         # Nếu không có mụn
#         if not advice or advice[0].get('severity') == 'healthy':
#             return {
#                 'severity': 'healthy',
#                 'recommendation': 'Da của bạn khá sạch! Tiếp tục chăm sóc như hiện tại.',
#                 'need_doctor': False,
#                 'affected_regions': 0
#             }
#
#         # Lấy thông tin từ summary_data
#         overall_severity = summary_data.get('overall_severity', 'mild')
#         acne_regions = summary_data.get('acne_regions', 0)
#
#         # Tạo recommendation dựa trên severity
#         if overall_severity == 'severe':
#             recommendation = '🚨 Da có nhiều mụn nghiêm trọng. Nên gặp bác sĩ da liễu để được tư vấn điều trị chuyên sâu.'
#             need_doctor = True
#
#         elif overall_severity == 'moderate':
#             recommendation = '⚠️ Da có mụn ở mức trung bình. Nên sử dụng sản phẩm trị mụn phù hợp và theo dõi tình trạng trong 2-3 tuần.'
#             need_doctor = False
#
#         elif overall_severity == 'mild':
#             recommendation = 'Da có mụn nhẹ. Duy trì vệ sinh da mặt và chế độ ăn uống lành mạnh, tránh stress.'
#             need_doctor = False
#
#         else:  # healthy
#             recommendation = 'Da của bạn khá sạch! Tiếp tục chăm sóc như hiện tại.'
#             need_doctor = False
#
#         return {
#             'severity': overall_severity,
#             'recommendation': recommendation,
#             'need_doctor': need_doctor,
#             'affected_regions': acne_regions
#         }