import google.generativeai as genai
from supabase import create_client
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
import time

load_dotenv()

class RAGService:
    def __init__(self):
        """Khởi tạo RAG Service với Gemini và Supabase"""
        # Configure Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.embed_model = "models/text-embedding-004"
        self.gen_model = genai.GenerativeModel("models/gemini-2.5-flash")
        
        # Configure Supabase
        self.supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Tạo embedding cho query của user
        
        Args:
            text: Câu hỏi của user
            
        Returns:
            Vector embedding 768 chiều
        """
        result = genai.embed_content(
            model=self.embed_model,
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    
    def search_similar_documents(
        self, 
        query: str, 
        top_k: int = 3,
        similarity_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Tìm kiếm documents tương tự với câu hỏi
        
        Args:
            query: Câu hỏi của user
            top_k: Số lượng documents trả về
            similarity_threshold: Ngưỡng similarity tối thiểu
            
        Returns:
            List các documents liên quan nhất
        """
        try:
            # 1. Tạo embedding cho query
            query_embedding = self.generate_embedding(query)
            
            # 2. Gọi function match_documents trong Supabase
            result = self.supabase.rpc(
                "match_documents",
                {
                    "query_embedding": query_embedding,
                    "match_count": top_k
                }
            ).execute()
            
            # 3. Filter by similarity threshold
            filtered_docs = [
                doc for doc in result.data 
                if doc['similarity'] >= similarity_threshold
            ]
            
            return filtered_docs
            
        except Exception as e:
            print(f"❌ Lỗi khi search documents: {e}")
            return []
    
    def classify_question(self, question: str) -> str:
        """
        Phân loại câu hỏi để xử lý phù hợp
        
        Returns: 
            'barbergo_specific' - Câu hỏi về chức năng app
            'beauty_related' - Câu hỏi về làm đẹp, cắt tóc
            'greeting' - Chào hỏi, xã giao
            'off_topic' - Hoàn toàn ngoài lề
        """
        # Keywords về BarberGo app
        barbergo_keywords = [
            'đặt lịch', 'hủy lịch', 'app', 'ứng dụng', 'barbergo',
            'thanh toán', 'đối tác', 'tài khoản', 'mật khẩu', 'đăng ký',
            'đăng nhập', 'quên mật khẩu', 'hủy tài khoản', 'cài đặt',
            'thông báo', 'ưu đãi', 'khuyến mãi', 'giảm giá'
        ]
        
        # Keywords về làm đẹp
        beauty_keywords = [
            'tóc', 'cắt', 'nhuộm', 'uốn', 'duỗi', 'gội', 'massage',
            'spa', 'nail', 'móng', 'làm đẹp', 'chăm sóc da', 'mặt',
            'wax', 'triệt lông', 'facial', 'mặt nạ', 'thợ', 'salon',
            'barber', 'tóc nam', 'tóc nữ', 'kiểu tóc', 'phong cách'
        ]
        
        # Keywords chào hỏi
        greeting_keywords = [
            'chào', 'hello', 'hi', 'xin chào', 'hey', 'hế lô',
            'khỏe không', 'bạn là ai', 'bạn tên gì', 'cảm ơn', 'thanks'
        ]
        
        question_lower = question.lower()
        
        # Check greeting
        if any(kw in question_lower for kw in greeting_keywords):
            return 'greeting'
        
        # Check BarberGo specific
        if any(kw in question_lower for kw in barbergo_keywords):
            return 'barbergo_specific'
        
        # Check beauty related
        if any(kw in question_lower for kw in beauty_keywords):
            return 'beauty_related'
        
        return 'off_topic'
    
    def generate_answer(
        self, 
        question: str, 
        contexts: List[Dict]
    ) -> str:
        """
        Generate câu trả lời dựa trên contexts từ knowledge base
        
        Args:
            question: Câu hỏi của user
            contexts: Các documents liên quan
            
        Returns:
            Câu trả lời từ Gemini
        """
        # Nếu có context với similarity cao (>0.65), trả lời từ knowledge base
        if contexts and len(contexts) > 0 and contexts[0]['similarity'] > 0.65:
            # Build context từ retrieved documents
            context_text = "\n\n".join([
                f"Thông tin {i+1}:\n{doc['metadata']['output']}"
                for i, doc in enumerate(contexts[:2])  # Chỉ lấy 2 contexts tốt nhất
            ])
            
            # Tạo prompt cho Gemini
            prompt = f"""Bạn là trợ lý ảo thông minh của ứng dụng BarberGo - ứng dụng đặt lịch cắt tóc và các dịch vụ làm đẹp tại Việt Nam.

THÔNG TIN TỪ KNOWLEDGE BASE:
{context_text}

CÂU HỎI CỦA KHÁCH HÀNG:
{question}

HƯỚNG DẪN TRẢ LỜI:
1. Dựa vào thông tin từ knowledge base để trả lời
2. Trả lời ngắn gọn, rõ ràng (2-4 câu)
3. Giọng điệu thân thiện, chuyên nghiệp
4. Nếu có nhiều bước, liệt kê rõ ràng
5. Trả lời trực tiếp, không nói "Dựa vào thông tin..."

CÂU TRẢ LỜI:"""
            
            try:
                time.sleep(0.5)  # Rate limiting
                response = self.gen_model.generate_content(prompt)
                return response.text
            except Exception as e:
                print(f"❌ Lỗi khi generate answer: {e}")
                # Fallback: trả về answer từ database
                return contexts[0]['metadata']['output']
        
        # Không có context phù hợp -> xử lý theo loại câu hỏi
        return self._generate_fallback_answer(question)
    
    def _generate_fallback_answer(self, question: str) -> str:
        """
        Generate câu trả lời khi không tìm thấy context phù hợp
        Xử lý thông minh dựa trên loại câu hỏi
        """
        question_type = self.classify_question(question)
        
        # 1. Chào hỏi, xã giao
        if question_type == 'greeting':
            prompt = f"""Bạn là trợ lý ảo thân thiện của BarberGo - app đặt lịch cắt tóc.

Khách hàng nói: {question}

Hãy:
1. Trả lời thân thiện, tự nhiên
2. Giới thiệu ngắn gọn bạn có thể giúp gì (về đặt lịch, dịch vụ làm đẹp)
3. Ngắn gọn 2-3 câu

VÍ DỤ:
- "Chào bạn! Mình là trợ lý ảo của BarberGo. Mình có thể giúp bạn đặt lịch cắt tóc, tìm salon gần nhà, hoặc giải đáp thắc mắc về dịch vụ. Bạn cần hỗ trợ gì không?"

CÂU TRẢ LỜI:"""
        
        # 2. Câu hỏi về làm đẹp chung (không có trong DB)
        elif question_type == 'beauty_related':
            prompt = f"""Bạn là chuyên gia làm đẹp của ứng dụng BarberGo.

Câu hỏi về làm đẹp: {question}

Hãy:
1. Trả lời ngắn gọn, hữu ích dựa trên kiến thức chung về làm đẹp (2-3 câu)
2. Gợi ý đặt lịch trên BarberGo để được tư vấn trực tiếp
3. Giọng điệu thân thiện, chuyên nghiệp

VÍ DỤ:
- Câu hỏi: "Tóc dài bao lâu nên cắt?"
- Trả lời: "Thông thường nên cắt tóc 4-6 tuần một lần để giữ kiểu đẹp và loại bỏ ngọn tóc hư tổn. Bạn có thể đặt lịch với stylist trên BarberGo để được tư vấn cụ thể dựa trên kiểu tóc và tình trạng tóc của mình nhé!"

CÂU TRẢ LỜI:"""
        
        # 3. Câu hỏi về BarberGo nhưng không có trong DB
        elif question_type == 'barbergo_specific':
            prompt = f"""Bạn là trợ lý của BarberGo.

Khách hàng hỏi về tính năng app: {question}

Bạn không có thông tin cụ thể trong hệ thống. Hãy:
1. Xin lỗi lịch sự
2. Gợi ý liên hệ bộ phận hỗ trợ (chat trong app hoặc hotline)
3. Ngắn gọn 2 câu

CÂU TRẢ LỜI:"""
        
        # 4. Hoàn toàn ngoài lề
        else:  # off_topic
            prompt = f"""Bạn là trợ lý của BarberGo - app đặt lịch làm đẹp.

Khách hàng hỏi: {question}

Câu hỏi này KHÔNG liên quan đến làm đẹp hoặc BarberGo. Hãy:
1. Lịch sự từ chối (không trả lời câu hỏi off-topic)
2. Nhắc bạn chỉ hỗ trợ về đặt lịch và dịch vụ làm đẹp
3. Hỏi xem có thể giúp gì về chủ đề này
4. Ngắn gọn, thân thiện (2 câu)

VÍ DỤ:
- "Tôi là trợ lý chuyên về đặt lịch cắt tóc và dịch vụ làm đẹp nên không thể trả lời câu hỏi này được. Bạn có cần hỗ trợ gì về BarberGo không? 😊"

CÂU TRẢ LỜI:"""
        
        try:
            time.sleep(0.5)  # Rate limiting
            response = self.gen_model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"❌ Lỗi fallback: {e}")
            
            # Hard fallback dựa trên loại
            fallback_responses = {
                'greeting': "Chào bạn! Mình là trợ lý ảo của BarberGo. Mình có thể giúp bạn đặt lịch cắt tóc, tìm salon, hoặc giải đáp thắc mắc về dịch vụ. Bạn cần hỗ trợ gì không?",
                'beauty_related': "Mình nghĩ bạn nên tham khảo ý kiến stylist chuyên nghiệp. Bạn có thể đặt lịch tư vấn miễn phí trên BarberGo để được hỗ trợ tốt nhất nhé!",
                'barbergo_specific': "Mình không có thông tin cụ thể về câu hỏi này. Bạn vui lòng liên hệ bộ phận hỗ trợ qua chat trong app để được tư vấn chi tiết hơn nhé!",
                'off_topic': "Tôi là trợ lý chuyên về đặt lịch cắt tóc và dịch vụ làm đẹp. Bạn có câu hỏi nào về BarberGo không? 😊"
            }
            return fallback_responses.get(question_type, fallback_responses['off_topic'])
    
    def query(
        self, 
        question: str, 
        top_k: int = 3,
        return_sources: bool = False
    ) -> Dict:
        """
        Main function để query RAG system
        
        Args:
            question: Câu hỏi của user
            top_k: Số lượng contexts để retrieve
            return_sources: Có trả về sources không
            
        Returns:
            Dict chứa answer và sources (nếu có)
        """
        # 1. Retrieve relevant documents
        relevant_docs = self.search_similar_documents(question, top_k)
        
        # 2. Generate answer
        answer = self.generate_answer(question, relevant_docs)
        
        # 3. Xác định confidence
        if relevant_docs and len(relevant_docs) > 0:
            max_similarity = relevant_docs[0]['similarity']
            if max_similarity > 0.75:
                confidence = "high"
            elif max_similarity > 0.5:
                confidence = "medium"
            else:
                confidence = "low"
        else:
            # Không có docs -> câu hỏi ngoài lề
            question_type = self.classify_question(question)
            confidence = "medium" if question_type in ['greeting', 'beauty_related'] else "low"
        
        # 4. Prepare response
        response = {
            "answer": answer,
            "confidence": confidence
        }
        
        if return_sources:
            response["sources"] = [
                {
                    "question": doc['metadata']['input'],
                    "answer": doc['metadata']['output'],
                    "similarity": round(doc['similarity'], 3)
                }
                for doc in relevant_docs
            ] if relevant_docs else []
        
        return response


# Singleton instance
rag_service = RAGService()