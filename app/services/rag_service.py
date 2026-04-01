"""
Hybrid RAG Service: Ollama + Gemini
- Xử lý lỗi Gemini quota properly
- Thêm method _record_gemini_usage()
- Fallback sang Ollama khi Gemini fail
"""

import google.generativeai as genai
from supabase import create_client
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
import json
import time
from collections import deque

# Import Ollama Client
from app.services.ollama_client import ollama_client

load_dotenv()


class HybridRAGService:
    def __init__(self):
        """Khởi tạo Hybrid RAG Service"""
        
        # Ollama config
        self.ollama_chat_model = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
        self.ollama_embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        
        # Gemini config 
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.gemini_gen_model = genai.GenerativeModel("models/gemini-2.0-flash-exp")
        
        # Supabase
        self.supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
        
        # Threshold
        self.similarity_threshold = 0.75
        self.gemini_rpm_limit = 5
        self.gemini_usage_window = deque()
        
        print(" Hybrid RAG Service initialized")
        print(f"   Ollama: {self.ollama_chat_model} + {self.ollama_embed_model}")
        print(f"   Gemini: gemini-2.0-flash-exp")
        print(f"   Threshold: {self.similarity_threshold}")
    
    # ==================== GEMINI RATE LIMITING ====================
    
    def _check_gemini_availability(self) -> bool:
        """Kiểm tra xem có được phép gọi Gemini không"""
        now = time.time()
        
        # Loại bỏ request cũ hơn 1 giờ
        while self.gemini_usage_window and now - self.gemini_usage_window[0] > 3600:
            self.gemini_usage_window.popleft()
        
        return len(self.gemini_usage_window) < self.gemini_rpm_limit
    
    def _record_gemini_usage(self):
        """ Ghi nhận usage khi gọi Gemini"""
        self.gemini_usage_window.append(time.time())
    
    # ==================== EMBEDDING ====================
    
    def generate_embedding(self, text: str, use_ollama: bool = True) -> List[float]:
        """Generate embedding"""
        try:
            if use_ollama:
                result = ollama_client.embeddings(
                    model=self.ollama_embed_model,
                    prompt=text
                )
                return result['embedding']
            else:
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document"
                )
                return result['embedding']
        except Exception as e:
            print(f" Embedding error: {e}")
            raise
    
    # ==================== SEARCH ====================
    
    def search_similar_documents(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.75
    ) -> List[Dict]:
        """Tìm kiếm documents tương tự"""
        try:
            query_embedding = self.generate_embedding(query, use_ollama=True)
            
            result = self.supabase.rpc(
                "match_documents",
                {
                    "query_embedding": query_embedding,
                    "match_count": top_k
                }
            ).execute()
            
            filtered_docs = []
            for doc in result.data:
                if doc.get('similarity', 0) < similarity_threshold:
                    continue
                
                metadata = doc.get('metadata', {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {"output": ""}
                
                doc['metadata'] = metadata
                filtered_docs.append(doc)
            
            return filtered_docs
        
        except Exception as e:
            print(f" Search error: {e}")
            return []
    
    # ==================== CLASSIFICATION ====================
    
    def classify_question(self, question: str) -> str:
        """Phân loại câu hỏi"""
        barbergo_keywords = [
            'đặt lịch', 'hủy lịch', 'app', 'ứng dụng', 'barbergo',
            'thanh toán', 'đối tác', 'tài khoản', 'mật khẩu', 'đăng ký',
            'đăng nhập', 'quên mật khẩu', 'hủy tài khoản', 'cài đặt',
            'thông báo', 'lịch hẹn', 'lịch đặt', 'lịch', 'dịch vụ', 'booking'
        ]
        
        beauty_keywords = [
            'tóc', 'cắt', 'nhuộm', 'uốn', 'duỗi', 'gội', 'massage',
            'spa', 'nail', 'móng', 'làm đẹp', 'chăm sóc da', 'mặt',
            'wax', 'triệt lông', 'facial', 'mặt nạ', 'thợ', 'salon',
            'barber', 'tóc nam', 'tóc nữ', 'kiểu tóc', 'phong cách'
        ]
        
        greeting_keywords = [
            'chào', 'hello', 'hi', 'xin chào', 'hey', 'hế lô',
            'khỏe không', 'bạn là ai', 'bạn tên gì', 'cảm ơn', 'thanks', 'alo', 'ê', 'bye'
        ]
        
        question_lower = question.lower()
        
        if any(kw in question_lower for kw in greeting_keywords):
            return 'greeting'
        if any(kw in question_lower for kw in barbergo_keywords):
            return 'barbergo_specific'
        if any(kw in question_lower for kw in beauty_keywords):
            return 'beauty_related'
        
        return 'off_topic'
    
    # ==================== ANSWER GENERATION ====================
    
    def generate_answer(
        self,
        question: str,
        contexts: List[Dict]
    ) -> str:
        
        
        if not contexts or len(contexts) == 0:
            # Không có context → Dùng Ollama fallback
            return self._generate_fallback_answer(question, use_ollama=True)
        
        top_similarity = contexts[0].get('similarity', 0)
        
        # Build context
        context_text = "\n\n".join([
            f"Thông tin {i + 1}:\n{doc['metadata']['output']}"
            for i, doc in enumerate(contexts[:2])
        ])
        
        prompt = f"""Bạn là trợ lý ảo của BarberGo - app đặt lịch cắt tóc.

QUAN TRỌNG: TRẢ LỜI HOÀN TOÀN BẰNG TIẾNG VIỆT.

THÔNG TIN TỪ KNOWLEDGE BASE:
{context_text}

CÂU HỎI:
{question}

HƯỚNG DẪN:
1. Trả lời BẰNG TIẾNG VIỆT
2. Dựa vào knowledge base
3. Ngắn gọn 2-4 câu
4. Thân thiện, chuyên nghiệp
5. Trả lời trực tiếp

CÂU TRẢ LỜI:"""
        
        use_gemini = False
        
        # Strategy: Similarity < threshold → muốn dùng Gemini
        if top_similarity < self.similarity_threshold:
            if self._check_gemini_availability():
                use_gemini = True
            else:
                print(f"⚠️ Gemini Rate Limit. Fallback to Ollama.")
        
        try:
            if not use_gemini:
                # OLLAMA
                print(f"🟢 Using Ollama (Sim: {top_similarity:.2f})")
                
                response = ollama_client.chat(
                    model=self.ollama_chat_model,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={
                        'temperature': 0.7,
                        'num_predict': 500
                    }
                )
                return response['message']['content']
            
            else:
                # GEMINI
                print(f" Using Gemini (Sim: {top_similarity:.2f})")
                
                self._record_gemini_usage()
                time.sleep(0.5)
                
                response = self.gemini_gen_model.generate_content(prompt)
                return response.text
        
        except Exception as e:
            print(f" Generate answer error: {e}")
            
            # 🔧 FIX: Nếu lỗi → Thử Ollama
            try:
                print(" Retrying with Ollama...")
                response = ollama_client.chat(
                    model=self.ollama_chat_model,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'temperature': 0.7, 'num_predict': 500}
                )
                return response['message']['content']
            except:
                # Cuối cùng: Trả về output từ context
                return contexts[0]['metadata']['output']
    
    def _generate_fallback_answer(self, question: str, use_ollama: bool = True) -> str:
     
        question_type = self.classify_question(question)
        
        prompts = {
            'greeting': f"""Bạn là trợ lý ảo của BarberGo. TRẢ LỜI BẰNG TIẾNG VIỆT.
Khách hàng: {question}
Hãy chào hỏi thân thiện, giới thiệu ngắn gọn bạn có thể giúp gì. 2-3 câu.""",
            
            'beauty_related': f"""Bạn là chuyên gia làm đẹp của BarberGo. TRẢ LỜI BẰNG TIẾNG VIỆT.
Câu hỏi: {question}
Trả lời ngắn gọn, gợi ý đặt lịch trên BarberGo. 2-3 câu.""",
            
            'barbergo_specific': f"""Bạn là trợ lý BarberGo. TRẢ LỜI BẰNG TIẾNG VIỆT.
Câu hỏi: {question}
Xin lỗi lịch sự, gợi ý liên hệ hỗ trợ. 2 câu.""",
            
            'off_topic': f"""Bạn là trợ lý BarberGo. TRẢ LỜI BẰNG TIẾNG VIỆT.
Câu hỏi: {question}
Lịch sự từ chối, nhắc chỉ hỗ trợ về làm đẹp. 2 câu."""
        }
        
        prompt = prompts.get(question_type, prompts['off_topic'])
        
        try:
            if use_ollama:
                # OLLAMA (ưu tiên vì ổn định)
                response = ollama_client.chat(
                    model=self.ollama_chat_model,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'temperature': 0.7, 'num_predict': 300}
                )
                return response['message']['content']
            else:
                # GEMINI (nếu có quota)
                if self._check_gemini_availability():
                    self._record_gemini_usage()
                    time.sleep(0.5)
                    response = self.gemini_gen_model.generate_content(prompt)
                    return response.text
                else:
                    # Hết quota → Fallback Ollama
                    return self._generate_fallback_answer(question, use_ollama=True)
        
        except Exception as e:
            print(f" Fallback error: {e}")
            
            fallback_responses = {
                'greeting': "Chào bạn! Mình là trợ lý ảo của BarberGo. Mình có thể giúp bạn đặt lịch cắt tóc, tìm salon, hoặc giải đáp thắc mắc về dịch vụ. Bạn cần hỗ trợ gì không?",
                'beauty_related': "Mình nghĩ bạn nên tham khảo ý kiến stylist chuyên nghiệp. Bạn có thể đặt lịch tư vấn miễn phí trên BarberGo để được hỗ trợ tốt nhất nhé!",
                'barbergo_specific': "Mình không có thông tin cụ thể về câu hỏi này. Bạn vui lòng liên hệ bộ phận hỗ trợ qua chat trong app để được tư vấn chi tiết hơn nhé!",
                'off_topic': "Tôi là trợ lý chuyên về đặt lịch cắt tóc và dịch vụ làm đẹp. Bạn có câu hỏi nào về BarberGo không? 😊"
            }
            return fallback_responses.get(question_type, fallback_responses['off_topic'])
    
    # ==================== MAIN QUERY ====================
    
    def query(
        self,
        question: str,
        user_id: str,
        session_id: Optional[str] = None,
        top_k: int = 3,
        return_sources: bool = False
    ) -> Dict:
       
        
        # 1. Tạo session
        if not session_id:
            session_id = self.create_chat_session(
                user_id=user_id,
                title=question[:50]
            )
        
        # 2. Lưu user message
        self.save_chat_message(
            session_id=session_id,
            role="user",
            content=question
        )
        
        try:
            # 3. RAG retrieve
            relevant_docs = self.search_similar_documents(question, top_k)
            
            if relevant_docs:
                print(f"Top document similarity: {relevant_docs[0]['similarity']:.3f}")
                print(f" Document content preview: {relevant_docs[0]['content'][:100]}")
            else:
                print(" No relevant documents found")
            
            # 4. Generate answer
            answer = self.generate_answer(question, relevant_docs)
            
            # 5. Calculate confidence
            if relevant_docs:
                similarity = relevant_docs[0]["similarity"]
                confidence = (
                    "high" if similarity > 0.7 else
                    "medium" if similarity > 0.45 else
                    "low"
                )
            else:
                confidence = "low"
            
            # 6. Lưu assistant message
            self.save_chat_message(
                session_id=session_id,
                role="assistant",
                content=answer,
                confidence=confidence
            )
            
            return {
                "session_id": session_id,
                "answer": answer,
                "confidence": confidence,
                "sources": relevant_docs if return_sources else None
            }
        
        except Exception as e:
            print(f" Query error: {e}")
            
            # Lưu error message
            error_message = "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau."
            self.save_chat_message(
                session_id=session_id,
                role="assistant",
                content=error_message,
                confidence="low"
            )
            
            return {
                "session_id": session_id,
                "answer": error_message,
                "confidence": "low",
                "sources": None
            }
    
    # ==================== SESSION MANAGEMENT ====================
    
    def create_chat_session(self, user_id: str, title: str) -> str:
        """Tạo session"""
        result = self.supabase.table("chat_sessions").insert({
            "user_id": user_id,
            "title": title
        }).execute()
        return result.data[0]["id"]
    
    def save_chat_message(
        self,
        session_id: str,
        role: str,
        content: str,
        confidence: Optional[str] = None
    ):
        """Lưu message"""
        self.supabase.table("chat_messages").insert({
            "session_id": session_id,
            "role": role,
            "content": content,
            "confidence": confidence
        }).execute()
    
    def get_chat_history(self, session_id: str) -> List[Dict]:
        """Lấy lịch sử chat"""
        result = self.supabase.table("chat_messages") \
            .select("id, role, content, confidence, created_at") \
            .eq("session_id", session_id) \
            .order("created_at", desc=False) \
            .execute()
        return result.data
    
    def get_user_sessions(self, user_id: str) -> List[Dict]:
        """Lấy sessions của user"""
        result = self.supabase.table("chat_sessions") \
            .select("id, user_id, title, created_at") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .execute()
        return result.data
    
    def delete_session(self, session_id: str) -> bool:
        """Xóa session"""
        try:
            self.supabase.table("chat_messages") \
                .delete() \
                .eq("session_id", session_id) \
                .execute()
            
            result = self.supabase.table("chat_sessions") \
                .delete() \
                .eq("id", session_id) \
                .execute()
            
            return bool(result.data)
        except Exception as e:
            print(f" Delete session error: {e}")
            return False
    
    def update_session_title(self, session_id: str, new_title: str) -> bool:
        """Đổi tên session"""
        try:
            result = self.supabase.table("chat_sessions") \
                .update({"title": new_title}) \
                .eq("id", session_id) \
                .execute()
            return bool(result.data)
        except Exception as e:
            print(f" Update session error: {e}")
            return False
    
    # ==================== DOCUMENT MANAGEMENT ====================
    
    def create_document(
        self,
        content: str,
        output: str,
        extra_metadata: Optional[dict] = None
    ) -> bool:
        """Tạo document"""
        try:
            metadata = {"input": content, "output": output}
            if extra_metadata:
                metadata.update(extra_metadata)
            
            self.supabase.table("documents").insert({
                "content": content,
                "embedding": self.generate_embedding(content, use_ollama=True),
                "metadata": json.dumps(metadata, ensure_ascii=False)
            }).execute()
            
            return True
        except Exception as e:
            print(f" Create document error: {e}")
            return False
    
    def update_document(
        self,
        document_id: int,
        new_content: Optional[str] = None,
        new_output: Optional[str] = None,
        new_metadata: Optional[dict] = None
    ) -> bool:
        """Cập nhật document"""
        try:
            update_data = {}
            
            if new_content:
                update_data["content"] = new_content
                update_data["embedding"] = self.generate_embedding(new_content, use_ollama=True)
            
            if new_output or new_metadata:
                old_doc = self.supabase.table("documents") \
                    .select("metadata") \
                    .eq("id", document_id) \
                    .single() \
                    .execute()
                
                metadata = json.loads(old_doc.data["metadata"]) \
                    if isinstance(old_doc.data["metadata"], str) \
                    else old_doc.data["metadata"]
                
                if new_output:
                    metadata["output"] = new_output
                if new_metadata:
                    metadata.update(new_metadata)
                
                update_data["metadata"] = json.dumps(metadata, ensure_ascii=False)
            
            result = self.supabase.table("documents") \
                .update(update_data) \
                .eq("id", document_id) \
                .execute()
            
            return bool(result.data)
        except Exception as e:
            print(f" Update document error: {e}")
            return False
    
    def delete_document(self, document_id: int) -> bool:
        """Xóa document"""
        try:
            result = self.supabase.table("documents") \
                .delete() \
                .eq("id", document_id) \
                .execute()
            return bool(result.data)
        except Exception as e:
            print(f" Delete document error: {e}")
            return False
    
    def get_all_documents(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Lấy tất cả documents"""
        try:
            result = self.supabase.table("documents") \
                .select("id, content, metadata") \
                .range(offset, offset + limit - 1) \
                .order("id") \
                .execute()
            
            documents = []
            for doc in result.data:
                parsed_doc = {
                    "id": doc["id"],
                    "content": doc["content"],
                    "metadata": json.loads(doc["metadata"]) \
                        if isinstance(doc.get("metadata"), str) \
                        else doc.get("metadata", {})
                }
                documents.append(parsed_doc)
            
            return documents
        except Exception as e:
            print(f" Get documents error: {e}")
            return []
    
    def get_document_by_id(self, document_id: int) -> Optional[Dict]:
        """Lấy document theo ID"""
        try:
            result = self.supabase.table("documents") \
                .select("id, content, metadata") \
                .eq("id", document_id) \
                .single() \
                .execute()
            
            doc = result.data
            if isinstance(doc.get("metadata"), str):
                doc["metadata"] = json.loads(doc["metadata"])
            
            return doc
        except Exception as e:
            print(f" Get document error: {e}")
            return None
    
    def search_documents_by_keyword(self, keyword: str) -> List[Dict]:
        """Search documents"""
        try:
            result = self.supabase.table("documents") \
                .select("id, content, metadata") \
                .ilike("content", f"%{keyword}%") \
                .execute()
            
            documents = []
            for doc in result.data:
                if isinstance(doc.get("metadata"), str):
                    doc["metadata"] = json.loads(doc["metadata"])
                documents.append(doc)
            
            return documents
        except Exception as e:
            print(f" Search documents error: {e}")
            return []


# Singleton instance
rag_service = HybridRAGService()