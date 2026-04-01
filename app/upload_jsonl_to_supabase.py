import json
import google.generativeai as genai
from supabase import create_client
import os
from dotenv import load_dotenv
from tqdm import tqdm
import time

# Load environment variables
load_dotenv()

# Configure services
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

class KnowledgeBaseUploader:
    def __init__(self, jsonl_file_path):
        self.jsonl_file_path = jsonl_file_path
        self.embed_model = "models/text-embedding-004"
    
    def generate_embedding(self, text: str):
        """Tạo embedding vector từ text sử dụng Gemini"""
        try:
            result = genai.embed_content(
                model=self.embed_model,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            print(f" Lỗi khi tạo embedding: {e}")
            return None
    
    def load_jsonl(self):
        """Đọc file JSONL với error handling tốt hơn"""
        documents = []
        line_number = 0
        
        with open(self.jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_number += 1
                line = line.strip()
                
                if not line:  # Bỏ qua dòng trống
                    continue
                
                try:
                    # Parse JSON với strict=False để cho phép các ký tự điều khiển
                    doc = json.loads(line, strict=False)
                    documents.append(doc)
                except json.JSONDecodeError as e:
                    print(f"⚠️ Lỗi parse JSON tại dòng {line_number}: {e}")
                    print(f"   Nội dung: {line[:100]}...")
                    
                    # Thử fix bằng cách escape các ký tự đặc biệt
                    try:
                        # Replace các ký tự xuống dòng trong string
                        fixed_line = line.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                        doc = json.loads(fixed_line, strict=False)
                        documents.append(doc)
                        print(f"   ✓ Đã fix thành công dòng {line_number}")
                    except:
                        print(f"   ✗ Không thể fix dòng {line_number}, bỏ qua")
                        continue
        
        return documents
    
    def prepare_document(self, doc):
        """Chuẩn bị document để upload"""
        # Clean text: remove extra whitespace và newlines
        input_text = ' '.join(doc['input'].split())
        output_text = ' '.join(doc['output'].split())
        
        content = f"""Câu hỏi: {input_text}
Câu trả lời: {output_text}"""
        
        metadata = {
            "instruction": doc.get("instruction", ""),
            "input": input_text,
            "output": output_text
        }
        
        return {
            "content": content,
            "metadata": metadata
        }
    
    def upload_to_supabase(self, documents):
        """Upload documents lên Supabase với embeddings"""
        print(f"\n🚀 Bắt đầu upload {len(documents)} documents lên Supabase...")
        
        success_count = 0
        fail_count = 0
        
        for i, doc in enumerate(tqdm(documents, desc="Uploading")):
            try:
                # 1. Tạo embedding cho content
                embedding = self.generate_embedding(doc['content'])
                
                if embedding is None:
                    print(f"⚠️ Bỏ qua document {i+1} do không tạo được embedding")
                    fail_count += 1
                    continue
                
                # 2. Chuẩn bị data để insert
                data = {
                    "content": doc['content'],
                    "metadata": doc['metadata'],
                    "embedding": embedding
                }
                
                # 3. Insert vào Supabase
                result = supabase.table("documents").insert(data).execute()
                
                success_count += 1
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"\n Lỗi khi upload document {i+1}: {e}")
                fail_count += 1
                continue
        
        print(f"\n Hoàn thành!")
        print(f"   - Thành công: {success_count} documents")
        print(f"   - Thất bại: {fail_count} documents")
    
    def run(self):
        """Chạy toàn bộ quá trình upload"""
        print("=" * 60)
        print(" KNOWLEDGE BASE UPLOADER")
        print("=" * 60)
        
        # 1. Load JSONL file
        print(f"\n Đọc file: {self.jsonl_file_path}")
        raw_documents = self.load_jsonl()
        print(f"    Đọc được {len(raw_documents)} documents")
        
        if len(raw_documents) == 0:
            print(" Không có document nào được load. Kiểm tra lại file JSONL!")
            return
        
        # 2. Prepare documents
        print(f"\n Chuẩn bị documents...")
        prepared_docs = [self.prepare_document(doc) for doc in raw_documents]
        print(f"    Đã chuẩn bị xong {len(prepared_docs)} documents")
        
        # 3. Upload to Supabase
        self.upload_to_supabase(prepared_docs)
        
        print("\n" + "=" * 60)
        print(" Hoàn tất quá trình upload!")
        print("=" * 60)


if __name__ == "__main__":
    # Đường dẫn đến file JSONL của bạn
    JSONL_FILE = "app/data/qa_pairs.jsonl"
    
    uploader = KnowledgeBaseUploader(JSONL_FILE)
    uploader.run()