import os
from dotenv import load_dotenv
import google.generativeai as genai
from supabase import create_client

load_dotenv()

# Configure
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

print("=" * 60)
print("🔍 RAG SYSTEM DEBUG")
print("=" * 60)

# Test 1: Embedding
print("\n1️⃣ Test Embedding...")
try:
    question = "Làm thế nào để đặt lịch?"
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=question,
        task_type="retrieval_query"
    )
    embedding = result['embedding']
    print(f"   ✅ Embedding OK - dimension: {len(embedding)}")
except Exception as e:
    print(f"   ❌ Embedding Error: {e}")
    exit(1)

# Test 2: Supabase Search
print("\n2️⃣ Test Supabase Search...")
try:
    search_result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": embedding,
            "match_count": 3
        }
    ).execute()
    
    docs = search_result.data
    print(f"   ✅ Search OK - found {len(docs)} documents")
    
    if len(docs) > 0:
        print(f"\n   📄 Top result:")
        print(f"      Similarity: {docs[0]['similarity']:.3f}")
        print(f"      Question: {docs[0]['metadata']['input'][:80]}...")
        print(f"      Answer: {docs[0]['metadata']['output'][:80]}...")
    else:
        print("   ⚠️ No documents found!")
        
except Exception as e:
    print(f"   ❌ Supabase Error: {e}")
    exit(1)

# Test 3: Gemini Generate Content
print("\n3️⃣ Test Gemini Generation...")
try:
    # Simple test first
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    simple_response = model.generate_content("Say hello in Vietnamese")
    print(f"   ✅ Simple generation OK: {simple_response.text[:50]}...")
    
except Exception as e:
    print(f"   ❌ Simple generation error: {e}")
    print("\n   Trying with gemini-1.5-flash instead...")
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        simple_response = model.generate_content("Say hello")
        print(f"   ✅ Works with gemini-1.5-flash: {simple_response.text[:50]}...")
        print("\n   💡 Solution: Change model to 'gemini-1.5-flash' in rag_service.py")
    except Exception as e2:
        print(f"   ❌ Still error: {e2}")
        exit(1)

# Test 4: Generate with Context
print("\n4️⃣ Test Generation with Context...")
try:
    if len(docs) > 0:
        context = docs[0]['metadata']['output']
        prompt = f"""Bạn là trợ lý ảo của BarberGo.

Dựa vào thông tin sau:
{context}

Trả lời câu hỏi: {question}

Câu trả lời ngắn gọn:"""

        response = model.generate_content(prompt)
        print(f"   ✅ Context generation OK!")
        print(f"\n   📝 Answer: {response.text}")
        
except Exception as e:
    print(f"   ❌ Context generation error: {e}")
    print(f"\n   Error type: {type(e).__name__}")
    
    # Chi tiết lỗi
    import traceback
    print("\n   Full traceback:")
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ Debug complete!")
print("=" * 60)