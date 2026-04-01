# regenerate_embeddings.py
from app.services.rag_service import rag_service
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

print(" Starting embedding regeneration...")

# 1. Lấy tất cả documents
result = supabase.table("documents").select("id, content").execute()
documents = result.data

print(f"Found {len(documents)} documents to update")

# 2. Regenerate từng document
for i, doc in enumerate(documents, 1):
    doc_id = doc['id']
    content = doc['content']
    
    print(f"\n[{i}/{len(documents)}] Processing doc ID {doc_id}")
    print(f"   Content: {content[:60]}...")
    
    try:
        # Generate NEW embedding với Ollama
        new_embedding = rag_service.generate_embedding(content, use_ollama=True)
        
        print(f"    Generated new embedding (length: {len(new_embedding)})")
        
        # Update vào DB
        supabase.table("documents").update({
            "embedding": new_embedding
        }).eq("id", doc_id).execute()
        
        print(f"    Updated in database")
        
    except Exception as e:
        print(f"  Error: {e}")
        continue

print("\n Done! All embeddings regenerated with Ollama.")