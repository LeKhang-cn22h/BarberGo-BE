"""
Barber RAG Service - Tách biệt hoàn toàn với rag_service.py cũ
- Bảng: barber_documents (không đụng bảng documents)
- Function: match_barber_documents
"""

import json
import os
from typing import List, Dict, Optional
from supabase import create_client
from dotenv import load_dotenv
from app.services.ollama_client import ollama_client

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
TABLE_NAME  = "barber_documents"


# ==========================================
# Embedding
# ==========================================

def create_embedding(text: str) -> list:
    result = ollama_client.embeddings(model=EMBED_MODEL, prompt=text)
    return result["embedding"]


# ==========================================
# Search
# ==========================================

def search_barber_documents(
    query: str,
    top_k: int = 3,
    threshold: float = 0.5
) -> List[Dict]:
    try:
        embedding = create_embedding(query)

        result = supabase.rpc("match_barber_documents", {
            "query_embedding": embedding,
            "match_threshold": threshold,
            "match_count": top_k
        }).execute()

        docs = []
        for doc in result.data:
            if isinstance(doc.get("metadata"), str):
                doc["metadata"] = json.loads(doc["metadata"])
            docs.append(doc)

        return docs

    except Exception as e:
        print(f" Search barber documents error: {e}")
        return []


# ==========================================
# CRUD
# ==========================================

def insert_barber_document(
    content: str,
    output: str,
    extra_metadata: Optional[dict] = None
) -> Optional[Dict]:
    try:
        metadata = {"input": content, "output": output}
        if extra_metadata:
            metadata.update(extra_metadata)

        result = supabase.table(TABLE_NAME).insert({
            "content":   content,
            "embedding": create_embedding(content),
            "metadata":  json.dumps(metadata, ensure_ascii=False)
        }).execute()

        return result.data[0] if result.data else None

    except Exception as e:
        print(f" Insert barber document error: {e}")
        return None


def get_all_barber_documents(
    limit: int = 100,
    offset: int = 0,
    doc_type: Optional[str] = None
) -> Dict:
    try:
        query = supabase.table(TABLE_NAME).select("id, content, metadata")

        # Filter theo type nếu có
        if doc_type:
            query = query.eq("metadata->>type", doc_type)

        result = query.range(offset, offset + limit - 1).order("id").execute()

        # Count
        count_query = supabase.table(TABLE_NAME).select("*", count="exact")
        if doc_type:
            count_query = count_query.eq("metadata->>type", doc_type)
        count_result = count_query.execute()

        docs = []
        for doc in result.data:
            if isinstance(doc.get("metadata"), str):
                doc["metadata"] = json.loads(doc["metadata"])
            docs.append(doc)

        return {
            "total":    count_result.count,
            "limit":    limit,
            "offset":   offset,
            "has_more": (offset + limit) < count_result.count,
            "documents": docs
        }

    except Exception as e:
        print(f"Get barber documents error: {e}")
        return {"total": 0, "documents": []}


def get_barber_document_by_id(doc_id: int) -> Optional[Dict]:
    try:
        result = supabase.table(TABLE_NAME) \
            .select("id, content, metadata") \
            .eq("id", doc_id) \
            .single() \
            .execute()

        doc = result.data
        if isinstance(doc.get("metadata"), str):
            doc["metadata"] = json.loads(doc["metadata"])

        return doc

    except Exception as e:
        print(f" Get barber document by id error: {e}")
        return None


def update_barber_document(
    doc_id: int,
    new_content: Optional[str] = None,
    new_output:  Optional[str] = None,
    new_metadata: Optional[dict] = None
) -> bool:
    try:
        update_data = {}

        # Re-embed nếu content thay đổi
        if new_content:
            update_data["content"]   = new_content
            update_data["embedding"] = create_embedding(new_content)

        # Merge metadata
        if new_output or new_metadata:
            old = supabase.table(TABLE_NAME) \
                .select("metadata") \
                .eq("id", doc_id) \
                .single() \
                .execute()

            meta = old.data.get("metadata", {})
            if isinstance(meta, str):
                meta = json.loads(meta)

            if new_output:
                meta["output"] = new_output
            if new_metadata:
                meta.update(new_metadata)

            update_data["metadata"] = json.dumps(meta, ensure_ascii=False)

        result = supabase.table(TABLE_NAME) \
            .update(update_data) \
            .eq("id", doc_id) \
            .execute()

        return bool(result.data)

    except Exception as e:
        print(f" Update barber document error: {e}")
        return False


def delete_barber_document(doc_id: int) -> bool:
    try:
        result = supabase.table(TABLE_NAME) \
            .delete() \
            .eq("id", doc_id) \
            .execute()
        return bool(result.data)
    except Exception as e:
        print(f" Delete barber document error: {e}")
        return False


def delete_barber_documents_by_barber_id(barber_id: str) -> int:
    """Xoá tất cả documents của 1 tiệm cụ thể"""
    try:
        # Lấy ids cần xoá
        result = supabase.table(TABLE_NAME) \
            .select("id, metadata") \
            .execute()

        ids_to_delete = [
            d["id"] for d in result.data
            if (json.loads(d["metadata"]) if isinstance(d["metadata"], str)
                else d.get("metadata", {})).get("barber_id") == barber_id
        ]

        if not ids_to_delete:
            return 0

        supabase.table(TABLE_NAME) \
            .delete() \
            .in_("id", ids_to_delete) \
            .execute()

        return len(ids_to_delete)

    except Exception as e:
        print(f" Delete by barber_id error: {e}")
        return 0


def search_barber_documents_by_keyword(keyword: str) -> List[Dict]:
    try:
        result = supabase.table(TABLE_NAME) \
            .select("id, content, metadata") \
            .ilike("content", f"%{keyword}%") \
            .execute()

        docs = []
        for doc in result.data:
            if isinstance(doc.get("metadata"), str):
                doc["metadata"] = json.loads(doc["metadata"])
            docs.append(doc)

        return docs

    except Exception as e:
        print(f" Keyword search error: {e}")
        return []