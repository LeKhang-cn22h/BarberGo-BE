from services.rag_service import rag_service
import json
import time

def clean_all_metadata():
    """
    Xóa các fields không cần thiết trong metadata:
    - Xóa: input, instruction
    - Giữ: output, type
    """
    print("=" * 60)
    print("🧹 CLEAN METADATA")
    print("=" * 60)
    
    # 1. Lấy tất cả documents
    print("\n📥 Đang lấy tất cả documents...")
    all_docs = rag_service.get_all_documents(limit=1000)
    print(f"✅ Tìm thấy {len(all_docs)} documents")
    
    if not all_docs:
        print("❌ Không có document nào!")
        return
    
    # 2. Process từng document
    success_count = 0
    fail_count = 0
    
    for i, doc in enumerate(all_docs, 1):
        doc_id = doc['id']
        old_metadata = doc.get('metadata', {})
        
        print(f"\n[{i}/{len(all_docs)}] Processing Document ID: {doc_id}")
        
        try:
            # Parse metadata if string
            if isinstance(old_metadata, str):
                try:
                    old_metadata = json.loads(old_metadata)
                except json.JSONDecodeError:
                    print(f"  ⚠️ Cannot parse metadata, skipping")
                    fail_count += 1
                    continue
            
            # Extract chỉ output và type
            output = old_metadata.get('output', 'Vui lòng liên hệ hỗ trợ.')
            doc_type = old_metadata.get('type', 'general')
            
            # Tạo metadata mới - CHỈ CÓ output và type
            new_metadata = {
                "output": output,
                "type": doc_type
            }
            
            print(f"  📋 Old metadata keys: {list(old_metadata.keys())}")
            print(f"  ✨ New metadata keys: {list(new_metadata.keys())}")
            print(f"  📝 Output: {output[:60]}...")
            print(f"  🏷️ Type: {doc_type}")
            
            # Update document - CHỈ SỬA METADATA, không đổi content và embedding
            result = rag_service.supabase.table("documents") \
                .update({"metadata": json.dumps(new_metadata, ensure_ascii=False)}) \
                .eq("id", doc_id) \
                .execute()
            
            if result.data:
                print(f"  ✅ Cleaned successfully!")
                success_count += 1
            else:
                print(f"  ❌ Update failed!")
                fail_count += 1
            
            # Rate limiting
            time.sleep(0.3)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            fail_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 CLEAN SUMMARY")
    print(f"✅ Success: {success_count}")
    print(f"❌ Failed: {fail_count}")
    print(f"📝 Total: {len(all_docs)}")
    print("=" * 60)


def preview_clean():
    """
    Preview metadata trước và sau khi clean
    """
    print("=" * 60)
    print("👀 PREVIEW CLEAN (Không update DB)")
    print("=" * 60)
    
    all_docs = rag_service.get_all_documents(limit=10)
    
    for i, doc in enumerate(all_docs, 1):
        old_metadata = doc.get('metadata', {})
        
        if isinstance(old_metadata, str):
            try:
                old_metadata = json.loads(old_metadata)
            except:
                old_metadata = {}
        
        # Extract
        output = old_metadata.get('output', '')
        doc_type = old_metadata.get('type', 'general')
        
        new_metadata = {
            "output": output,
            "type": doc_type
        }
        
        print(f"\n[Document #{doc['id']}]")
        print(f"OLD metadata:")
        print(f"  Keys: {list(old_metadata.keys())}")
        for key, value in old_metadata.items():
            print(f"  - {key}: {str(value)[:60]}...")
        
        print(f"\nNEW metadata:")
        print(f"  Keys: {list(new_metadata.keys())}")
        for key, value in new_metadata.items():
            print(f"  - {key}: {str(value)[:60]}...")
        
        print("-" * 60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--preview":
        # Preview mode
        preview_clean()
    else:
        # Real cleaning
        confirm = input("⚠️ Bạn có chắc muốn clean metadata TẤT CẢ documents? (yes/no): ")
        if confirm.lower() == 'yes':
            clean_all_metadata()
        else:
            print("❌ Cleaning cancelled")