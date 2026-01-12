from services.rag_service import rag_service
import json
import time

def migrate_all_documents():
    """
    Migrate tất cả documents sang format mới:
    - content: chỉ chứa câu hỏi (input)
    - metadata: {"output": "câu trả lời", "type": "loại"}
    - embedding: vector mới từ content mới
    """
    print("=" * 60)
    print("🔄 MIGRATE DOCUMENTS")
    print("=" * 60)
    
    # 1. Lấy tất cả documents hiện tại
    print("\n📥 Đang lấy tất cả documents...")
    all_docs = rag_service.get_all_documents(limit=1000)
    print(f"✅ Tìm thấy {len(all_docs)} documents")
    
    if not all_docs:
        print("❌ Không có document nào để migrate!")
        return
    
    # 2. Process từng document
    success_count = 0
    fail_count = 0
    
    for i, doc in enumerate(all_docs, 1):
        doc_id = doc['id']
        old_content = doc['content']
        old_metadata = doc.get('metadata', {})
        
        print(f"\n[{i}/{len(all_docs)}] Processing Document ID: {doc_id}")
        print(f"  Old content: {old_content[:80]}...")
        
        try:
            # Parse old metadata if string
            if isinstance(old_metadata, str):
                try:
                    old_metadata = json.loads(old_metadata)
                except json.JSONDecodeError:
                    print(f"  ⚠️ Cannot parse metadata, using empty dict")
                    old_metadata = {}
            
            # Extract input và output từ old metadata hoặc content
            new_content = old_metadata.get('input', '')
            output = old_metadata.get('output', '')
            
            # Nếu không có input trong metadata, extract từ content
            if not new_content:
                # Try to extract from "Câu hỏi: ... Câu trả lời: ..." format
                if 'Câu hỏi:' in old_content and 'Câu trả lời:' in old_content:
                    parts = old_content.split('Câu trả lời:')
                    new_content = parts[0].replace('Câu hỏi:', '').strip()
                    if not output:
                        output = parts[1].strip() if len(parts) > 1 else ''
                else:
                    # Fallback: use old content as is
                    new_content = old_content
            
            if not output:
                output = "Vui lòng liên hệ bộ phận hỗ trợ để được tư vấn chi tiết."
            
            # Determine type based on content
            doc_type = classify_document_type(new_content, output)
            
            # Create new metadata
            new_metadata = {
                "output": output,
                "type": doc_type
            }
            
            print(f"  ✏️ New content: {new_content[:60]}...")
            print(f"  ✏️ Output: {output[:60]}...")
            print(f"  ✏️ Type: {doc_type}")
            
            # Update document (sẽ tự động tạo embedding mới)
            success = rag_service.update_document(
                document_id=doc_id,
                new_content=new_content,
                new_output=output,
                new_metadata={"type": doc_type}
            )
            
            if success:
                print(f"  ✅ Updated successfully!")
                success_count += 1
            else:
                print(f"  ❌ Update failed!")
                fail_count += 1
            
            # Rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            fail_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 MIGRATION SUMMARY")
    print(f"✅ Success: {success_count}")
    print(f"❌ Failed: {fail_count}")
    print(f"📝 Total: {len(all_docs)}")
    print("=" * 60)


def classify_document_type(content: str, output: str) -> str:
    """
    Phân loại document dựa vào content và output
    
    Returns:
        - "app": Về chức năng app (đặt lịch, hủy lịch, thanh toán...)
        - "beauty": Về làm đẹp, chăm sóc (cắt tóc, nhuộm, spa...)
        - "policy": Về chính sách (bảo mật, điều khoản...)
        - "general": Chung chung
    """
    text = (content + " " + output).lower()
    
    # Keywords cho từng loại
    app_keywords = [
        'đặt lịch', 'hủy lịch', 'app', 'ứng dụng', 'tài khoản',
        'đăng nhập', 'đăng ký', 'quên mật khẩu', 'thanh toán',
        'thông báo', 'lịch hẹn', 'booking', 'payment'
    ]
    
    beauty_keywords = [
        'tóc', 'cắt', 'nhuộm', 'uốn', 'duỗi', 'gội', 'massage',
        'spa', 'nail', 'móng', 'làm đẹp', 'chăm sóc', 'wax',
        'facial', 'kiểu tóc', 'stylist', 'barber', 'salon'
    ]
    
    policy_keywords = [
        'chính sách', 'bảo mật', 'quyền riêng tư', 'điều khoản',
        'quy định', 'thỏa thuận', 'hợp đồng', 'pháp lý'
    ]
    
    # Check
    if any(kw in text for kw in app_keywords):
        return "app"
    elif any(kw in text for kw in beauty_keywords):
        return "beauty"
    elif any(kw in text for kw in policy_keywords):
        return "policy"
    else:
        return "general"


def preview_migration():
    """
    Preview những thay đổi mà không thực sự update DB
    """
    print("=" * 60)
    print("👀 PREVIEW MIGRATION (Không update DB)")
    print("=" * 60)
    
    all_docs = rag_service.get_all_documents(limit=10)
    
    for i, doc in enumerate(all_docs, 1):
        old_content = doc['content']
        old_metadata = doc.get('metadata', {})
        
        if isinstance(old_metadata, str):
            try:
                old_metadata = json.loads(old_metadata)
            except:
                old_metadata = {}
        
        new_content = old_metadata.get('input', '')
        output = old_metadata.get('output', '')
        
        if not new_content:
            if 'Câu hỏi:' in old_content:
                parts = old_content.split('Câu trả lời:')
                new_content = parts[0].replace('Câu hỏi:', '').strip()
                if not output:
                    output = parts[1].strip() if len(parts) > 1 else ''
            else:
                new_content = old_content
        
        doc_type = classify_document_type(new_content, output)
        
        print(f"\n[Document #{doc['id']}]")
        print(f"OLD content: {old_content[:100]}...")
        print(f"NEW content: {new_content[:100]}...")
        print(f"Output: {output[:100]}...")
        print(f"Type: {doc_type}")
        print("-" * 60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--preview":
        # Preview mode
        preview_migration()
    else:
        # Real migration
        confirm = input("⚠️ Bạn có chắc muốn migrate TẤT CẢ documents? (yes/no): ")
        if confirm.lower() == 'yes':
            migrate_all_documents()
        else:
            print("❌ Migration cancelled")