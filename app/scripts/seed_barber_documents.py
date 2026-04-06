"""
Script seed dữ liệu barber vào bảng barber_documents
Chạy: python scripts/seed_barber_documents.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supabase import create_client
from dotenv import load_dotenv
from app.services.barber_rag_service import (
    insert_barber_document,
    delete_barber_documents_by_barber_id
)
import json

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


def fetch_all_data():
    barbers  = supabase.table("barbers").select("*").eq("status", True).execute().data
    result   = []
    for b in barbers:
        services = supabase.table("services").select("*") \
            .eq("barber_id", b["id"]).eq("status", True).execute().data
        ratings  = supabase.table("ratings").select("score, comment") \
            .eq("barber_id", b["id"]).execute().data
        avg_score = (
            round(sum(r["score"] for r in ratings) / len(ratings), 1)
            if ratings else None
        )
        result.append({"barber": b, "services": services,
                        "ratings": ratings, "avg_score": avg_score})
    return result


def generate_qa_pairs(data: list) -> list:
    pairs = []
    days_vn = {
        "Monday": "Thứ 2", "Tuesday": "Thứ 3", "Wednesday": "Thứ 4",
        "Thursday": "Thứ 5", "Friday": "Thứ 6",
        "Saturday": "Thứ 7", "Sunday": "Chủ nhật"
    }

    for item in data:
        b         = item["barber"]
        services  = item["services"]
        ratings   = item["ratings"]
        avg_score = item["avg_score"]

        name     = b["name"]
        area     = b.get("area", "")
        address  = b.get("address", "")
        open_t   = b.get("opening_time", "")
        close_t  = b.get("closing_time", "")
        wdays    = b.get("working_days", [])
        wdays_str = ", ".join(days_vn.get(d, d) for d in (wdays if isinstance(wdays, list) else []))

        meta = {"barber_id": b["id"], "barber_name": name}

        # Giờ mở cửa
        pairs.append((
            f"Tiệm {name} mở cửa mấy giờ? Giờ làm việc {name}?",
            f"Tiệm {name} mở từ {open_t} đến {close_t}, các ngày: {wdays_str}. "
            f"Đặt lịch trên BarberGo để chọn slot phù hợp nhé!",
            {**meta, "type": "opening_hours"}
        ))

        # Địa chỉ
        pairs.append((
            f"Tiệm {name} ở đâu? Địa chỉ {name}? Barber khu {area}?",
            f"Tiệm {name} tại {address}, khu vực {area}. "
            f"Xem bản đồ và đặt lịch ngay trên BarberGo!",
            {**meta, "type": "location"}
        ))

        # Bảng giá tổng
        if services:
            lines     = "\n".join(f"  - {s['service_name']}: {s['price']:,}đ ({s['duration_min']} phút)" for s in services)
            min_price = min(s["price"] for s in services)
            max_price = max(s["price"] for s in services)
            pairs.append((
                f"Bảng giá tiệm {name}? Dịch vụ {name} giá bao nhiêu?",
                f"Tiệm {name} có các dịch vụ:\n{lines}\n"
                f"Giá từ {min_price:,}đ – {max_price:,}đ. Đặt lịch trên BarberGo!",
                {**meta, "type": "services"}
            ))

            # Chi tiết từng dịch vụ
            for s in services:
                pairs.append((
                    f"{s['service_name']} tại {name} giá bao nhiêu? Mất bao lâu?",
                    f"{s['service_name']} tại {name}: {s['price']:,}đ, "
                    f"khoảng {s['duration_min']} phút. Đặt lịch trước để không chờ!",
                    {**meta, "type": "service_detail",
                     "service_name": s["service_name"], "price": s["price"]}
                ))

        # Đánh giá
        if avg_score:
            comments = [r["comment"] for r in ratings if r.get("comment")][:3]
            cmt_text = " | ".join(f'"{c}"' for c in comments) if comments else ""
            pairs.append((
                f"Tiệm {name} có tốt không? Review {name}?",
                f"Tiệm {name} đạt {avg_score}/5 sao ({len(ratings)} đánh giá). "
                f"{cmt_text} Thử đặt lịch trên BarberGo và để lại review nhé!",
                {**meta, "type": "rating", "avg_score": avg_score}
            ))

        # Gợi ý theo khu vực
        svc_preview = ", ".join(s["service_name"] for s in services[:3]) if services else "nhiều dịch vụ"
        pairs.append((
            f"Gợi ý tiệm barber {area}? Tìm tiệm tóc ở {area}?",
            f"Tiệm {name} ở {area} là lựa chọn tốt! Chuyên: {svc_preview}. "
            f"Mở cửa {open_t}–{close_t}, {wdays_str}. "
            f"{f'Đánh giá {avg_score}/5. ' if avg_score else ''}"
            f"Đặt lịch ngay trên BarberGo!",
            {**meta, "type": "recommendation"}
        ))

    return pairs


def main():
    print(" Seed barber_documents...\n")
    data  = fetch_all_data()
    print(f" {len(data)} tiệm barber\n")

    pairs = generate_qa_pairs(data)
    print(f" {len(pairs)} Q&A pairs\n")

    ok = 0
    for i, (content, output, meta) in enumerate(pairs):
        print(f" [{i+1}/{len(pairs)}] {meta['barber_name']} — {meta['type']}")
        doc = insert_barber_document(content, output, meta)
        if doc:
            ok += 1
        else:
            print(f"   Thất bại")

    print(f"\n {ok}/{len(pairs)} documents đã seed vào barber_documents!")


if __name__ == "__main__":
    main()