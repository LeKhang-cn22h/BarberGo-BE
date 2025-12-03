from app.database.supabase_client import supabase
from app.schemas.ratings_schema import RatingCreate, RatingUpdate
from fastapi import HTTPException
from datetime import datetime

# ==================== Create Rating ====================
def create_rating(data: RatingCreate):
    """Tạo đánh giá mới"""
    try:
        # Kiểm tra barber có tồn tại không
        barber_check = supabase.table("barbers").select("id").eq("id", data.barber_id).execute()
        if not barber_check.data:
            raise HTTPException(status_code=404, detail="Barber không tồn tại")
        
        # Kiểm tra user có tồn tại không
        user_check = supabase.table("users").select("id").eq("id", data.user_id).execute()
        if not user_check.data:
            raise HTTPException(status_code=404, detail="User không tồn tại")
        
        # Kiểm tra user đã đánh giá barber này chưa
        existing_rating = supabase.table("ratings")\
            .select("*")\
            .eq("barber_id", data.barber_id)\
            .eq("user_id", data.user_id)\
            .execute()
        
        if existing_rating.data:
            raise HTTPException(status_code=400, detail="Bạn đã đánh giá barber này rồi. Vui lòng cập nhật đánh giá cũ.")
        
        # Tạo rating
        rating_data = {
            "barber_id": data.barber_id,
            "user_id": data.user_id,
            "score": float(data.score),
            "comment": data.comment,
            "created_at": datetime.now().isoformat()
        }
        
        response = supabase.table("ratings").insert(rating_data).execute()
        
        if not response.data:
            raise HTTPException(status_code=400, detail="Tạo đánh giá thất bại")
        
        # ← CẬP NHẬT RANK CỦA BARBER
        new_rank = update_barber_rank(data.barber_id)
        
        return {
            "message": "Đánh giá thành công",
            "rating": response.data[0],
            "barber_new_rank": new_rank  # Trả về rank mới của barber
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Tạo đánh giá thất bại: {str(e)}")
# ==================== Get Ratings ====================

def get_all_ratings():
    """Lấy danh sách tất cả đánh giá"""
    try:
        response = supabase.table("ratings")\
            .select("*, barbers(full_name), users(full_name, email)")\
            .order("created_at", desc=True)\
            .execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {str(e)}")


def get_rating_by_id(rating_id: int):
    """Lấy thông tin đánh giá theo ID"""
    try:
        response = supabase.table("ratings")\
            .select("*, barbers(full_name), users(full_name, email)")\
            .eq("id", rating_id)\
            .execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy đánh giá")
        
        return response.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {str(e)}")


def get_ratings_by_barber(barber_id: str):
    """Lấy danh sách đánh giá của 1 barber"""
    try:
        response = supabase.table("ratings")\
            .select("*, users(full_name, email, avatar_url)")\
            .eq("barber_id", barber_id)\
            .order("created_at", desc=True)\
            .execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {str(e)}")


def get_ratings_by_user(user_id: str):
    """Lấy danh sách đánh giá của 1 user"""
    try:
        response = supabase.table("ratings")\
            .select("*, barbers(full_name, avatar_url)")\
            .eq("user_id", user_id)\
            .order("created_at", desc=True)\
            .execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {str(e)}")


def get_barber_average_rating(barber_id: str):
    """Lấy điểm trung bình và tổng số đánh giá của barber"""
    try:
        # Kiểm tra barber có tồn tại không
        barber_check = supabase.table("barbers").select("id, full_name").eq("id", barber_id).execute()
        if not barber_check.data:
            raise HTTPException(status_code=404, detail="Barber không tồn tại")
        
        # Lấy tất cả ratings của barber
        ratings = supabase.table("ratings")\
            .select("score")\
            .eq("barber_id", barber_id)\
            .execute()
        
        if not ratings.data:
            return {
                "barber_id": barber_id,
                "barber_name": barber_check.data[0]["full_name"],
                "average_score": 0,
                "total_ratings": 0
            }
        
        # Tính điểm trung bình
        total_score = sum(float(r["score"]) for r in ratings.data)
        total_ratings = len(ratings.data)
        average_score = round(total_score / total_ratings, 2)
        
        return {
            "barber_id": barber_id,
            "barber_name": barber_check.data[0]["full_name"],
            "average_score": average_score,
            "total_ratings": total_ratings
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {str(e)}")


# ==================== Update Rating ====================

def update_rating(rating_id: int, data: RatingUpdate):
    """Cập nhật đánh giá"""
    try:
        # Lấy thông tin rating cũ để biết barber_id
        old_rating = supabase.table("ratings").select("barber_id").eq("id", rating_id).execute()
        if not old_rating.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy đánh giá")
        
        barber_id = old_rating.data[0]["barber_id"]
        
        # Chỉ update các field không None
        update_data = {}
        
        if data.score is not None:
            update_data["score"] = float(data.score)
        
        if data.comment is not None:
            update_data["comment"] = data.comment
        
        if not update_data:
            raise HTTPException(status_code=400, detail="Không có dữ liệu để cập nhật")
        
        response = supabase.table("ratings").update(update_data).eq("id", rating_id).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy đánh giá")
        
        # ← CẬP NHẬT RANK CỦA BARBER
        new_rank = update_barber_rank(barber_id)
        
        return {
            "message": "Cập nhật đánh giá thành công",
            "rating": response.data[0],
            "barber_new_rank": new_rank
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cập nhật thất bại: {str(e)}")

# ==================== Delete Rating ====================

def delete_rating(rating_id: int):
    """Xóa đánh giá"""
    try:
        # Lấy thông tin rating để biết barber_id
        rating = supabase.table("ratings").select("barber_id").eq("id", rating_id).execute()
        if not rating.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy đánh giá")
        
        barber_id = rating.data[0]["barber_id"]
        
        # Xóa rating
        response = supabase.table("ratings").delete().eq("id", rating_id).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy đánh giá")
        
        # ← CẬP NHẬT RANK CỦA BARBER
        new_rank = update_barber_rank(barber_id)
        
        return {
            "message": "Xóa đánh giá thành công",
            "barber_new_rank": new_rank
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Xóa thất bại: {str(e)}")
# ==================== Update Barber Rank ====================

def update_barber_rank(barber_id: str):
    """Tính lại và cập nhật rank của barber dựa trên ratings"""
    try:
        # Lấy tất cả ratings của barber
        ratings = supabase.table("ratings")\
            .select("score")\
            .eq("barber_id", barber_id)\
            .execute()
        
        if not ratings.data or len(ratings.data) == 0:
            # Không có rating nào → rank = 0
            new_rank = 0.0
        else:
            # Tính điểm trung bình
            total_score = sum(float(r["score"]) for r in ratings.data)
            total_ratings = len(ratings.data)
            new_rank = round(total_score / total_ratings, 2)
        
        # Cập nhật rank vào bảng barbers
        supabase.table("barbers")\
            .update({"rank": new_rank})\
            .eq("id", barber_id)\
            .execute()
        
        return new_rank
    except Exception as e:
        print(f"Lỗi cập nhật rank: {str(e)}")
        return None