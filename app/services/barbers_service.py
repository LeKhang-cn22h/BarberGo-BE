from app.database.supabase_client import supabase
from fastapi import HTTPException, UploadFile
from app.schemas.barbers_schema import BarberCreate, BarberUpdate, BarberResponse
from typing import List, Optional
from uuid import UUID
import uuid
from pathlib import Path

BUCKET_NAME = "imgBarber"

def upload_barber_image(file: UploadFile, barber_id: str) -> str:
    """Upload ảnh lên Supabase Storage và trả về public URL"""
    try:
        # Kiểm tra file type
        allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
            )
        
        # Kiểm tra file size (max 5MB)
        file.file.seek(0, 2)  # Di chuyển con trỏ đến cuối file
        file_size = file.file.tell()  # Lấy vị trí = size
        file.file.seek(0)  # Reset con trỏ về đầu
        
        max_size = 5 * 1024 * 1024  # 5MB
        if file_size > max_size:
            raise HTTPException(status_code=400, detail="File size exceeds 5MB")
        
        # Tạo file name unique
        file_extension = Path(file.filename).suffix
        file_name = f"{barber_id}_{uuid.uuid4()}{file_extension}"
        file_path = f"barbers/{file_name}"
        
        # Đọc file content
        file_content = file.file.read()
        
        # Upload lên Supabase Storage
        response = supabase.storage.from_(BUCKET_NAME).upload(
            path=file_path,
            file=file_content,
            file_options={"content-type": file.content_type}
        )
        
        # Lấy public URL
        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(file_path)
        
        return public_url
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading image: {str(e)}")


def delete_barber_image(image_url: str) -> bool:
    """Xóa ảnh từ Supabase Storage"""
    try:
        if not image_url:
            return True
        
        # Extract file path from public URL
        # URL format: https://...supabase.co/storage/v1/object/public/imgBarber/barbers/filename.jpg
        if BUCKET_NAME in image_url:
            parts = image_url.split(f"{BUCKET_NAME}/")
            if len(parts) >= 2:
                file_path = parts[1]
                supabase.storage.from_(BUCKET_NAME).remove([file_path])
                return True
        
        return False
        
    except Exception as e:
        print(f"Error deleting image: {str(e)}")
        return False


def create_barber(data: BarberCreate, image_file: Optional[UploadFile] = None) -> BarberResponse:
    """Tạo barber mới với upload ảnh"""
    barber_id = str(uuid.uuid4())
    image_url = None
    
    try:
        # Upload ảnh nếu có
        if image_file:
            image_url = upload_barber_image(image_file, barber_id)
        
        barber_data = {
            "id": barber_id,
            "name": data.name,
            "location": data.location,
            "area": data.area,
            "address": data.address,
            "imagepath": image_url,
            "user_id": str(data.user_id),
            "rank": 0,
            "status": True
        }
        
        response = supabase.table("barbers").insert(barber_data).execute()
        
        if not response.data:
            # Nếu tạo barber fail thì xóa ảnh đã upload
            if image_url:
                delete_barber_image(image_url)
            raise HTTPException(status_code=400, detail="Failed to create barber")
        
        return BarberResponse(**response.data[0])
        
    except HTTPException:
        # Cleanup ảnh nếu có lỗi
        if image_url:
            delete_barber_image(image_url)
        raise
    except Exception as e:
        # Cleanup ảnh nếu có lỗi
        if image_url:
            delete_barber_image(image_url)
        raise HTTPException(status_code=500, detail=f"Error creating barber: {str(e)}")


def get_barber_by_id(barber_id: UUID) -> BarberResponse:
    try:
        response = supabase.table("barbers").select("*").eq("id", str(barber_id)).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Barber not found")
        
        return BarberResponse(**response.data[0])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching barber: {str(e)}")


def get_all_barbers(
    skip: int = 0, 
    limit: int = 100, 
    status: Optional[bool] = None,
    area: Optional[str] = None
) -> List[BarberResponse]:
    try:
        query = supabase.table("barbers").select("*")
        
        if status is not None:
            query = query.eq("status", status)
        
        if area:
            query = query.eq("area", area)
        
        response = query.range(skip, skip + limit - 1).execute()
        
        return [BarberResponse(**barber) for barber in response.data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching barbers: {str(e)}")


def get_top_barbers(limit: int = 2) -> List[BarberResponse]:
    """Lấy danh sách barbers có rank cao nhất"""
    try:
        response = (
            supabase.table("barbers")
            .select("*")
            .eq("status", True)
            .order("rank", desc=True)
            .limit(limit)
            .execute()
        )
        
        return [BarberResponse(**barber) for barber in response.data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching top barbers: {str(e)}")


def get_barbers_by_user(user_id: UUID) -> List[BarberResponse]:
    try:
        response = supabase.table("barbers").select("*").eq("user_id", str(user_id)).execute()
        
        return [BarberResponse(**barber) for barber in response.data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching user barbers: {str(e)}")


def update_barber(
    barber_id: UUID, 
    data: BarberUpdate, 
    image_file: Optional[UploadFile] = None
) -> BarberResponse:
    """Update barber với option upload ảnh mới"""
    new_image_url = None
    
    try:
        # Lấy thông tin barber hiện tại
        current_barber = get_barber_by_id(barber_id)
        old_image_url = current_barber.imagepath
        
        # Chuẩn bị dữ liệu update
        update_data = {k: v for k, v in data.dict(exclude_unset=True).items() if v is not None}
        
        # Upload ảnh mới nếu có
        if image_file:
            new_image_url = upload_barber_image(image_file, str(barber_id))
            update_data["imagepath"] = new_image_url
        
        if not update_data:
            raise HTTPException(status_code=400, detail="No data to update")
        
        # Convert Decimal to float nếu có
        if "rank" in update_data:
            update_data["rank"] = float(update_data["rank"])
        
        response = supabase.table("barbers").update(update_data).eq("id", str(barber_id)).execute()
        
        if not response.data:
            # Nếu update fail thì xóa ảnh mới đã upload
            if new_image_url:
                delete_barber_image(new_image_url)
            raise HTTPException(status_code=404, detail="Barber not found")
        
        # Xóa ảnh cũ nếu đã upload ảnh mới thành công
        if new_image_url and old_image_url:
            delete_barber_image(old_image_url)
        
        return BarberResponse(**response.data[0])
        
    except HTTPException:
        # Cleanup ảnh mới nếu có lỗi
        if new_image_url:
            delete_barber_image(new_image_url)
        raise
    except Exception as e:
        # Cleanup ảnh mới nếu có lỗi
        if new_image_url:
            delete_barber_image(new_image_url)
        raise HTTPException(status_code=500, detail=f"Error updating barber: {str(e)}")


def soft_delete_barber(barber_id: UUID) -> BarberResponse:
    """Soft delete bằng cách set status = False"""
    try:
        response = supabase.table("barbers").update({"status": False}).eq("id", str(barber_id)).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Barber not found")
        
        return BarberResponse(**response.data[0])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error soft deleting barber: {str(e)}")


def delete_barber(barber_id: UUID) -> dict:
    """Xóa barber vĩnh viễn và ảnh của barber"""
    try:
        # Lấy thông tin barber để lấy image path
        barber = get_barber_by_id(barber_id)
        
        # Xóa record trong database
        response = supabase.table("barbers").delete().eq("id", str(barber_id)).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Barber not found")
        
        # Xóa ảnh từ storage
        if barber.imagepath:
            delete_barber_image(barber.imagepath)
        
        return {"message": "Barber deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting barber: {str(e)}")


def get_unique_locations() -> List[str]:
    """Lấy danh sách các location không trùng lặp"""
    try:
        response = (
            supabase.table("barbers")
            .select("location")
            .eq("status", True)
            .execute()
        )
        
        locations = set()
        for barber in response.data:
            if barber.get("location"):
                locations.add(barber["location"])
        
        return sorted(list(locations))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching locations: {str(e)}")


def get_unique_areas() -> List[str]:
    """Lấy danh sách các area không trùng lặp"""
    try:
        response = (
            supabase.table("barbers")
            .select("area")
            .eq("status", True)
            .execute()
        )
        
        areas = set()
        for barber in response.data:
            if barber.get("area"):
                areas.add(barber["area"])
        
        return sorted(list(areas))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching areas: {str(e)}")


def get_barbers_by_location(location: str) -> List[BarberResponse]:
    """Lấy tất cả barbers theo location cụ thể"""
    try:
        response = (
            supabase.table("barbers")
            .select("*")
            .eq("status", True)
            .eq("location", location)
            .order("rank", desc=True)
            .execute()
        )
        
        return [BarberResponse(**barber) for barber in response.data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching barbers by location: {str(e)}")


def get_barbers_by_area(area: str) -> List[BarberResponse]:
    """Lấy tất cả barbers theo area cụ thể"""
    try:
        response = (
            supabase.table("barbers")
            .select("*")
            .eq("status", True)
            .eq("area", area)
            .order("rank", desc=True)
            .execute()
        )
        
        return [BarberResponse(**barber) for barber in response.data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching barbers by area: {str(e)}")