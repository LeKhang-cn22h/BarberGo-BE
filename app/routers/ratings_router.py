from fastapi import APIRouter, Depends
from app.schemas.ratings_schema import RatingCreate, RatingUpdate
from app.services import rating_service
from app.dependencies.current_user import get_current_user

router = APIRouter(prefix="/ratings", tags=["Ratings"])


# ==================== Create ====================

@router.post("/", dependencies=[Depends(get_current_user)])
def create_rating(data: RatingCreate):
    """
    Tạo đánh giá mới
    - Yêu cầu đăng nhập
    - Mỗi user chỉ đánh giá 1 barber 1 lần
    """
    return rating_service.create_rating(data)


# ==================== Read ====================

@router.get("/")
def get_all_ratings():
    """
    Lấy danh sách tất cả đánh giá
    - Không cần đăng nhập
    """
    return rating_service.get_all_ratings()


@router.get("/{rating_id}")
def get_rating(rating_id: int):
    """
    Lấy thông tin chi tiết 1 đánh giá
    - Không cần đăng nhập
    """
    return rating_service.get_rating_by_id(rating_id)


@router.get("/barber/{barber_id}")
def get_barber_ratings(barber_id: str):
    """
    Lấy danh sách đánh giá của 1 barber
    - Không cần đăng nhập
    """
    return rating_service.get_ratings_by_barber(barber_id)


@router.get("/barber/{barber_id}/average")
def get_barber_average(barber_id: str):
    """
    Lấy điểm trung bình và tổng số đánh giá của barber
    - Không cần đăng nhập
    """
    return rating_service.get_barber_average_rating(barber_id)


@router.get("/user/{user_id}", dependencies=[Depends(get_current_user)])
def get_user_ratings(user_id: str):
    """
    Lấy danh sách đánh giá của 1 user
    - Yêu cầu đăng nhập
    """
    return rating_service.get_ratings_by_user(user_id)


# ==================== Update ====================

@router.put("/{rating_id}", dependencies=[Depends(get_current_user)])
def update_rating(rating_id: int, data: RatingUpdate):
    """
    Cập nhật đánh giá
    - Yêu cầu đăng nhập
    - Chỉ user tạo đánh giá mới được update
    """
    return rating_service.update_rating(rating_id, data)


# ==================== Delete ====================

@router.delete("/{rating_id}", dependencies=[Depends(get_current_user)])
def delete_rating(rating_id: int):
    """
    Xóa đánh giá
    - Yêu cầu đăng nhập
    - Chỉ user tạo đánh giá hoặc admin mới được xóa
    """
    return rating_service.delete_rating(rating_id)