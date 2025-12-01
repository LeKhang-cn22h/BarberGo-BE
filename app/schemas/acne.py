from pydantic import BaseModel
from typing import List, Dict

class AcneDetail(BaseModel):
    type: str
    confidence: float
    bbox: List[float]

class RegionResult(BaseModel):
    count: Dict[str, int]
    total: int
    details: List[AcneDetail]

class AdviceItem(BaseModel):
    zone: str
    severity: str
    acne_count: int
    tips: List[str]

class AcneDetectionResponse(BaseModel):
    success: bool
    data: Dict
