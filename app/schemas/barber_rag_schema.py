from pydantic import BaseModel
from typing import Optional, List, Any

class BarberDocumentCreate(BaseModel):
    content: str
    output:  str
    extra_metadata: Optional[dict] = None

class BarberDocumentUpdate(BaseModel):
    new_content:  Optional[str]  = None
    new_output:   Optional[str]  = None
    new_metadata: Optional[dict] = None

class BarberDocumentItem(BaseModel):
    id:       int
    content:  str
    metadata: Optional[dict] = None

class BarberDocumentListResponse(BaseModel):
    total:     int
    limit:     int
    offset:    int
    has_more:  bool
    documents: List[BarberDocumentItem]

class BarberSearchResult(BaseModel):
    id:         int
    content:    str
    metadata:   Optional[dict] = None
    similarity: float

class BarberSearchResponse(BaseModel):
    query:   str
    total:   int
    results: List[BarberSearchResult]