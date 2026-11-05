from datetime import datetime

from pydantic import BaseModel, Field


class BaseDocument(BaseModel):
    id: str | None = Field(alias="_id", default=None)
    rev: str | None = Field(alias="_rev", default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True # Permite usar los alias de CouchDB (_id, _rev)