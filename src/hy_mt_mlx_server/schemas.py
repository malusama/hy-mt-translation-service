from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DetectRequest(BaseModel):
    text: str


class DetectResponse(BaseModel):
    language: str


class TranslateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str
    from_lang: Optional[str] = Field(default=None, alias="from")
    to: str


class TranslateResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str
    from_lang: str = Field(alias="from")
    to: str


class ImmersiveRequest(BaseModel):
    source_lang: Optional[str] = None
    target_lang: str
    text_list: List[str]


class ImmersiveItem(BaseModel):
    detected_source_lang: str
    text: str


class ImmersiveResponse(BaseModel):
    translations: List[ImmersiveItem]


class HcfyRequest(BaseModel):
    text: str
    source: Optional[str] = None
    destination: List[str]


class HcfyResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str
    from_lang: str = Field(alias="from")
    to: str
    result: List[str]


class DeeplxRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str


class DeeplxResponse(BaseModel):
    code: int
    id: int
    data: str
    alternatives: List[str]
    source_lang: str
    target_lang: str
    method: str

