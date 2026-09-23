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
    # Router metadata is additive: existing clients ignore unknown fields.
    engine: Optional[str] = None
    reason: Optional[str] = None
    issues: Optional[List[str]] = None


class ImmersiveRequest(BaseModel):
    source_lang: Optional[str] = None
    target_lang: str
    text_list: List[str]


class ImmersiveItem(BaseModel):
    detected_source_lang: str
    text: str
    engine: Optional[str] = None
    reason: Optional[str] = None
    issues: Optional[List[str]] = None
    similarity: Optional[float] = None


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


class RouteRequest(BaseModel):
    """Inspect the router's decision without running a forward pass."""

    text: Optional[str] = None
    text_list: Optional[List[str]] = None
    source_lang: Optional[str] = None
    target_lang: str = "zh"


class RouteDecision(BaseModel):
    index: int
    engine: str
    reason: str = ""
    chunks: int = 1
    source_lang: str = ""
    target_lang: str = ""
    glossary_terms: List[str] = []
    reference_attached: bool = False
    similarity: Optional[float] = None


class RouteResponse(BaseModel):
    routes: List[RouteDecision]


class ShortlistEntryModel(BaseModel):
    source: str
    target: str
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None


class ShortlistAddRequest(BaseModel):
    entries: List[ShortlistEntryModel] = []


class GlossaryTermModel(BaseModel):
    source_term: str
    target_term: str
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None


class GlossaryAddRequest(BaseModel):
    terms: List[GlossaryTermModel] = []


class MemoryAddResponse(BaseModel):
    added: int
    shortlist_entries: int
    glossary_entries: int
    persisted: bool = False
