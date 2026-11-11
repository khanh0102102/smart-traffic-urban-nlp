"""Pydantic request / response schemas."""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AdminLoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=256)


class AdminTokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int


class PredictRequest(BaseModel):
    text: str = Field(min_length=3, max_length=5000)
    include_probabilities: bool = True

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Nội dung phản ánh không được để trống.")
        return value


class ReportCreateRequest(PredictRequest):
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    image_url: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def coordinates_must_be_paired(self):
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude và longitude phải cùng có hoặc cùng không có.")
        return self


class ModerationRequest(BaseModel):
    category_code: str = Field(pattern=r"^R(0[1-9]|1[01])$")
    note: str | None = Field(default=None, max_length=2000)


class CategoryOut(BaseModel):
    code: str
    name: str


class PredictionOut(BaseModel):
    predicted_category: CategoryOut
    confidence: float
    needs_review: bool
    model_name: str
    model_version: str
    probabilities: dict[str, float] | None = None


class ReportCreateOut(PredictionOut):
    report_id: int
    report_code: str
    status: str


class ReportOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    report_code: str
    text: str
    latitude: float | None
    longitude: float | None
    image_url: str | None
    status: str
    created_at: str
    predicted_category: CategoryOut | None = None
    confidence: float | None = None
    needs_review: bool | None = None
    confirmed_category: CategoryOut | None = None
