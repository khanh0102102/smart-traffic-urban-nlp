"""Local image upload endpoint for the demonstration application."""

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status

router = APIRouter(tags=["uploads"])

UPLOAD_DIR = Path(__file__).resolve().parents[4] / "uploads"
MAX_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED_TYPES = {
    "image/jpeg": (".jpg", lambda data: data.startswith(b"\xff\xd8\xff")),
    "image/png": (".png", lambda data: data.startswith(b"\x89PNG\r\n\x1a\n")),
    "image/webp": (".webp", lambda data: data.startswith(b"RIFF") and data[8:12] == b"WEBP"),
}


@router.post("/uploads/image", status_code=status.HTTP_201_CREATED)
async def upload_image(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Chỉ chấp nhận ảnh JPEG, PNG hoặc WebP.")

    content = await file.read(MAX_IMAGE_BYTES + 1)
    await file.close()
    if not content:
        raise HTTPException(status_code=400, detail="Tệp ảnh rỗng.")
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Ảnh vượt quá giới hạn 5 MB.")

    extension, signature_check = ALLOWED_TYPES[file.content_type]
    if not signature_check(content):
        raise HTTPException(status_code=415, detail="Định dạng nội dung ảnh không khớp MIME type.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid4().hex}{extension}"
    (UPLOAD_DIR / filename).write_bytes(content)
    return {"image_url": f"/media/{filename}", "filename": filename}
