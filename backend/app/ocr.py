"""OCR for scanned PDF pages and image files.

Two providers, so they can be compared (`python -m eval.ocr_eval`):
- "gemini": a vision LLM reads the page image. Handles tables, odd layouts
  and poor scans well, but each page is an API call (rate limits, latency) and
  the image is sent to Google.
- "tesseract": the classic open-source OCR engine, running locally. Free,
  private and unlimited, but weaker on tables and complex layouts.
"""

import base64
import logging
from pathlib import Path

import anyio

from app.config import settings
from app.parsing import IMAGE_EXTENSIONS

log = logging.getLogger(__name__)

OCR_PROMPT = """Transcribe all text in this image exactly as written.
- Preserve the reading order and paragraph breaks.
- Render tables as rows with " | " between cells.
- Do not summarize, translate, correct, or add commentary. Output only the transcribed text.
- If the image has no text, output nothing."""

_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


class OCRError(RuntimeError):
    pass


async def ocr_document(filename: str, data: bytes, pages: list[int], provider: str | None = None) -> dict[int, str]:
    """Return {page_number: text} for the requested pages (or the image itself)."""
    provider = provider or settings.ocr_provider
    ext = Path(filename).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        images = {1: (data, _MIME[ext])}
    elif ext == ".pdf":
        rendered = await anyio.to_thread.run_sync(render_pdf_pages, data, pages, settings.ocr_dpi)
        images = {p: (png, "image/png") for p, png in rendered.items()}
    else:
        raise OCRError(f"OCR not supported for {ext}")

    results: dict[int, str] = {}
    for page, (image, mime) in images.items():
        log.info("OCR %s page %s via %s", filename, page, provider)
        results[page] = await ocr_image(image, mime, provider)
    return results


def render_pdf_pages(data: bytes, pages: list[int], dpi: int) -> dict[int, bytes]:
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return {p: doc[p - 1].get_pixmap(dpi=dpi).tobytes("png") for p in pages}


async def ocr_image(image: bytes, mime: str, provider: str) -> str:
    if provider == "tesseract":
        return await anyio.to_thread.run_sync(_tesseract, image)
    if provider == "gemini":
        try:
            return await _vision_llm(image, mime)
        except Exception as e:
            # Rate limited / overloaded / no key: a slightly worse transcript
            # beats a failed document. Tesseract runs locally and always works.
            log.warning("Gemini OCR failed (%s); falling back to Tesseract", e)
            return await anyio.to_thread.run_sync(_tesseract, image)
    raise OCRError(f"Unknown OCR provider: {provider}")


def _tesseract(image: bytes) -> str:
    import io

    import pytesseract
    from PIL import Image

    return pytesseract.image_to_string(Image.open(io.BytesIO(image)))


async def _vision_llm(image: bytes, mime: str) -> str:
    if settings.llm_provider == "fake":
        raise OCRError("OCR_PROVIDER=gemini needs a real LLM (LLM_PROVIDER=fake); use OCR_PROVIDER=tesseract")
    from app.llm import _get_client  # same OpenAI-compatible client as chat

    data_url = f"data:{mime};base64,{base64.b64encode(image).decode()}"
    resp = await _get_client().chat.completions.create(
        model=settings.ocr_model,
        temperature=0,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": OCR_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
    )
    return resp.choices[0].message.content or ""
