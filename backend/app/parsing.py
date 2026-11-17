"""Turn uploaded files into plain text sections.

Each Section keeps its page number (when the format has pages) so answers
can cite "file.pdf, page 4". Parsing quality caps retrieval quality: if the
text extracted here is garbled, no amount of clever retrieval fixes it.
"""

import io
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".md", ".markdown", ".csv", ".json", ".html", ".htm",
} | IMAGE_EXTENSIONS

# A PDF page with less text than this is treated as scanned and sent to OCR.
MIN_PAGE_CHARS = 20


class ParseError(ValueError):
    pass


@dataclass
class Section:
    text: str
    page: int | None = None
    # For tables (CSV): the header row. The chunker can repeat it on every
    # chunk so rows are never separated from their column names.
    header: str | None = None


@dataclass
class ParsedDocument:
    sections: list[Section]
    num_pages: int | None = None
    # Pages (1-based) with no text layer that need OCR. For image files: [1].
    ocr_pages: list[int] = field(default_factory=list)


def parse_document(filename: str, data: bytes) -> ParsedDocument:
    """Extract text. Doesn't do OCR itself (that's an async API call); it reports
    which pages need it in `ocr_pages`, and the worker runs OCR on them."""
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ParseError(f"Unsupported file type: {ext or '(none)'}")

    if ext == ".pdf":
        parsed = _parse_pdf(data)
    elif ext in IMAGE_EXTENSIONS:
        return ParsedDocument([], num_pages=1, ocr_pages=[1])
    elif ext == ".docx":
        parsed = ParsedDocument([Section(_parse_docx(data))])
    elif ext in {".html", ".htm"}:
        parsed = ParsedDocument([Section(_parse_html(data))])
    elif ext == ".csv":
        parsed = ParsedDocument([_parse_csv(data)])
    else:
        parsed = ParsedDocument([Section(_decode_text(data))])

    parsed.sections = [s for s in parsed.sections if s.text.strip()]
    if not parsed.sections and not parsed.ocr_pages:
        raise ParseError("No extractable text found")
    return parsed


def _parse_pdf(data: bytes) -> ParsedDocument:
    import pymupdf

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:  # corrupt / encrypted files
        raise ParseError(f"Could not open PDF: {e}") from e
    with doc:
        sections, ocr_pages = [], []
        for i, page in enumerate(doc):
            text = page.get_text("text")
            # Scanned page = (almost) no text layer, but an image. A short page
            # that's genuinely text (e.g. a title page) keeps its text, and a
            # blank page is skipped instead of wasting an OCR call.
            if len(text.strip()) < MIN_PAGE_CHARS and page.get_images():
                ocr_pages.append(i + 1)
            elif text.strip():
                sections.append(Section(text, page=i + 1))
        return ParsedDocument(sections, num_pages=doc.page_count, ocr_pages=ocr_pages)


def _parse_csv(data: bytes) -> Section:
    text = _decode_text(data).strip()
    header, _, rows = text.partition("\n")
    return Section(rows, header=header) if rows else Section(header)


def _parse_docx(data: bytes) -> str:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(io.BytesIO(data))
    lines: list[str] = []
    # Walk the body in order so tables stay next to the paragraphs around them.
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(child, document)
            text = para.text.strip()
            if not text:
                continue
            style = (para.style.name if para.style is not None else "") or ""
            if style.startswith("Heading"):
                level = style.removeprefix("Heading").strip()
                hashes = "#" * int(level) if level.isdigit() else "#"
                text = f"{hashes} {text}"
            lines.append(text)
        elif tag == "tbl":
            for row in Table(child, document).rows:
                cells = [c.text.strip() for c in row.cells]
                lines.append(" | ".join(cells))
    return "\n\n".join(lines)


def _parse_html(data: bytes) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(data, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text("\n")


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")
