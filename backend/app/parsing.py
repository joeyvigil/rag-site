"""Turn uploaded files into plain text sections.

Each Section keeps its page number (when the format has pages) so answers
can cite "file.pdf, page 4". Parsing quality caps retrieval quality: if the
text extracted here is garbled, no amount of clever retrieval fixes it.
"""

import io
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown", ".csv", ".json", ".html", ".htm"}


class ParseError(ValueError):
    pass


@dataclass
class Section:
    text: str
    page: int | None = None


@dataclass
class ParsedDocument:
    sections: list[Section]
    num_pages: int | None = None


def parse_document(filename: str, data: bytes) -> ParsedDocument:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ParseError(f"Unsupported file type: {ext or '(none)'}")

    if ext == ".pdf":
        parsed = _parse_pdf(data)
    elif ext == ".docx":
        parsed = ParsedDocument([Section(_parse_docx(data))])
    elif ext in {".html", ".htm"}:
        parsed = ParsedDocument([Section(_parse_html(data))])
    else:
        parsed = ParsedDocument([Section(_decode_text(data))])

    parsed.sections = [s for s in parsed.sections if s.text.strip()]
    if not parsed.sections:
        hint = " (scanned PDF? OCR isn't supported yet)" if ext == ".pdf" else ""
        raise ParseError(f"No extractable text found{hint}")
    return parsed


def _parse_pdf(data: bytes) -> ParsedDocument:
    import pymupdf

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:  # corrupt / encrypted files
        raise ParseError(f"Could not open PDF: {e}") from e
    with doc:
        sections = [Section(page.get_text("text"), page=i + 1) for i, page in enumerate(doc)]
        return ParsedDocument(sections, num_pages=doc.page_count)


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
