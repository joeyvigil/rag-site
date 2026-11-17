import io

import pytest

from app.parsing import ParseError, parse_document


def test_txt():
    parsed = parse_document("notes.txt", "Hello ünïcode".encode())
    assert parsed.sections[0].text == "Hello ünïcode"


def test_html_strips_scripts():
    html = b"<html><script>evil()</script><body><h1>Title</h1><p>Body text</p></body></html>"
    text = parse_document("page.html", html).sections[0].text
    assert "Title" in text and "Body text" in text and "evil" not in text


def test_pdf_pages():
    import pymupdf

    doc = pymupdf.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), f"Content of page {i + 1}")
    data = doc.tobytes()

    parsed = parse_document("report.pdf", data)
    assert parsed.num_pages == 3
    assert [s.page for s in parsed.sections] == [1, 2, 3]
    assert "Content of page 2" in parsed.sections[1].text


def test_docx_headings_and_tables():
    import docx

    d = docx.Document()
    d.add_heading("Intro", level=1)
    d.add_paragraph("First paragraph.")
    table = d.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Name"
    table.rows[0].cells[1].text = "Value"
    buf = io.BytesIO()
    d.save(buf)

    text = parse_document("doc.docx", buf.getvalue()).sections[0].text
    assert text.splitlines()[0] == "# Intro"
    assert "First paragraph." in text
    assert "Name | Value" in text


def test_empty_pdf_raises():
    import pymupdf

    doc = pymupdf.open()
    doc.new_page()  # blank page, no text
    with pytest.raises(ParseError, match="No extractable text"):
        parse_document("scan.pdf", doc.tobytes())


def test_unsupported_extension():
    with pytest.raises(ParseError):
        parse_document("image.png", b"\x89PNG")
