from app.chunking import chunk_sections, split_text
from app.parsing import Section


def test_short_text_is_single_chunk():
    assert split_text("Hello world.", size=100, overlap=10) == ["Hello world."]


def test_chunks_respect_size_and_overlap():
    sentences = [f"Sentence number {i} has some words in it." for i in range(200)]
    text = " ".join(sentences)
    chunks = split_text(text, size=300, overlap=60)

    assert len(chunks) > 1
    # A chunk can exceed size by at most the overlap carried over.
    assert all(len(c) <= 300 + 60 for c in chunks)
    # Consecutive chunks share some text (the overlap).
    for a, b in zip(chunks, chunks[1:]):
        assert any(s in a and s in b for s in sentences), "expected overlapping sentence"
    # No content is lost.
    for s in sentences:
        assert any(s in c for c in chunks)


def test_prefers_paragraph_boundaries():
    para_a = "A" * 80
    para_b = "B" * 80
    chunks = split_text(f"{para_a}\n\n{para_b}", size=100, overlap=0)
    assert chunks == [para_a, para_b]


def test_hard_cut_when_no_separators():
    chunks = split_text("x" * 250, size=100, overlap=0)
    assert [len(c) for c in chunks] == [100, 100, 50]


def test_chunks_keep_page_numbers_and_global_index():
    sections = [Section("Page one text.", page=1), Section("Page two text.", page=2)]
    chunks = chunk_sections(sections, size=100, overlap=10)
    assert [(c.index, c.page) for c in chunks] == [(0, 1), (1, 2)]


def test_csv_header_repeated_in_every_chunk():
    rows = "\n".join(f"SKU-{i},Widget {i},{i}.00" for i in range(100))
    section = Section(rows, header="sku,name,price")
    chunks = chunk_sections([section], size=300, overlap=0, repeat_headers=True)
    assert len(chunks) > 1
    assert all(c.text.startswith("sku,name,price\n") for c in chunks)
    assert all(len(c.text) <= 300 for c in chunks)


def test_csv_header_only_first_chunk_when_disabled():
    rows = "\n".join(f"SKU-{i},Widget {i},{i}.00" for i in range(100))
    chunks = chunk_sections([Section(rows, header="sku,name,price")], size=300, overlap=0, repeat_headers=False)
    assert chunks[0].text.startswith("sku,name,price\n")
    assert not chunks[1].text.startswith("sku,name,price")


def test_whitespace_is_normalized():
    assert split_text("a   b\n\n\n\n c", size=100, overlap=0) == ["a b\n\nc"]
