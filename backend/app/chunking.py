"""Recursive character chunking.

Strategy: try to split on the "biggest" natural boundary first (blank line,
then newline, then sentence end, then space) and only fall back to a hard
character cut when a piece is still too long. Pieces are then packed back
together up to `size` characters, with `overlap` characters repeated between
neighbouring chunks so a fact that straddles a boundary isn't lost.

Chunks never cross a page boundary, which keeps page citations exact.
"""

from dataclasses import dataclass

from app.parsing import Section

SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]


@dataclass
class ChunkData:
    index: int
    text: str
    page: int | None

    @property
    def token_estimate(self) -> int:
        return max(1, len(self.text) // 4)  # ~4 chars per token for English


def chunk_sections(
    sections: list[Section], size: int, overlap: int, repeat_headers: bool = True
) -> list[ChunkData]:
    """Split sections into chunks.

    Sections with a `header` (CSV tables): if `repeat_headers`, every chunk
    starts with the header row, so each chunk is self-describing. Otherwise
    the header is only at the top of the first chunk (naive behaviour).
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks: list[ChunkData] = []
    for section in sections:
        if section.header and repeat_headers:
            prefix = section.header.strip() + "\n"
            budget = max(size - len(prefix), overlap + 1)
            texts = [prefix + t for t in split_text(section.text, budget, overlap)]
        elif section.header:
            texts = split_text(section.header + "\n" + section.text, size, overlap)
        else:
            texts = split_text(section.text, size, overlap)
        for text in texts:
            chunks.append(ChunkData(index=len(chunks), text=text, page=section.page))
    return chunks


def split_text(text: str, size: int, overlap: int) -> list[str]:
    text = _normalize_whitespace(text)
    if not text:
        return []
    pieces = _split_recursive(text, size, SEPARATORS)
    return _merge(pieces, size, overlap)


def _normalize_whitespace(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.replace("\r\n", "\n").split("\n")]
    out: list[str] = []
    blank = 0
    for line in lines:
        if line:
            blank = 0
            out.append(line)
        else:
            blank += 1
            if blank == 1:
                out.append("")  # keep at most one blank line (paragraph break)
    return "\n".join(out).strip()


def _split_recursive(text: str, size: int, separators: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    if not separators:  # no boundary left: hard cut
        return [text[i : i + size] for i in range(0, len(text), size)]

    sep, rest = separators[0], separators[1:]
    if sep not in text:
        return _split_recursive(text, size, rest)

    parts = text.split(sep)
    pieces: list[str] = []
    for i, part in enumerate(parts):
        piece = part + sep if i < len(parts) - 1 else part  # keep the separator
        if not piece:
            continue
        if len(piece) <= size:
            pieces.append(piece)
        else:
            pieces.extend(_split_recursive(piece, size, rest))
    return pieces


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    window: list[str] = []
    window_len = 0
    for piece in pieces:
        if window and window_len + len(piece) > size:
            chunks.append("".join(window).strip())
            # Drop pieces from the front until only ~`overlap` chars remain
            # (and the next piece fits).
            while window and (window_len > overlap or window_len + len(piece) > size):
                window_len -= len(window.pop(0))
        window.append(piece)
        window_len += len(piece)
    if window:
        chunks.append("".join(window).strip())
    return [c for c in chunks if c]
