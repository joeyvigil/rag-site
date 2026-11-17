"""Compare OCR providers on 'scanned' copies of the employee handbook.

  docker compose exec api python -m eval.ocr_eval
  docker compose exec api python -m eval.ocr_eval --providers tesseract   # offline only

We know the true text of every handbook page, so we can turn the PDF into
image-only "scans" (a clean one, and a degraded one: low resolution, blur,
heavy JPEG compression), push them through the real pipeline
(parse_document -> ocr_document), and measure:

- CER (character error rate): character edits needed to turn the OCR output
  into the truth, divided by the truth's length. Lower is better.
- WER (word error rate): the same, counted in words.
- Evidence survival: of the golden-dataset evidence strings on these pages,
  how many appear verbatim in the OCR text. This is what matters for RAG: a
  typo inside "accrue 22 days" breaks keyword search on exactly that fact.
- Seconds per page.
"""

import argparse
import asyncio
import io
import json
import time
from datetime import UTC, datetime

from app.ocr import ocr_document
from app.parsing import parse_document
from eval.build_corpus import CORPUS_DIR
from eval.metrics import normalize
from eval.run import DATASET_PATH, RESULTS_DIR

SOURCE = CORPUS_DIR / "employee_handbook.pdf"


def make_scan(pdf: bytes, degraded: bool) -> bytes:
    """Render every page to an image and build an image-only PDF (no text layer)."""
    import pymupdf
    from PIL import Image, ImageFilter

    out = pymupdf.open()
    with pymupdf.open(stream=pdf, filetype="pdf") as src:
        for page in src:
            pix = page.get_pixmap(dpi=90 if degraded else 200)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
            if degraded:
                img = img.filter(ImageFilter.GaussianBlur(0.8))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=30 if degraded else 90)
            new = out.new_page(width=page.rect.width, height=page.rect.height)
            new.insert_image(new.rect, stream=buf.getvalue())
    return out.tobytes()


def edit_distance(a: list | str, b: list | str) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def score(truth: str, ocr: str, evidence: list[str]) -> dict:
    t, o = normalize(truth), normalize(ocr)
    tw, ow = t.split(), o.split()
    found = [e for e in evidence if normalize(e) in o]
    return {
        "cer": edit_distance(t, o) / max(1, len(t)),
        "wer": edit_distance(tw, ow) / max(1, len(tw)),
        "evidence_found": len(found),
        "evidence_total": len(evidence),
        "evidence_missing": [e for e in evidence if e not in found],
    }


async def main(args) -> None:
    original = SOURCE.read_bytes()
    truth = {s.page: s.text for s in parse_document(SOURCE.name, original).sections}
    truth_all = "\n".join(truth[p] for p in sorted(truth))
    evidence = sorted({
        e for q in map(json.loads, DATASET_PATH.read_text().splitlines())
        for e in q["evidence"] if normalize(e) in normalize(truth_all)
    })
    print(f"Ground truth: {len(truth)} pages, {len(truth_all)} chars, {len(evidence)} evidence strings\n")

    rows = []
    for variant in ("clean", "degraded"):
        scan = make_scan(original, degraded=variant == "degraded")
        parsed = parse_document("scan.pdf", scan)
        assert not parsed.sections and parsed.ocr_pages == sorted(truth), "scan should have no text layer"
        for provider in args.providers:
            start = time.perf_counter()
            try:
                pages = await ocr_document("scan.pdf", scan, parsed.ocr_pages, provider=provider)
            except Exception as e:
                print(f"{provider:10} {variant:9} ERROR {e}")
                continue
            elapsed = time.perf_counter() - start
            text = "\n".join(pages[p] for p in sorted(pages))
            row = {"provider": provider, "variant": variant, "sec_per_page": elapsed / len(pages),
                   **score(truth_all, text, evidence)}
            rows.append(row)
            print(f"{provider:10} {variant:9} CER {row['cer']:.3f}  WER {row['wer']:.3f}  "
                  f"evidence {row['evidence_found']}/{row['evidence_total']}  {row['sec_per_page']:.1f}s/page"
                  + (f"  missing: {row['evidence_missing']}" if row["evidence_missing"] else ""))

    out_dir = RESULTS_DIR / "ocr"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}_ocr.json"
    path.write_text(json.dumps(rows, indent=2))
    from eval.run import _match_owner
    _match_owner(out_dir, path)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--providers", nargs="+", default=["tesseract", "gemini"])
    asyncio.run(main(p.parse_args()))
