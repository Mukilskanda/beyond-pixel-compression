"""Offline mechanism check for `textpath.compress.compress_pages` (cross-page
line dedup) — the follow-up to `pdf_batched_text_route_check.py`'s negative
result, which found batching pages into one call doesn't win because text
calls have no large fixed cost to amortize away. Cross-page dedup targets a
different, real cost: a genuinely repeated header/footer/watermark line that
survives on every page today, because `compress_markdown` resets its `seen`
set per page and DocVQA (independent single-page documents) never had one
to find in the first place.

**What this script can and cannot tell you — read before trusting a number
out of it.** There is no cached real multi-page-same-document dataset in
this project. DocVQA's own schema was checked for one — it has `docId` and
`ucsf_document_page_no` fields that looked promising — but a stream scan of
15,000 real validation rows found ZERO docIds with more than one distinct
page number (see this module's git history / the session that added this
comment for the exact scan). DocVQA's per-question images are apparently
always a single page per docId in this split, not a real multi-page corpus
this project can extract from for free. ChartQA is single-chart images by
construction, same gap. So this remains a SYNTHETIC-fixture check, not a
real-document measurement, and that's a confirmed absence, not an
unexplored option.

Given that, this version doesn't report one arbitrary made-up scenario —
it SWEEPS the two things that actually determine whether cross-page dedup
helps (how much boilerplate repeats, and how much genuine content
surrounds it), so the output is a sensitivity range, not a single invented
data point. This can prove:
  - `compress_pages` is *correct* on real extracted text (via
    `textpath.pdf.extract_pdf_pages`, real PyMuPDF extraction, not mocked).
  - How the token savings SCALE with boilerplate density and page count,
    which is a real, general mathematical property of the dedup mechanism
    (more repeated lines relative to content -> more savings; more pages
    -> more savings, converging as boilerplate amortizes) — this part is
    NOT dataset-dependent, it holds for any document with this structure.
It still CANNOT prove:
  - How much boilerplate REAL documents actually contain (invoices,
    reports, contracts, forms this project has no samples of) — that's an
    empirical fact about the world, not something a sweep over invented
    ratios can supply. The sweep tells you the payoff AT a given density;
    it doesn't tell you which density is realistic.
  - Real billed tokens (tiktoken is an offline proxy, not the real
    Gemini/Groq tokenizer, same caveat as `pdf_batched_text_route_check.py`).
  - Fidelity impact of dropping a repeated line — `compress_pages`'s own
    docstring names the exact-string (not semantic) over-deletion risk.
So: this is a mechanism + sensitivity proof, not a real-world savings
claim. Report it as such, with the sweep, not a single headline ratio.

Usage:
    python experiments/cross_page_dedup_check.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import fitz
import tiktoken

from textpath.compress import compress_markdown, compress_pages
from textpath.pdf import extract_pdf_pages

_ENC = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


# Three named boilerplate densities spanning what's plausible for a
# template document — not claimed to be THE real distribution (unknown,
# see module docstring), but a spread from "light" to "heavy" so a reader
# can see how the payoff moves rather than trusting one invented middle
# value.
BOILERPLATE_PROFILES = {
    "light": ["CONFIDENTIAL"],
    "medium": ["ACME CORP — QUARTERLY COMPLIANCE REPORT", "CONFIDENTIAL — DO NOT DISTRIBUTE"],
    "heavy": [
        "ACME CORP — QUARTERLY COMPLIANCE REPORT",
        "Prepared by the Office of Regulatory Affairs",
        "CONFIDENTIAL — DO NOT DISTRIBUTE OUTSIDE ACME CORP",
        "This document is subject to the Acme Corp data retention policy v4.2",
    ],
}

# Two body lengths — a short-form page (a single line of real content) and
# a long-form page (several lines) — since the dedup payoff is a function
# of boilerplate-to-content ratio, not boilerplate alone.
BODY_PROFILES = {
    "short_body": lambda i: [f"Section {i + 1}: revenue this quarter was ${1000 + i * 137}.42."],
    "long_body": lambda i: [
        f"Section {i + 1}: revenue this quarter was ${1000 + i * 137}.42, up from last quarter.",
        f"Operating costs for the period totaled ${500 + i * 41}.18, a change of {i}.3 percent.",
        f"Headcount at the end of the period was {120 + i} full-time employees across all divisions.",
        f"The board approved item {i + 1} of the quarterly agenda without dissent.",
    ],
}


def _build_synthetic_multi_page_pdf(path: Path, n_pages: int, boilerplate: list[str], body_fn) -> None:
    """Boilerplate appears once per page (as a header), not twice — putting
    it at both top and bottom of the SAME page would let within-page dedup
    catch it too, muddying the cross-page-only effect this check exists to
    isolate."""
    doc = fitz.open()
    for i in range(n_pages):
        body_lines = body_fn(i)
        text = "\n\n".join(boilerplate + body_lines)
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def run(n_pages: int, profile: str = "medium", body: str = "short_body") -> dict:
    boilerplate = BOILERPLATE_PROFILES[profile]
    body_fn = BODY_PROFILES[body]

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "synthetic_multi_page.pdf"
        _build_synthetic_multi_page_pdf(pdf_path, n_pages, boilerplate, body_fn)

        pages = extract_pdf_pages(pdf_path)
        raw_texts = [p.text for p in pages]

    no_dedup_tokens = sum(_count_tokens(t) for t in raw_texts)

    within_page_only = [compress_markdown(t) for t in raw_texts]
    within_page_tokens = sum(_count_tokens(t) for t in within_page_only)

    cross_page = compress_pages(raw_texts)
    cross_page_tokens = sum(_count_tokens(t) for t in cross_page)

    return {
        "n_pages": n_pages,
        "profile": profile,
        "body": body,
        "no_dedup_tokens": no_dedup_tokens,
        "within_page_dedup_tokens": within_page_tokens,
        "cross_page_dedup_tokens": cross_page_tokens,
        "cross_page_vs_none": round(no_dedup_tokens / cross_page_tokens, 2) if cross_page_tokens else None,
        "cross_page_vs_within_page": (
            round(within_page_tokens / cross_page_tokens, 2) if cross_page_tokens else None
        ),
    }


def sweep() -> list[dict]:
    """The actual deliverable: a grid over boilerplate density x body
    length x page count, not one cherry-picked scenario."""
    results = []
    for profile in BOILERPLATE_PROFILES:
        for body in BODY_PROFILES:
            for n in (2, 4, 8, 16):
                results.append(run(n, profile=profile, body=body))
    return results


if __name__ == "__main__":
    import json

    rows = sweep()
    for r in rows:
        print(json.dumps(r))

    print("\n--- summary: cross_page_vs_none at n_pages=16, by profile/body ---")
    for row in rows:
        if row["n_pages"] == 16:
            print(f"  {row['profile']:6s} / {row['body']:10s}: {row['cross_page_vs_none']}x")

    print(
        "\nCAVEAT: these ratios are a function of the invented boilerplate/body "
        "profiles above, not a measurement of any real document corpus — see "
        "module docstring. No real multi-page-same-document data was "
        "available offline to check this against (DocVQA's docId grouping "
        "was checked and found not to contain multi-page documents in a "
        "15,000-row scan of the validation split)."
    )
