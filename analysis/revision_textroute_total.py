"""INDICON 2026 revision (Reviewer 2, end-to-end cost): text route vs
image route on input + billed output tokens, same pairs as the input-only
ratios in Sec. IV-C (unreduced vs text_compressed, non-truncated).

Billed output: Groq = completion_tokens (Qwen3.6 reasoning is inside the
returned text, so it is counted there); Gemini = candidates + thoughts,
with thoughts recovered from the cached raw usage_metadata. Pairs whose
Gemini thinking count cannot be recovered are reported and excluded.
Zero API calls.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict

import numpy as np

RATES = {"qwen/qwen3.6-27b": (0.60, None), "gemini-2.5-flash": (0.30, 2.50)}
RNG = np.random.default_rng(0)


def gemini_thoughts() -> dict[str, int]:
    out = {}
    for (r,) in sqlite3.connect("results/cache.sqlite").execute(
        "SELECT response_json FROM cache WHERE model='gemini-2.5-flash' AND created_at < '2026-10-18'"
    ):
        rr = json.loads(r)
        um = (rr.get("raw") or {}).get("usage_metadata")
        if um is not None:
            # Keyed by response text, so only calls before revision 2 (18 Oct 2026): later
            # calls with the same short answer would overwrite them. Revision-2 rows log
            # thoughts_tokens directly.
            out[rr.get("text") or ""] = um.get("thoughts_token_count") or 0
    return out


def boot(x):
    x = np.array(x)
    m = [RNG.choice(x, len(x)).mean() for _ in range(10000)]
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main() -> None:
    th = gemini_thoughts()
    rows = (
        sqlite3.connect("results/results.sqlite")
        .execute(
            "SELECT dataset, image_path, method, model, input_tokens, output_tokens, response_text FROM runs "
            "WHERE method IN ('unreduced','text_compressed') AND truncated=0 AND dataset='docvqa' "
            "AND model IN ('qwen/qwen3.6-27b','gemini-2.5-flash')"
        )
        .fetchall()
    )
    by = defaultdict(dict)
    for ds, ip, m, mo, i, o, txt in rows:
        extra = 0
        if mo == "gemini-2.5-flash":
            extra = th.get(txt or "")
            if extra is None:
                continue
        by[(ip, mo)][m] = (i, (o or 0) + extra)
    for mo in ("qwen/qwen3.6-27b", "gemini-2.5-flash"):
        pairs = [v for (ip, m), v in by.items() if m == mo and "unreduced" in v and "text_compressed" in v]
        if not pairs:
            continue
        ui = [p["unreduced"][0] for p in pairs]
        uo = [p["unreduced"][1] for p in pairs]
        ti = [p["text_compressed"][0] for p in pairs]
        to = [p["text_compressed"][1] for p in pairs]
        r_in = [a / b for a, b in zip(ui, ti)]
        r_tot = [(a + c) / (b + d) for a, b, c, d in zip(ui, ti, uo, to)]
        print(
            f"{mo}: n={len(pairs)}  mean in {np.mean(ui):.0f}->{np.mean(ti):.0f}  out {np.mean(uo):.0f}->{np.mean(to):.0f}"
        )
        print("   input-only ratio  mean %.2f  CI [%.2f, %.2f]" % boot(r_in))
        print("   total-token ratio mean %.2f  CI [%.2f, %.2f]" % boot(r_tot))


if __name__ == "__main__":
    main()
