"""Mode-selection optimizer — the paper's actual savings mechanism under the
characterization framing (paper/draft.md §9.5, §10): on providers where cost
is fixed/tiered rather than content-scaled, the only real lever is picking
the cheapest DISCRETE mode that still clears a fidelity floor, not
pixel-level region selection.

Built from real, provider-verified tier data only:
- Gemini 2.5 Flash `media_resolution`: "low" -> 74 tokens, unset/default ->
  258 tokens (real-verified, tests/fixtures/gemini_token_cost_2026-08-16.json
  + paper/draft.md §3.5/§7/§9.3). No other media_resolution values (e.g.
  "medium", "high") have been empirically tested — do not add tiers here
  that haven't been measured; extend GEMINI_MEDIA_RESOLUTION_TOKENS only
  after a real call confirms a new value.
- Groq/qwen3.6-27b: `detail` has zero measured effect (§9.3) - there is no
  mode lever on this provider, so ModeOptimizer.select() for Groq always
  returns the single available "unset" mode. This is a real, verified
  finding, not an oversight.

This module intentionally does NOT touch pixelpath/regions.py or
packing.py's knapsack machinery - that code remains valid for a
content-sensitive-pricing provider, which the two providers tested are not
(see packing.py's module docstring). Mode selection is a separate,
independent lever.
"""

from __future__ import annotations

from dataclasses import dataclass

# Real, provider-verified token cost per Gemini `media_resolution` setting.
# Only entries with an actual measured call behind them belong here.
GEMINI_MEDIA_RESOLUTION_TOKENS: dict[str, int] = {
    "low": 74,
    "unset": 258,  # also covers "high"/"default" — not separately distinguished from unset by measurement
}

# Groq/qwen3.6-27b: `detail` parameter measured to have zero effect on cost
# (§9.3) - only one real mode exists.
GROQ_DETAIL_TOKENS: dict[str, int] = {
    "unset": 1827,  # value from the specific sample tested; cost is flat but not provider-universal like Gemini's 258
}


@dataclass
class ModeOption:
    mode: str
    token_cost: int
    fidelity: float | None  # None = not yet measured for this mode on this sample/category


@dataclass
class ModeSelection:
    chosen: ModeOption
    rejected: list[ModeOption]
    reason: str


class ModeOptimizer:
    """Picks the cheapest available discrete mode whose measured fidelity is
    >= fidelity_floor. Requires real fidelity measurements per mode - does
    NOT assume a mode's fidelity from its token cost alone (that would be
    the same "content should correlate with cost" assumption this paper's
    entire finding falsifies).
    """

    def __init__(self, mode_token_costs: dict[str, int]):
        if not mode_token_costs:
            raise ValueError("mode_token_costs must have at least one entry")
        self.mode_token_costs = mode_token_costs

    def select(self, fidelity_by_mode: dict[str, float], fidelity_floor: float) -> ModeSelection:
        """fidelity_by_mode: real measured fidelity (e.g. ANLS) per mode name,
        for the sample/category being routed. Every key must exist in
        self.mode_token_costs - this optimizer does not guess costs for
        unmeasured modes.
        """
        unknown = set(fidelity_by_mode) - set(self.mode_token_costs)
        if unknown:
            raise ValueError(f"fidelity_by_mode has modes with no known token cost: {unknown}")

        options = [
            ModeOption(mode=m, token_cost=self.mode_token_costs[m], fidelity=fidelity_by_mode[m])
            for m in fidelity_by_mode
        ]
        options.sort(key=lambda o: o.token_cost)

        passing = [o for o in options if o.fidelity is not None and o.fidelity >= fidelity_floor]
        if passing:
            chosen = passing[0]  # cheapest passing mode
            rejected = [o for o in options if o is not chosen]
            reason = (
                f"cheapest mode ({chosen.mode}, {chosen.token_cost} tokens) with fidelity "
                f"{chosen.fidelity:.3f} >= floor {fidelity_floor:.3f}"
            )
        else:
            # nothing clears the floor - fall back to the highest-fidelity mode measured,
            # not the cheapest, since silently serving a below-floor answer is worse than
            # paying for the mode most likely to be right.
            measured = [o for o in options if o.fidelity is not None]
            if not measured:
                raise ValueError("fidelity_by_mode had no measured (non-None) fidelity values")
            chosen = max(measured, key=lambda o: o.fidelity)
            rejected = [o for o in options if o is not chosen]
            reason = (
                f"no mode cleared fidelity floor {fidelity_floor:.3f}; fell back to "
                f"highest-measured-fidelity mode ({chosen.mode}, fidelity {chosen.fidelity:.3f})"
            )

        return ModeSelection(chosen=chosen, rejected=rejected, reason=reason)
