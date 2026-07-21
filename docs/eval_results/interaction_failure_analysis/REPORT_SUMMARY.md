# Model Interaction Failure Summary

## Result in brief

The four conditions fail in different ways. Gemini usually reaches its
32-step limit; the three direct-MCP conditions usually stop with `DONE` while
fields remain incomplete. At field level, Gemini is split between unattempted
and attempted-but-blank fields, while unattempted fields dominate the other
conditions.

The strongest shared stress tests are 14 non-dropdown questions missed by all
four conditions. The hardest pooled forms are `club_event_planning` and
`purchase_request` (27.5% field correctness), followed by `club_application`
(32.5%), `internship_app` (35.42%), and `data_annotation` (36.11%).

## Failure matrix

| Model | Missed fields | Dominant behavioral failure | Count | Share |
|---|---:|---|---:|---:|
| Gemini 3.5 Flash | 113 | Attempted but blank | 44 | 38.94% |
| OpenCUA direct-MCP | 119 | Not attempted | 94 | 78.99% |
| Qwen3 Text | 161 | Not attempted | 114 | 70.81% |
| Qwen3-VL | 126 | Not attempted | 97 | 76.98% |

| Model | Incomplete trials | Dominant stop reason | Count | Share |
|---|---:|---|---:|---:|
| Gemini 3.5 Flash | 38 | Max steps exceeded | 28 | 73.68% |
| OpenCUA direct-MCP | 35 | DONE while incomplete | 29 | 82.86% |
| Qwen3 Text | 42 | DONE while incomplete | 33 | 78.57% |
| Qwen3-VL | 39 | DONE while incomplete | 29 | 74.36% |

Dropdown container text is excluded from the behavioral failure label because
the stored selected-state audit shows that it is a verifier error, not a proven
model error.

## Dropdown and widget findings

All 100 dropdown targets were scored wrong by the original verifier. A
stored-artifact audit now confirms 79 correct selections and no wrong
selection. The remaining 21 are unresolved because their bounded snapshot
excerpt does not expose the dropdown state. The minimum confirmed rates are
Gemini 100% (25/25, manually reviewed post-action screenshots), Qwen3-VL 84%
(21/25), OpenCUA 80% (20/25), and Qwen3 Text 52% (13/25). These are lower
bounds, not point estimates for the unresolved cases.

On the 25 forms without dropdowns, exact full-fill rates are OpenCUA 60%,
Gemini 48%, Qwen3-VL 44%, and Qwen3 Text 32%. On the 25 dropdown forms, the
artifact-confirmed full-fill lower bounds are OpenCUA 13/25, Gemini 6/25,
Qwen3-VL 11/25, and Qwen3 Text 1/25. If every unresolved dropdown were correct,
the corresponding upper bounds would be 15/25, 6/25, 12/25, and 3/25.

Excluding dropdowns, time fields have the highest shared failure rate (53.45%).
Gemini also misses 22 of 45 paragraph fields, and its failure rate rises from
3.73% in the first form third to 54.48% in the final third.

## Action findings

Primary counts include only state-changing interactions. In incomplete trials,
exact adjacent repetition is 20.19% for OpenCUA, 29.15% for Qwen3 Text, and
28.20% for Qwen3-VL; it is 0% in every direct-MCP full-fill subset. Gemini's
problem is broader navigation inefficiency: incomplete trials use 5.33 actions
per verified field versus 2.93 in full fills.

The normalization matters most for Qwen3-VL, where 122 snapshots and one wait
are removed from 714 raw calls. The raw-to-normalized audit remains available.

## Submission and ideal-reference checks

The canonical 200 trials made zero submit attempts, so confirmation pages did
not affect them. In the historical baseline, all 519 submitted trials ended
with final-page zero. Pre-submit logs recover 512 nonzero scores and identify
153 verified-perfect submissions; seven remain unrecovered.

The ideal dataset contains 300/300 successful scripted submissions. Its median
run takes 56.44 seconds and 10 normalized interactions. Matched median model
time ratios are 11.78× ideal for Gemini, 5.21× for OpenCUA, 3.62× for Qwen3
Text, and 3.86× for Qwen3-VL. Action ratios alone are not rankings because
incomplete trials can stop early and calls differ in granularity.

## Paper-safe interpretation

- Cite exact completion directly for forms without dropdowns; use the audited
  lower/upper bounds for forms with dropdowns.
- Treat the failure matrix as descriptive final-state evidence, not causal
  attribution.
- Report normalized interactions together with completion and time.
- Separate capability on usable trials from operational reliability.

The verifier now reads `aria-selected`, asserts the option after selection, and
preserves a verified value when a later form page makes the field invisible.
Remaining recommendations are a completion check before `DONE`, progress-aware
loop limits, and per-action before/after field logging.
See `report.html` for the supporting tables and `METHODS_AND_LIMITATIONS.md` for
definitions and caveats.
