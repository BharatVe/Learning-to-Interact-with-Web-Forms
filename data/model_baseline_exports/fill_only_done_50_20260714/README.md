# Fill-only/DONE 50-form comparison

Generated from compact per-trial `summary.json` artifacts. Raw screenshots, videos, and model traces are intentionally excluded from Git.

No form was submitted. `full_fill_successes` means the visible fields were verified as correct before returning DONE.

| Model | Usable forms | Full fills | Field correctness | Model actions |
|---|---:|---:|---:|---:|
| Gemini 3.5 Flash | 50/50 | 12/50 | 296/409 (72.37%) | 1382 |
| OpenCUA direct-MCP | 50/50 | 15/50 | 290/409 (70.90%) | 684 |
| Qwen3 Text | 50/50 | 8/50 | 248/409 (60.64%) | 577 |
| Qwen3-VL | 39/50 | 8/39 | 222/318 (69.81%) | 478 |

## Files

- `trials.csv`: one row per canonical model/form trial, including usability and source artifact path.
- `aggregate.json`: aggregate metrics, stop-reason counts, and missing-form lists.

Recreate this export with:

```bash
.venv/bin/python scripts/export_fill_only_done_50_results.py
```
