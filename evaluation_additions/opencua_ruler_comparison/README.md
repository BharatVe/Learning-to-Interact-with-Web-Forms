# OpenCUA pixel-ruler comparison

This directory is a compact, GitHub-trackable evidence bundle for the matched 50-form OpenCUA ruler evaluation. Open `report.html` in any current browser for the primary technical report.

## Result

The tested pixel ruler did not improve aggregate verified correctness: all-field accuracy changed from **39.1%** to **38.1%** (-1.0 percentage points). The paired form bootstrap interval is **-4.6 to +1.8 points**. This supports a descriptive negative result, not a causal or variability claim.

## Files

- `report.html`: self-contained portable report for local viewing.
- `artifact.json`: canonical source for the portable report.
- `comparison_summary.csv`: condition-level metrics.
- `paired_forms.csv`: one matched row per form.
- `paired_fields.csv`: one matched row per field; values are hashed to avoid exposing answer contents.
- `widget_summary.csv` and `position_summary.csv`: exploratory breakdowns.
- `robustness.json`: bootstrap, paired transitions, and exact McNemar result.
- `protocol_comparison.json`: recorded matched settings, intervention, and comparability caveats.
- `source_manifest.json`: source and output hashes, row counts, keys, and commands.

## Regenerate and validate

```bash
python3 scripts/analyze_opencua_ruler_comparison.py
python3 scripts/analyze_opencua_ruler_comparison.py --check
```

The data exporter uses only the Python standard library. `report.html` is already packaged and does not need Python, Node.js, a local server, or network access to open.

## Report validation

The portable-report packager passed artifact validation, packaging, and structural verification. Automated Chromium rendering was not completed in this environment: automatic browser discovery was unavailable, and an explicit attempt with the experiment Chromium timed out after 11.2 seconds. Open `report.html` locally for the remaining visual inspection before publication.

## Paper-safe wording

> In a matched descriptive comparison across 50 forms, adding labeled pixel rulers to the OpenCUA screenshots did not improve verified field accuracy (38.1% with rulers versus 39.1% without). The mean paired form difference was -1.3 percentage points (form-bootstrap 95% interval -4.6 to +1.8). Because each form-condition setting was evaluated once and source commits were not recorded in the original cohort metadata, this result does not establish a causal ruler effect or run-to-run model variability.
