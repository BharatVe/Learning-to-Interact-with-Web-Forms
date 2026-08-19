# LocalForms: dataset creation methodology for the platform comparison

This is the methods note for the second-platform arm of the interface comparison
(see `docs/ALTERNATIVE_PLATFORM_PLAN.md` for the original proposal and rationale).
It answers one question precisely: **did we use FormFactory as-is, or modify it,
and how.**

## Short answer

We imported FormFactory's *implementation pattern* (Flask app, one route per form,
Jinja templates extending a shared base, Bootstrap 4 form markup, JSON submission
logging) exactly. We did **not** use FormFactory's own 25/40 forms, its own gold
answers, or its own evaluator. Instead, the pattern was populated with this
project's own 50 forms (from `src/forms/*/spec.json`) and paired with this
project's own scoring pipeline, so that platform is the only thing that varies
between this condition and the existing Google Forms condition. Two small,
additive markup conventions were added to the generated pages so this project's
existing verifier can read them; one functional bug (native date/time inputs) was
found and fixed during validation. All changes are documented below with exact
rationale.

## 1. What FormFactory is, and why it's the reference

[FormFactory](https://arxiv.org/abs/2506.01520) (ACM MM '25;
[github.com/formfactory-ai/formfactory](https://github.com/formfactory-ai/formfactory),
commit `b7ef0d669d77d2afd41ff4d51ab31becbdeff30b`) is a published benchmark that
hosts its own forms as a **Flask app**, not a third-party SaaS product: `app.py`
defines one route per form; `templates/*.html` are Jinja templates extending a
shared `base.html`; forms use Bootstrap 4 classes (`mb-3`/`form-label`/
`form-control`/`form-select`/`form-check`); POST submissions are logged to
`submission/<form>.json`. This makes it the strongest available precedent for
"recreate our own forms as locally-hosted HTML" — the exact thing this comparison
needed — rather than an uncontrolled SaaS platform with no reproducibility
guarantee.

The pristine clone is kept for reference/citation at
`evaluation_additions/formfactory_import/upstream/` (git metadata stripped, plain
files only).

## 2. What was imported verbatim

- The **application architecture**: Flask, one route per form, template
  inheritance from a shared base, Bootstrap 4 markup and class names, JSON
  submission logging (`save_submission_to_json`).
- The **dependency pin**: `Flask>=2.3,<3.0`, exactly as FormFactory's
  `requirements.txt` specifies (installed as Flask 2.3.3).
- The **visual idiom**: `.mb-3` field wrapper divs, `form-label`, `form-control`
  / `form-select` / `form-check`/`form-check-input`/`form-check-label`, matching
  upstream templates such as `A11.html` and `D11.html` field-for-field in class
  usage.

## 3. What was generated instead of copied

FormFactory's own 25 forms are a different dataset (different domains, field
counts, and content) — using them would confound *platform* with *content*, which
defeats the purpose of a controlled comparison. So instead of FormFactory's
`templates/*.html`, a generator script produces the equivalent Jinja-wrapped HTML
for **this project's own 50 forms**:

```
src/forms/<form_id>/spec.json  →  evaluation_additions/formfactory_import/site/templates/lf_<form_id>.html
                                →  evaluation_additions/formfactory_import/site/app.py   (route table)
                                →  src/forms_localforms/lf_<form_id>/spec.json           (isolated form spec)
data/answers/<form_id>/runs.json → data/answers_localforms/lf_<form_id>/runs.json         (copied, form_id renamed)
```

Generator: `scripts/generate_localforms.py`. Deterministic and idempotent — rerun
any time the 50 source specs change; it fully regenerates all four outputs from
scratch (`shutil.rmtree` + rebuild) so there is never a stale/partial mix.

Widget mapping (all native HTML, per FormFactory's own style — its templates use
plain inputs/selects, not custom JS widgets):

| `q_type` in `spec.json` | Generated control |
|---|---|
| `SHORT_TEXT` | `<input type="text">` |
| `PARAGRAPH` | `<textarea>` |
| `SINGLE_CHOICE` | `<input type="radio">` per option, in `.form-check` |
| `MULTI_CHOICE` | `<input type="checkbox">` per option, in `.form-check` |
| `DROPDOWN` | `<select><option>...</option></select>` |
| `DATE` | `<input type="text" placeholder="YYYY-MM-DD">` (see §5) |
| `TIME` | `<input type="text" placeholder="HH:MM">` (see §5) |

Question labels, help text, required flags, and option strings are copied
verbatim from `spec.json` (which is itself synced from
`From Generator - Questions.csv`), so wording is identical to the Google Forms
condition. All 50 forms are currently generated and present under
`src/forms_localforms/lf_*/` and `data/answers_localforms/lf_*/`.

## 4. Deliberate, documented modifications — and why each was necessary

The constraint driving every modification below: this comparison must vary
**platform only**. That means reusing this project's own scoring pipeline
(`src/engine/mcp_browser_engine.py`, `src/engine/form_engine.py`) unchanged in
its *scoring policy*, which in turn means the generated pages must satisfy the
two DOM conventions that pipeline already assumes (because Google Forms happens
to satisfy them natively).

| Modification | Why | Risk to Google Forms condition |
|---|---|---|
| Each question wrapped in `<div role="listitem">` inside `<div role="list">` | The verifier locates a question by scanning `div[role='listitem']` for matching label text. Valid ARIA; Google Forms itself does this. | None — purely additive; only affects LocalForms pages. |
| Explicit `role="radio"` / `role="checkbox"` + `aria-label="<option text>"` on native radio/checkbox inputs | The verifier reads selection state via a literal `[role='radio']` CSS attribute selector. Native inputs carry these roles *implicitly* in the real accessibility tree (so the model's own tool calls, which use `aria-ref`, already see them correctly) — but a raw CSS attribute selector does not match implicit roles, only explicit ones. Adding the attribute is standard, redundant-but-harmless ARIA practice. | None — Google Forms' own custom widgets already carry these roles. |
| `DATE`/`TIME` rendered as plain `<input type="text">` with a format hint, instead of native `<input type="date">`/`<input type="time">` | See §5 — native versions silently lost their value under keystroke-simulated typing. | None — code path only used for LocalForms text inputs. |

Two small, additive CLI flags were also added to
`src/baselines/run_qwen_direct_mcp_eval.py`: `--forms-root` (default unchanged:
`src/forms`) and `--form-url` (overrides the spec's baked-in URL). Both default
to prior behavior when omitted, so the existing Google Forms invocation is
untouched. They exist so a Slurm job can point the same eval script at
`src/forms_localforms` and a dynamically-chosen local server port without a
second copy of the script.

## 5. A real bug found and fixed during validation (not a design choice)

Before spending any GPU time, the recreation was validated with this project's
own **scripted** Playwright engine (`--interaction-mode local`, target answers
known in advance, no model, no GPU) run against the generated pages end-to-end.
This caught a genuine defect: native `<input type="date">` /
`<input type="time">` do not reliably accept character-by-character simulated
typing in headless Chromium. A direct probe confirmed it —

```
after type("10:15") on <input type="time">:      ''            (empty)
after type("2026-09-01") on <input type="date">:  '60901-02-02' (garbled)
after fill("11:45") on the same time input:        '11:45'       (correct)
```

Both the model-driven MCP tools and this project's scripted reference engine type
character-by-character rather than calling `.fill()` directly, so native
date/time inputs would have silently zeroed a real 73 of 409 fields (18%, all
DATE + TIME questions) across the whole 50-form comparison — a defect that would
not have shown up as an error, just as wrong-looking model output, and could
easily have been mis-read as a model capability finding instead of a platform bug.

Fix: DATE/TIME questions are generated as plain text inputs (matching upstream
FormFactory's own style, which doesn't use native date/time pickers either), with
a `Format: YYYY-MM-DD` / `Format: HH:MM (24-hour)` hint. Two verifier/engine
functions gained a small, additive fallback (a single-text-input read/fill
branch, mirroring a fallback `readDateValue` already had) so both engines handle
a lone text field for DATE and TIME identically regardless of platform:
`mcp_browser_engine.py`'s `readTimeValue`, and `form_engine.py`'s `_handle_time` /
`_read_time_value`. Neither change alters behavior for Google Forms, which never
presents a lone `input[type=text]` for a date/time question.

## 6. Validation performed, in order

1. **Static render check** — all 50 generated pages served locally (Flask dev
   server), confirmed HTTP 200 and correct field markup by inspection.
2. **Zero-GPU scripted round-trip** — `src/engine/runner.py --interaction-mode
   local` (known answers, no model) run against all 10 pilot forms end-to-end
   (fill → verify → submit). This is what caught the date/time bug in §5; after
   the fix, all 10 passed clean.
3. **Environment/toolchain regression check** — existing unit test suite (94
   tests across `test_baseline_eval_contract`, `test_browser_language_defaults`,
   `test_qwen_direct_mcp_eval`, `test_opencua_direct_eval`,
   `test_mcp_browser_engine`) run after the CLI/engine edits — all pass.
4. **1-form GPU smoke test** (OpenCUA-32B, direct-MCP, fill-only, `conf_interest`)
   — first attempt failed on an unrelated environment bug (the LocalForms Flask
   server was launched with a bare-system Python 3.9 build that is ABI-incompatible
   with the OpenSSL brought in by the compute node's `module load` stack); fixed
   by launching it with the same module-provided Python already used for the rest
   of the job (`.venv-opencua`), verified under the actual loaded module
   environment, then rerun clean: 0 invalid tool calls, 0 tool errors, 7/7 fields
   verified.
5. **10-form and 40-form batch jobs** — in progress at time of writing; see
   `docs/ALTERNATIVE_PLATFORM_PLAN.md` / session log for current job IDs and
   results as they land.

## 7. Where everything lives

| Artifact | Path |
|---|---|
| Upstream reference (unmodified) | `evaluation_additions/formfactory_import/upstream/` |
| Generated Flask site | `evaluation_additions/formfactory_import/site/` |
| Generator script | `scripts/generate_localforms.py` |
| Generated form specs (50) | `src/forms_localforms/lf_*/spec.json` |
| Generated answer sets (50) | `data/answers_localforms/lf_*/runs.json` |
| Slurm job scripts | `scripts/slurm_opencua_direct_mcp_localforms.sbatch`, `scripts/run_opencua_direct_mcp_localforms_matrix.sh`, `scripts/run_localforms_server.sh` |
| Implementation notes | `evaluation_additions/formfactory_import/README.md` |
| Original proposal / alternatives considered | `docs/ALTERNATIVE_PLATFORM_PLAN.md` |
| This methodology report | `docs/LOCALFORMS_METHODOLOGY.md` |

Nothing here has been committed to git yet — these are working-tree changes,
ready to review with `git status` / `git diff` and commit when you're ready.
