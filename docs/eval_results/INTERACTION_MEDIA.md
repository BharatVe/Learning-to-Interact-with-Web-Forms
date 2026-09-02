# Interaction Media (videos & labelled screenshots)

Curated screen recordings and step screenshots of model form-filling interactions,
kept for thesis analysis and presentations so they do not have to be dug out of the
raw run trees (`data/model_baselines/**`, ~60 GB, git-ignored on purpose).

Unlike the rest of `data/model_baselines/`, this small reviewed subset **is committed**
to the repo, under [`interaction_media/`](interaction_media/):

```
docs/eval_results/interaction_media/
├── *.webm   (4 screen recordings, ~125 MB total)
├── *.png    (15 labelled step screenshots)
└── MANIFEST.md   (asset -> source-trial provenance)
```

> Size note: the two `*_lab_safety_FAILED-loop.webm` files are ~53 MB each. GitHub
> shows a "large file" warning above 50 MB but still accepts them (hard limit 100 MB).

## What is covered

Two models, one **completed** interaction and one **failed** interaction each.

| Model | Run label | Model id | Task mode |
|---|---|---|---|
| OpenCUA-32B | `opencua_localforms_direct_mcp_submit_*_20260827` | `computer_use_opencua_32b_direct_mcp` | fill **and submit** (LocalForms platform) |
| Gemini 3.5 Flash | `gemini_35_flash_fill_only_done_50_completion_20260713_r2_step32` | `computer_use_gemini_35_flash_lowcost` | **fill only** (FormFactory-style platform) |

> Platform caveat: the two models were run against different form platforms and task
> modes because no Gemini 3.5 Flash run exists on the LocalForms fill-and-submit setup.
> OpenCUA's "completed" ends on a real submission-confirmation page; Gemini's "completed"
> ends on a fully-filled form (fill-only task, `stop_reason = filled_without_submit`).

## Videos

| File | Model | Form | Outcome | Steps | Source trial |
|---|---|---|---|---|---|
| `interaction_media/opencua32b_alumni_checkin_COMPLETED.webm` | OpenCUA-32B | `lf_alumni_checkin` | completed — submitted, 7/7 fields correct | 8 | `data/model_baselines/opencua_localforms_direct_mcp_submit_40_20260827/computer_use_opencua_32b_direct_mcp/lf_alumni_checkin/run_0002/trial_20260901T102532672160Z` |
| `interaction_media/opencua32b_lab_safety_FAILED-loop.webm` | OpenCUA-32B | `lf_lab_safety` | failed — `max_steps_exceeded` (128), stuck on a required dropdown | 128 | `data/model_baselines/opencua_localforms_direct_mcp_submit_10_20260827/computer_use_opencua_32b_direct_mcp/lf_lab_safety/run_0002/trial_20260901T103326186594Z` |
| `interaction_media/gemini35flash_alumni_checkin_COMPLETED.webm` | Gemini 3.5 Flash | `alumni_checkin` | completed — `filled_without_submit`, `success=true` | 13 | `data/model_baselines/gemini_35_flash_fill_only_done_50_completion_20260713_r2_step32/computer_use_gemini_35_flash_lowcost/alumni_checkin/run_0002/trial_20260714T050343266100Z` |
| `interaction_media/gemini35flash_lab_safety_FAILED-loop.webm` | Gemini 3.5 Flash | `lab_safety` | failed — `max_steps_exceeded` (32) | 32 | `data/model_baselines/gemini_35_flash_fill_only_done_50_completion_20260713_r2_step32/computer_use_gemini_35_flash_lowcost/lab_safety/run_0002/trial_20260714T004036234866Z` |

## Screenshots

Each interaction is captured at first step, a middle step, and the last step, plus a
fourth frame (submission confirmation, final filled state, or the validation-error
screenshot) where one exists. All under [`interaction_media/`](interaction_media/).

### OpenCUA-32B — `lf_alumni_checkin` — COMPLETED
| File | Frame |
|---|---|
| `opencua32b_alumni_checkin_COMPLETED_01-first_step0000.png` | first — step 0000 |
| `opencua32b_alumni_checkin_COMPLETED_02-middle_step0004.png` | middle — step 0004 |
| `opencua32b_alumni_checkin_COMPLETED_03-last_step0008.png` | last — step 0008 |
| `opencua32b_alumni_checkin_COMPLETED_04-final-confirmation.png` | "Your response has been recorded" |

### OpenCUA-32B — `lf_lab_safety` — FAILED (loop)
| File | Frame |
|---|---|
| `opencua32b_lab_safety_FAILED-loop_01-first_step0000.png` | first — step 0000 |
| `opencua32b_lab_safety_FAILED-loop_02-middle_step0064.png` | middle — step 0064 |
| `opencua32b_lab_safety_FAILED-loop_03-last_step0127.png` | last — step 0127 (step cap hit) |
| `opencua32b_lab_safety_FAILED-loop_04-error.png` | required dropdown "Lab area you need access to" left unselected — "Please select an item in the list." |

### Gemini 3.5 Flash — `alumni_checkin` — COMPLETED
| File | Frame |
|---|---|
| `gemini35flash_alumni_checkin_COMPLETED_01-first_step0000.png` | first — step 0000 |
| `gemini35flash_alumni_checkin_COMPLETED_02-middle_step0006.png` | middle — step 0006 |
| `gemini35flash_alumni_checkin_COMPLETED_03-last_step0012.png` | last — step 0012 |
| `gemini35flash_alumni_checkin_COMPLETED_04-final-filled-state.png` | fully filled form (fill-only task, not submitted) |

### Gemini 3.5 Flash — `lab_safety` — FAILED (loop)
| File | Frame |
|---|---|
| `gemini35flash_lab_safety_FAILED-loop_01-first_step0000.png` | first — step 0000 |
| `gemini35flash_lab_safety_FAILED-loop_02-middle_step0016.png` | middle — step 0016 |
| `gemini35flash_lab_safety_FAILED-loop_03-last_step0032.png` | last — step 0032 (step cap hit) |

## Provenance

Every file's exact source path within the raw run tree is listed in
[`interaction_media/MANIFEST.md`](interaction_media/MANIFEST.md). The raw trials
themselves live on the HPC workspace only (git-ignored).
