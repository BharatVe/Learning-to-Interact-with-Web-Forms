# interaction-media-v1 — provenance manifest

Repo: BharatVe/Learning-to-Interact-with-Web-Forms
Proposed release tag: interaction-media-v1
Staged: 2026-09-01

All source paths are relative to the repo root
(`.../Learning-to-Interact-with-Web-Forms/`). Sources are under the git-ignored
`data/model_baselines/` tree.

## Source trials

| key | model | form | outcome | source trial dir |
|---|---|---|---|---|
| opencua_alumni    | OpenCUA-32B (`computer_use_opencua_32b_direct_mcp`) | lf_alumni_checkin | success=true, submit_success=true, stop=submitted, 7/7 | data/model_baselines/opencua_localforms_direct_mcp_submit_40_20260827/computer_use_opencua_32b_direct_mcp/lf_alumni_checkin/run_0002/trial_20260901T102532672160Z |
| opencua_labsafety | OpenCUA-32B (`computer_use_opencua_32b_direct_mcp`) | lf_lab_safety    | success=false, stop=max_steps_exceeded (128), 7/8       | data/model_baselines/opencua_localforms_direct_mcp_submit_10_20260827/computer_use_opencua_32b_direct_mcp/lf_lab_safety/run_0002/trial_20260901T103326186594Z |
| gemini_alumni     | Gemini 3.5 Flash (`computer_use_gemini_35_flash_lowcost`) | alumni_checkin | success=true, submit_success=false, stop=filled_without_submit | data/model_baselines/gemini_35_flash_fill_only_done_50_completion_20260713_r2_step32/computer_use_gemini_35_flash_lowcost/alumni_checkin/run_0002/trial_20260714T050343266100Z |
| gemini_labsafety  | Gemini 3.5 Flash (`computer_use_gemini_35_flash_lowcost`) | lab_safety     | success=false, stop=max_steps_exceeded (32)             | data/model_baselines/gemini_35_flash_fill_only_done_50_completion_20260713_r2_step32/computer_use_gemini_35_flash_lowcost/lab_safety/run_0002/trial_20260714T004036234866Z |

## Asset -> source file

### videos/
| asset | source (within trial dir) |
|---|---|
| opencua32b_alumni_checkin_COMPLETED.webm      | opencua_alumni    : videos/0f3a7f5463f6b5dfa186a44c34282e3c.webm |
| opencua32b_lab_safety_FAILED-loop.webm        | opencua_labsafety : videos/d664d3bc6d72e9a7caa43c7ddcae290a.webm |
| gemini35flash_alumni_checkin_COMPLETED.webm   | gemini_alumni     : alumni_checkin_trial_20260714T050343266100Z.webm |
| gemini35flash_lab_safety_FAILED-loop.webm     | gemini_labsafety  : lab_safety_trial_20260714T004036234866Z.webm |

### screenshots/
| asset | source (within trial dir) |
|---|---|
| opencua32b_alumni_checkin_COMPLETED_01-first_step0000.png     | opencua_alumni    : observations/step_0000_vlm.png |
| opencua32b_alumni_checkin_COMPLETED_02-middle_step0004.png    | opencua_alumni    : observations/step_0004_vlm.png |
| opencua32b_alumni_checkin_COMPLETED_03-last_step0008.png      | opencua_alumni    : observations/step_0008_vlm.png |
| opencua32b_alumni_checkin_COMPLETED_04-final-confirmation.png | opencua_alumni    : observations/final.png |
| opencua32b_lab_safety_FAILED-loop_01-first_step0000.png       | opencua_labsafety : observations/step_0000_vlm.png |
| opencua32b_lab_safety_FAILED-loop_02-middle_step0064.png      | opencua_labsafety : observations/step_0064_vlm.png |
| opencua32b_lab_safety_FAILED-loop_03-last_step0127.png        | opencua_labsafety : observations/step_0127_vlm.png |
| opencua32b_lab_safety_FAILED-loop_04-error.png                | opencua_labsafety : observations/error.png |
| gemini35flash_alumni_checkin_COMPLETED_01-first_step0000.png       | gemini_alumni    : observations/step_0000.png |
| gemini35flash_alumni_checkin_COMPLETED_02-middle_step0006.png      | gemini_alumni    : observations/step_0006.png |
| gemini35flash_alumni_checkin_COMPLETED_03-last_step0012.png        | gemini_alumni    : observations/step_0012.png |
| gemini35flash_alumni_checkin_COMPLETED_04-final-filled-state.png   | gemini_alumni    : final.png |
| gemini35flash_lab_safety_FAILED-loop_01-first_step0000.png    | gemini_labsafety : observations/step_0000.png |
| gemini35flash_lab_safety_FAILED-loop_02-middle_step0016.png   | gemini_labsafety : observations/step_0016.png |
| gemini35flash_lab_safety_FAILED-loop_03-last_step0032.png     | gemini_labsafety : observations/step_0032.png |
