# Dropdown Failure Analysis

## Paper-ready result

The original evaluator assigned zero credit to all 100 dropdown targets. A
retrospective audit of stored interaction artifacts confirms that 79 were
selected correctly, confirms none as wrong, and leaves 21 unresolved. The raw
0% dropdown score is therefore invalid and must not be interpreted as model
failure.

| Model/interface | Targets | Confirmed correct | Unresolved | Confirmed wrong | Confirmed lower bound |
|---|---:|---:|---:|---:|---:|
| Gemini 3.5 Flash | 25 | 25 | 0 | 0 | 100% |
| OpenCUA direct-MCP | 25 | 20 | 5 | 0 | 80% |
| Qwen3 Text | 25 | 13 | 12 | 0 | 52% |
| Qwen3-VL | 25 | 21 | 4 | 0 | 84% |

For the 25 forms containing one dropdown, artifact-confirmed full fills are
Gemini 6, OpenCUA 13, Qwen3 Text 1, and Qwen3-VL 11. Treating unresolved values
as an interval gives upper bounds of 6, 15, 3, and 12, respectively.

## Where the failure occurred

1. The model selected an option.
2. The verifier read the custom listbox's complete descendant text. Hidden
   options remain in the Google Forms DOM, so this returned every option rather
   than the element marked `aria-selected="true"`.
3. The exact-value comparison rejected the option-list string.
4. On some multi-page forms, final verification could no longer see an earlier
   page and replaced prior state with `container_not_visible`.

This explains both original patterns: expected values embedded inside long
option-list strings and attempted dropdowns recorded as blank.

## Evidence and decision rule

- Direct-MCP trials are confirmed only when a saved accessibility snapshot has
  the expected option marked selected or the saved encoded form state contains
  the exact expected value.
- Gemini trials are confirmed from manually reviewed post-action screenshots
  showing the expected value in the collapsed dropdown.
- A click on the intended option is not counted as proof of final correctness.
- If the bounded saved excerpt omits the control, the outcome remains
  unresolved. Unresolved is never converted to correct or wrong.

The row-level audit, including evidence paths and steps, is
`data/dropdown_selected_state_audit.csv`. Aggregate lower and upper bounds are
in `data/dropdown_selected_state_summary.csv`. Regenerate both with
`audit_dropdown_selected_state.py`; no model API call or new trial is required.

## Implemented fix

Both browser backends now read the selected option through `aria-selected`
instead of listbox text. The native browser backend also asserts selection
immediately after an option click. Verification state is preserved when a
later page makes an already-verified field invisible. Regression tests cover
the selected-option reader and multi-page state preservation.

## Suggested manuscript wording

> The initial evaluator returned zero correct dropdowns because it compared the
> expected option with the full text of the custom listbox, whose DOM retained
> hidden alternatives. We therefore audited stored, pre-existing interaction
> artifacts without rerunning models. Selected accessibility state, encoded
> form state, or post-action screenshots confirmed 79 of 100 dropdown targets
> as correct; none was confirmed wrong, and 21 remained unobservable in bounded
> snapshot excerpts. We report confirmed correctness as a lower bound and use
> an interval for dropdown-form completion rather than treating unresolved
> observations as failures.
