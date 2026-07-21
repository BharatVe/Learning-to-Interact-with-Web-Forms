# Methods and Limitations

## Evaluation scope

- Canonical capability comparison: 200 fill-only trials, 50 forms for each of
  four model/interface conditions, with 1,636 target-field outcomes.
- Historical interaction context: 978 primary baseline trials.
- Ideal reference: 300 successful scripted runs, covering 50 forms and six
  answer sets per form.

## Metric corrections

### Action count

The primary count includes navigation, clicks, typing/fill, scrolling,
selection/checking, keypresses, dragging, hovering, uploads, and scripted fill
or submit operations. Screenshots, accessibility snapshots, waits, `DONE`,
setup, and close are excluded. Raw calls remain in
`data/action_count_correction_by_model.csv` for audit.

This removes observation overhead but does not make every call equal: one
scripted or form-fill call can do more work than one click. Compare action
counts with completion and elapsed time.

### Submitted-trial score

Successful submission replaces the form with a confirmation page, so final
verification can incorrectly produce `0/N`. Historical trials therefore use,
in order: the pre-successful-submit score, the pre-first-submit score, an
existing recovered score, and only then the final-page score.

This recovers nonzero scores for 512 of 519 submitted trials and proves 153
trials perfect. Partial submitted trials are repaired at task level; field
identity is restored only when the aggregate proves every field correct. The
canonical fill-only cohort is unaffected because all 200 trials made zero
submit attempts.

### Dropdown score

The original custom-dropdown verifier read the listbox's full descendant text.
Google Forms keeps hidden options in that subtree, so the recorded value became
the complete option list. On multi-page forms, a later `container_not_visible`
result could also overwrite a value verified on an earlier page.

The retrospective audit uses only stored artifacts: the selected option or
encoded form state in direct-MCP accessibility snapshots, and manually reviewed
post-action screenshots for Gemini. It confirms 79 correct selections, no
wrong selections, and leaves 21 unresolved where the bounded snapshot excerpt
does not expose the control. `data/dropdown_selected_state_audit.csv` records
the evidence path and step for every target. Confirmed rates are lower bounds.

Use exact completion on `performance_forms_without_dropdown.csv`. For forms
with dropdowns, report the audited full-fill interval in
`data/dropdown_selected_state_summary.csv`; do not assume unresolved selections
are correct or wrong.

## What the failure labels mean

`not_attempted`, `attempted_but_blank`, and `wrong_value` describe the final
field state. They do not identify the causal action. An empty field could result
from missed navigation, lost focus, a failed type operation, or later value
loss. Causal action attribution requires per-action target and before/after
field state.

## Ideal comparison

The ideal reference's median is 10 normalized interactions and 56.44 seconds.
It includes submission, whereas the canonical model task stops after filling.
Time comparisons are therefore strong directional efficiency evidence, not
perfectly identical-task estimates. A model/ideal action ratio below one can
also reflect early stopping rather than efficiency.

## Remaining limitations

1. Twenty-one historical dropdown outcomes remain unresolved because saved
   snapshot excerpts are bounded; no new run is needed for the other 79.
2. Widget type, form length, and field position are correlated; unadjusted cuts
   are descriptive rather than isolated causal effects.
3. The canonical matrix replaces or excludes infrastructure failures. It
   estimates capability conditional on a usable run, not scheduled-attempt
   reliability.
4. Direct-MCP logs do not consistently capture an action target and verified
   before/after state, limiting action-level causal attribution.
5. Repeated randomized runs are unavailable; uncertainty reflects this fixed
   observed cohort rather than run-to-run variance.

## Recommended reporting rules

- Primary: non-dropdown field correctness, no-dropdown full fills, stop reason,
  matched completion time, and normalized actions with outcome.
- Secondary/audit: raw calls, unresolved dropdown outcomes, and raw scripted
  trace-event counts.
- Always state denominators and separate usable-trial capability from
  operational reliability.

The detailed machine-readable gap register remains in
`data/methodology_gap_register.csv`.
