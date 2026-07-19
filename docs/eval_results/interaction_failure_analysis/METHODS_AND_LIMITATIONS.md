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

The custom-dropdown verifier often records every option label instead of the
selected option. All 100 dropdown targets were scored wrong; 89 contain the
expected option in this ambiguous container text and 11 are blank or
unattempted. The 89 cannot be safely reclassified as correct.

Use exact completion on `performance_forms_without_dropdown.csv`. For forms
with dropdowns, use `performance_forms_with_dropdown.csv` and report only
non-dropdown correctness until selection can be read from `aria-selected`, the
selected option node, or the collapsed trigger label.

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

1. Dropdown correctness is unresolved until selected-option verification is
   repaired and tested with selected and unselected fixtures.
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
