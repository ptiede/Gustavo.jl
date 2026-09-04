# Scope: per-scan (time-varying) bandpass component

Status: scoped, not started. Blocked on `DESIGN_REVIEW_PLAN.md` CHUNK-006
(Bandpass model surface) and CHUNK-003/CHUNK-008 (station scoping) — the
configuration entry point for this feature is decided there. Everything below
concerns the solver side, which is independent of how the component is spelled.

## Motivation

ALMA's bandpass in the 2022 EHT data changes late in the track (seen in
`plot_stability` per-scan series on AA baselines of `hops_3809_M87.uvfits`).
Two facts pin this on a genuine instrumental effect the current model cannot
express:

- The a priori (ANTAB) calibration does not explain it. AA's per-channel SEFD
  table in `e22f27_b3_proc.AN` (32 channels, ~6 s sampling, 2701 rows over
  ~8.3 h) has a large static spectral shape (±12% in sqrt-SEFD, i.e. in
  visibility amplitude) but that shape is stable in time to 0.3–0.6% across
  the whole track. Applying the ANTAB removes a *fixed* amplitude-bandpass
  component, not a late-track change.
- The pre-streaming implementation reached the same conclusion and handled it:
  AA's production config gave the *phase* bandpass per-scan freedom
  (`Flat ⊕ Poly × ChannelBlocks(4)`, both components per-scan). The current
  pipeline's bandpass is hard-wired `GlobalTime()`, so scan-to-scan bandpass
  evolution averages into the template and shows up as late-scan residuals.

A plausible instrumental origin: phased-ALMA phasing efficiency is
frequency-dependent and degrades toward low elevation (late in an M87 track).

## Reference implementation

The deleted `src/Bandpass/` module contains a complete, tested per-scan
bandpass solver. Read it at commit `ce55f83` (the tree just before
`5d6f96b` "Remove old bandpass for streaming approach"):

    git show ce55f83:src/Bandpass/Solver.jl

Key locations at that revision:

| What | Where |
|---|---|
| Per-scan basis columns + G3 demeaning | `Solver.jl:694-902` (demean site `:770-780`) |
| Template/per-scan solver split | `refine_bandpass!`, `Solver.jl:2137-2195` |
| Residual fit against frozen template | `_fit_phase_track_with_frozen`, `Solver.jl:881-902` |
| Post-fit demean + deterministic unwrap seed | `Solver.jl:788-830`, `:243-265` |
| DOF accounting for demeaned Flat | `_segmented_freedom`, `Solver.jl:1648-1662` |
| Underdetermined-scan warning | `Solver.jl:1699-1774` |
| Merge of template + per-scan gains | `merge_scan_gains!`, `Solver.jl:1488-1502` |

What AA actually ran with (production TOML and tuned interactive variant, in
the M87 2022 driver repository): phase = `Flat ⊕ Poly` per 4-channel (per-IF)
block, degree 1 (tuned) or 2 (reports), **both components per-scan**;
amplitude = global per-channel. The mixed-time form (`Flat` global,
`Poly` per-scan) was tried and lost ~1.0 in χ² — the per-IF stair itself
drifts in time. Note the current need is an *amplitude* change, so the new
implementation should support per-scan amplitude with the same basis form
from the start.

## Architecture decision

Solve the per-scan component **sequentially after** the time-invariant joint
ALS, inside the existing `Bandpass` step's `finish_pass!` — the direct port of
the old `refine_template`/`refine_scans` split. Not inside the joint ALS
(ALMA's per-scan flat mode and the frequency-flat source coherence `S` would
trade freely between iterations), and not as a separate pipeline step
(`process_scan!` already collects per-scan full-channel accumulators for every
scan; a separate step would re-stream the whole dataset to rebuild them).

Prerequisite plumbing: `process_scan!`'s per-scan results carry no time index
(`finish_pass!` gets a positional list). Thread `first(win.ti_idx)` through —
required by any variant of this feature.

The evaluate/apply side needs no changes: `PerScan` components index the θ
leaf's time-segment axis via `plan.tseg_id[ti]`, and `component_dimarray`
gives the solved component a scan-epoch `Ti` axis automatically. The current
`ts = 1` hardcoding in the three `_write_*_bandpass!` writers only ever
touches the `GlobalTime` template component and stays as is.

Station scoping uses the allocate-for-all / write-only-scoped convention
(θ = 0 reads back as unit gain; `_write_joint_bandpass!` already relies on
unwritten-means-identity), pending CHUNK-003's decision on the spelling.

## Solve algorithm (per scan, after the joint ALS converges)

For each scan, for each station with per-scan freedom:

1. Divide the solved template gains out of that scan's segment-reduced
   accumulators. The residual *is* the deviation — the old code's
   `frozen`-track bookkeeping is unnecessary here.
2. Alternate a few iterations: closed-form flat `S` per (baseline, pol) given
   current deviations (reuse `_update_source_coherence!` restricted to one
   scan); closed-form per-segment deviation estimate with Fisher weights
   `denom·|ĝ|²` (the same linearization both existing tiers use).
3. Project the (log-amp, wrapped-difference phase) tracks onto the demeaned
   `Flat ⊕ Poly` basis by WLS; write coefficients into the leaf at
   `ts = tseg_id[ti]`.

A station with fewer than `min_baselines` baselines in a scan gets unit gain
and status `nodata` — never a wild fit (port the old
`warn_underdetermined_per_scan_supports` logic as a solve-time guard, not
just a warning).

## Identifiability (the design core)

**Flat-mode degeneracy — must handle.** `S` is one frequency-flat complex
number per (scan, baseline, pol), so the band-mean of any per-scan station
gain — phase and log-amplitude both — is exactly degenerate with it. Port the
old G3 discipline verbatim, including its subtleties:

- Per-scan basis columns are demeaned with the **uniform** mean over all
  channels (not weighted, not valid-only).
- The fitted track is demeaned again post-fit with the same uniform mean.
- The phase unwrap seed is deterministic (first finite channel).

These three sites must agree: a weighted demean makes the WLS fit
non-invariant under 2π shifts of the unwrapped track. The payoff is the G3
invariant — the per-(antenna, feed) band-mean of the bandpass is
time-invariant by construction, so applying the bandpass cannot alter the
temporal structure of band-averaged per-baseline quantities. The old driver's
`g3_gain_diagnostic` (band-mean spread < 1e-6 rad / 1e-9 in log-amp across
scans) becomes a package test.

**Delay-slope degeneracy — decided posture: report, don't project.** A
per-scan linear-in-frequency phase is a residual fringe delay, degenerate with
the fringe step's per-scan delays. The old code never handled this and it was
live in its shipped default. First cut: match the old behavior (block-local
`Poly` slopes, no global-slope projection — data arrives fringe-corrected, so
what remains is genuine per-IF structure), but compute and report each scan's
implied global delay from the fitted per-scan phase so an ownership fight with
the fringe step is visible instead of silent. An optional projection flag can
follow if the diagnostic shows drift.

**Phase gauge.** With demeaned bases and per-scan freedom restricted to scoped
stations, no additional per-scan pin is needed — the per-scan common phase
lives entirely in `S`, and `_joint_bandpass_pins`' analysis for the template
is unaffected because the template solve is untouched.

## Application semantics

`PerScan` components resolve on a foreign grid by scan name; applying to a
scan the solve never saw is a hard error (correct, fail-fast). Consequence:
the bandpass step's scan selection must cover every scan to be corrected.
`CoverageTopup`'s "mixing sources is safe" justification is explicitly a
time-invariance argument and does not extend to the per-scan component — a
topped-up scan from another source gets template-only (status `nodata`), never
a fitted deviation.

## Reporting

`bandpass_track_report` hardcodes `(Ant, Feed, band)` status arrays; per-scan
tracks need a scan axis (or a separate per-scan record) carried through the
step info and HDF5 serialization. Add the implied-delay diagnostic (above) to
the report.

## Work breakdown (after CHUNK-006/008 land)

1. Thread scan `ti` through `process_scan!` results (tiny).
2. Per-scan component config through the CHUNK-006 model surface, scoped via
   the CHUNK-008 primitive (small).
3. G3 basis machinery: demeaned per-scan design columns from
   `Polynomial`/`ConstantTerm` × `ChannelBlocks`, uniform-mean discipline,
   rank trim (medium — port from `ce55f83` `Solver.jl:694-902`).
4. Per-scan residual solve + θ writer at `ts`, per-scan status, sparse-scan
   guard (the bulk).
5. Report/serialization: per-scan status, HDF5, implied-delay diagnostic
   (small–medium).
6. Tests: synthetic injection/recovery, G3 invariant, `S`-unchanged degeneracy
   check, absent-station and sparse-scan edges (medium).

Rough total 600–900 lines including tests.

## Open questions

- Per-scan on amplitude, phase, or both for AA? (Old config: phase only; the
  current observation is amplitude — scope both, enable per observation.)
- Default basis: `Flat ⊕ Poly1 × ChannelBlocks(4)` both per-scan (the tuned
  config) vs the TOML's degree 2?
- Is report-don't-project acceptable long-term for the delay slope, or should
  the projection flag land in the first cut?
