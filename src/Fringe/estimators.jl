# ── Fringe estimator seam ────────────────────────────────────────────────────
#
# The strategy that turns scan data into station fringe parameters (delays,
# rates, phases) is pluggable. Today's implementation — a per-baseline
# delay/rate matched-filter search followed by a closure-screened per-station
# WLS ("stationization") — is ONE estimator; a Schwab–Cotton-style global least
# squares fit of station parameters directly to the visibilities is another,
# and needs no stationization at all. The search/closure machinery therefore
# belongs to the ESTIMATOR that uses it, not to the `FringeFit` step itself.
#
# The concrete estimator types (`MatchedFilter`, wrapping the existing
# search + `solve_station_systems!` path) and the `estimate!` contract land
# with the FringeFit step carve-out; this file declares the seam.

"""
    AbstractFringeEstimator

The strategy a `FringeFit` step uses to estimate station fringe parameters
from scan data. Implementations own their machinery entirely — e.g. the
matched-filter estimator carries a search configuration and a `Stationization`;
a global least-squares estimator would carry neither.
"""
abstract type AbstractFringeEstimator end
