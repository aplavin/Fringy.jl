
export instrumental

"""
    instrumental(cfg, ld; cable_apply = cfg.cable_apply) -> (; applied, tones, summary)

The session's instrumental calibration, derived from its PHASE-CAL tones.

* `applied::CapturedAppliedPhase` — the delay and phase the calibration installs, per
  (antenna, receptor slot, time cell, frequency cell), together with the channel comb they are evaluated
  on. This is exactly what [`total_delays`](@ref) must add back (or retain, per `cfg.captured_phase_mode`) and
  exactly what [`fringefit`](@ref) installs before it measures.
* `tones` — the PHASE-CAL table as [`pick_tones`](@ref) left it: unchanged on a two-tone session,
  cut to the outermost healthy pair on a comb mode.
* `summary` — the census the step measures as data rather than as a printed line:
  `cells`, `filled`, `delay_median`, `delay_max` [s] and `phase_median` [rad] over the filled cells.

`cable_apply` is a keyword with the configured value as its default because it is the arm of a
DIAGNOSTIC experiment (which halves of the tone measurement the CABLE_CAL reading corrects —
the cable-application diagnostic), flipped per run by that experiment, not a second production
setting. `cfg.cable_antennas` is not optional even at `cable_apply = :none`: an antenna in the list with
no reading in a cell has that cell SKIPPED, which precedes the mode switch and changes the filled set.
"""
function instrumental(cfg, ld; cable_apply = cfg.cable_apply)
    tones = pick_tones(cfg, ld.pcal)
    sol, _ = apply_step(PcalInit(tones; antennas = ld.antennas, cable = cfg.cable_antennas,
                                 cable_apply, amplitude_floor = cfg.tone_floor),
                        instrumental_states(ld))
    applied = CapturedAppliedPhase(sol)
    d, p = applied.delay, applied.phase
    dnz, pnz = filter(!iszero, d), filter(!iszero, p)
    summary = (; cells = length(d), filled = count(!iszero, d),
                 delay_median = isempty(dnz) ? NaN : median(abs.(dnz)),
                 delay_max = maximum(abs, d),
                 phase_median = isempty(pnz) ? NaN : median(abs.(pnz)))
    (; applied, tones, summary)
end

"""
    instrumental_states(ld) -> Solution

The empty three-component solution the instrumental calibration is decoded into: `PhaseOffset` and
`Delay` per (scan, IF) — one tone pair measures one delay and one band-centre phase per (scan, IF),
which is what `PcalInit` asserts — and `Rate` over the whole band, which the tones say nothing about
and which the fringe search fills.
"""
instrumental_states(ld) = Solution(ld.ds; components = (
    PhaseOffset(; time = ld.byscan, frequency = ld.byif),
    Delay(; time = ld.byscan, frequency = ld.byif),
    Rate(; time = ld.byscan, frequency = ld.wholeband)))
