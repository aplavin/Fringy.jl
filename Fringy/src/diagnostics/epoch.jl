
export epoch_consistency, epoch_summary

"""
    epoch_consistency(cfg, fringes, instr, astrometry_inputs; shift = 5e-4, captured_phase_mode = cfg.captured_phase_mode)
        -> NamedTuple

The canonical-epoch consistency of the model layer on this session's own observables.

Returns

  * `poly_minus_model_clock` — `(; median_ns, robust_sigma_ns, n)` of
    `τ_correlator − τ_geometric_hydrostatic + τ_correlator_clock` over finite rows;
  * `consistent_shift_max_ps` — the largest `|Δ(poly − model)|` under a shift of `shift` seconds
    applied consistently to both sides. It is the number that must be small;
  * `polynomial_only_max_ps` — the largest `|Δτ_correlator|` under the same shift, i.e. what a split epoch
    would have injected. It is the number that makes the first one mean something.

`fringes` is [`fringefit`](@ref)'s result, `instr` [`instrumental`](@ref)'s, and `astrometry_inputs`
[`astrometry_models`](@ref)'s — step results as values.
"""
function epoch_consistency(cfg, fringes, instr, astrometry_inputs; shift = 5e-4, captured_phase_mode = cfg.captured_phase_mode)
    ai = captured_applied_phase(instr)
    epoch_summary(total_delays(fringes.obs, ai, astrometry_inputs.correlator_model, astrometry_inputs.astrometry_model; captured_phase_mode),
                  total_delays(fringes.obs, ai, astrometry_inputs.correlator_model, astrometry_inputs.astrometry_model; captured_phase_mode, epoch_shift = shift);
                  shift)
end

"""
    epoch_summary(O, O_shifted; shift) -> NamedTuple

The three numbers of [`epoch_consistency`](@ref), reduced from the two observable tables — the
arithmetic on its own, so that it is statable on hand-written tables:

  * `poly_minus_model_clock` — the distribution of
    `τ_correlator − τ_geometric_hydrostatic + τ_correlator_clock` over finite rows;
  * `consistent_shift_max_ps` — the largest `|Δ(poly − model)|` between the two tables. It is the
    number that must be small;
  * `polynomial_only_max_ps` — the largest `|Δτ_correlator|`, i.e. what a split epoch would have
    injected. It is the number that makes the first one mean something, and reporting it is what
    keeps the comparison from passing vacuously.
"""
function epoch_summary(O, O_shifted; shift)
    ok = findall(isfinite, O.τ_residual)
    resid = O.τ_correlator[ok] .- O.τ_geometric_hydrostatic[ok] .+ O.τ_correlator_clock[ok]
    d = filter(isfinite, (O_shifted.τ_correlator .- O_shifted.τ_geometric_hydrostatic) .- (O.τ_correlator .- O.τ_geometric_hydrostatic))
    dsplit = filter(isfinite, O_shifted.τ_correlator .- O.τ_correlator)
    (; poly_minus_model_clock = (; median_ns = 1e9median(resid),
                                   robust_sigma_ns = 1e9robust_sigma(resid), n = length(resid)),
       consistent_shift_max_ps = 1e12maximum(abs, d),
       polynomial_only_max_ps = 1e12maximum(abs, dsplit),
       shift_s = float(shift), n_rows = length(O))
end
