
export delay_fit_spec, solve

"""
    delay_fit_spec(cfg; elevation_cutoff, min_snr, reweight = true, reject = true) -> GlobalDelayFit

The estimator specification, entirely from the session configuration. `GlobalDelayFit`
has no defaults of its own — all sixteen of its keywords are required, deliberately — so every one of
them is a config field, and the two that are not (`reweight`, `reject`) stay function keywords
because they are variant switches of a single solve, flipped per solve by the validation suite, not
session configuration.
"""
delay_fit_spec(cfg; elevation_cutoff = cfg.elevation_cutoff, min_snr = cfg.min_snr,
               reweight = true, reject = true) =
    GlobalDelayFit(; reference_antenna = cfg.reference_antenna,
                   reference_receptor_slot = cfg.reference_receptor_slot,
                   elevation_cutoff, min_snr,
                   clock_node = cfg.clock_node, zwd_node = cfg.zwd_node,
                   clock_rate_constraint = cfg.clock_rate_constraint,
                   zwd_rate_constraint = cfg.zwd_rate_constraint,
                   gradient_constraint = cfg.gradient_constraint,
                   reweight, reject, outlier_nsigma = cfg.outlier_nsigma,
                   robust_iterations = cfg.robust_iterations, sigma_floor_init = cfg.sigma_floor_init,
                   min_baseline_obs = cfg.min_baseline_obs, gn_iterations = cfg.gn_iterations)

"""
    solve(cfg, obs, ctx = nothing; structure = nothing, spec = delay_fit_spec(cfg)) -> AstrometryFit

The global astrometric solve on an observable table.

`ctx` is `(; fringes, instr, astrometry_inputs)` — exactly what the observables step took — and its purpose is
the Gauss–Newton relinearization, which re-forms the table at the current position estimate and must
therefore carry the SAME `structure` this solve is against. Pass `nothing` to solve on the linearized
table alone (the stationarity check's control).

With `structure = nothing` this is the point-source solve; with a `StructureModels` it is one
registration solve against that model set. The weighting procedure — the per-baseline σ_floor
reweighting and the outlier rejection with restoration — is re-run from scratch either way, which is
what makes two solves comparable.
"""
function solve(cfg, obs, ctx = nothing; structure = nothing, spec = delay_fit_spec(cfg))
    relin = isnothing(ctx) ? nothing :
            offsets -> observables(cfg, ctx.fringes, ctx.instr, ctx.astrometry_inputs;
                                   structure, source_offsets = offsets)
    fit(spec, obs; relinearize = relin)
end
