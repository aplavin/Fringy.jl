
export closure_evaluation

"""
    closure_evaluation(cfg, calibration, sets, sources; verbose = true)
        -> Dictionary{Tuple{String, Symbol}, NamedTuple}

`(model set, source) ⇒ (; chi2_phase, n_phase, chi2_amp, n_amp)` for every pair the catalogue has a row
for: every source for the point set, and the sources it carries a model for otherwise.

`calibration` is [`calibrate_and_average`](@ref)'s result — `avg` and the `weight_scale` measured on it, so that nothing
here needs an imaging product to know what a weight is worth in flux units.
"""
function closure_evaluation(cfg, calibration, sets::ModelSets, sources; verbose = true)
    vis = stokes_i(calibration.avg)
    ws = calibration.weight_scale
    rows_of = Dictionary{Symbol, Vector{Int}}()
    for s in sources
        set!(rows_of, s, findall(==(s), vis.source))
    end
    pairs_ = [(l, s) for (l, set) in pairs(sets) for s in sources
              if (set.kind === :point || haskey(set.models, s)) && !isempty(rows_of[s])]
    res = Vector{NamedTuple}(undef, length(pairs_))
    Threads.@threads for i in eachindex(pairs_)
        (l, s) = pairs_[i]
        v = view(vis, rows_of[s])
        m = get(sets[l].models, s, nothing)
        ph = closure_chi2(v, m; weight_scale = ws)
        am = closure_amp_chi2(v, m; weight_scale = ws, min_snr = cfg.closure_amp_min_snr)
        res[i] = (; chi2_phase = ph.chi2, n_phase = ph.n, chi2_amp = am.chi2, n_amp = am.n)
    end
    out = Dictionary{Tuple{String, Symbol}, NamedTuple}()
    for (i, (l, s)) in enumerate(pairs_)
        insert!(out, (l, s), res[i])
    end
    if verbose
        for l in keys(sets)
            v = [out[(l, s)].chi2_phase for s in sources if haskey(out, (l, s))]
            isempty(v) && continue
            f = filter(isfinite, v)
            @printf("  %-34s closure φ χ² median %7.3f  p90 %8.3f  max %9.3f  (n = %d)\n",
                    l, isempty(f) ? NaN : median(f), isempty(f) ? NaN : quantile(f, 0.9),
                    isempty(f) ? NaN : maximum(f), length(v))
        end
    end
    out
end
