
export SetResult, Registration, register, wrap_matched, fit_wrms

"""
    SetResult

One model set's registration.

| field | what |
|---|---|
| `label`, `kind` | the model set this is the registration of |
| `F` | the `AstrometryFit` — the solve, with its own weighting procedure |
| `P` | `positions(F)`: the registered offset of each source's model origin |
| `tau` | per source: median/max \\|τ_structure\\| [ps], the wrap-excluded count and the row count |
| `n_wrap` | rows the phase-wrap guard excluded, over all sources |
| `n_no_model` | sources this set says nothing about (all of them, for the point set — that is what it IS) |
| `sources` | per source: the post-fit wrms [ps] and accepted count of THIS solve |
| `baselines` | per baseline: length, σ_floor [ps], post-fit wrms [ps], accepted count |
| `summary` | wrms, χ²/dof, selected/accepted counts, median σ_floor |
| `reference_model_set`, `reference_wrms_ps` | the solve this set's wrms is read against, and its value |
"""
struct SetResult{F, P, S, B}
    label::String
    kind::Symbol
    F::F
    P::P
    tau::Dictionary{Symbol, NamedTuple}
    n_wrap::Int
    n_no_model::Int
    sources::S
    baselines::B
    summary::NamedTuple
    reference_model_set::String
    reference_wrms_ps::Float64
end

"""
    Registration

Label ⇒ [`SetResult`](@ref) for every configured model set, plus:

* `reference_model_set` — the label whose solve every other set's wrms is read against (`"point"`);
* `obs` — the reference model set's observable table. One table, not six: every set is formed from the same
  fringe rows in the same order, so the row identity, the a priori `(ra, dec)` and every non-structure
  column are shared, and only `τ_structure`/`τ_corrected`/`τ_residual` differ between sets — which is exactly what the
  per-set `tau` census reports.
* `sources` — the session's source list, sorted;
* `radec` — source ⇒ the a priori `(ra, dec)` [rad] the offsets are relative to.
"""
struct Registration{S, O}
    sets::Dictionary{String, SetResult}
    reference_model_set::String
    obs::O
    sources::Vector{Symbol}
    radec::Dictionary{Symbol, Tuple{Float64, Float64}}
    summary::S
end

Base.show(io::IO, r::Registration) = print(io,
    "Registration(", length(r.sets), " model sets, reference model set ", repr(r.reference_model_set), ": ",
    join(keys(r.sets), ", "), ")")

"""
    wrap_matched(O_reference, O_set) -> StructArray

`O_reference` restricted to the rows the wrap guard kept in `O_set`: the same table, with `τ_residual` set to
`NaN` wherever `O_set.τ_structure` is.

The per-observable wrap guard (`cfg.structure_wrap`) drops rows from a STRUCTURE-corrected solve only.
Comparing that solve's wrms against the reference model set's solve over all rows would confound "the structure was
removed" with "a different set of observations was used", so the reference model set is re-run with those rows
removed. `total_delays` maps over `eachindex(obs)`, so row *i* of one table is row *i* of the other and
the match is positional — which is the only reason this is three lines.
"""
wrap_matched(O_reference, O_set) =
    StructArray(map(i -> merge(NamedTuple(O_reference[i]),
                               (; τ_residual = isnan(O_set.τ_structure[i]) ? NaN : O_reference.τ_residual[i])),
                    eachindex(O_reference)))

"""
    fit_wrms(F, sel) -> (wrms_ps, n)

Weighted rms [ps] of the accepted rows of `F` selected by `sel`, a predicate on the OBSERVABLE index
(`F.selected[j]` is the observable row of fit row `j`). `NaN` when the selection is empty — an absence
reported, never a zero.
"""
function fit_wrms(F, sel)
    sw = 0.0; swr = 0.0; n = 0
    for (j, i) in enumerate(F.selected)
        (F.accepted[j] && sel(i)) || continue
        sw += F.weight[j]; swr += F.weight[j] * F.residual[j]^2; n += 1
    end
    (n == 0 ? NaN : 1e12sqrt(swr / sw), n)
end

"""
    register(cfg, fringes, instr, astrometry_inputs, sets::ModelSets; reference_model_set = "point", verbose = true)
        -> Registration

Register this session against every model set, in the order [`models`](@ref) produced them.

For each set, the same five steps and no branch: build the `StructureModels` under `cfg.structure_wrap`,
form the observable table, solve with the weighting procedure re-run from scratch and the Gauss–Newton
relinearization carrying the SAME structure, take the positions, and census the structure delays and
the fit residuals per source and per baseline.

`reference_model_set` names the set every other set's wrms is read against; it must be configured, and it is
`"point"` because that is the solve a registration is a correction TO. The comparison is made on
MATCHED ROWS: where a set's wrap guard excluded nothing, the matched reference solve is the reference model set's
own solve, an identity rather than an approximation, and no extra solve runs. Where the guard did
fire, that set pays one extra reference solve
on its own row set and the comparison stays honest.
"""
function register(cfg, fringes, instr, astrometry_inputs, sets::ModelSets; reference_model_set = "point", verbose = true)
    haskey(sets, reference_model_set) ||
        error("register: the reference model set $(repr(reference_model_set)) is not configured — every registration " *
              "is read against the solve with no structure model, so `PointSource()` belongs in " *
              "`structure_models` (configured: $(join(keys(sets), ", ")))")
    ctx = (; fringes, instr, astrometry_inputs)
    blen(a1, a2) = norm(astrometry_inputs.astrometry_model.antenna_geometry[a1].xyz - astrometry_inputs.astrometry_model.antenna_geometry[a2].xyz) / 1e3

    O_reference = nothing
    results = Dictionary{String, SetResult}()
    order = [reference_model_set; [l for l in keys(sets) if l != reference_model_set]]
    for label in order
        set = sets[label]
        SM = StructureModels(set.models; wrap_threshold = cfg.structure_wrap)
        t = time()
        O = observables(cfg, fringes, instr, astrometry_inputs; structure = SM)
        F = solve(cfg, O, ctx; structure = SM)
        label == reference_model_set && (O_reference = O)
        P = positions(F)
        srcs = sort!(unique(O.source))
        n_no_model = count(s -> !haskey(set.models, s), srcs)

        tau = Dictionary{Symbol, NamedTuple}()
        for s in srcs
            ix = findall(==(s), O.source)
            v = filter(isfinite, O.τ_structure[ix])
            set!(tau, s, (; median_ps = isempty(v) ? 0.0 : 1e12median(abs.(v)),
                            max_ps = isempty(v) ? 0.0 : 1e12maximum(abs, v),
                            n_wrap = count(isnan, O.τ_structure[ix]), n_obs = length(ix)))
        end
        n_wrap = count(isnan, O.τ_structure)

        reference_F = label == reference_model_set ? F :
                      n_wrap == 0 ? results[reference_model_set].F :
                      solve(cfg, wrap_matched(O_reference, O), ctx)

        srows = StructArray([(; source = String(s),
                                (w = fit_wrms(F, i -> O.source[i] === s);
                                 (; wrms_ps = w[1], n_accepted = w[2]))...) for s in srcs])
        bl = sort(unique(collect(zip(O.a1, O.a2))); by = k -> blen(k...))
        brows = StructArray([(; a1 = String(a1), a2 = String(a2), km = blen(a1, a2),
                                sigma_floor_ps = 1e12get(F.sigma_floor, (a1, a2), NaN),
                                (w = fit_wrms(F, i -> O.a1[i] === a1 && O.a2[i] === a2);
                                 (; wrms_ps = w[1], n_accepted = w[2]))...)
                             for (a1, a2) in bl if haskey(F.sigma_floor, (a1, a2))])

        summary = (; wrms_ps = 1e12F.wrms, chi2 = F.chi2, n_selected = length(F.accepted),
                     n_accepted = count(F.accepted),
                     rejected_pct = 100 * (1 - count(F.accepted) / length(F.accepted)),
                     sigma_floor_median_ps = 1e12median(collect(F.sigma_floor)),
                     n_sources = length(srcs), seconds = time() - t)
        insert!(results, label,
                SetResult(label, set.kind, F, P, tau, n_wrap, n_no_model, srows, brows, summary,
                          reference_model_set, 1e12reference_F.wrms))
        verbose && @printf("  %-34s wrms %6.3f ps (reference %6.3f, %+.3f)  χ²/dof %.4f  accepted %6d / %6d  |τ| med %6.2f max %8.2f ps  wrap %d  no model %d  [%.0f s]\n",
                           label, summary.wrms_ps, 1e12reference_F.wrms,
                           summary.wrms_ps - 1e12reference_F.wrms, F.chi2, summary.n_accepted,
                           summary.n_selected,
                           median([t.median_ps for t in tau]), maximum([t.max_ps for t in tau]),
                           n_wrap, n_no_model, summary.seconds)
    end

    ordered = Dictionary{String, SetResult}()
    for label in keys(sets)
        insert!(ordered, label, results[label])
    end
    srcs = sort!(unique(O_reference.source))
    radec = Dictionary{Symbol, Tuple{Float64, Float64}}()
    for i in eachindex(O_reference)
        haskey(radec, O_reference.source[i]) ||
            insert!(radec, O_reference.source[i], (O_reference.ra[i], O_reference.dec[i]))
    end
    Registration(ordered, reference_model_set, O_reference, srcs, radec,
                 (; n_sets = length(ordered), n_observables = length(O_reference),
                    n_sources = length(srcs), wrap_threshold = cfg.structure_wrap,
                    captured_phase_mode = cfg.captured_phase_mode))
end

"""
    registered_position(reg, label, source) -> (ra, dec)

The absolute position [rad] of `source`'s model origin in model set `label` — the a priori position plus the
fitted offset, with the East offset converted at the fitted declination. This IS the value the set's
image should carry as its WCS reference; a brightness feature's position follows from it through
[`absolute_position`](@ref).
"""
function registered_position(reg::Registration, label::AbstractString, source::Symbol)
    P = reg.sets[label].P
    k = findfirst(==(source), P.source)
    isnothing(k) && error("registered_position: $source is not in the solve of model set $(repr(label))")
    α0, δ0 = reg.radec[source]
    dec = δ0 + P.Δδ_mas[k] * MAS
    (α0 + P.Δα★_mas[k] * MAS / cos(dec), dec)
end
