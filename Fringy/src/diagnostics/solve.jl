
export with_captured_phase_mode, solve_variant, respec, fit_metrics, source_radec, compare_positions,
       accepted_view, per_source_stats,
       residual_breakdown, split_solves, antenna_jackknife, residual_forensics,
       parameterization_stress, systematics_transfer, constraint_chi2


"""
    with_captured_phase_mode(obs, mode; current_mode = :retain) -> StructArray

The observable table `obs` algebraically transformed to `mode ∈ (:retain, :add_back)` from
`current_mode` (the `cfg.captured_phase_mode` of the run that wrote it).

  * `:add_back` — add the captured applied-phase slope to `τ_total`, `τ_corrected`, and `τ_residual`;
  * `:retain` — remove that slope from those columns.

The conversion changes only those existing table columns by the reported `τ_applied_phase` slope; it neither
undoes IF alignment or other uncaptured terms nor reruns or refits the fringe pass. It is exact as that
table-column transform, so a mode comparison costs a solve rather than a fringe pass. This is the identity
[`instrument_fraction_scan`](@ref) sweeps continuously.
"""
function with_captured_phase_mode(obs, mode::Symbol; current_mode::Symbol = :retain)
    mode in (:retain, :add_back) ||
        error("with_captured_phase_mode: mode must be :retain or :add_back, got $mode")
    current_mode in (:retain, :add_back) ||
        error("with_captured_phase_mode: the table's current mode must be :retain or :add_back, got $current_mode")
    current_mode === mode && return obs
    s = mode === :retain ? -1.0 : +1.0
    c = StructArrays.components(obs)
    StructArray(merge(c, (; τ_total = c.τ_total .+ s .* c.τ_applied_phase,
                            τ_corrected = c.τ_corrected .+ s .* c.τ_applied_phase,
                            τ_residual = c.τ_residual .+ s .* c.τ_applied_phase)))
end

"""
    with_residual(obs, τ_residual) -> StructArray

`obs` with its fitted quantity replaced and every other column left alone — how an injection, a
scan point or a term switched off is expressed as a VALUE rather than as a mode.
"""
with_residual(obs, τ_residual) = StructArray(merge(StructArrays.components(obs), (; τ_residual)))


"""
    solve_variant(cfg, obs; spec = delay_fit_spec(cfg), mask = nothing, relinearize = nothing)
        -> (; F, P, sub, metrics)

One solve plus the summary every comparison in this family quotes. `mask` selects the rows (a split
arm, a jackknife, an acceptance sweep); `spec` is the estimator specification, which
[`respec`](@ref) makes it cheap to vary one field of.
"""
function solve_variant(cfg, obs; spec = delay_fit_spec(cfg), mask = nothing, relinearize = nothing)
    sub = isnothing(mask) ? obs : obs[mask]
    F = fit(spec, sub; relinearize)
    P = positions(F)
    (; F, P, sub, metrics = fit_metrics(F, P))
end

"""
    respec(spec::GlobalDelayFit; kwargs...) -> GlobalDelayFit

The same specification with the named fields replaced. `GlobalDelayFit` has no defaults at all — all
sixteen keywords are required, deliberately — so this restates the fifteen that do not change rather
than letting a variant inherit a default nobody wrote down.
"""
respec(spec::GlobalDelayFit; kwargs...) =
    GlobalDelayFit(; (f => get(kwargs, f, getfield(spec, f)) for f in fieldnames(GlobalDelayFit))...)

"""
    fit_metrics(F, P = positions(F)) -> NamedTuple

The summary of one solve: the post-fit and starting wrms [ps] and χ², the selected/accepted counts
and the rejected percentage, the source count and degrees of freedom, the median formal σ [mas] and
the median per-baseline σ_floor [ps].
"""
function fit_metrics(F, P = positions(F))
    nacc = count(F.accepted)
    (; wrms_ps = 1e12F.wrms, chi2 = F.chi2, chi2_start = F.log[1].chi2,
       wrms_start_ps = 1e12F.log[1].wrms,
       n_obs = length(F.selected), n_accepted = nacc,
       rejected_pct = 100 * (1 - nacc / length(F.accepted)),
       n_sources = length(P), ndof = F.ndof,
       sigma_ra_mas = median(P.σ_Δα★_mas), sigma_dec_mas = median(P.σ_Δδ_mas),
       floor_median_ps = 1e12 * median(collect(F.sigma_floor)))
end


"A priori (ra, dec) [rad] per source, from the observable table's own columns."
function source_radec(obs)
    d = Dictionary{Symbol, Tuple{Float64, Float64}}()
    for i in eachindex(obs)
        haskey(d, obs.source[i]) || insert!(d, obs.source[i], (obs.ra[i], obs.dec[i]))
    end
    d
end

"""
    compare_positions(a, b, radec) -> NamedTuple

Per-source difference of two position tables over their common sources, with the pull against the
quadrature-summed formal errors AND the decomposition into a rigid frame rotation plus per-source
scatter about it.

The decomposition is the substance. Two subsets of one observation see the sky through different
geometry, so they can differ by a frame ROTATION with every source perfectly consistent; a comparison
that reported only the rms would read that rotation as a per-source inconsistency and the error
budget would inherit it. `rotation` is the three-vector [mas] and `rot_rms_*` the scatter left after
removing it.
"""
function compare_positions(a, b, radec)
    common = sort!(collect(intersect(Set(a.source), Set(b.source))))
    ia = [findfirst(==(s), a.source) for s in common]
    ib = [findfirst(==(s), b.source) for s in common]
    dα = a.Δα★_mas[ia] .- b.Δα★_mas[ib]
    dδ = a.Δδ_mas[ia] .- b.Δδ_mas[ib]
    σα = sqrt.(a.σ_Δα★_mas[ia] .^ 2 .+ b.σ_Δα★_mas[ib] .^ 2)
    σδ = sqrt.(a.σ_Δδ_mas[ia] .^ 2 .+ b.σ_Δδ_mas[ib] .^ 2)
    rf = rotation_fit(common, radec, dα, dδ)
    pulls = vcat(dα ./ σα, dδ ./ σδ)
    (; sources = common, dα, dδ, σα, σδ,
       rms_α = sqrt(mean(abs2, dα)), rms_δ = sqrt(mean(abs2, dδ)),
       mean_α = mean(dα), mean_δ = mean(dδ), std_α = std(dα), std_δ = std(dδ),
       total = sqrt.(dα .^ 2 .+ dδ .^ 2),
       pull_rms = sqrt(mean(abs2, pulls)),
       pull_rms_α = sqrt(mean(abs2, dα ./ σα)), pull_rms_δ = sqrt(mean(abs2, dδ ./ σδ)),
       rotation = rf.rot, rot_rms_α = rf.rms_after[1], rot_rms_δ = rf.rms_after[2],
       res_α = rf.res_α, res_δ = rf.res_δ)
end


"Weighted rms of residuals `r` under weights `w`."
wrms_of(r, w) = sqrt(sum(w .* r .^ 2) / sum(w))

"The project's own robust scatter estimator, about the sample median."
robust_sigma(x) = isempty(x) ? NaN : 1.4826 * median(abs.(x .- median(x)))

"""
    accepted_view(F) -> (; r, w, ix)

Post-fit residuals, weights and the rows of the observable table they belong to, over the ACCEPTED
observations only — which is the only set any residual statistic may be read over, because a rejected
row is one the solve decided not to believe.
"""
function accepted_view(F)
    acc = findall(F.accepted)
    (; r = F.residual[acc], w = F.weight[acc], ix = F.selected[acc])
end

"Per-source `(n_obs, n_scans, wrms [ps], median/max SNR, n_baselines)` over the accepted rows."
function per_source_stats(F, obs)
    v = accepted_view(F)
    out = Dictionary{Symbol, NamedTuple}()
    src = obs.source[v.ix]
    for s in unique(src)
        m = src .== s
        rows = v.ix[m]
        set!(out, s, (; n_obs = count(m), n_scans = length(unique(obs.scan[rows])),
                        wrms_ps = 1e12wrms_of(v.r[m], v.w[m]),
                        snr_median = median(obs.snr[rows]), snr_max = maximum(obs.snr[rows]),
                        n_baselines = length(unique(zip(obs.a1[rows], obs.a2[rows])))))
    end
    out
end


const DEFAULT_ELEVATION_BANDS = (7, 15, 25, 45, 90)
const DEFAULT_HOUR_BANDS = 0:4:24

"""
    residual_breakdown(F, obs; time0, elevation_bands, hour_bands) -> NamedTuple

Where the post-fit wrms sits: rows per antenna, per selected diagonal pair of receptor slots, per elevation band and per UT band,
plus the per-baseline σ_floor census.

`(; antennas, receptor_slots, elevation, hour, floors, n_accepted, wrms_ps, chi2)`, each of the first four
a `Vector{NamedTuple}` with `n` and `wrms_ps`; `floors` is `(key, floor_ps)` sorted worst first.
`time0` is the run's own canonical epoch origin, which is what turns a row's `t` into UT.

Every number is a measurement of whatever dataset it is pointed at — there is no expected value here,
which is exactly what makes it a diagnostic and not a gate.
"""
function residual_breakdown(F, obs; time0, elevation_bands = DEFAULT_ELEVATION_BANDS,
                            hour_bands = DEFAULT_HOUR_BANDS)
    v = accepted_view(F)
    r, w, ix = v.r, v.w, v.ix
    band(m) = (; n = count(m), wrms_ps = count(m) == 0 ? NaN : 1e12wrms_of(r[m], w[m]))
    antennas = [(; antenna = st, band((obs.a1[ix] .== st) .| (obs.a2[ix] .== st))...)
                for st in sort(F.layout.antennas)]
    receptor_slots = [(; receptor_slots = receptor_slot_pair,
                         band(obs.receptor_slots[ix] .== Ref(receptor_slot_pair))...)
                      for receptor_slot_pair in sort(unique(obs.receptor_slots[ix]))]
    elmin = rad2deg.(min.(obs.el1[ix], obs.el2[ix]))
    elevation = [(; lo = float(elevation_bands[k]), hi = float(elevation_bands[k + 1]),
                    band((elmin .≥ elevation_bands[k]) .& (elmin .< elevation_bands[k + 1]))...)
                 for k in 1:length(elevation_bands) - 1]
    hours = ut_hours(obs.t[ix], time0)
    hb = collect(hour_bands)
    hour = [(; lo = float(hb[k]), hi = float(hb[k + 1]),
               band((hours .≥ hb[k]) .& (hours .< hb[k + 1]))...) for k in 1:length(hb) - 1]
    floors = sort!([(; baseline = k, floor_ps = 1e12v) for (k, v) in pairs(F.sigma_floor)];
                   by = x -> -x.floor_ps)
    (; antennas, receptor_slots, elevation, hour, floors,
       n_accepted = length(ix), wrms_ps = 1e12F.wrms, chi2 = F.chi2,
       floor_median_ps = median([f.floor_ps for f in floors]))
end

"UT hours of `t` [s after `time0`], counted from 0h of the session's first day."
ut_hours(t, time0) = t ./ 3600 .+ Dates.value(Second(time0 - DateTime(Date(time0)))) / 3600


"""
    split_solves(cfg, obs, splits; spec = delay_fit_spec(cfg)) -> (; groups, pairs, radec)

One solve per named split group, and the full position comparison in each split's explicit
`difference` direction.

Each split carries `kind`, a named `groups` tuple of masks, and an explicit `difference` identifying
the minuend and subtrahend. Two independent subsets must agree within their joint formal errors, and
where they do not the difference is a systematic the formal errors do not describe; `pairs` carries
the full decomposition of [`compare_positions`](@ref) so that a common frame shift is never read as
a per-source inconsistency.
"""
function split_solves(cfg, obs, splits; spec = delay_fit_spec(cfg))
    radec = source_radec(obs)
    groups = NamedTuple[]
    prs = NamedTuple[]
    for split in splits
        group_ids = keys(split.groups)
        solved = NamedTuple{group_ids}(map(group_ids) do group_id
            solve_variant(cfg, obs; spec, mask = getproperty(split.groups, group_id))
        end)
        for group_id in group_ids
            push!(groups, (; kind = split.kind, group_id,
                             getproperty(solved, group_id).metrics...))
        end
        minuend = getproperty(solved, split.difference.minuend)
        subtrahend = getproperty(solved, split.difference.subtrahend)
        c = compare_positions(minuend.P, subtrahend.P, radec)
        group_metrics = map(x -> x.metrics, solved)
        push!(prs, (; kind = split.kind, difference = split.difference,
                      group_metrics, n_sources = length(c.sources), comparison = c,
                      sigma_median_α = median(c.σα), sigma_median_δ = median(c.σδ)))
    end
    (; groups, pairs = prs, radec)
end

"""
    standard_splits(obs) -> Vector{NamedTuple}

The three splits every session's internal consistency is read on: receptor slot 2 minus receptor
receptor slot 1, the earlier half minus the later half, and odd scans minus even scans. Group storage order is
natural and independent of subtraction direction. Each split therefore carries a structured `kind`,
named `groups`, and an explicit `difference = (; id, minuend, subtrahend)`.

The splits are derived from the table itself — no session name, epoch, or antenna appears — so a
session with only one present diagonal receptor slot simply produces a shorter list.
"""
function standard_splits(obs)
    out = NamedTuple[]
    receptor1 = collect(obs.receptor_slots .== Ref((1, 1)))
    receptor2 = collect(obs.receptor_slots .== Ref((2, 2)))
    any(receptor1) && any(receptor2) &&
        push!(out, (; kind = :receptor_slots, groups = (; receptor1, receptor2),
                      difference = (; id = :receptor2_minus_receptor1,
                                      minuend = :receptor2, subtrahend = :receptor1)))
    tmid = median(obs.t)
    earlier = collect(obs.t .< tmid)
    later = collect(obs.t .≥ tmid)
    push!(out, (; kind = :session_halves, groups = (; earlier, later),
                  difference = (; id = :earlier_minus_later,
                                  minuend = :earlier, subtrahend = :later)))
    scans = sort(unique(obs.scan))
    odd_scans = Set(scans[1:2:end])
    odd = [s in odd_scans for s in obs.scan]
    even = [!(s in odd_scans) for s in obs.scan]
    push!(out, (; kind = :scan_parity, groups = (; odd, even),
                  difference = (; id = :odd_minus_even,
                                  minuend = :odd, subtrahend = :even)))
    out
end


"""
    antenna_jackknife(cfg, obs, antenna_names = all_antennas(obs); spec, reference = nothing)
        -> Vector{NamedTuple}

Drop each antenna in turn and compare the resulting positions with the full solve: one row per
dropped antenna with its metrics and the full [`compare_positions`](@ref) decomposition.

Dropping the reference antenna needs a new clock gauge. Which one is irrelevant to the positions —
fixing one antenna's clock is a pure gauge choice on the clock block — so the first antenna left is
taken, and the substitution is recorded in the row.
"""
function antenna_jackknife(cfg, obs, antenna_names = all_antennas(obs);
                           spec = delay_fit_spec(cfg), reference = nothing)
    radec = source_radec(obs)
    full = isnothing(reference) ? solve_variant(cfg, obs; spec) : reference
    out = NamedTuple[]
    for st in antenna_names
        mask = (obs.a1 .!= st) .& (obs.a2 .!= st)
        regauged = st === spec.reference_antenna
        sp = regauged ? respec(spec; reference_antenna = first(filter(!=(st), antenna_names))) : spec
        j = solve_variant(cfg, obs; spec = sp, mask)
        c = compare_positions(j.P, full.P, radec)
        push!(out, (; dropped = st, regauged, reference_antenna = sp.reference_antenna,
                      j.metrics..., n_common = length(c.sources), comparison = c,
                      max_total = maximum(c.total)))
    end
    out
end

"Every antenna named by an observable table, sorted."
all_antennas(obs) = sort(unique(vcat(collect(obs.a1), collect(obs.a2))))


"""
The binned censuses of the post-fit residual: the edges of each, and the column of the observable
table each is taken over. Elevation is the LOWER of the two antennas' (the limiting one), the hour is
UT, `iono_abs` and `sigma_iono` are in ps, and `snr` and `baseline_km` speak for themselves.
"""
const DEFAULT_RESIDUAL_BINS = (
    elevation   = [7, 10, 15, 20, 25, 35, 50, 90],
    hour_ut     = collect(0.0:2:24),
    iono_abs    = [0, 5, 10, 20, 40, 80, 160, 1000],
    sigma_iono  = [0, 5, 10, 15, 20, 30, 50, 500],
    snr         = [7, 15, 30, 60, 120, 250, 500, 1e6],
    baseline_km = [0, 1500, 2500, 3500, 5000, 6500, 9000])

"""
    residual_forensics(F, obs; time0, antenna_geometry = nothing, bins = DEFAULT_RESIDUAL_BINS,
                       periods = 1.0:0.05:30.0) -> NamedTuple

The full post-fit forensics of one solve, as data:

  * `baselines` — per baseline: length [km] (when `antenna_geometry` is given), n, wrms, the per-baseline
    σ_floor, the mean residual and the median SNR;
  * `antennas` — per antenna: n, wrms, and the SCAN-MEAN wander of the antenna-signed residual
    (`scan_rms_ps`) with its scan-to-scan step, which is the antenna-common structure a per-baseline
    noise floor cannot represent;
  * `sources` — [`per_source_stats`](@ref) with each source's declination;
  * `bins` — one row per (kind, bin) with n, wrms, mean and median |residual|;
  * `spectra` — per antenna, the amplitude of a sinusoid of each trial period fitted to the
    antenna-signed scan-mean residual: the diurnal check;
  * `closure` — the triangle closure of the ACCEPTED post-fit residuals, per (scan, receptor-slot pair), with
    the summary of its distribution.

`antenna_geometry` is the a priori antenna table (`astrometry_models(...).astrometry_model.antenna_geometry`) and is optional: without
it the baseline lengths are `NaN` and the `baseline_km` census is skipped, which is what a caller
holding only an observable table gets.
"""
function residual_forensics(F, obs; time0, antenna_geometry = nothing, bins = DEFAULT_RESIDUAL_BINS,
                            periods = collect(1.0:0.05:30.0))
    v = accepted_view(F)
    r, w, ix = v.r, v.w, v.ix
    sts = all_antennas(view(obs, ix))
    len(a, b) = isnothing(antenna_geometry) ? NaN : norm(antenna_geometry[a].xyz - antenna_geometry[b].xyz) / 1e3

    baselines = NamedTuple[]
    for (a1, a2) in sort(unique(collect(zip(obs.a1[ix], obs.a2[ix]))))
        m = (obs.a1[ix] .== a1) .& (obs.a2[ix] .== a2)
        push!(baselines, (; a1, a2, length_km = len(a1, a2), n = count(m),
                            wrms_ps = 1e12wrms_of(r[m], w[m]),
                            floor_ps = 1e12get(F.sigma_floor, minmax(a1, a2), NaN),
                            mean_ps = 1e12mean(r[m]), snr_median = median(obs.snr[ix[m]])))
    end
    sort!(baselines; by = x -> -x.wrms_ps)

    signed = Dictionary{Symbol, NamedTuple}()
    for st in sts
        sel = findall(k -> obs.a1[ix[k]] === st || obs.a2[ix[k]] === st, eachindex(ix))
        isempty(sel) && continue
        sgn = [obs.a2[ix[k]] === st ? 1.0 : -1.0 for k in sel]
        rs = r[sel] .* sgn
        sc = obs.scan[ix[sel]]
        us = sort(unique(sc))
        μ = [mean(rs[sc .== s]) for s in us]
        tμ = [median(obs.t[ix[sel][sc .== s]]) for s in us] ./ 3600
        set!(signed, st, (; sel, μ, tμ))
    end
    antenna_rows = [(; antenna = st, n = length(signed[st].sel),
                      wrms_ps = 1e12wrms_of(r[signed[st].sel], w[signed[st].sel]),
                      scan_rms_ps = 1e12sqrt(mean(abs2, signed[st].μ)),
                      scan_step_ps = length(signed[st].μ) < 2 ? NaN :
                                     1e12sqrt(mean(abs2, diff(signed[st].μ)) / 2),
                      n_scans = length(signed[st].μ)) for st in sts if haskey(signed, st)]

    ps = per_source_stats(F, obs)
    radec = source_radec(obs)
    sources = [(; source = s, ps[s]..., dec_deg = rad2deg(radec[s][2]))
               for s in sort(collect(keys(ps)))]

    values_for = Dict{Symbol, Vector{Float64}}(
        :elevation => rad2deg.(min.(obs.el1[ix], obs.el2[ix])),
        :hour_ut => ut_hours(obs.t[ix], time0),
        :iono_abs => 1e12abs.(obs.τ_ionosphere[ix]),
        :sigma_iono => 1e12 .* obs.σ_τ_ionosphere[ix],
        :snr => collect(float.(obs.snr[ix])))
    isnothing(antenna_geometry) ||
        (values_for[:baseline_km] = [len(obs.a1[i], obs.a2[i]) for i in ix])
    binrows = NamedTuple[]
    for (kind, edges) in pairs(bins)
        haskey(values_for, kind) || continue
        x = values_for[kind]
        for k in 1:length(edges) - 1
            m = (x .≥ edges[k]) .& (x .< edges[k + 1])
            count(m) == 0 && continue
            push!(binrows, (; kind, lo = float(edges[k]), hi = float(edges[k + 1]), n = count(m),
                              wrms_ps = 1e12wrms_of(r[m], w[m]), mean_ps = 1e12mean(r[m]),
                              median_abs_ps = 1e12median(abs.(r[m]))))
        end
    end

    specrows = NamedTuple[]
    for st in sts
        haskey(signed, st) || continue
        μ = signed[st].μ .- mean(signed[st].μ)
        tt = signed[st].tμ
        length(μ) < 4 && continue
        for P in periods
            ω = 2π / P
            A = [cos.(ω .* tt) sin.(ω .* tt)]
            c = A \ μ
            push!(specrows, (; antenna = st, period_h = P, amplitude_ps = 1e12 * hypot(c...)))
        end
    end

    closure = residual_closure(r, ix, obs)
    (; baselines, antennas = antenna_rows, sources, bins = binrows, spectra = specrows, closure)
end

"""
    residual_closure(r, ix, obs) -> (; triangles, summary)

Triangle sums of the post-fit residuals per (scan, receptor-slot pair). Antenna-based terms cancel exactly in a
triangle, so what is left is measurement noise plus whatever does not close — the same statistic as
[`closure_floor`](@ref) applied to the residual rather than to the raw delay.
"""
function residual_closure(r, ix, obs)
    byscan = Dict{Tuple{Int,NTuple{2,Int}}, Dict{Tuple{Symbol,Symbol}, Float64}}()
    for k in eachindex(ix)
        i = ix[k]
        d = get!(() -> Dict{Tuple{Symbol,Symbol}, Float64}(), byscan,
                 (obs.scan[i], obs.receptor_slots[i]))
        d[(obs.a1[i], obs.a2[i])] = r[k]
    end
    cl = Float64[]
    rows = NamedTuple[]
    for ((sc, receptor_slots), d) in byscan
        ants = sort(unique(vcat([k[1] for k in keys(d)], [k[2] for k in keys(d)])))
        for i in 1:length(ants), j in i+1:length(ants), k in j+1:length(ants)
            p, q, s = ants[i], ants[j], ants[k]
            (haskey(d, (p, q)) && haskey(d, (q, s)) && haskey(d, (p, s))) || continue
            c = d[(p, q)] + d[(q, s)] - d[(p, s)]
            push!(cl, c)
            push!(rows, (; scan = sc, receptor_slots, a1 = p, a2 = q, a3 = s,
                           closure_ps = 1e12c))
        end
    end
    summary = isempty(cl) ?
        (; n = 0, median_abs_ps = NaN, robust_sigma_ps = NaN, rms_ps = NaN, p99_ps = NaN) :
        (; n = length(cl), median_abs_ps = 1e12median(abs.(cl)),
           robust_sigma_ps = 1e12 * 1.4826 * median(abs.(cl .- median(cl))),
           rms_ps = 1e12sqrt(mean(abs2, cl)), p99_ps = 1e12quantile(abs.(cl), 0.99))
    (; triangles = rows, summary)
end


"""
    constraint_chi2(F, spec) -> NamedTuple

The constraint blocks' own χ² per row — `clock`, `zwd`, `gradient` and each one's maximum pull. This
is where prior-versus-data tension shows: a constraint whose χ² is far below 1 is not constraining
anything, and one far above it is fighting the data.
"""
function constraint_chi2(F, spec)
    L = F.layout
    dtc = L.clock_nodes[2] - L.clock_nodes[1]
    dtz = L.zwd_nodes[2] - L.zwd_nodes[1]
    pc = vcat([diff(v) ./ dtc ./ spec.clock_rate_constraint for v in antenna_clocks(F)]...)
    pz = vcat([diff(v) ./ dtz ./ spec.zwd_rate_constraint for v in antenna_zwd(F)]...)
    pg = vcat([collect(g) ./ spec.gradient_constraint for g in antenna_gradients(F)]...)
    (; clock = mean(abs2, pc), zwd = mean(abs2, pz), gradient = mean(abs2, pg),
       clock_max = maximum(abs, pc), zwd_max = maximum(abs, pz), gradient_max = maximum(abs, pg))
end

"""
The estimator knobs the stress sweep moves, and the acceptance knobs swept beside them. Each is a
FIELD of `GlobalDelayFit` with the factors to multiply it by; the acceptance entries are absolute
values, because an elevation cutoff of half of ten degrees is not a meaningful thing to ask for.
"""
const DEFAULT_STRESS_KNOBS = (
    scaled = [(:zwd_rate_constraint, "ZWD rate σ"), (:clock_rate_constraint, "clock rate σ"),
              (:gradient_constraint, "gradient σ"), (:zwd_node, "ZWD node"),
              (:clock_node, "clock node")],
    factors = (0.5, 2.0),
    absolute = [("elevation 5°", (; elevation_cutoff = deg2rad(5))),
                ("elevation 10°", (; elevation_cutoff = deg2rad(10))),
                ("elevation 15°", (; elevation_cutoff = deg2rad(15))),
                ("SNR ≥ 15", (; min_snr = 15.0)), ("SNR ≥ 30", (; min_snr = 30.0))])

"""
    parameterization_stress(cfg, obs, knobs = DEFAULT_STRESS_KNOBS; spec, reference = nothing,
                            iono_off = true) -> Vector{NamedTuple}

Re-solve with each constraint σ and node spacing halved and doubled, with each acceptance knob at
its swept value, and — when `iono_off` — with the ionospheric correction switched off, and report how
far the positions move against the reference solve.

One row per variant: the metrics, the constraint χ², the position movement (rms, max and pull) and
the full comparison. What it measures is how much of the answer is the data and how much is the
parameterization; a knob that moves the positions by more than their σ is a knob the result depends
on.
"""
function parameterization_stress(cfg, obs, knobs = DEFAULT_STRESS_KNOBS;
                                 spec = delay_fit_spec(cfg), reference = nothing, iono_off = true)
    radec = source_radec(obs)
    base = isnothing(reference) ? solve_variant(cfg, obs; spec) : reference
    cb = constraint_chi2(base.F, spec)
    rows = NamedTuple[(; knob = "reference", factor = 1.0, base.metrics...,
                         shift_α = 0.0, shift_δ = 0.0, max_shift = 0.0, pull = 0.0,
                         c_clock = cb.clock, c_zwd = cb.zwd, c_gradient = cb.gradient,
                         comparison = nothing)]
    record(label, factor, sp) = begin
        j = solve_variant(cfg, obs; spec = sp)
        c = compare_positions(j.P, base.P, radec)
        cc = try constraint_chi2(j.F, sp) catch; (; clock = NaN, zwd = NaN, gradient = NaN) end
        push!(rows, (; knob = label, factor, j.metrics...,
                       shift_α = c.rms_α, shift_δ = c.rms_δ, max_shift = maximum(c.total),
                       pull = c.pull_rms, c_clock = cc.clock, c_zwd = cc.zwd,
                       c_gradient = cc.gradient, comparison = c))
    end
    for (field, label) in knobs.scaled, factor in knobs.factors
        record(label, factor, respec(spec; (field => getfield(spec, field) * factor,)...))
    end
    for (label, kw) in knobs.absolute
        record(label, NaN, respec(spec; kw...))
    end
    if iono_off
        Ono = with_residual(obs, obs.τ_residual .+ obs.τ_ionosphere)
        j = solve_variant(cfg, Ono; spec)
        c = compare_positions(j.P, base.P, radec)
        cc = constraint_chi2(j.F, spec)
        push!(rows, (; knob = "τ_ionosphere off", factor = NaN, j.metrics...,
                       shift_α = c.rms_α, shift_δ = c.rms_δ, max_shift = maximum(c.total),
                       pull = c.pull_rms, c_clock = cc.clock, c_zwd = cc.zwd,
                       c_gradient = cc.gradient, comparison = c))
    end
    rows
end


"""
    systematics_transfer(cfg, obs, injections; spec, reference = nothing) -> Vector{NamedTuple}

How much of a known, physically-shaped delay pattern reaches the positions.

`injections` is a vector of `(; label, family, delta)` — `delta` a per-row delay [s] added to the
fitted quantity `τ_residual`. Each is re-solved with the estimator's own machinery
fully active (clock and
ZWD piecewise-linear models, gradients, per-baseline reweighting, outlier rejection), so what is
measured is the part the model does NOT absorb, which is exactly the quantity the error budget needs.

The response is linear in the injected amplitude — the solve is a linear WLS at fixed weights — so
each row is reported both as an absolute shift and as a TRANSFER COEFFICIENT per 10 ps of injected
rms.

The injections themselves are built by the caller ([`injection_families`](@ref) builds the standard
set), which is what keeps this function free of any dataset knowledge.
"""
function systematics_transfer(cfg, obs, injections; spec = delay_fit_spec(cfg), reference = nothing)
    radec = source_radec(obs)
    base = isnothing(reference) ? solve_variant(cfg, obs; spec) : reference
    map(injections) do inj
        amp = 1e12sqrt(mean(abs2, inj.delta))
        j = solve_variant(cfg, with_residual(obs, obs.τ_residual .+ inj.delta); spec)
        c = compare_positions(j.P, base.P, radec)
        (; inj.label, inj.family, amplitude_ps = amp, j.metrics...,
           shift_α = c.rms_α, shift_δ = c.rms_δ, max_shift = maximum(c.total),
           mean_α = c.mean_α, mean_δ = c.mean_δ, pull = c.pull_rms,
           rotation = c.rotation, rot_rms_α = c.rot_rms_α, rot_rms_δ = c.rot_rms_δ,
           per10_α = c.rms_α * 10 / amp, per10_δ = c.rms_δ * 10 / amp, comparison = c)
    end
end

"""
    injection_families(obs, astrometry_model, cfg; reps = 5, rng = MersenneTwister(1), diurnal_ps = 15.0,
                       zwd_ps = 20.0, iono_scales = (-0.2, 0.2)) -> Vector{NamedTuple}

Standard systematic split families represented as values on this observable table:

  * `diurnal` / `semidiurnal` — a per-antenna sinusoid at the scale of the unattributed model
    residual, random phase per antenna, `reps` realizations;
  * `iono_product` — the differentials between the a priori model's ionosphere products (the
    3-product spread, not a proxy). Requires `astrometry_model`; skipped when it is `nothing`;
  * `iono_scale` — the applied correction mis-scaled, which is the honest one-parameter stand-in for
    a residual ionosphere the product spread cannot certify;
  * `zwd_turbulence` — per-antenna Kolmogorov-like zenith wet delay (PSD ∝ f^{-8/3}) beyond the PWL
    node spacing, mapped through the observation's own wet mapping function.

Random phases are a STATED condition of the measurement, not synthetic data standing in for real
data: the injected pattern is added to the real observable table and the real solve runs on it. The
`rng` is an argument so a caller can reproduce a run exactly.
"""
function injection_families(obs, astrometry_model, cfg; reps = 5, rng = MersenneTwister(1),
                            diurnal_ps = 15.0, zwd_ps = 20.0, iono_scales = (-0.2, 0.2))
    sts = all_antennas(obs)
    T0, T1 = extrema(obs.t)
    antenna_delta(f) = [f(obs.a2[i], i) - f(obs.a1[i], i) for i in eachindex(obs)]
    out = NamedTuple[]
    for (family, ω) in (("diurnal", 2π / 86400), ("semidiurnal", 4π / 86400))
        for rep in 1:reps
            φ = Dictionary(sts, 2π .* rand(rng, length(sts)))
            δ = antenna_delta((st, i) -> 1e-12diurnal_ps * sin(ω * obs.t[i] + φ[st]))
            push!(out, (; label = "$family $(diurnal_ps) ps #$rep", family, delta = δ))
        end
    end
    if !isnothing(astrometry_model)
        cache = Dict{Tuple{Float64,Symbol,Symbol}, Any}()
        ionoof(st, i) = get!(cache, (obs.t[i], obs.source[i], st)) do
            antenna_delay_terms(astrometry_model, st, obs.ra[i], obs.dec[i], obs.t[i]).τ_ionosphere
        end
        for other in keys(astrometry_model.ionex)
            other === astrometry_model.iono_primary && continue
            δ = antenna_delta((st, i) -> (v = ionoof(st, i); v[other] - v[astrometry_model.iono_primary]))
            push!(out, (; label = "τ_ionosphere $(other)−$(astrometry_model.iono_primary)", family = "iono_product",
                          delta = δ))
        end
    end
    for f in iono_scales
        push!(out, (; label = @sprintf("τ_ionosphere scale %+.0f%%", 100f), family = "iono_scale",
                      delta = f .* collect(obs.τ_ionosphere)))
    end
    for rep in 1:reps
        z = Dictionary(sts, [turbulent_series(rng, obs.t, 1e-12zwd_ps, T0, T1) for _ in sts])
        δ = [obs.wet_mapping2[i] * z[obs.a2[i]][i] - obs.wet_mapping1[i] * z[obs.a1[i]][i] for i in eachindex(obs)]
        push!(out, (; label = "ZWD turbulence $(zwd_ps) ps zenith #$rep",
                      family = "zwd_turbulence", delta = δ))
    end
    out
end

"""
Spectral synthesis of one antenna's zenith wet delay: `nmodes` sinusoids over the session length with
amplitudes ∝ f^{-4/3} (PSD ∝ f^{-8/3}, the frozen-Kolmogorov-screen delay spectrum) and random
phases, normalized to `rms` seconds. Evaluated directly at the observation epochs, so nothing is
smoothed away by an interpolation grid.
"""
function turbulent_series(rng, ts, rms, T0, T1; nmodes = 200)
    fs = (1 / (T1 - T0)) .* (1:nmodes)
    a = fs .^ (-4 / 3)
    φ = 2π .* rand(rng, nmodes)
    y = [sum(a[k] * cos(2π * fs[k] * (t - T0) + φ[k]) for k in 1:nmodes) for t in ts]
    y .-= mean(y)
    y .* (rms / sqrt(mean(abs2, y)))
end

"""
    transfer_summary(rows) -> Vector{NamedTuple}

Per injection family: the mean transfer coefficient per 10 ps and its scatter over the realizations,
plus the absolute shift at the injected level. This is the form the error budget quotes.
"""
transfer_summary(rows) =
    map(unique(r.family for r in rows)) do fam
        g = [r for r in rows if r.family == fam]
        sd(v) = length(v) > 1 ? std(v) : 0.0
        (; family = fam, n = length(g),
           per10_α = mean(x -> x.per10_α, g), per10_α_std = sd([x.per10_α for x in g]),
           per10_δ = mean(x -> x.per10_δ, g), per10_δ_std = sd([x.per10_δ for x in g]),
           shift_α = mean(x -> x.shift_α, g), shift_δ = mean(x -> x.shift_δ, g))
    end


"""
    split_transfer(cfg, obs, splits, injections; spec) -> Vector{NamedTuple}

Each injection propagated THROUGH each split: an injected systematic that drives a split difference
must CHANGE that difference by as much as the split shows. One row per (injection, split) with the
reference split difference, the injected one, and the change between them.

This is the arbiter between the two explanations of a large split: a time-correlated systematic, or
the geometry a subset loses. A systematic that changes the split difference by a tenth of what the
split shows is not what drives it.
"""
function split_transfer(cfg, obs, splits, injections; spec = delay_fit_spec(cfg))
    radec = source_radec(obs)
    ref = Dict{Symbol, Any}()
    for split in splits
        solutions = Dict(group_id => solve_variant(cfg, obs; spec,
                                                   mask = getproperty(split.groups, group_id))
                         for group_id in keys(split.groups))
        ref[split.kind] = compare_positions(solutions[split.difference.minuend].P,
                                            solutions[split.difference.subtrahend].P, radec)
    end
    rows = NamedTuple[]
    for inj in injections
        Oi = with_residual(obs, obs.τ_residual .+ inj.delta)
        for split in splits
            solutions = Dict(group_id => solve_variant(cfg, Oi; spec,
                                                       mask = getproperty(split.groups, group_id))
                             for group_id in keys(split.groups))
            c = compare_positions(solutions[split.difference.minuend].P,
                                  solutions[split.difference.subtrahend].P, radec)
            r = ref[split.kind]
            push!(rows, (; injection = inj.label, inj.family, kind = split.kind,
                           difference = split.difference.id,
                           minuend = split.difference.minuend,
                           subtrahend = split.difference.subtrahend,
                           amplitude_ps = 1e12sqrt(mean(abs2, inj.delta)),
                           rms_α = c.rms_α, rms_δ = c.rms_δ,
                           ref_rms_α = r.rms_α, ref_rms_δ = r.rms_δ,
                           change_α = sqrt(mean(abs2, c.dα .- r.dα)),
                           change_δ = sqrt(mean(abs2, c.dδ .- r.dδ))))
        end
    end
    (; rows, reference = ref)
end

"""
    split_geometry(obs, splits, reference) -> Vector{NamedTuple}

What each source LOSES in a split subset: scans, hour-angle span and time span in the thinner arm,
against how far that source moved. A source seen in two scans of a half-session is degenerate with
the antenna clock in a way a 24-hour source is not, and this is the table that says so.
"""
function split_geometry(obs, splits, reference)
    T0 = minimum(obs.t)
    geom(mask) = begin
        out = Dictionary{Symbol, NamedTuple}()
        idx = findall(mask)
        for s in unique(obs.source[idx])
            ir = [i for i in idx if obs.source[i] === s]
            ha = [mod(2π * (obs.t[i] - T0) / 86164.0905 - obs.ra[i], 2π) for i in ir]
            el = vcat(obs.el1[ir], obs.el2[ir])
            set!(out, s, (; n_scans = length(unique(obs.scan[ir])), n_obs = length(ir),
                            ha_span = maximum(ha) - minimum(ha),
                            el_span = rad2deg(maximum(el) - minimum(el)),
                            t_span = (maximum(obs.t[ir]) - minimum(obs.t[ir])) / 3600))
        end
        out
    end
    rows = NamedTuple[]
    for split in splits
        group_geometry = Dict(group_id => geom(getproperty(split.groups, group_id))
                              for group_id in keys(split.groups))
        minuend_geometry = group_geometry[split.difference.minuend]
        subtrahend_geometry = group_geometry[split.difference.subtrahend]
        c = reference[split.kind]
        for (k, s) in enumerate(c.sources)
            (haskey(minuend_geometry, s) && haskey(subtrahend_geometry, s)) || continue
            push!(rows, (; kind = split.kind, difference = split.difference.id,
                           minuend = split.difference.minuend,
                           subtrahend = split.difference.subtrahend, source = s,
                           n_scans_min = min(minuend_geometry[s].n_scans,
                                             subtrahend_geometry[s].n_scans),
                           ha_span_min_deg = rad2deg(min(minuend_geometry[s].ha_span,
                                                        subtrahend_geometry[s].ha_span)),
                           t_span_min_h = min(minuend_geometry[s].t_span,
                                               subtrahend_geometry[s].t_span),
                           dα = c.dα[k], dδ = c.dδ[k], total = c.total[k],
                           σα = c.σα[k], σδ = c.σδ[k]))
        end
    end
    rows
end
