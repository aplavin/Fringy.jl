
export instrumental_arm, pcal_sign_regression, instrumental_arms, interif_structure,
       cable_application_scan, instrument_fraction_scan, rfc_agreement


"""
    instrumental_arm(instr, scale) -> (; applied, tones, summary)

[`instrumental`](@ref)'s result with its installed states scaled by `scale`: `1` is the run's own arm,
`0` the UNCALIBRATED control, `-1` the sign-inverted one.

An arm of a DIAGNOSTIC experiment, exactly as `cable_apply` is (`steps/instrumental.jl`) — the step is
unchanged and the arm goes into the unmodified [`fringefit`](@ref), which reads nothing of `instr` but
its `applied` states. `CapturedAppliedPhase` is a plain value with six array fields, so the scaling is
exact and total.

    fringefit(cfg, ld, instrumental_arm(instr, 0))     # the uncalibrated control
    fringefit(cfg, ld, instrumental_arm(instr, -1))    # the sign-inverted arm

`summary` gains `arm_scale`, so a product built from an arm says which arm it is.
"""
instrumental_arm(instr, scale) =
    (; applied = CapturedAppliedPhase(; instr.applied.ν, instr.applied.if_of, instr.applied.fcell,
                                       instr.applied.νref,
                                       delay = scale .* instr.applied.delay,
                                       phase = scale .* instr.applied.phase),
       instr.tones,
       summary = merge(instr.summary, (; arm_scale = float(scale))))


"""
The nuisance bases the sign regression is measured against, in increasing flexibility: a session
quadratic, then piecewise-linear clock models on 6 h and 3 h nodes.

A finer basis suppresses the troposphere/clock residual — hence sharpens the slope — but also starts
absorbing the instrumental wander itself, which lives on 0.2–3 h scales. The slope is therefore
reported for ALL of them and must stay ≈ +1 across them for the conclusion to mean anything.
"""
const DEFAULT_NUISANCE_BASES = ["quadratic (session)" => nothing, "PWL 6 h nodes" => 6.0,
                                "PWL 3 h nodes" => 3.0]

"""
    pcal_sign_regression(applied, obs; nuisance_bases = DEFAULT_NUISANCE_BASES,
                         sigma_gate = 10e-12, min_series = 12, targeted = true) -> NamedTuple

The empirical sign of the applied phase-cal, as a matched-filter regression of the UNCALIBRATED
whole-band delays on the pcal prediction.

`PcalInit` stores the tones AS MEASURED and this project's components are the CORRUPTION, so applying
that state SUBTRACTS the instrument. That the stored direction is the one that REMOVES it — rather
than doubles it — is not a convention to reason about but a measurement, and this is that measurement.

  * `applied` is the calibrated run's [`CapturedAppliedPhase`](@ref) (or the whole `instrumental`
    result): the prediction;
  * `obs` is the tidy peak table of an UNCALIBRATED pass — `fringefit(cfg, ld,
    instrumental_arm(instr, 0)).obs`, the arm above.

Per (baseline, selected diagonal receptor slot) the prediction is the pcal delay difference `τ_p − τ_q`, band-combined with
the SAME per-IF weights the fringe sum used; a nuisance basis in time is projected out of both it and
the measured delay; a common slope is then fitted globally. Slope **+1** means the uncalibrated data
CONTAINS the pcal delay with the stored sign, i.e. dividing it out is what removes it; **−1** would
mean the stored sign is inverted and `PcalInit` needs flipping.

The observable identifies each product by its ordered structural receptor-slot pair; this diagnostic does not
inspect antenna receptor labels.

Returns `(; bases, best, targeted, verdict, n_series)`, where each basis row carries `slope`, `se`,
`corr`, `n`, the prediction and measurement rms and the three residual rms (with this sign, with no
pcal, with the opposite), and `verdict` is `:confirmed` / `:inverted` / `:inconclusive`.

Two caveats the numbers carry with them. (a) The prediction is the weighted mean of the per-IF
DELAYS, which equals the whole-band delay of the installed states only because the instrumental
variation is band-common — a term present in one and not the other would not show up here. (b) `se`
is the textbook slope error and is OPTIMISTIC: the projected residuals are correlated within each
series and the nuisance projection has not been charged for its degrees of freedom. Read it as a
scale, not as a p-value.
"""
function pcal_sign_regression(applied, obs; nuisance_bases = DEFAULT_NUISANCE_BASES,
                              sigma_gate = 10e-12, min_series = 12, targeted = true)
    ai = captured_applied_phase(applied)
    P = ai.delay
    ifcells = 1:size(P, 4)
    groups = Dict{Tuple{Symbol,NTuple{2,Int}}, Vector{NTuple{3,Float64}}}()
    for i in eachindex(obs)
        obs.sigma_delay[i] < sigma_gate || continue
        p, q = obs.baseline_ix[i].antennas
        receptor_slots = obs.receptor_slots[i]
        receptor_slot = _astrometric_receptor_slot(receptor_slots,
                                                    "pcal_sign_regression: row $i")
        w = obs.ifweights[i]
        sum(w) > 0 || continue
        x = sum(w[k] * (P[p, receptor_slot, obs.tcell[i], k] -
                        P[q, receptor_slot, obs.tcell[i], k]) for k in ifcells) /
            sum(w)
        push!(get!(() -> NTuple{3,Float64}[], groups,
                   (Symbol(obs.baseline[i]), receptor_slots)),
              (obs.t[i], x, obs.delay[i]))
    end
    isempty(groups) &&
        error("pcal_sign_regression: no observable survived the σ_τ < $(1e12sigma_gate) ps gate — " *
              "nothing to regress")
    bases = [(; label, result = _project_series(groups, spacing; min_series))
             for (label, spacing) in nuisance_bases]
    best = first(bases).result
    (; bases, best, n_series = length(groups),
       targeted = targeted ? _targeted_window(groups, ai, obs; min_series) : nothing,
       verdict = best.slope > 0.5 ? :confirmed : best.slope < -0.5 ? :inverted : :inconclusive)
end

"Triangular (hat) basis functions of a piecewise-linear model on `nodes`."
function pwl_basis(t, nodes)
    A = zeros(length(t), length(nodes))
    for (j, c) in enumerate(nodes), i in eachindex(t)
        h = j == 1 ? nodes[2] - nodes[1] : nodes[j] - nodes[j - 1]
        h2 = j == length(nodes) ? nodes[end] - nodes[end - 1] : nodes[j + 1] - nodes[j]
        A[i, j] = max(0.0, 1 - (t[i] < c ? (c - t[i]) / h : (t[i] - c) / h2))
    end
    A
end

"""
The common slope over every series, after projecting the nuisance basis out of BOTH the prediction and
the measurement. `spacing` is the PWL node spacing in hours, or `nothing` for a session quadratic.

The nodes must SPAN each series: a last node short of the last sample would force the model to decay
to zero there and leave nanoseconds of "residual" that are pure basis artefact.
"""
function _project_series(groups, spacing; min_series = 12)
    X = Float64[]; Y = Float64[]
    for (_, v) in groups
        length(v) ≥ min_series || continue
        t = [s[1] for s in v] ./ 3600
        A = isnothing(spacing) ? [ones(length(t)) (t .- mean(t)) (t .- mean(t)) .^ 2] :
            pwl_basis(t, range(minimum(t), maximum(t);
                               length = max(2, ceil(Int, (maximum(t) - minimum(t)) / spacing) + 1)))
        Q = I - A * pinv(A)
        append!(X, Q * [s[2] for s in v]); append!(Y, Q * [s[3] for s in v])
    end
    _slope_report(X, Y)
end

function _slope_report(X, Y)
    (isempty(X) || sum(abs2, X) == 0) &&
        return (; slope = NaN, se = NaN, corr = NaN, n = length(X), xrms = NaN, yrms = NaN,
                  with = NaN, against = NaN)
    slope = sum(X .* Y) / sum(abs2, X)
    resid = Y .- slope .* X
    (; slope, se = sqrt(sum(abs2, resid) / (length(X) - 1) / sum(abs2, X)),
       corr = cor(X, Y), n = length(X), xrms = sqrt(mean(abs2, X)), yrms = sqrt(mean(abs2, Y)),
       with = sqrt(mean(abs2, Y .- X)), against = sqrt(mean(abs2, Y .+ X)))
end

"""
The same regression restricted to the one antenna and the one time window where the instrument moves
MOST — a feature no clock polynomial can follow, so over a short window a per-baseline LINE is enough
nuisance and the prediction is not being fitted away.

BOTH the antenna and the window come from THIS session's own pcal series, never inherited: the antenna
with the largest pcal delay scatter, and the contiguous run of cells around its largest departure from
its own median where the departure is still at least half as large, padded by its own half-width on
each side so that the linear nuisance has quiet data on both flanks. Reported, never gated: one
antenna over a few hours is a small sample, and a session with no localized excursion has no sharp
version of this test to run.
"""
function _targeted_window(groups, ai, obs; min_series = 12)
    name_of = Dict{Int, Symbol}()
    for i in eachindex(obs)
        p, q = obs.baseline_ix[i].antennas
        n1, n2 = obs.baseline[i].antennas
        name_of[p] = n1; name_of[q] = n2
    end
    cells = sort(unique(obs.tcell))
    tref = Dict(c => median(obs.t[findall(==(c), obs.tcell)]) for c in cells)
    P = ai.delay
    ifcells = 1:size(P, 4)
    series = Dict{Int, Vector{Float64}}()
    for p in keys(name_of)
        series[p] = map(cells) do c
            vals = [P[p, i, c, k] for i in 1:size(P, 2), k in ifcells if !iszero(P[p, i, c, k])]
            isempty(vals) ? NaN : mean(vals)
        end
    end
    rms_of(s) = (f = filter(isfinite, s); length(f) < 3 ? 0.0 : std(f))
    isempty(series) && return nothing
    st = argmax(p -> rms_of(series[p]), collect(keys(series)))
    s = series[st]
    rms_of(s) == 0.0 && return nothing
    m = median(filter(isfinite, s))
    dev = map(x -> isfinite(x) ? abs(x - m) : 0.0, s)
    kmax = argmax(dev)
    half = 0.5dev[kmax]
    klo = something(findprev(<(half), dev, kmax), 0) + 1
    khi = something(findnext(<(half), dev, kmax), length(dev) + 1) - 1
    pad = (tref[cells[khi]] - tref[cells[klo]]) / 2 + 1800.0
    t_lo, t_hi = tref[cells[klo]] - pad, tref[cells[khi]] + pad
    X = Float64[]; Y = Float64[]
    for (key, v) in groups
        occursin(string(name_of[st]), string(key[1])) || continue
        w = filter(x -> t_lo ≤ x[1] ≤ t_hi, v)
        length(w) ≥ min_series || continue
        t = [x[1] for x in w] ./ 3600; t = t .- mean(t)
        Q = I - [ones(length(t)) t] * pinv([ones(length(t)) t])
        append!(X, Q * [x[2] for x in w]); append!(Y, Q * [x[3] for x in w])
    end
    (; antenna = name_of[st], delay_rms = rms_of(s), excursion = dev[kmax],
       t_excursion = tref[cells[kmax]], t_lo, t_hi, _slope_report(X, Y)...)
end


"""
    interif_structure(rows; snr_gate) -> NamedTuple

The per-IF delay structure of a per-IF fringe table, split into a session-STATIC pattern and the
scan-to-scan VARIATION about it.

Per (baseline, receptor-slot pair, scan) the per-IF delays are reduced to their deviations from the band mean,
which cancels every antenna term common to the band (clock, troposphere, ionosphere, source position)
and leaves exactly the per-IF instrumental structure. Its scan-to-scan scatter about each
(baseline, IF) session mean is what per-scan phase-cal is for.

`rows` is `fringefit(...; frequency = :if).table.rows`.
"""
function interif_structure(rows; snr_gate)
    key = Dict{Tuple{Symbol,Int,Int}, Vector{Float64}}()
    nkept = 0
    for r in rows, ip in (1, 2)
        d = [r.cells[k][ip, ip].delay for k in eachindex(r.fcells)]
        s = [r.cells[k][ip, ip].snr for k in eachindex(r.fcells)]
        (all(isfinite, d) && all(≥(snr_gate), s)) || continue
        nkept += 1
        m = mean(d)
        for k in eachindex(d)
            push!(get!(() -> Float64[], key, (Symbol(r.baseline), ip, k)), d[k] - m)
        end
    end
    statics = Float64[]; varies = Float64[]
    for (_, v) in key
        length(v) ≥ 3 || continue
        push!(statics, mean(v))
        append!(varies, v .- mean(v))
    end
    (; n = nkept, ngroups = length(statics),
       static_rms = isempty(statics) ? NaN : sqrt(mean(abs2, statics)),
       var_rms = isempty(varies) ? NaN : sqrt(mean(abs2, varies)),
       var_mad = isempty(varies) ? NaN : 1.4826 * median(abs.(varies)))
end

"""
    instrumental_arms(cfg, ld, instr; arms = (0, 1, -1), snr_gate, min_snr = cfg.min_snr)
        -> (; arms, lobe, reading)

Empirically validate phase-cal application by running the identical chain
on each ARM of [`instrumental_arm`](@ref) — `0` no pcal, `1` as `PcalInit` stores it, `-1` negated —
and comparing two things the instrument controls:

  (a) whole-band fringe SNR, which must not degrade;
  (b) the scan-to-scan variation of the per-IF delay structure ([`interif_structure`](@ref)).

Reading (b) honestly is part of the diagnostic and is why it returns numbers rather than a verdict:
the per-IF deviations isolate the per-IF RELATIVE instrumental delay, whose per-scan estimate is
dominated by the tone-pair noise itself, so applying it adds tone noise in quadrature to this
statistic SYMMETRICALLY in sign — it does not decide the sign, and expecting it to would be an error.
The band-common wander that phase-cal is for cancels exactly in an inter-IF deviation and is measured
by [`pcal_sign_regression`](@ref) instead.

The static part of the inter-IF pattern is unchanged (the tone-pair delays are ambiguous
modulo the tone spacing), and whether that is harmless depends entirely on how big it is against the
WIDTH OF THE WHOLE-BAND MAIN LOBE: small, it is a per-(antenna, receptor slot) bias the estimator's clock
absorbs; approaching the lobe width, it starts DECIDING WHICH LOBE the whole-band search locks onto,
deterministically and all session, and no clock parameter can undo that. `reading` is that ratio and
its interpretation, computed rather than asserted.
"""
function instrumental_arms(cfg, ld, instr; arms = (0, 1, -1), snr_gate = 150.0,
                           min_snr = cfg.min_snr)
    out = NamedTuple[]
    for scale in arms
        a = instrumental_arm(instr, scale)
        band = fringefit(cfg, ld, a; frequency = :band)
        perif = fringefit(cfg, ld, a; frequency = :if)
        snr = collect(float.(band.obs.snr))
        push!(out, (; arm_scale = float(scale), n_peaks = length(snr),
                      snr_median = median(snr),
                      detection_fraction = count(≥(min_snr), snr) / length(snr),
                      interif = interif_structure(perif.table.rows; snr_gate)))
    end
    ν = ld.ds.freq.ν
    B = maximum(ν) - minimum(ν)
    lobe = 1 / B
    ref = out[findfirst(a -> a.arm_scale == 1.0, out)]
    ratio = ref.interif.static_rms / lobe
    (; arms = out, lobe, bandwidth = B, static_ratio = ratio,
       reading = ratio < 0.4 ? :small_against_the_main_lobe : :not_negligible)
end


"""
    cable_application_scan(cfg, arms; spec = delay_fit_spec(cfg), rfc_references = nothing) -> Vector{NamedTuple}

The cable-application / captured-phase-mode experiment: one row per ARM, where an arm is
`(; name, obs, captured_phase_mode)` — an observable table of one `cable_apply` run, expressed in one
observable form.

Two orthogonal choices are decided here on evidence over the same observable set: captured phase
(`captured_phase_mode = :add_back`, which algebraically adds the captured tone-derived applied-phase
slope `τ_applied_phase` to the total, corrected, and residual columns, or `:retain`, which leaves it out) and cable application
(`cable_apply = :both`, `:delay`, or `:none`). Each arm reports the full solve, the
split by structural receptor-slot identifier with its frame shift between receptor slots, the two time splits, and — when `rfc_references` are
given — the external-catalogue agreement after a rigid rotation.

Every arm is a value the caller supplies: the arms of the cable-application experiment are separate RUNS of the
pipeline under a different `cable_apply`, which is a work directory each, and finding them is the
entry script's business, not this function's.
"""
function cable_application_scan(cfg, arms; spec = delay_fit_spec(cfg), rfc_references = nothing)
    map(arms) do arm
        O = with_captured_phase_mode(arm.obs, arm.captured_phase_mode;
                                     current_mode = get(arm, :current_mode, cfg.captured_phase_mode))
        radec = source_radec(O)
        full = solve_variant(cfg, O; spec)
        sp = split_solves(cfg, O, standard_splits(O); spec)
        (; arm.name, captured_phase_mode = arm.captured_phase_mode, full.metrics...,
           splits = sp.pairs, positions = full.P, radec,
           external = isnothing(rfc_references) ? nothing :
                      rfc_agreement(full.P, radec, rfc_references))
    end
end

"""
    rfc_agreement(P, radec, rfc_references) -> NamedTuple

Agreement of a position table with an external absolute catalogue: the rms of (solve − catalogue) over
the sources the references carry, and the same after removing a rigid rotation. The rotation is separated
because a whole-frame offset between two catalogues is not a per-source disagreement.
"""
function rfc_agreement(P, radec, rfc_references)
    srcs = [s for s in P.source if haskey(rfc_references, s)]
    ia = [findfirst(==(s), P.source) for s in srcs]
    dα = Float64[]; dδ = Float64[]
    for (k, s) in zip(ia, srcs)
        α0, δ0 = radec[s]
        r = rfc_references[s]
        solved = ICRSCoords(α0 + P.Δα★_mas[k] * MAS_IN_RAD / cos(δ0),
                            δ0 + P.Δδ_mas[k] * MAS_IN_RAD)
        Δ = separation(SphericalOffsetFlat, U.value(r.coords), solved)
        push!(dα, Δ[1] / MAS_IN_RAD)
        push!(dδ, Δ[2] / MAS_IN_RAD)
    end
    rf = rotation_fit(srcs, radec, dα, dδ)
    (; n = length(srcs), rms_α = rf.rms_before[1], rms_δ = rf.rms_before[2],
       rot = rf.rot, rot_rms_α = rf.rms_after[1], rot_rms_δ = rf.rms_after[2],
       sources = srcs, dα, dδ, res_α = rf.res_α, res_δ = rf.res_δ)
end

"One milliarcsecond in radians."
const MAS_IN_RAD = deg2rad(1e-3 / 3600)

"""
    instrument_fraction_scan(cfg, obs, predictor, scales; spec = delay_fit_spec(cfg))
        -> Vector{NamedTuple}

How much of a measured instrumental correction the SKY signal actually contains, scanned continuously
on a cached observable table — no extra fringe pass, because the applied calibration enters the
observable linearly and analytically:

    τ_residual(s) = τ_residual(0) − s · predictor
    Δcable = τ_applied_phase(cable both) − τ_applied_phase(cable none)

One row per scale: the solve's metrics, the receptor-slot split's scatter and shift, and the half-session
split. The diagnostic is the shape of this curve—a parabola in wrms with a minimum
at the empirical optimum, which [`scan_minimum`](@ref) reads off it.
"""
function instrument_fraction_scan(cfg, obs, predictor, scales; spec = delay_fit_spec(cfg))
    radec = source_radec(obs)
    map(scales) do s
        Os = with_residual(obs, obs.τ_residual .- s .* predictor)
        full = solve_variant(cfg, Os; spec)
        sp = split_solves(cfg, Os, standard_splits(Os); spec)
        receptor = findfirst(p -> p.kind === :receptor_slots, sp.pairs)
        half = findfirst(p -> p.kind === :session_halves, sp.pairs)
        (; scale = float(s), full.metrics...,
           receptor_comparison = isnothing(receptor) ? nothing : sp.pairs[receptor].comparison,
           halves = isnothing(half) ? nothing : sp.pairs[half].comparison)
    end
end

"""
    scan_minimum(rows) -> NamedTuple

The vertex of the parabola through the three lowest points of a `wrms_ps`-versus-`scale` scan — the
empirical optimum the scan exists to locate. `nothing` for the scale when the minimum sits at an end
of the swept range, because a parabola through an endpoint extrapolates rather than interpolates.
"""
function scan_minimum(rows)
    w = [r.wrms_ps for r in rows]; s = [r.scale for r in rows]
    j = argmin(w)
    (1 < j < length(w)) || return (; scale = nothing, wrms_ps = w[j], at_edge = true)
    x = s[j-1:j+1]; y = w[j-1:j+1]
    c = [x .^ 2 x ones(3)] \ y
    (; scale = -c[2] / (2c[1]), wrms_ps = c[3] - c[2]^2 / (4c[1]), at_edge = false)
end

"""
    match_observable_rows(A, B) -> (ia, ib)

Row-match two observable tables built from different fringe passes of the same session, on
(scan, antenna pair, receptor-slot pair). Two passes of one session measure the same rows; a row present in one
and not the other is a real difference and is left out rather than silently aligned by position.
"""
function match_observable_rows(A, B)
    key(O, i) = (O.scan[i], O.a1[i], O.a2[i], O.receptor_slots[i])
    db = Dict(key(B, j) => j for j in eachindex(B))
    ia = Int[]; ib = Int[]
    for i in eachindex(A)
        j = get(db, key(A, i), 0)
        j == 0 && continue
        push!(ia, i); push!(ib, j)
    end
    (ia, ib)
end
