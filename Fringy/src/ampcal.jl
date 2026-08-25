
export AmpSelfCal, Gains

"""
    Gains()

Product-KIND singleton: wrapping an [`AmpSelfCal`](@ref) in `Capture(Gains() => :name, step)`
materializes the per-cell solve diagnostics (see [`apply_step`](@ref)).
"""
struct Gains end

"""
    AmpSelfCal(LogAmplitudeBandpass; models, weight_scale, min_snr, floor, regularization,
               normalize, selection = Selection())

Model-based amplitude self-calibration. Solves, per (receptor slot, time cell, frequency cell) of the
solution's `LogAmplitudeBandpass` component, the weighted least-squares system

    log|V_pq| − log|V_model(u_pq)|  ≈  a_p + a_q ,   weight  w·|V_model|² ,

over the antennas present in that cell, and installs the result as an increment on the component's
current values (`values += a`). The residual is formed on data with the solution's OWN gains already
divided out ([`calibrated_dataset`](@ref) with `jones_terms(sol)`), so the step composes: run it on a
fresh zero solution over an already-calibrated dataset, or on the `TsysInit` solution itself, and it
means the same thing either way.

Only diagonal matrix cells `(1, 1)` and `(2, 2)` enter; off-diagonal products are unused. This is a
structural receptor-slot selection: receptor labels are not inspected. Each selected receptor slot is compared with the
same scalar source model and solved separately because residual calibration after `TsysInit` can
differ by receptor slot.

`models` maps source names to `InterferometricModels` models (anything supporting `haskey`/`getindex`;
every source the dataset carries rows for must have one, and no model may be empty) — the same input
[`model_dataset`](@ref) takes, and normally the same CLEAN models.

Conditions, all explicit and all required:

- `weight_scale` is the `k` of [`noise_scale`](@ref), for which `k·w` is `1/σ²` in the visibilities'
  own flux units. It is needed only for `min_snr`.
- `min_snr` drops a datum whose MODEL amplitude is below `min_snr` thermal σ. The cut is on the model
  and not on the data on purpose: `log|V|` of a noise-dominated datum is both meaningless and biased,
  and selecting on the data's own amplitude would keep the upward fluctuations and throw the downward
  ones away, biasing the gains high. It is also the cut that is gain-invariant — an antenna whose gain
  is wrong by a factor has its weights wrong by the same factor squared, so `|V_model|·√(k·w)` is the
  datum's true amplitude SNR whatever the calibration error is. What survives the cut is the noise bias
  of `log|V|` itself, `+1/(4·SNR²)` — 1 % at `min_snr = 5` and falling as SNR⁻², which is why a cut and
  not a correction. The weight is exactly inverse-variance in the same currency: `log|V|` has variance
  `1/(2·k·w·|V_model|²)`, so `w·|V_model|²` is the WLS weight up to the constant `2k`.
- `floor` (a fraction of the model's total flux) drops data at a NULL of the model, where `log|V_model|`
  diverges — the same guard, with the same meaning, as `model_dataset`'s.
- `regularization` (dimensionless, small ≈ 1e-3) adds `λ·Σ a_p²` with `λ = regularization ×` the cell's
  mean per-antenna weight. The normal matrix `D + B` of this system is the graph's signless Laplacian,
  `xᵀ(D+B)x = Σ w_pq(x_p+x_q)²`: positive SEMI-definite always, and singular exactly when some connected
  component of the cell's baseline graph is bipartite (a single baseline, a chain, any tree) — one odd
  cycle, e.g. a triangle, makes that component definite. The prior
  makes every solvable cell determined and lets directions the data does not constrain revert smoothly
  to no correction, exactly as `StEFCal`'s does.
- A cell with fewer than three antennas carrying data is not solved (its correction stays zero and it
  is counted in the product): with two antennas the single baseline determines only `a_p + a_q`, which
  `normalize = :percell` then removes in full.

`normalize` fixes what the solve is allowed to change about the FLUX SCALE — the choice the docstring
of this step exists for:

- `:percell` — in every cell, the information-weighted mean of `a` over the antennas present is
  subtracted, so the weighted mean baseline correction is exactly zero and the cell's flux scale is
  untouched. Only the inter-antenna relative gains are corrected, and no amount of iteration can
  move flux between the map and the gains. The right choice when a cell carries a SINGLE source, and
  it costs the per-interval common mode entirely.
- `:global` — one constant over the whole solve: the ensemble flux scale is fixed once, and each
  interval's common mode is free. The right choice when a cell carries SEVERAL sources, whose models
  then determine that common mode jointly; on single-source cells it is exactly the freedom that lets
  one bad model rescale its own map, so it needs the intervals to be coarse enough to mix sources.
- `:none` — the model sets everything, including the absolute scale.

`selection` restricts the data. All keyword arguments are required except `selection`.
"""
struct AmpSelfCal{M, S} <: Step
    targets::Tuple
    models::M
    weight_scale::Float64
    min_snr::Float64
    floor::Float64
    regularization::Float64
    normalize::Symbol
    selection::S
end

const _AMP_NORMALIZE = (:percell, :global, :none)

function AmpSelfCal(targets...; models, weight_scale, min_snr, floor, regularization, normalize,
                    selection = Selection())
    Set(targets) == Set((LogAmplitudeBandpass,)) ||
        error("AmpSelfCal targets must be exactly (LogAmplitudeBandpass,); got $(map(nameof, targets))")
    weight_scale > 0 || error("AmpSelfCal: weight_scale must be positive, got $weight_scale")
    min_snr > 0 || error("AmpSelfCal: min_snr must be positive, got $min_snr")
    0 ≤ floor < 1 || error("AmpSelfCal: `floor` is a fraction of the model's total flux, got $floor")
    regularization > 0 || error("AmpSelfCal: regularization must be > 0, got $regularization")
    normalize in _AMP_NORMALIZE ||
        error("AmpSelfCal: normalize must be one of $(_AMP_NORMALIZE); got $(repr(normalize))")
    AmpSelfCal((LogAmplitudeBandpass,), models, Float64(weight_scale), Float64(min_snr), Float64(floor),
               Float64(regularization), Symbol(normalize), selection)
end

function _amp_condense(step::AmpSelfCal, sol::Solution)
    ds = dataset(sol)
    def = sol[LogAmplitudeBandpass].definition
    ntc = ncells(def.time); nfc = ncells(def.frequency); nant = nantennas(ds)

    seldata = select(ds, step.selection)
    nrows(seldata) ≥ 1 || error("AmpSelfCal: selection matched no rows")
    comps, thr = _model_columns(seldata, step.models, step.floor; who = "AmpSelfCal")
    cal = calibrated_dataset(sol, seldata; terms = jones_terms(sol))
    prows = parentrows(seldata); pchans = parentchannels(seldata); νs = seldata.freq.ν
    blix = seldata.rows.baseline_ix; sids = seldata.rows.source_ix; uvws = seldata.rows.uvw
    vcol = cal.rows.visibility; wcol = cal.rows.weight

    Sw = zeros(nant, nant, 2, ntc, nfc)
    Sr = zeros(nant, nant, 2, ntc, nfc)
    Sq = zeros(nant, nant, 2, ntc, nfc)
    Sn = zeros(Int, nant, nant, 2, ntc, nfc)
    nused = 0; nnull = 0; nweak = 0
    fcell = map(c -> cell_of_channel(def.frequency, pchans[c]), eachindex(νs))
    for j in 1:nrows(seldata)
        p0, q0 = blix[j].antennas
        p, q = minmax(p0, q0)
        tc = cell_of_row(def.time, prows[j])
        cs = comps[sids[j]]; nullthr = thr[sids[j]]
        uvw = uvws[j]; Vrow = vcol[j]; Wrow = wcol[j]
        for c in eachindex(νs)
            uv = uv_of(uvw, νs[c])
            M = abs(_model_visibility(cs, uv[1], uv[2]))
            if M < nullthr
                nnull += 1
                continue
            end
            fc = fcell[c]; V = Vrow[c]; W = Wrow[c]
            for i in 1:2
                w = W[i, i]
                w > 0 || continue
                if M * sqrt(step.weight_scale * w) < step.min_snr
                    nweak += 1
                    continue
                end
                a = abs(V[i, i])
                (isfinite(a) && a > 0) || continue
                r = log(a) - log(M)
                wt = w * M^2
                @inbounds Sw[p, q, i, tc, fc] += wt
                @inbounds Sr[p, q, i, tc, fc] += wt * r
                @inbounds Sq[p, q, i, tc, fc] += wt * r^2
                @inbounds Sn[p, q, i, tc, fc] += 1
                nused += 1
            end
        end
    end
    nused > 0 ||
        error("AmpSelfCal: no datum survived the model null floor ($nnull dropped) and the " *
              "min_snr = $(step.min_snr) cut ($nweak dropped) — nothing to solve against")
    (; Sw, Sr, Sq, Sn, nant, ntc, nfc, weight_scale = step.weight_scale,
       counts = (; used = nused, null = nnull, weak = nweak))
end

function _amp_solve!(a, info, ws, regularization::Real)
    (; Sw, Sr, Sq, Sn, nant, ntc, nfc) = ws
    cells = NamedTuple[]
    d = zeros(nant); b = zeros(nant)
    for fc in 1:nfc, tc in 1:ntc, i in 1:2
        fill!(d, 0.0); fill!(b, 0.0)
        for p in 1:nant, q in (p + 1):nant
            w = Sw[p, q, i, tc, fc]
            w > 0 || continue
            s = Sr[p, q, i, tc, fc]
            d[p] += w; d[q] += w; b[p] += s; b[q] += s
        end
        present = findall(>(0), d)
        n = length(present)
        n ≥ 3 || continue
        λ = regularization * (sum(@view d[present]) / n)
        A = zeros(n, n)
        for (ia, p) in enumerate(present)
            A[ia, ia] = d[p] + λ
            for (ib, q) in enumerate(present)
                ia == ib && continue
                A[ia, ib] = Sw[minmax(p, q)..., i, tc, fc]
            end
        end
        x = Symmetric(A) \ @view(b[present])
        all(isfinite, x) ||
            error("AmpSelfCal: the amplitude system of (receptor slot $i, time cell $tc, frequency cell " *
                  "$fc) solved to $(x) — a non-finite gain is not a calibration")
        for (ia, p) in enumerate(present)
            a[p, i, tc, fc] = x[ia]
            info[p, i, tc, fc] = d[p]
        end
        Q = 0.0; Qa = 0.0; Wtot = 0.0; ndata = 0
        for p in 1:nant, q in (p + 1):nant
            w = Sw[p, q, i, tc, fc]
            w > 0 || continue
            g = a[p, i, tc, fc] + a[q, i, tc, fc]
            Q += Sq[p, q, i, tc, fc]
            Qa += Sq[p, q, i, tc, fc] - 2g * Sr[p, q, i, tc, fc] + g^2 * w
            Wtot += w; ndata += Sn[p, q, i, tc, fc]
        end
        push!(cells, (; receptor_slot = i, tcell = tc, fcell = fc, nantennas = n, ndata, weight = Wtot,
                        rms_before = sqrt(max(Q, 0.0) / Wtot), rms_after = sqrt(max(Qa, 0.0) / Wtot),
                        rms_noise = sqrt(ndata / (2 * ws.weight_scale * Wtot))))
    end
    isempty(cells) &&
        error("AmpSelfCal: no (receptor slot, time cell, frequency cell) carries the three antennas the " *
              "amplitude system needs — check the partitions against the data")
    StructArray(identity.(cells))
end

function _amp_normalize!(a, info, mode::Symbol)
    mode === :none && return 0.0
    if mode === :global
        Σ = sum(info); Σ > 0 || return 0.0
        ā = sum(info .* a) / Σ
        a .-= ā .* (info .> 0)
        return ā
    end
    tot = 0.0; Σall = 0.0
    for fc in axes(a, 4), tc in axes(a, 3), i in axes(a, 2)
        Σ = 0.0; num = 0.0
        for p in axes(a, 1)
            Σ += info[p, i, tc, fc]; num += info[p, i, tc, fc] * a[p, i, tc, fc]
        end
        Σ > 0 || continue
        ā = num / Σ
        for p in axes(a, 1)
            info[p, i, tc, fc] > 0 && (a[p, i, tc, fc] -= ā)
        end
        tot += ā * Σ; Σall += Σ
    end
    Σall > 0 ? tot / Σall : 0.0
end

"""
    apply_step(step::AmpSelfCal, sol, capture::Union{Nothing,Gains}) -> (sol′, product_or_nothing)

Solve the per-cell amplitude system of [`AmpSelfCal`](@ref) and install the result as an increment on
the solution's `LogAmplitudeBandpass` values, publishing one new `Solution`. Cells that were not solved
(fewer than three antennas with data) keep their previous values — zero correction.

Under `Capture(Gains() => :name, step)` the product is `(; cells, counts, normalization)`.

`cells` is a `StructArray` with one row per solved `(receptor_slot, tcell, fcell)`: `receptor_slot` is the
structural receptor-slot index; `nantennas`, `ndata`,
the total `weight`, and three log-amplitude residual levels that read together —

| column | what it is |
|---|---|
| `rms_before` | the weighted rms of `log|V| − log|V_model|` as the cell stands |
| `rms_after` | what is left once the antenna-decomposable part is taken out — the solve's own fit |
| `rms_noise` | what the THERMAL noise alone would leave, `√(ndata/(2·weight_scale·weight))` |

The 2 in `rms_noise` is [`map_sigma`](@ref)'s: `k·w` is `1/σ²` of the COMPLEX visibility and an
amplitude sees half of that variance, so one datum's `log|V|` has variance `1/(2·k·w·|V_model|²)`.
Note the DIRECTION this one leans: [`noise_scale`](@ref)'s σ is an upper bound on the thermal one, so
`rms_noise` is an upper bound on the thermal floor — unlike a χ², where the same conservatism makes the
number too SMALL, here it makes the fit look CLOSER to the noise than it is. Read `rms_after/rms_noise`
as a lower bound on how far above thermal a cell sits, and compare cells rather than trusting the 1.

`rms_after` down at `rms_noise` means the cell's amplitude error was entirely antenna gains at this
granularity; `rms_after` well above it means the rest is either the model or an antenna gain varying
faster than the cell.

`counts` is the `(; used, null, weak, cells_skipped)` data census, and `normalization` the mean
log-gain the `normalize` constraint removed.
"""
function apply_step(step::AmpSelfCal, sol::Solution, capture::Union{Nothing, Gains})
    haskind(sol, LogAmplitudeBandpass) ||
        error("AmpSelfCal needs a LogAmplitudeBandpass component in the solution")
    st = sol[LogAmplitudeBandpass]
    ws = _amp_condense(step, sol)
    a = zeros(ws.nant, 2, ws.ntc, ws.nfc)
    info = zeros(ws.nant, 2, ws.ntc, ws.nfc)
    cells = _amp_solve!(a, info, ws, step.regularization)
    ā = _amp_normalize!(a, info, step.normalize)

    comps = merge(sol.components, (; LogAmplitudeBandpass = ComponentState(st.definition, st.values .+ a)))
    sol′ = Solution(sol.dataset, comps)
    isnothing(capture) && return (sol′, nothing)
    skipped = 2 * ws.ntc * ws.nfc - length(cells)
    (sol′, (; cells, counts = (; ws.counts..., cells_skipped = skipped), normalization = ā))
end
