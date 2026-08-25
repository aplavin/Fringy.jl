
export StEFCal

"""
    StEFCal(PhaseBandpass, LogAmplitudeBandpass; coherence, iterations, regularization,
            reference=nothing, selection=Selection())

Model-free bandpass alignment stage. `targets` must be exactly the two bandpass kinds `PhaseBandpass`
and `LogAmplitudeBandpass` (both present in the solution, sharing time & frequency partitions —
validated). `coherence = (; time::TimePartition, frequency::FreqPartition)` are the (root-bound) nuisance
tiles over which the intrinsic baseline visibility is assumed constant: `coherence.time` must refine the
bandpass time partition (each tile finalizes one bandpass time cell) and `coherence.frequency` must not
split an IF (every IF's channels lie in one coherence frequency cell). `iterations` is the exact number
of profile/gain alternations (no convergence test). `selection` restricts the data.

`regularization` (required, dimensionless, small ≈ 1e-3) is the strength of a prior pseudo-observation
pulling every bandpass cell toward unity: `λ = regularization·S`, with `S` the `Sw`-weighted 0.9-quantile
of the per-entry `|Swv|²/Sw` — the data's own information scale, so the solve is invariant under both
weight and flux rescaling. Well-determined directions are biased by `~regularization`; directions the data
leaves unconstrained revert smoothly to unity. Nothing is thresholded — the solution is a continuous
function of the weights.

`reference` is optional and purely cosmetic (the absolute phase is unobservable in the calibration).
`nothing` keeps the solve's own initialization gauge, near unity for a fresh solution, drifting
toward unity by ~λ/den per iteration — so stored values depend mildly on `iterations` while calibration
products do not (only gauge-invariant quantities are exactly stable). A
`(antenna name, receptor slot)` node like `(:AA, 1)` triggers a post-hoc rotation: per (time cell,
frequency cell) every antenna and both receptor slots are multiplied by the one common phase that zeroes
that node's phase, leaving their relative phase untouched. The node must contribute
data somewhere in the solve (fail loud otherwise); cells where it has none solve to unity and so
rotate by the identity.

Every cell is installed: a cell with no selected data solves to exactly unity — zero correction, up to
the cell's common reference rotation.
"""
struct StEFCal{CT, S} <: Step
    targets::Tuple
    coherence::CT
    iterations::Int
    regularization::Float64
    reference::Union{Nothing, Tuple{Symbol, Int}}
    selection::S
end

function StEFCal(targets...; coherence, iterations::Integer, regularization::Real,
                 reference = nothing, selection = Selection())
    Set(targets) == Set((PhaseBandpass, LogAmplitudeBandpass)) ||
        error("StEFCal (alignment) targets must be exactly (PhaseBandpass, LogAmplitudeBandpass); got $(map(nameof, targets))")
    (hasproperty(coherence, :time) && hasproperty(coherence, :frequency)) ||
        error("StEFCal coherence must be a NamedTuple (; time, frequency)")
    iterations ≥ 1 || error("StEFCal iterations must be ≥ 1")
    regularization > 0 || error("StEFCal regularization must be > 0; got $regularization")
    isnothing(reference) || (reference isa Tuple && length(reference) == 2) ||
        error("StEFCal reference must be nothing or an (antenna, receptor slot) node like (:AA, 1); got $(repr(reference))")
    isnothing(reference) || reference[2] ∈ (1, 2) ||
        error("StEFCal reference receptor slot must be 1 or 2; got $(reference[2])")
    StEFCal((PhaseBandpass, LogAmplitudeBandpass), coherence, iterations, regularization,
            isnothing(reference) ? nothing : (Symbol(reference[1]), reference[2]), selection)
end

function _validate_stefcal(step::StEFCal, sol::Solution)
    haskind(sol, PhaseBandpass) || error("StEFCal: solution has no PhaseBandpass component")
    haskind(sol, LogAmplitudeBandpass) || error("StEFCal: solution has no LogAmplitudeBandpass component")
    pb = sol[PhaseBandpass].definition
    la = sol[LogAmplitudeBandpass].definition
    (ncells(pb.time) == ncells(la.time) && pb.time.lookup == la.time.lookup) ||
        error("StEFCal: PhaseBandpass and LogAmplitudeBandpass must share the same time partition")
    (ncells(pb.frequency) == ncells(la.frequency) && pb.frequency.lookup == la.frequency.lookup) ||
        error("StEFCal: PhaseBandpass and LogAmplitudeBandpass must share the same frequency partition")
    assert_refines(step.coherence.time, pb.time)
    assert_refines(partition(dataset(sol), ByIF()), step.coherence.frequency)
    pb
end

function _weighted_quantile(samples, weights, q::Real)
    o = sortperm(samples)
    edges = [0.0; cumsum(weights[o]) ./ sum(weights)]
    width = sum(abs2, weights) / sum(weights)^2
    lo = q - width / 2; hi = q + width / 2
    ov = @views max.(0.0, min.(edges[2:end], hi) .- max.(edges[1:end - 1], lo))
    sum(ov .* samples[o]) / sum(ov)
end

function _condensed_workspace(step::StEFCal, sol::Solution)
    ds = dataset(sol)
    refnode = isnothing(step.reference) ? nothing :
              (_resolve_reference(ds, step.reference[1]), step.reference[2])
    pb = _validate_stefcal(step, sol)
    la = sol[LogAmplitudeBandpass]
    pbst = sol[PhaseBandpass]

    states = values(Base.structdiff(sol.components, NamedTuple{(:PhaseBandpass, :LogAmplitudeBandpass)}))

    ctime = step.coherence.time; cfreq = step.coherence.frequency
    btime = pb.time; bfreq = pb.frequency
    nant = nantennas(ds); ntc = ncells(btime); nk = ncells(bfreq)

    seldata = select(ds, step.selection)
    nrows(seldata) ≥ 1 || error("StEFCal: selection matched no rows")
    prows = parentrows(seldata); pchans = parentchannels(seldata)
    row_h  = map(r -> cell_of_row(ctime, r), prows)
    row_tc = map(r -> cell_of_row(btime, r), prows)
    chan_a = map(c -> cell_of_channel(cfreq, pchans[c]), 1:nchannels(seldata))
    chan_k = map(c -> cell_of_channel(bfreq, pchans[c]), 1:nchannels(seldata))
    νs = seldata.freq.ν; ts = seldata.rows.t
    blix = seldata.rows.baseline_ix; sids = seldata.rows.source_ix
    vcol = seldata.rows.visibility; wcol = seldata.rows.weight

    key2idx = Dictionary{NTuple{8, Int}, Int}()
    src = Int[]; P = Int[]; Q = Int[]; I = Int[]; J = Int[]; H = Int[]; A = Int[]; TC = Int[]; K = Int[]
    Swv = ComplexF64[]; Sw = Float64[]

    for j in 1:nrows(seldata)
        p, q = blix[j].antennas
        pr = prows[j]; t = ts[j]; sid = sids[j]; h = row_h[j]; tc = row_tc[j]
        Vrow = vcol[j]; Wrow = wcol[j]
        for c in 1:nchannels(seldata)
            ν = νs[c]; pc = pchans[c]; a = chan_a[c]; k = chan_k[c]
            W = Wrow[c]; V = Vrow[c]
            gp, gq = _gpq(states, p, q, pr, pc, t, ν)
            for i in 1:2, jj in 1:2
                W[i, jj] > 0 || continue
                wp  = W[i, jj] * abs2(gp[i]) * abs2(gq[jj])
                wpv = W[i, jj] * V[i, jj] * conj(gp[i]) * gq[jj]
                key = (sid, p, q, i, jj, h, a, k)
                idx = get(key2idx, key, 0)
                if idx == 0
                    push!(src, sid); push!(P, p); push!(Q, q); push!(I, i); push!(J, jj)
                    push!(H, h); push!(A, a); push!(TC, tc); push!(K, k)
                    push!(Swv, wpv); push!(Sw, wp)
                    set!(key2idx, key, length(Swv))
                else
                    @inbounds Swv[idx] += wpv; @inbounds Sw[idx] += wp
                end
            end
        end
    end
    isempty(Swv) && error("StEFCal: no present visibilities in the selection")

    raw = StructArray((; src, p = P, q = Q, i = I, j = J, h = H, a = A, tc = TC, k = K, Swv, Sw))
    order = sortperm(eachindex(raw); by = r -> (src[r], P[r], Q[r], I[r], J[r], H[r], A[r], K[r]))
    tbl = raw[order]

    isnothing(refnode) || any(e -> (e.p, e.i) == refnode || (e.q, e.j) == refnode, tbl) ||
        error("StEFCal: reference node $(step.reference) contributes no data to the solve " *
              "(no condensed entry touches it) — wrong antenna/receptor slot, or fully flagged/deselected.")

    gkey(r) = (tbl.src[r], tbl.p[r], tbl.q[r], tbl.i[r], tbl.j[r], tbl.h[r], tbl.a[r])
    n = length(tbl)
    starts = [1; filter(r -> gkey(r) != gkey(r - 1), 2:n); n + 1]
    groups = [starts[g]:(starts[g + 1] - 1) for g in 1:length(starts) - 1]

    lin = LinearIndices((nant, 2, ntc, nk))
    lin_p = [lin[tbl.p[r], tbl.i[r], tbl.tc[r], tbl.k[r]] for r in 1:n]
    lin_q = [lin[tbl.q[r], tbl.j[r], tbl.tc[r], tbl.k[r]] for r in 1:n]

    S = _weighted_quantile(abs2.(tbl.Swv) ./ tbl.Sw, tbl.Sw, 0.9)
    b0 = exp.(complex.(la.values, pbst.values))
    (; tbl, groups, lin_p, lin_q, S, refnode, b0)
end

function _solve_bandpass!(b, ws, iterations::Integer, λ::Real)
    tbl = ws.tbl; Swv = tbl.Swv; Sw = tbl.Sw
    lin_p = ws.lin_p; lin_q = ws.lin_q
    num = Vector{ComplexF64}(undef, length(b))
    den = Vector{Float64}(undef, length(b))
    bprev = copy(b)
    for it in 1:iterations
        copyto!(bprev, b)
        fill!(num, λ); fill!(den, λ)
        for grp in ws.groups
            N = zero(ComplexF64); D = 0.0
            @inbounds for r in grp
                bp = bprev[lin_p[r]]; bq = bprev[lin_q[r]]
                N += conj(bp) * bq * Swv[r]
                D += abs2(bp) * abs2(bq) * Sw[r]
            end
            D > 0 || continue
            Ĉ = N / D
            @inbounds for r in grp
                lp = lin_p[r]; lq = lin_q[r]
                bp = bprev[lp]; bq = bprev[lq]; w = Sw[r]; d = Swv[r]
                z = Ĉ * conj(bq)
                num[lp] += conj(z) * d
                den[lp] += w * abs2(z)
                z2 = conj(Ĉ) * conj(bp)
                num[lq] += conj(z2) * conj(d)
                den[lq] += w * abs2(z2)
            end
        end
        @inbounds for m in eachindex(b)
            b[m] = num[m] / den[m]
        end
        iseven(it) && (b .= (b .+ bprev) ./ 2)
    end
    b
end

_reference_rotate!(b, ::Nothing) = b
function _reference_rotate!(b, refnode::Tuple{Integer, Integer})
    p, i = refnode
    for k in axes(b, 4), tc in axes(b, 3)
        u = b[p, i, tc, k]
        b[:, :, tc, k] .*= conj(u / abs(u))
    end
    b
end

"""
    apply_step(step::StEFCal, sol, ::Nothing) -> (sol′, nothing)

Run the model-free bandpass solve and install the recovered total bandpass into the two bandpass
components (`PhaseBandpass = angle(b)`, `LogAmplitudeBandpass = log|b|`) in every cell — cells the
selection had no data for hold unity — publishing one new `Solution`. `StEFCal` has no captured product.
"""
function apply_step(step::StEFCal, sol::Solution, ::Nothing)
    ws = _condensed_workspace(step, sol)
    b = copy(ws.b0)
    _solve_bandpass!(b, ws, step.iterations, step.regularization * ws.S)
    _reference_rotate!(b, ws.refnode)

    comps = merge(sol.components, (;
        PhaseBandpass = ComponentState(sol[PhaseBandpass].definition, angle.(b)),
        LogAmplitudeBandpass = ComponentState(sol[LogAmplitudeBandpass].definition, log.(abs.(b)))))
    (Solution(sol.dataset, comps), nothing)
end
