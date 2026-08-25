

function _resolve_terms(sol::Solution, terms::Tuple)
    all(K -> K isa Type && K <: ComponentKind, terms) ||
        error("calibrated_dataset: terms must be component kinds, got $(terms)")
    allunique(terms) || error("calibrated_dataset: duplicate kind(s) in terms $(terms)")
    map(K -> sol[K], terms)
end

@inline function _gpq(states, p::Integer, q::Integer, pr::Integer, pc::Integer, t::Real, ν::Real)
    gp = SVector(gain(states, p, 1, pr, pc, t, ν), gain(states, p, 2, pr, pc, t, ν))
    gq = SVector(gain(states, q, 1, pr, pc, t, ν), gain(states, q, 2, pr, pc, t, ν))
    (gp, gq)
end

@inline function _cal_vis(states, p, q, pr, pc, t, ν, V::SMatrix)
    gp, gq = _gpq(states, p, q, pr, pc, t, ν)
    V ./ (gp * gq')
end

@inline function _cal_wgt(states, p, q, pr, pc, t, ν, W::SMatrix)
    gp, gq = _gpq(states, p, q, pr, pc, t, ν)
    wf = abs2.(gp) * abs2.(gq)'
    ifelse.(W .> 0, W .* wf, zero(eltype(W)))
end

"""
    calibrated_dataset(sol::Solution, data::Dataset; terms::Tuple) -> Dataset

Divide the solved diagonal gains of the requested `terms` (a tuple of component kinds, each resolved
against `sol`; fail loud on a non-kind, a duplicate, or an absent kind) out of `data`, LAZILY: the
returned `Dataset` shares every scalar column and only swaps `visibility`/`weight` for `mapview`s that
compute per row/channel `V′[i,j] = V[i,j]/(g_{p,i}·conj(g_{q,j}))` and
`w′[i,j] = w[i,j]·|g_{p,i}|²·|g_{q,j}|²` on demand (per-cell 2×2 = allocation-free `SMatrix`). An empty
`terms` tuple ⇒ identity gains ⇒ a lazy pass-through of the original values.

`data` may be `sol`'s root dataset or a parent-identity-preserving subset: partition lookups are routed
through `parentrows`/`parentchannels` (ROOT indices), while `t`/`ν` come from the subset's own columns.
"""
function calibrated_dataset(sol::Solution, data::Dataset; terms::Tuple)
    states = _resolve_terms(sol, terms)
    prows = parentrows(data); pchans = parentchannels(data); νs = data.freq.ν
    blix = data.rows.baseline_ix; ts = data.rows.t
    vcol = data.rows.visibility; wcol = data.rows.weight
    vis = mapview(eachindex(vcol)) do j
        p, q = blix[j].antennas; pr = prows[j]; t = ts[j]; V = vcol[j]
        mapview(c -> _cal_vis(states, p, q, pr, pchans[c], t, νs[c], V[c]), eachindex(νs))
    end
    wgt = mapview(eachindex(wcol)) do j
        p, q = blix[j].antennas; pr = prows[j]; t = ts[j]; W = wcol[j]
        mapview(c -> _cal_wgt(states, p, q, pr, pchans[c], t, νs[c], W[c]), eachindex(νs))
    end
    ds = @set data.rows.visibility = vis
    @set ds.rows.weight = wgt
end


"""
    Averaging(; time::TimePartition, frequency::FreqPartition)

Specification for [`average`](@ref): the (root-bound) time and frequency partitions whose cells become
the output time cells (one output row per occupied `(source, baseline, time-cell)`) and output channels
(one per frequency cell).
"""
@kwdef struct Averaging{T<:TimePartition, F<:FreqPartition}
    time::T
    frequency::F
end

mutable struct _AvgCell{UVW,IT}
    const source_ix::Int
    const source::Symbol
    const baseline::Baseline{Symbol}
    const baseline_ix::Baseline{Int}
    const tcell::Int
    const SwV::Vector{SMatrix{2,2,ComplexF64,4}}
    const Sw::Vector{SMatrix{2,2,Float64,4}}
    Σw_row::Float64
    Σw_uvw::UVW
    Σuvw::UVW
    Σit::IT
    n::Int
    scans::Set{Int}
end
_AvgCell{UVW,IT}(sid, src, bl, blix, tc, nfc) where {UVW,IT} = _AvgCell{UVW,IT}(
    sid, src, bl, blix, tc,
    fill(zero(SMatrix{2,2,ComplexF64,4}), nfc), fill(zero(SMatrix{2,2,Float64,4}), nfc),
    0.0, zero(UVW), zero(UVW), zero(IT), 0, Set{Int}())

"""
    average(data::Dataset, av::Averaging) -> Dataset

Condense `data` onto `av`'s partition cells in one streaming pass into an in-memory root `Dataset`: one
output row per occupied `(source, baseline, time-cell)`, one channel per frequency cell. Per correlation the
output visibility is the weighted mean `V̄ = Σw·V/Σw` with weight `Σw` (empty correlation ⇒ `V̄ = NaN`,
`w = 0`). Output coordinates: `datetime`/`t` from the time-cell reference; `uvw` the row-weight average of
contributing rows (equal weighting if fully flagged); `int_time` their SUM; `scan` the common scan (0 if
mixed). `ν` = frequency-cell references, `if_of` = the enclosing IF where contributing channels agree (else
0); `Δν`/`windows`/antennas/sources kept from `data`. `data` may be the root or a parent-identity-preserving
subset.
"""
function average(data::Dataset, av::Averaging)
    nfc = ncells(av.frequency)
    prows = parentrows(data); pchans = parentchannels(data)
    maximum(prows; init = 0) ≤ length(av.time.lookup) ||
        error("average: time partition does not cover the data's parent rows (built on a different root?)")
    maximum(pchans; init = 0) ≤ length(av.frequency.lookup) ||
        error("average: frequency partition does not cover the data's parent channels (built on a different root?)")

    chan_fc = map(c -> cell_of_channel(av.frequency, pchans[c]), 1:nchannels(data))
    UVW = eltype(data.rows.uvw); IT = eltype(data.rows.int_time)
    key2idx = Dictionary{NTuple{4,Int}, Int}()
    accs = _AvgCell{UVW,IT}[]

    blix = data.rows.baseline_ix; sids = data.rows.source_ix; srcs = data.rows.source
    bls = data.rows.baseline; scans = data.rows.scan; uvws = data.rows.uvw; its = data.rows.int_time
    vcol = data.rows.visibility; wcol = data.rows.weight
    for j in 1:nrows(data)
        p, q = blix[j].antennas
        tcell = cell_of_row(av.time, prows[j])
        key = (sids[j], p, q, tcell)
        idx = get(key2idx, key, 0)
        if idx == 0
            push!(accs, _AvgCell{UVW,IT}(sids[j], srcs[j], bls[j], blix[j], tcell, nfc))
            idx = length(accs); set!(key2idx, key, idx)
        end
        acc = accs[idx]
        Vrow = vcol[j]; Wrow = wcol[j]
        rw = 0.0
        for c in 1:nchannels(data)
            W = Wrow[c]; fc = chan_fc[c]
            @inbounds acc.SwV[fc] += ifelse.(W .> 0, W .* Vrow[c], zero(ComplexF64))
            @inbounds acc.Sw[fc] += W
            rw += sum(W)
        end
        acc.Σw_row += rw
        acc.Σw_uvw += rw * uvws[j]
        acc.Σuvw += uvws[j]; acc.Σit += its[j]
        acc.n += 1; push!(acc.scans, scans[j])
    end

    trefs = references(av.time)
    outrows = StructArray(map(accs) do acc
        dt = datetime_at(data, trefs[acc.tcell])
        uvw = acc.Σw_row > 0 ? acc.Σw_uvw / acc.Σw_row : acc.Σuvw / acc.n
        it = acc.Σit
        vis = map(fc -> ifelse.(acc.Sw[fc] .> 0, acc.SwV[fc] ./ acc.Sw[fc], ComplexF64(NaN, NaN)), 1:nfc)
        (
            baseline = acc.baseline, baseline_ix = acc.baseline_ix,
            source = acc.source, source_ix = acc.source_ix,
            scan = length(acc.scans) == 1 ? only(acc.scans) : 0,
            datetime = dt, uvw = uvw, int_time = it,
            visibility = vis, weight = acc.Sw,
        )
    end)

    νout = ustrip.(u"Hz", references(av.frequency))
    if_of_out = map(1:nfc) do fc
        ifs = unique(data.freq.if_of[c] for c in 1:nchannels(data) if chan_fc[c] == fc)
        length(ifs) == 1 ? only(ifs) : 0
    end
    freq = (; data.freq.windows, ν = νout, if_of = if_of_out, data.freq.Δν)
    Dataset(outrows, data.antennas, data.sources, freq; time0 = data.time0)
end
