
_antennas(a::AbstractVector{Symbol}) = [Antenna(; name) for name in a]
_antennas(a::AbstractVector{<:Antenna}) = collect(a)

"""
    simulate(; antennas, times, freq, scans, sources, baselines, time0, int_time, uvw,
             weights=Returns(1.0), single_receptor_antennas=Int[]) -> Dataset

Build a valid in-memory root `Dataset` with a grid of `(time, baseline)` rows and zero visibilities
(ready for [`corrupt`](@ref)). `antennas` is a vector of names (or `Antenna`s); `times` are seconds from
`time0`; `freq` is a NamedTuple `(; ν, if_of, Δν[, windows])`. `scans` and `sources` are per-time
(default a single scan/source), so each scan carries one source. `baselines` defaults to every `p<q`
pair. Per-correlation weights come from `weights(p, q, i, j, t, ν)::Real`; antennas listed in
`single_receptor_antennas` carry only receptor slot 1 (receptor slot 2 gets weight 0), so absence stays emergent from the
weight mask.

`uvw` is either one constant `SVector{3}` (the default — the alignment stages are uv-blind) or a
callable `(p, q, t) -> SVector{3}` giving each row its own baseline vector, which is what the imaging
gates need: a simulated sky brightness distribution is only meaningful over a real uv coverage.
"""
function simulate(;
    antennas,
    times::AbstractVector{<:Real},
    freq,
    scans::AbstractVector{<:Integer} = ones(Int, length(times)),
    sources::AbstractVector{Symbol} = fill(:SRC, length(times)),
    baselines = nothing,
    time0::DateTime = DateTime(2020, 1, 1),
    int_time = 1.0u"s",
    uvw = SVector(0.0, 0.0, 0.0)u"m",
    weights = Returns(1.0),
    single_receptor_antennas = Int[],
)
    ants = _antennas(antennas)
    antnames = map(a -> a.name, ants)
    nant = length(ants)
    length(times) == length(scans) == length(sources) ||
        error("simulate: times, scans, sources must have equal length")
    bls = isnothing(baselines) ? [(p, q) for p in 1:nant for q in (p+1):nant] : baselines
    srcnames = sort!(unique(sources))
    srcidx = Dictionary(srcnames, eachindex(srcnames))
    srctable = StructArray(map((k, nm) -> (; name = nm, coords = (ra = 0.0, dec = 0.0), rawid = k),
                               eachindex(srcnames), srcnames))
    νs = collect(Float64, freq.ν)
    nchan = length(νs)
    single_receptor_antenna = Set(single_receptor_antennas)
    receptor_available(a, receptor) = receptor == 1 || !(a in single_receptor_antenna)
    wmat(p, q, t, ν) = SMatrix{2,2,Float64,4}(
        receptor_available(p, 1) && receptor_available(q, 1) ? weights(p, q, 1, 1, t, ν) : 0.0,
        receptor_available(p, 2) && receptor_available(q, 1) ? weights(p, q, 2, 1, t, ν) : 0.0,
        receptor_available(p, 1) && receptor_available(q, 2) ? weights(p, q, 1, 2, t, ν) : 0.0,
        receptor_available(p, 2) && receptor_available(q, 2) ? weights(p, q, 2, 2, t, ν) : 0.0,
    )

    uvwf = _uvwf(uvw)
    rows = StructArray([
        let t = float(times[it]), p = pq[1], q = pq[2]
            (
                baseline = Baseline((antnames[p], antnames[q])),
                baseline_ix = Baseline((p, q)),
                source = sources[it], source_ix = srcidx[sources[it]],
                scan = scans[it], datetime = datetime_at(time0, t),
                uvw = uvwf(p, q, t), int_time = int_time,
                visibility = [zero(SMatrix{2,2,ComplexF64,4}) for _ in 1:nchan],
                weight = [wmat(p, q, t, νs[c]) for c in 1:nchan],
            )
        end
        for it in eachindex(times) for pq in bls
    ])
    fmeta = (; windows = hasproperty(freq, :windows) ? freq.windows : nothing,
             ν = νs, if_of = collect(Int, freq.if_of), Δν = Float64(freq.Δν))
    Dataset(rows, ants, srctable, fmeta; time0)
end

_uvwf(x::StaticVector) = Returns(x)
_uvwf(f) = f

_coh(c::AbstractMatrix, ::Type) = (sid, uvw, ν) -> SMatrix{2,2,ComplexF64,4}(c)
_coh(c, ::Type{U}) where {U} = hasmethod(c, Tuple{Int, U, Float64}) ? c : (sid, uvw, ν) -> c(sid)

"""
    corrupt(rng::AbstractRNG, ds::Dataset, truth::Solution; coherency, noise=0.0) -> Dataset

Apply the forward model to `ds`, returning a `Dataset` (sharing all scalar columns and weights) whose
visibilities are `V_pq[i,j] = g_{p,i}·C[i,j]·conj(g_{q,j}) + noise·z` with `g` the truth `gain` (the SAME
evaluator the solver uses), `C` the `coherency` and `z` a standard complex-normal draw from `rng` (drawn
in a fixed row-major, column-major-within-cell order, so a given seed is reproducible; `noise = 0` draws
nothing and is exact).

`coherency` is a 2×2 matrix (the same sky for every source), a callable `source_ix -> 2×2` (a
per-source constant), or a callable `(source_ix, uvw, ν) -> 2×2` — the last being a resolved sky, e.g.
`(sid, uvw, ν) -> visibility(models[sid], uv_of(uvw, ν)) * SMatrix(1,0,0,1)`, which is what the imaging
gates simulate.
"""
function corrupt(rng::AbstractRNG, ds::Dataset, truth::Solution; coherency, noise::Real = 0.0)
    uvws = ds.rows.uvw
    cohf = _coh(coherency, eltype(uvws))
    blix = ds.rows.baseline_ix; ts = ds.rows.t; sids = ds.rows.source_ix; νs = ds.freq.ν
    newvis = map(eachindex(ds.rows)) do j
        p, q = blix[j].antennas; t = ts[j]; sid = sids[j]; uvw = uvws[j]
        map(eachindex(νs)) do c
            ν = νs[c]
            C = SMatrix{2,2,ComplexF64,4}(cohf(sid, uvw, ν))
            gp = SVector(gain(truth, p, 1, j, c, t, ν), gain(truth, p, 2, j, c, t, ν))
            gq = SVector(gain(truth, q, 1, j, c, t, ν), gain(truth, q, 2, j, c, t, ν))
            model = (gp * gq') .* C
            iszero(noise) ? model :
                model .+ noise .* SMatrix{2,2,ComplexF64,4}(randn(rng, ComplexF64), randn(rng, ComplexF64),
                                                            randn(rng, ComplexF64), randn(rng, ComplexF64))
        end
    end
    @set ds.rows.visibility = newvis
end
