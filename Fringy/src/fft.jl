
using FFTW: plan_fft
using AbstractFFTs: fftfreq, fftshift
using LinearAlgebra: mul!, eigen, Symmetric, Diagonal, isposdef

export fringe_plane, LocalML, NoRefine


"""
    infer_step(coords)

Robust unit step of `coords`: the unit cluster (within [0.5, 1.5]·lower-quartile) of positive successive
differences, averaged. Gaps (larger multiples) are excluded; jitter averages out.
"""
function infer_step(coords)
    u = sort(unique(coords))
    length(u) ≤ 1 && error("cannot infer a lattice step: axis has <2 distinct values")
    d = filter(>(0), diff(u))
    isempty(d) && error("degenerate axis (no positive successive difference)")
    base = quantile(d, 0.25)
    cluster = filter(x -> 0.5base ≤ x ≤ 1.5base, d)
    isempty(cluster) ? base : mean(cluster)
end

"""
    uniform_lattice(coords) -> AbstractRange

Uniform physical lattice `origin:Δ:(≥max)` spanning `coords`; an IF gap / dropped integration becomes
zero-filled cells at the correct spacing. `(origin, Δ)` are LSQ-refined against integer indices so
crude-`infer_step` jitter doesn't drift over many cells. Idempotent on an already-uniform grid. A
degenerate axis (a single distinct sample) yields a length-1 lattice with an arbitrary step — it carries
no conjugate (rate/delay) information, so the transform sees a flat, gated plane.
"""
function uniform_lattice(coords)
    u = unique!(sort(float.(coords)))
    length(u) < 2 && return range(only(u); step = 1.0, length = 1)
    Δ0 = infer_step(u)
    lo = first(u)
    k = round.((u .- lo) ./ Δ0)
    if maximum(k) > minimum(k)
        k̄ = mean(k); ū = mean(u)
        Δ = sum((k .- k̄) .* (u .- ū)) / sum(abs2, k .- k̄)
        origin = ū - Δ * k̄
    else
        Δ = Δ0; origin = lo
    end
    n = round(Int, (last(u) - origin) / Δ)
    range(origin; step = Δ, length = n + 1)
end

snap(lat::AbstractRange, x) =
    clamp(round(Int, (x - first(lat)) / step(lat), RoundNearestTiesAway) + 1, 1, length(lat))

"""
    snap_residual(lat, coords) -> (; rms, max)

Distance of each `coord` to its nearest `lat` point, in `lat`'s units, summarised as `rms` and `max`
over all points. ~0 (float jitter) on a commensurate axis; grows toward Δ/2 as the axis drifts off comb.
"""
function snap_residual(lat::AbstractRange, coords)
    Δ = step(lat)
    res = map(c -> abs((c - first(lat)) - round((c - first(lat)) / Δ) * Δ), coords)
    (; rms = sqrt(sum(abs2, res) / length(res)), max = maximum(res))
end

"""
    assert_on_lattice(lat, coords, name; rmstol=0.25)

Fail loud unless `coords` are uniform at `lat`'s step. Per-point snap residuals are ≤ Δ/2 by
construction (a per-point test can never fire), so a WRONG inferred step is caught by the RMS residual
piling up toward Δ/2; assert RMS ≤ `rmstol·Δ`. Mild jitter passes; a non-uniform axis fails. Used for
the TIME axis (strict cadence); the FREQUENCY axis snaps-with-warning instead (see [`fringe_transform`]).
"""
function assert_on_lattice(lat::AbstractRange, coords, name::AbstractString; rmstol::Real = 0.25)
    Δ = step(lat)
    rms = snap_residual(lat, coords).rms
    rms ≤ rmstol * Δ || error(
        "$name samples are NOT uniform at the inferred step Δ=$Δ (RMS snap residual " *
        "$(round(rms/Δ; digits=3))·Δ > $(rmstol)·Δ): the axis cannot be gridded onto a single " *
        "uniform physical lattice — check for irregular / multi-rate sampling.")
    nothing
end

"""
    freq_lattice(freq) -> AbstractRange

Uniform channel comb spanning the PRESENT channels `freq.ν`, stepped by the metadata `freq.Δν` (READ,
not inferred). Within an IF channels are uniform at Δν; an inter-IF gap becomes zero-filled cells at
that spacing (delay is conjugate to this comb, so Δν IS the delay scale). Channels that are not
commensurate with Δν (e.g. zoom-band IF offsets) are snapped to the nearest comb point at gather, with
a single warning quantifying the induced phase error (see [`fringe_transform`]).
"""
function freq_lattice(freq)
    Δν = freq.Δν
    lo, hi = extrema(freq.ν)
    range(lo; step = Δν, length = round(Int, (hi - lo) / Δν) + 1)
end


"""
    fringe_grid(data::Dataset, states::Tuple) -> (; grids, D, σν, Dif, ifs, tlat, νlat, νsnap)

Gather ONE baseline's rows onto its uniform physical (time × frequency) lattice. `grids[a,b]` is the
2×2 `SMatrix` for the pairs of receptor slots, containing the weighted calibrated value Σ(w′·V′)
at lattice cell (a,b), and `D` is the 2×2 `SMatrix` of Σw′ (a positive scalar for each pair,
independent of any (τ,r)). Here
w′ = w·|g_p,i|²·|g_q,j|² and V′ = V/(g_p,i·conj(g_q,j)), so per cell w′·V′ = w·V·conj(g_p,i)·g_q,j
(identical to `_cal_wgt .* _cal_vis`). `states` are resolved `ComponentState`s (root-index gains) or `()`
for raw data (identity gains). Time lattice is per-baseline (its own cadence); the frequency lattice
spans the present channels with zero-filled inter-IF gaps. `νsnap = (; rms, max)` is the channel snap
residual against `νlat` (Hz), for the transform's non-commensurability warning. The TIME axis is strict
(two records on one time cell fail loud); frequency channels sharing a cell simply accumulate.

Two by-products of the same accumulation, which are not recoverable from downstream peaks:
`σν[i,j]` is the w′-weighted rms of the PHYSICAL channel frequency about its w′-weighted mean [Hz] —
the effective bandwidth of this receptor-slot pair's actual comb, which sets the thermal delay precision
σ_τ = 1/(2π·snr·σν) — and `Dif[k][i,j]` is Σw′ restricted to the channels of IF `ifs[k]` (`ifs` = the
present IFs of `data`, ascending; `sum(Dif) == D`). `σν = 0` and `Dif = 0` where `D == 0`.
"""
function fringe_grid(data::Dataset, states::Tuple)
    nrows(data) ≥ 1 || error("fringe_grid: no rows")
    nchannels(data) ≥ 1 || error("fringe_grid: no channels")
    allequal(data.rows.baseline_ix) || error("fringe_grid: rows span multiple baselines")
    p, q = data.rows.baseline_ix[1].antennas

    ts = data.rows.t; νs = data.freq.ν
    νlat = freq_lattice(data.freq)
    tlat = uniform_lattice(ts)
    assert_on_lattice(tlat, ts, "time")
    νsnap = snap_residual(νlat, νs)
    n_t = length(tlat); n_ν = length(νlat)

    ifs = sort!(unique(data.freq.if_of))
    ifpos = Dictionary(ifs, eachindex(ifs))
    ifof_c = map(a -> ifpos[a], data.freq.if_of)

    prows = parentrows(data); pchans = parentchannels(data)
    vcol = data.rows.visibility; wcol = data.rows.weight
    grids = fill(zero(SMatrix{2,2,ComplexF64,4}), n_t, n_ν)
    Dif = fill(zero(SMatrix{2,2,Float64,4}), length(ifs))
    D = zero(SMatrix{2,2,Float64,4})
    ν0lat = first(νlat)
    Σν = zero(SMatrix{2,2,Float64,4}); Σν² = zero(SMatrix{2,2,Float64,4})
    trow = falses(n_t)
    for j in 1:nrows(data)
        t = ts[j]; a = snap(tlat, t); pr = prows[j]
        trow[a] && error("fringe_grid: two records map to time lattice cell t=$(tlat[a]) s — duplicate/non-unique time")
        trow[a] = true
        Vrow = vcol[j]; Wrow = wcol[j]
        for c in 1:length(νs)
            ν = νs[c]; b = snap(νlat, ν)
            W = Wrow[c]; V = Vrow[c]
            gp, gq = _gpq(states, p, q, pr, pchans[c], t, ν)
            mask = W .> 0
            wpVp = ifelse.(mask, W .* V .* (conj.(gp) * transpose(gq)), zero(ComplexF64))
            wp = ifelse.(mask, W .* (abs2.(gp) * transpose(abs2.(gq))), 0.0)
            @inbounds grids[a, b] += wpVp
            D += wp
            dν = ν - ν0lat
            Σν += wp * dν; Σν² += wp * dν^2
            @inbounds Dif[ifof_c[c]] += wp
        end
    end
    σν = SMatrix{2,2,Float64,4}(ntuple(4) do k
        @inbounds D[k] > 0 ? sqrt(max(Σν²[k] / D[k] - (Σν[k] / D[k])^2, 0.0)) : 0.0
    end)
    (; grids, D, σν, Dif, ifs, tlat, νlat, νsnap)
end


const _FFT_PLAN = typeof(plan_fft(Matrix{ComplexF64}(undef, 1, 1), 1))
const _FFT_CACHE = Dict{Tuple{NTuple{2,Int},Int}, Tuple{_FFT_PLAN, Matrix{ComplexF64}, Matrix{ComplexF64}}}
const _FFT_CACHE_KEY = :Fringy_fft_cache

_fft_cache() = get!(_FFT_CACHE, task_local_storage(), _FFT_CACHE_KEY)::_FFT_CACHE

function _plan_for(shape::NTuple{2,Int}, region::Int)
    get!(_fft_cache(), (shape, region)) do
        buf = zeros(ComplexF64, shape)
        (plan_fft(buf, region), buf, similar(buf))
    end
end

function _batched_fft(S::AbstractMatrix, N::Integer, region::Int)
    shape = region == 2 ? (size(S, 1), N) : (N, size(S, 2))
    plan, inbuf, outbuf = _plan_for(shape, region)
    fill!(inbuf, 0)
    @views inbuf[axes(S)...] .= S
    mul!(outbuf, plan, inbuf)
    outbuf
end

_padlen(os, n) = nextprod((2, 3, 5), max(round(Int, os * n), n))

_stripval(x::Quantity, u) = ustrip(u, x)
_stripval(x::Real, _) = float(x)
_as_interval(iv, u) = _stripval(leftendpoint(iv), u)..(_stripval(rightendpoint(iv), u))

"""
    fringe_transform(S, tlat, νlat; window, oversample, freq_snap=(; rms=0.0, max=0.0)) -> (; G, rate, delay)

Pruned frequency-first windowed 2-D transform of the plane `S` for one pair of receptor slots
(n_t × n_ν, holding w′·V′)
onto the retained (rate × delay) rectangle. Zero-pads freq to `N_ν = nextprod((2,3,5), oversample.delay·n_ν)`,
batched-FFTs along freq, retains delay columns in `window.delay`; then zero-pads time to `N_t` and
batched-FFTs along time, retaining rate rows in `window.rate`. `G[i,j]` equals the direct DFT
`Σ S[a,b]·exp(-2πi[(t_a−t₁)·rate[i] + (ν_b−ν₁)·delay[j]])` (referenced to the lattice origins). Physical
axes `rate` [Hz], `delay` [s]. `window`/`oversample` are NamedTuples `(; rate, delay)`; window endpoints
may be Unitful or plain (Hz/s). Fails loud if a window retains zero bins.

`freq_snap = (; rms, max)` is the channel snap residual from [`fringe_grid`]. When it exceeds a tight
exactness tolerance the channels were not commensurate with Δν and were snapped onto the comb; one
warning quantifies the worst-case per-datum phase error 2π·|τ|·|δν| at the delay-window edge.
"""
function fringe_transform(S::AbstractMatrix, tlat, νlat; window, oversample, freq_snap = (; rms = 0.0, max = 0.0))
    n_t, n_ν = size(S)
    n_t ≥ 1 && n_ν ≥ 1 || error("fringe_transform: empty grid $(size(S))")
    Δt = step(tlat); Δν = step(νlat)

    wdelay = _as_interval(window.delay, u"s")
    wrate = _as_interval(window.rate, u"Hz")

    if freq_snap.rms > 1e-3 * Δν
        τedge = max(abs(leftendpoint(wdelay)), abs(rightendpoint(wdelay)))
        @warn "frequency channels are not commensurate with Δν=$(round(Δν/1e6; digits=4)) MHz; snapping to the lattice " *
              "(RMS |δν|=$(round(freq_snap.rms/Δν; digits=3))·Δν, max=$(round(freq_snap.max/Δν; digits=3))·Δν) — " *
              "worst-case phase error 2π·|τ|·|δν| ≈ $(round(360*τedge*freq_snap.max; digits=1))° at the delay-window " *
              "edge |τ|=$(round(τedge*1e9; digits=1)) ns" maxlog=1
    end

    N_ν = _padlen(oversample.delay, n_ν)
    Gf = _batched_fft(S, N_ν, 2)
    delay_full = fftshift(fftfreq(N_ν, 1 / Δν))
    dsel = findall(x -> x in wdelay, delay_full)
    isempty(dsel) && error("fringe_transform: delay window $(wdelay) s retains zero bins (axis " *
                           "$(extrema(delay_full)) s, Δ=$(step(delay_full)) s — window too narrow or beyond Nyquist)")
    G1 = Gf[:, fftshift(1:N_ν)[dsel]]
    delay = delay_full[dsel]

    N_t = _padlen(oversample.rate, n_t)
    G2 = _batched_fft(G1, N_t, 1)
    rate_full = fftshift(fftfreq(N_t, 1 / Δt))
    rsel = findall(x -> x in wrate, rate_full)
    isempty(rsel) && error("fringe_transform: rate window $(wrate) Hz retains zero bins (axis " *
                          "$(extrema(rate_full)) Hz, Δ=$(step(rate_full)) Hz — window too narrow or beyond Nyquist)")
    G = G2[fftshift(1:N_t)[rsel], :]
    rate = rate_full[rsel]
    (; G, rate, delay)
end


"""
    noise_region(dims::Tuple{Int,Int}, ir, id) -> Vector{CartesianIndex{2}}

The bins of a `dims = (rate, delay)` search plane over which the noise is estimated: **the quarter of
the plane most separated from the peak in both delay and rate**.

Each bin is ranked by the SMALLER of its delay- and rate-separation from the lattice peak `(ir, id)`
— each separation counted in bins and divided by the length of its own axis, so that the two are
comparable — and the quarter of the plane with the largest rank is taken. Where the boundary rank is
shared by more bins than the quarter has room for, they are taken in bin order: a quarter of the
plane is a quarter of the plane, and which bins of one equal-separation band complete it cannot
matter to a median over it.

The shape follows from the rule and is not separately imposed. A peak in the middle of the plane puts
the region in the four corners, a quarter of each axis on each side; a peak in a corner puts it in the
opposite quadrant, the far corner of the plane included. In between the region slides continuously
between those two: **moving the peak by one bin moves the region's boundary by one bin**, which is
what makes the noise estimate a continuous function of the data. Its predecessor took one fixed
quarter-window corner chosen by a hard midpoint test, so a last-bit change in the peak search could
replace the whole region and materially move the reported SNR.

The peak position is the LATTICE argmax, never the refined one: the noise estimate then does not
depend on the refinement algorithm, and a tie between two neighbouring lattice bins moves the region's
boundary by one bin rather than replacing the region.
"""
function noise_region(dims::Tuple{Int,Int}, ir::Integer, id::Integer)
    n_r, n_d = dims
    k = (n_r * n_d) ÷ 4
    k == 0 && return CartesianIndex{2}[]
    sep_rate = [abs(i - ir) / n_r for i in 1:n_r]
    sep_delay = [abs(j - id) / n_d for j in 1:n_d]
    levels = sort!(unique(vcat(sep_rate, sep_delay)); rev = true)
    boundary = levels[end]
    for ℓ in levels
        if count(≥(ℓ), sep_rate) * count(≥(ℓ), sep_delay) ≥ k
            boundary = ℓ
            break
        end
    end
    region = CartesianIndex{2}[]; band = CartesianIndex{2}[]
    for j in 1:n_d, i in 1:n_r
        separation = min(sep_rate[i], sep_delay[j])
        separation > boundary ? push!(region, CartesianIndex(i, j)) :
            separation == boundary && push!(band, CartesianIndex(i, j))
    end
    append!(region, @view band[1:(k - length(region))])
end

function _noise_sigma(G::AbstractMatrix, ir::Integer, id::Integer)
    region = noise_region(size(G), ir, id)
    mags = [abs(G[I]) for I in region]
    n = length(mags)
    σ = n > 0 ? median(mags) / sqrt(2 * log(2)) : 0.0
    (σ, n)
end

const _QUAD_UV = [(u, v) for u in (-1.0, 0.0, 1.0) for v in (-1.0, 0.0, 1.0)]
const _QUAD_A = reduce(vcat, [[1.0 u v u^2 v^2 u*v] for (u, v) in _QUAD_UV])

function _psd_clamp(Q::SMatrix{2,2,Float64})
    E = eigen(Symmetric(Matrix(Q)))
    vals = max.(E.values, 0.0)
    SMatrix{2,2,Float64}(E.vectors * Diagonal(vals) * transpose(E.vectors))
end

function _quadfit(G::AbstractMatrix, rate, delay, ir::Integer, id::Integer, D::Real)
    r_win, d_win = size(G)
    r0 = rate[ir]; d0 = delay[id]
    onboundary = ir == 1 || ir == r_win || id == 1 || id == d_win
    Δr = r_win > 1 ? rate[2] - rate[1] : 1.0
    Δd = d_win > 1 ? delay[2] - delay[1] : 1.0
    onboundary && return (d0, r0, zero(SMatrix{2,2,Float64,4}))

    b = [abs2(G[ir + Int(u), id + Int(v)]) / D for (u, v) in _QUAD_UV]
    c = _QUAD_A \ b
    a1, a2, a3, a4, a5 = c[2], c[3], c[4], c[5], c[6]
    H_uv = @SMatrix [2a3 a5; a5 2a4]
    H_phys = @SMatrix [2a3/Δr^2 a5/(Δr*Δd); a5/(Δr*Δd) 2a4/Δd^2]
    Q = _psd_clamp(-0.5 * H_phys)
    if isposdef(-H_uv)
        δ = -(H_uv \ SVector(a1, a2))
        δr = clamp(δ[1], -1.0, 1.0); δd = clamp(δ[2], -1.0, 1.0)
        (d0 + δd * Δd, r0 + δr * Δr, Q)
    else
        (d0, r0, Q)
    end
end

function _derotated(S::AbstractMatrix, tlat, νlat, t0, ν0, τ̂, r̂)
    nt, nν = size(S)
    ctv = Vector{ComplexF64}(undef, nt)
    @inbounds for a in 1:nt; ctv[a] = cis(-2π * (tlat[a] - t0) * r̂); end
    N = zero(ComplexF64)
    @inbounds for b in 1:nν
        cνb = cis(-2π * (νlat[b] - ν0) * τ̂)
        acc = zero(ComplexF64)
        for a in 1:nt
            Sab = S[a, b]
            iszero(Sab) && continue
            acc += Sab * ctv[a]
        end
        N += acc * cνb
    end
    N
end


"""
    LocalML(; iterations)

Refinement strategy for [`fringe_measure`](@ref)/[`FringeFit`](@ref): `iterations` Gauss–Newton steps that
locally maximize the coherent sum |N(τ,r)| = |Σ S·exp(−2πi[(ν−ν0)τ + (t−t0)r])| starting from the
interpolated FFT peak. `iterations = 2` is the validated setting (the third step moves the estimate by
≪ the thermal error).

*Why*: the 3×3 quadratic interpolation of the padded FFT plane carries a bin-quantization bias of ~1–2%
of a delay bin (a few ps at oversample 4 on a 512 MHz band) — comparable to the 1–5 ps thermal
precision of a high-SNR observable, hence a systematic at the astrometric level. The refinement removes
it: each step derotates the gathered plane by the current (τ,r), collapses it onto the frequency and
time axes, and adds the |·|²-weighted linear phase slope of each collapsed axis (divided by 2π).

*Failure mode*: the phases of the collapsed axis sums are read relative to their own vector mean, so a
starting point further than half a turn of residual slope across the band/scan (i.e. the FFT peak in the
WRONG lattice cell, only possible at SNR ≲ 5) would refine into a neighbouring fringe. It never moves
the estimate out of the lattice cell it starts in at usable SNR, and it is a no-op on an exactly-flat
plane.
"""
@kwdef struct LocalML
    iterations::Int
end

"""
    NoRefine()

Refinement strategy for [`fringe_measure`](@ref)/[`FringeFit`](@ref): keep the interpolated FFT peak as
the measurement (see [`LocalML`](@ref) for what that costs).
"""
struct NoRefine end

function _phase_slope(A::AbstractVector{ComplexF64}, x, N::ComplexF64)
    W = 0.0; Wx = 0.0
    @inbounds for k in eachindex(A)
        w = abs2(A[k]); W += w; Wx += w * x[k]
    end
    W > 0 || return 0.0
    x̄ = Wx / W
    num = 0.0; den = 0.0
    @inbounds for k in eachindex(A)
        w = abs2(A[k])
        w > 0 || continue
        dx = x[k] - x̄
        num += w * dx * angle(A[k] * conj(N))
        den += w * dx^2
    end
    den > 0 ? num / den : 0.0
end

_refine_peak(::NoRefine, S, tlat, νlat, t0, ν0, τ̂, r̂) = (τ̂, r̂)

function _refine_peak(m::LocalML, S::AbstractMatrix, tlat, νlat, t0, ν0, τ̂::Float64, r̂::Float64)
    nt, nν = size(S)
    sp = zeros(ComplexF64, nν)
    tsum = zeros(ComplexF64, nt)
    ctv = Vector{ComplexF64}(undef, nt)
    for _ in 1:m.iterations
        fill!(sp, 0); fill!(tsum, 0)
        @inbounds for a in 1:nt; ctv[a] = cis(-2π * (tlat[a] - t0) * r̂); end
        @inbounds for b in 1:nν
            cνb = cis(-2π * (νlat[b] - ν0) * τ̂)
            acc = zero(ComplexF64)
            for a in 1:nt
                Sab = S[a, b]
                iszero(Sab) && continue
                z = Sab * ctv[a] * cνb
                acc += z
                tsum[a] += z
            end
            sp[b] = acc
        end
        N = sum(sp)
        iszero(N) && break
        τ̂ += _phase_slope(sp, νlat, N) / 2π
        r̂ += _phase_slope(tsum, tlat, N) / 2π
    end
    (τ̂, r̂)
end

"""
    fringe_measure(S, G, rate, delay, tlat, νlat, t0, ν0, D; σν, refine) -> NamedTuple

Peak analysis on the retained plane `G` (holds the FFT numerator; only `|G|` is used, its phase
reference is irrelevant). `S` is the gathered plane for one pair of receptor slots (w′·V′),
`D = Σw′` (scalar), `σν` the
pair's effective bandwidth [Hz] (from [`fringe_grid`](@ref)) and `refine` a `LocalML`/`NoRefine`.
Returns `(; delay=τ̂, rate=r̂, snr, sigma_delay, coeff=ĝ, Q, q_ab, D, lat_delay, lat_rate)`: sub-bin
`(τ̂,r̂)` — quadratic interpolation, then `refine`d — + curvature `Q=−½·Hess(|N|²/D)` from
the interpolation stencil; committed complex peak `ĝ=N(τ̂,r̂)/D` by one direct derotated sum about the
tile references `(t0,ν0)`; empirical SNR `snr=|G_latpeak|/σ̂` (numerator = the lattice peak
magnitude since σ̂ is estimated on the same plane, denominator = the median over [`noise_region`](@ref),
the quarter of the plane most separated from the peak in both delay and rate); the thermal delay
precision `sigma_delay = 1/(2π·snr·σν)` [s] in that empirical-noise SNR convention (∞ when either
factor is 0);
`q_ab=|N(τ̂,r̂)|²/D` (edge weight).
"""
function fringe_measure(S::AbstractMatrix, G::AbstractMatrix, rate, delay, tlat, νlat, t0, ν0, D::Real;
                        σν::Real, refine)
    ir = 1; id = 1; pmax = -1.0
    @inbounds for j in axes(G, 2), i in axes(G, 1)
        p = abs2(G[i, j])
        p > pmax && (pmax = p; ir = i; id = j)
    end
    τ̂, r̂, Q = _quadfit(G, rate, delay, ir, id, D)
    τ̂, r̂ = _refine_peak(refine, S, tlat, νlat, t0, ν0, τ̂, r̂)
    N = _derotated(S, tlat, νlat, t0, ν0, τ̂, r̂)
    ĝ = N / D
    q_ab = abs2(N) / D
    σ, nbins = _noise_sigma(G, ir, id)
    nbins < 512 && @warn "fringe_measure: the noise region has only $nbins bins (<512); σ̂ is noisy" maxlog=1
    snr = σ > 0 ? abs(G[ir, id]) / σ : 0.0
    sigma_delay = snr > 0 && σν > 0 ? 1 / (2π * snr * σν) : Inf
    (; delay = τ̂, rate = r̂, snr, sigma_delay, coeff = ĝ, Q, q_ab, D,
       lat_delay = delay[id], lat_rate = rate[ir])
end


_pair(bl::Baseline) = bl.antennas
_pair(t::Tuple{Integer,Integer}) = t

"""
    fringe_plane(data::Dataset, baseline, rp::Tuple{Int,Int};
                 window, oversample, refine, solution=nothing, terms=()) -> NamedTuple

Retained rate×delay plane for one `(baseline, receptor-slot pair)`, treating ALL passed selected `data` as a
single tile with envelope-midpoint references. `baseline` is a `Baseline{Int}` or a dense `(p,q)` tuple
(normalized to p<q); `rp=(i,j)` is the structural receptor-slot pair. Rows are filtered to the requested baseline
internally. With `solution` + `terms` (a tuple of component kinds) the gathered data is CALIBRATED (gains
divided out, weights transformed); without, raw data is gathered. `window=(; rate, delay)` and
`oversample=(; rate, delay)` are NamedTuples (window endpoints Unitful or plain Hz/s); `refine`
(`LocalML`/`NoRefine`, required) is the peak refinement, exactly as in [`FringeFit`](@ref).

Returns `(; rate::Vector [Hz], delay::Vector [s], magnitude::KeyedArray[rate,delay] (= |G|/D), peak, D)`
where `peak = (; delay, rate, snr, sigma_delay, coeff::ComplexF64)` (`coeff` = the committed
ĝ = N(τ̂,r̂)/D) and `D` is Σw′ for the selected pair of receptor slots. Fails loud when the baseline has no positive-weight
data for `rp`.
"""
function fringe_plane(data::Dataset, baseline, rp::Tuple{Int,Int};
                      window, oversample, refine, solution = nothing, terms = ())
    a, b = _pair(baseline); p, q = min(a, b), max(a, b)
    sub = select(data, Selection(@o _.baseline_ix.antennas == (p, q)))
    nrows(sub) ≥ 1 || error("fringe_plane: no rows for baseline ($p,$q)")
    states = isnothing(solution) ? () : _resolve_terms(solution, terms)

    g = fringe_grid(sub, states)
    i, j = rp
    Dij = g.D[i, j]
    Dij > 0 || error("fringe_plane: no positive-weight data for receptor-slot pair $rp on baseline ($p,$q)")
    S = getindex.(g.grids, i, j)
    tr = fringe_transform(S, g.tlat, g.νlat; window, oversample, freq_snap = g.νsnap)
    t0 = (minimum(sub.rows.t) + maximum(sub.rows.t)) / 2
    ν0 = (minimum(sub.freq.ν) + maximum(sub.freq.ν)) / 2
    pk = fringe_measure(S, tr.G, tr.rate, tr.delay, g.tlat, g.νlat, t0, ν0, Dij; σν = g.σν[i, j], refine)
    magnitude = KeyedArray(abs.(tr.G) ./ Dij; rate = tr.rate, delay = tr.delay)
    (; rate = tr.rate, delay = tr.delay, magnitude,
       peak = (; pk.delay, pk.rate, pk.snr, pk.sigma_delay, pk.coeff), D = Dij)
end
