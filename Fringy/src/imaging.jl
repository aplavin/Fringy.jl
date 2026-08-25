
export ImageGrid, SourceImage, stokes_i, uv_of, dirty_map, dirty_beam, fit_beam, hogbom, restore,
       image_source, model_dataset, significant_components, closure_chi2, closure_amp_chi2,
       uniform_weights, grid_axes, cleaned_flux, noise_scale, map_sigma, peak_position


"""
    UV_SIGN

The sign relating the correlator's `uvw` to the uv coordinate of the imaging/model convention
(see the file header). Fixed at `−1` and pinned by the coordinate-sign tests.
"""
const UV_SIGN = -1.0

"""
    uv_of(uvw, ν) -> SVector{2,Float64}

The uv point [inverse milliarcseconds] of a datum with baseline vector `uvw` (metres, the correlator's
own, unitful or plain) at frequency `ν` [Hz]: `UV_SIGN·uvw₁,₂·ν/c` converted from radians⁻¹ to mas⁻¹.
The ONE place the imaging path turns a baseline into a uv coordinate — the dirty map, the model
division and structure delays must all use the same one, sign included.
"""
@inline function uv_of(uvw, ν::Real)
    s = UV_SIGN * ν / C_LIGHT_M_S * MAS
    SVector(_metres(uvw[1]) * s, _metres(uvw[2]) * s)
end
_metres(x::Quantity) = ustrip(u"m", x)
_metres(x::Real) = float(x)


"""
    stokes_i(ds::Dataset) -> StructArray

The diagonal-product visibilities of an averaged `Dataset`, one row per usable
`(data row, channel)`:

| column | meaning |
|---|---|
| `source, source_ix, baseline_ix, scan, t, chan` | identity of the datum |
| `u, v` | `uvw₁,₂·ν/c` in inverse milliarcseconds (see the file header) |
| `V` | `(V[1,1] + V[2,2])/2`, or the single present diagonal cell |
| `w` | `4·w[1,1]·w[2,2]/(w[1,1] + w[2,2])`, or the single present diagonal cell's weight |

Cells with neither diagonal product present (`present` weight 0) are dropped; off-diagonal products
are unused. This is a structural matrix-cell selection: receptor labels are not inspected. The
current imaging pipeline interprets the result as Stokes I under its assumption that the diagonal
cells are the two parallel-hand products whose half-sum is I; `stokes_i` itself does not establish
that polarization meaning. The two diagonal cells keep independent per-scan phase solutions and are
coupled only through the hybrid loop's shared model.
"""
function stokes_i(ds::Dataset)
    νs = ds.freq.ν
    rows = NamedTuple[]
    for j in 1:nrows(ds)
        V = ds.rows.visibility[j]; W = ds.rows.weight[j]
        uvw = ds.rows.uvw[j]
        for c in eachindex(νs)
            wR = W[c][1, 1]; wL = W[c][2, 2]
            (wR > 0 || wL > 0) || continue
            Vi, wi = if wR > 0 && wL > 0
                (V[c][1, 1] + V[c][2, 2]) / 2, 4 * wR * wL / (wR + wL)
            elseif wR > 0
                V[c][1, 1], wR
            else
                V[c][2, 2], wL
            end
            isfinite(Vi) || error("stokes_i: non-finite visibility with positive weight at row $j channel $c")
            uv = uv_of(uvw, νs[c])
            push!(rows, (; source = ds.rows.source[j], source_ix = ds.rows.source_ix[j],
                           baseline_ix = ds.rows.baseline_ix[j], scan = ds.rows.scan[j],
                           t = ds.rows.t[j], chan = c,
                           u = uv[1], v = uv[2], V = ComplexF64(Vi), w = Float64(wi)))
        end
    end
    isempty(rows) && error("stokes_i: no usable diagonal correlation product in the dataset")
    StructArray(identity.(rows))
end


"""
    ImageGrid(; npix, pixel)

A square image grid: `npix` × `npix` pixels of `pixel` milliarcseconds, centred on the phase centre
(pixel `npix ÷ 2 + 1` in both axes). Axis 1 is `x` (East), axis 2 is `y` (North).
"""
@kwdef struct ImageGrid
    npix::Int
    pixel::Float64
end
ImageGrid(npix::Integer, pixel::Real) = ImageGrid(; npix = Int(npix), pixel = Float64(pixel))

"""
    grid_axes(g::ImageGrid) -> (xs, ys)

The pixel-centre coordinates [mas] of the image axes.
"""
grid_axes(g::ImageGrid) = (_gaxis(g), _gaxis(g))
_gaxis(g::ImageGrid) = ((1:g.npix) .- (g.npix ÷ 2 + 1)) .* g.pixel

_baxis(g::ImageGrid) = ((1:(2g.npix - 1)) .- g.npix) .* g.pixel


const _DFT_BLOCK = 256

function _dft_grid(u::AbstractVector{Float64}, v::AbstractVector{Float64}, z::AbstractVector{ComplexF64},
                   xs::AbstractVector{Float64}, ys::AbstractVector{Float64})
    nx = length(xs); ny = length(ys); nvis = length(u)
    out = zeros(Float64, nx, ny)
    nvis == 0 && return out
    nchunk = min(4 * Threads.nthreads(), ny)
    edges = round.(Int, range(0, ny; length = nchunk + 1))
    nb = min(_DFT_BLOCK, nvis)
    ar = Matrix{Float64}(undef, nx, nb); ai = Matrix{Float64}(undef, nx, nb)
    for k0 in 1:nb:nvis
        kn = min(nb, nvis - k0 + 1)
        _foreach_tile(kn) do b
            k = k0 + b - 1
            zk = z[k]
            iszero(zk) && return
            zr = real(zk); zi = imag(zk); ku = -2π * u[k]
            @inbounds for ix in 1:nx
                s, c = sincos(ku * xs[ix])
                ar[ix, b] = zr * c - zi * s
                ai[ix, b] = zr * s + zi * c
            end
        end
        _foreach_tile(nchunk) do ic
            jlo = edges[ic] + 1; jhi = edges[ic + 1]
            jlo > jhi && return
            @inbounds for b in 1:kn
                k = k0 + b - 1
                iszero(z[k]) && continue
                kv = -2π * v[k]
                for jy in jlo:jhi
                    sb, cb = sincos(kv * ys[jy])
                    @simd for ix in 1:nx
                        out[ix, jy] += ar[ix, b] * cb - ai[ix, b] * sb
                    end
                end
            end
        end
    end
    out
end

"""
    dirty_map(vis, grid::ImageGrid; weights = vis.w) -> Matrix{Float64}

The dirty map [Jy/beam] of the Stokes-I table `vis` (columns `u`, `v`, `V`, `w`) on `grid`: the
weighted adjoint DFT `Σ w·Re[V·cis(−2π(ux + vy))] / Σ w` (file header). Natural weighting is the
default; pass [`uniform_weights`](@ref) for the uniform variant.
"""
function dirty_map(vis, grid::ImageGrid; weights = vis.w)
    xs, ys = grid_axes(grid)
    Σw = sum(weights)
    Σw > 0 || error("dirty_map: total weight is zero")
    _dft_grid(vis.u, vis.v, weights .* vis.V, xs, ys) ./ Σw
end

"""
    dirty_beam(vis, grid::ImageGrid; weights = vis.w) -> (; patch, beam)

The dirty beam of the same uv coverage on the difference grid ((2npix−1)², centred on zero, unit peak)
together with the elliptical-Gaussian restoring beam of that coverage ([`fit_beam`](@ref)).

Computed ONCE PER SOURCE and reused across hybrid iterations: the uv coverage does not change when the
calibration does.
"""
function dirty_beam(vis, grid::ImageGrid; weights = vis.w)
    bs = _baxis(grid)
    Σw = sum(weights)
    Σw > 0 || error("dirty_beam: total weight is zero")
    patch = _dft_grid(vis.u, vis.v, complex.(weights), bs, bs) ./ Σw
    n = grid.npix
    patch[n, n] ≈ 1 ||
        error("dirty_beam: the central pixel is $(patch[n, n]), not 1 — non-finite weights?")
    (; patch, beam = fit_beam(vis.u, vis.v, weights))
end

"""
    uniform_weights(vis, grid::ImageGrid) -> Vector{Float64}

Uniform weights: each visibility's natural weight divided by the total natural weight in its cell of
the uv grid conjugate to `grid` (cell size `1/(npix·pixel)`). A resolution-favouring stability-check
variant of natural weighting, which is the default elsewhere.
"""
function uniform_weights(vis, grid::ImageGrid)
    Δ = 1 / (grid.npix * grid.pixel)
    cell(k) = (round(Int, vis.u[k] / Δ), round(Int, vis.v[k] / Δ))
    dens = Dictionary{Tuple{Int,Int},Float64}()
    for k in eachindex(vis.w)
        c = cell(k)
        set!(dens, c, get(dens, c, 0.0) + vis.w[k])
    end
    map(k -> vis.w[k] / dens[cell(k)], eachindex(vis.w))
end


"""
    fit_beam(u, v, weights) -> InterferometricModels.Beam

The elliptical-Gaussian restoring beam of a uv coverage, in closed form from its WEIGHTED SECOND
MOMENTS:

    Σ = (4π² · Σ w·[u v]ᵀ[u v] / Σ w)⁻¹ ,

whose eigendecomposition gives the major/minor axes and the position angle (from North through East,
`InterferometricModels`' `pa_major`). The returned `Beam` has unit peak intensity, so restoring puts a
component's own flux at its own pixel.

**Why this is the beam.** The dirty beam is `B(x) = Σ w·cos(2π·u·x) / Σ w` (the file header), whose
expansion about its own centre is `1 − 2π²·xᵀ⟨uuᵀ⟩x`; the Gaussian `exp(−xᵀQx)` with `Q = 2π²⟨uuᵀ⟩`
— i.e. `Σ = (2Q)⁻¹` as above — therefore has exactly the dirty beam's central curvature in every
direction, and for a Gaussian uv distribution it IS the dirty beam, exactly and everywhere. It is a
second-moment beam: systematically a few per cent narrower than a half-power fit, and weighted towards
the longest baselines when the weight distribution has tails.

**Why not the half-power contour.** A flood-filled half-power region is a discrete selection: a small
weight change can connect a sidelobe to the core and make the fitted beam jump. The moment beam has no
threshold or connectivity decision, so it varies smoothly with the weights. It intentionally does not
describe sidelobe structure or a half-power contour. Registration is beam-free; the restoring beam
affects restored maps, beam metadata, and peaks read from those maps.
"""
function fit_beam(u::AbstractVector{<:Real}, v::AbstractVector{<:Real}, weights::AbstractVector{<:Real})
    length(u) == length(v) == length(weights) ||
        error("fit_beam: u, v and weights have lengths $(length(u)), $(length(v)), $(length(weights))")
    Σw = zero(Float64); Suu = zero(Float64); Svv = zero(Float64); Suv = zero(Float64)
    @inbounds for k in eachindex(u, v, weights)
        w = Float64(weights[k]); uk = Float64(u[k]); vk = Float64(v[k])
        Σw += w; Suu += w * uk * uk; Svv += w * vk * vk; Suv += w * uk * vk
    end
    Σw > 0 || error("fit_beam: total weight is zero")
    M = SMatrix{2,2,Float64}(Suu, Suv, Suv, Svv) ./ Σw
    (M[1, 1] > 0 && M[2, 2] > 0 && det(M) > 0) ||
        error("fit_beam: the weighted uv second moments $(M) are not positive definite — the coverage " *
              "is a point or lies along one line through the origin, and has no beam in the " *
              "perpendicular direction")
    Σ = inv(4π^2 * M)
    E = eigen(Symmetric(Σ))
    λ = E.values; vmaj = E.vectors[:, 2]
    Beam(EllipticGaussian; σ_major = sqrt(λ[2]), ratio_minor_major = sqrt(λ[1] / λ[2]),
         pa_major = atan(vmaj[1], vmaj[2]))
end


function _peak(residual::Matrix{Float64}, window, rows::UnitRange{Int}, cols::UnitRange{Int})
    ip = 0; jp = 0; best = -1.0
    @inbounds for j in cols, i in rows
        (window === nothing || window[i, j]) || continue
        a = abs(residual[i, j])
        a > best && (best = a; ip = i; jp = j)
    end
    ip == 0 && error("hogbom: the CLEAN window is empty")
    (ip, jp, residual[ip, jp])
end

function _window_box(window, nx::Int, ny::Int)
    window === nothing && return (1:nx, 1:ny)
    size(window) == (nx, ny) ||
        error("hogbom: the window is $(size(window)), expected $((nx, ny))")
    any(window) || error("hogbom: the CLEAN window is empty")
    is = [i for i in 1:nx if any(@view window[i, :])]
    js = [j for j in 1:ny if any(@view window[:, j])]
    (first(is):last(is), first(js):last(js))
end

"""
    hogbom(dirty, patch, grid; gain, niter, threshold, window = nothing) -> (; model, residual, flux_map, niter, rms)

Image-plane Högbom CLEAN: repeatedly take the strongest residual pixel (by absolute value, inside
`window` when given), record `gain` × its value as a component and subtract that fraction of the dirty
beam `patch` centred there. No major cycles are needed, and not only "at these sizes": the beam patch
spans the whole difference grid, so the subtraction is EXACT — the image-plane residual equals the
dirty map of the model-subtracted visibilities to round-off, whatever the depth (pinned by a test).

Stops when the residual peak falls below `threshold` — an ABSOLUTE level in the map's own units
[Jy/beam], normally a multiplier × [`map_sigma`](@ref) — or after `niter` ITERATIONS, whichever comes
first. One component pixel can be hit many times, so the returned model usually has far fewer
components than the iteration count; `niter` is a runaway guard, not the intended stopping rule, and a
CLEAN that hits it has not converged.

**Absolute, deliberately.** The obvious alternative — a multiplier on the CURRENT residual rms — is
scale-free in the source flux: the rms of a source-dominated residual is proportional to the flux still
in the map, so such a loop halts at a fixed residual peak-to-rms set by the dirty beam's sidelobes and
never at the noise. `threshold` is therefore absolute rather than residual-relative.

Returns the accumulated per-pixel components as a `MultiComponentModel` of `Point`s (Jy, mas), the
residual map, the per-pixel component fluxes, the number of iterations used and the achieved residual
rms.

`window` (a `BitMatrix` over the image grid) restricts where components may be placed — the mechanism
the significance policy uses to keep noise components out of the structure-delay model.
"""
function hogbom(dirty::Matrix{Float64}, patch::Matrix{Float64}, grid::ImageGrid;
                gain::Real, niter::Integer, threshold::Real, window = nothing)
    nx, ny = size(dirty)
    (nx, ny) == (grid.npix, grid.npix) ||
        error("hogbom: the dirty map is $(size(dirty)), expected $((grid.npix, grid.npix))")
    size(patch) == (2nx - 1, 2ny - 1) ||
        error("hogbom: the beam patch is $(size(patch)), expected $((2nx - 1, 2ny - 1))")
    0 < gain ≤ 1 || error("hogbom: loop gain must be in (0, 1], got $gain")
    threshold > 0 || error("hogbom: threshold must be positive [Jy/beam], got $threshold")
    rows, cols = _window_box(window, nx, ny)
    residual = copy(dirty)
    fluxmap = zeros(nx, ny)
    used = 0
    for _ in 1:niter
        ip, jp, pk = _peak(residual, window, rows, cols)
        abs(pk) ≥ threshold || break
        δ = gain * pk
        fluxmap[ip, jp] += δ
        @inbounds for j in 1:ny
            bj = j - jp + ny
            @simd for i in 1:nx
                residual[i, j] -= δ * patch[i - ip + nx, bj]
            end
        end
        used += 1
    end
    xs, ys = grid_axes(grid)
    ord = sort(findall(!iszero, fluxmap); by = I -> -abs(fluxmap[I]))
    comps = [Point(fluxmap[I], SVector(xs[I[1]], ys[I[2]])) for I in ord]
    (; model = MultiComponentModel(comps), residual, flux_map = fluxmap, niter = used, rms = _rms(residual))
end

_rms(m::AbstractMatrix{Float64}) = sqrt(sum(abs2, m) / length(m))

"""
    significant_components(model, rms; nsigma) -> MultiComponentModel

The components of `model` whose flux reaches `nsigma`×`rms`. Unwindowed cleaning into the noise puts
spurious components in every map, and a fake
component at 10 mas would inject tens of picoseconds of fake structure delay on the long baselines.
"""
significant_components(model, rms::Real; nsigma::Real) =
    MultiComponentModel(filter(c -> abs(flux(c)) ≥ nsigma * rms, collect(components(model))))

"""
    restore(model, beam, residual, grid) -> Matrix{Float64}

The restored image [Jy/beam]: the CLEAN components convolved with the fitted restoring `beam`
(`InterferometricModels.convolve`, evaluated out to 4 major-axis FWHM of each component) plus the
residual map.
"""
function restore(model, beam::Beam, residual::Matrix{Float64}, grid::ImageGrid)
    xs, ys = grid_axes(grid)
    out = copy(residual)
    conv = convolve(model, beam)
    for c in components(conv)
        f = intensity(c)
        r = 4 * fwhm_max(c)
        cx, cy = coords(c)
        for j in eachindex(ys)
            abs(ys[j] - cy) ≤ r || continue
            for i in eachindex(xs)
                abs(xs[i] - cx) ≤ r || continue
                out[i, j] += f(SVector(xs[i], ys[j]))
            end
        end
    end
    out
end


"""
    SourceImage

One source's imaging product: the `grid`, the `dirty`/`residual`/`restored` maps [Jy/beam], the CLEAN
components as an `InterferometricModels.MultiComponentModel` of `Point`s (Jy, mas — the ONE
representation that also drives the model division and, later, the structure delays), the fitted
restoring `beam`, the residual `rms`, the peak flux and its position [mas], and a `provenance`
NamedTuple (visibility count, CLEAN settings, iteration, timings — whatever the driver records).
"""
struct SourceImage{M, B, P}
    source::Symbol
    grid::ImageGrid
    dirty::Matrix{Float64}
    residual::Matrix{Float64}
    restored::Matrix{Float64}
    model::M
    beam::B
    rms::Float64
    peak::Float64
    peak_xy::SVector{2,Float64}
    nvis::Int
    provenance::P
end

Base.show(io::IO, si::SourceImage) = print(io, "SourceImage(", si.source, ", ",
    si.grid.npix, "² × ", si.grid.pixel, " mas, ", length(components(si.model)), " components, ",
    round(cleaned_flux(si.model); digits = 3), " Jy cleaned, peak ", round(si.peak; digits = 3),
    " Jy/beam, rms ", round(1e3 * si.rms; digits = 2), " mJy/beam)")

"""
    cleaned_flux(model) -> Float64

Total CLEANed flux [Jy] of a component model (0 for an empty one, where `flux` has nothing to sum).
"""
cleaned_flux(model) = sum(flux, components(model); init = 0.0)

"""
    image_source(vis, grid, db; source, gain, niter, threshold, window = nothing,
                 weights = vis.w, provenance = (;)) -> SourceImage

The full per-source imaging step: dirty map → Högbom CLEAN against the cached dirty beam `db` (the
`(; patch, beam)` of [`dirty_beam`](@ref)) → restore. Every CLEAN setting is required; `threshold` is
the absolute stopping level [Jy/beam], see [`hogbom`](@ref) and [`map_sigma`](@ref). `weights` must be
the SAME weighting `db` was built with — the CLEAN loop deconvolves the map with that beam.

`window` is either a `BitMatrix` or a CALLABLE `(dirty, grid) -> BitMatrix` evaluated on this map. The
callable form is what lets a window policy depend on the map itself (the analysis's is "a disk around
the peak") without the caller having to compute the dirty map — the expensive DFT — a second time.
"""
function image_source(vis, grid::ImageGrid, db; source::Symbol, gain, niter, threshold,
                      window = nothing, weights = vis.w, provenance = (;))
    dirty = dirty_map(vis, grid; weights)
    win = window isa Union{Nothing,AbstractMatrix} ? window : window(dirty, grid)
    cl = hogbom(dirty, db.patch, grid; gain, niter, threshold, window = win)
    restored = restore(cl.model, db.beam, cl.residual, grid)
    I = argmax(restored)
    SourceImage(source, grid, dirty, cl.residual, restored, cl.model, db.beam, cl.rms,
                restored[I], peak_position(restored, grid), length(vis.u),
                merge(provenance, (; gain, niter, threshold, cleaned = cl.niter,
                                     window_pixels = win === nothing ? grid.npix^2 : count(win))))
end

function _subpixel(fm::Real, f0::Real, fp::Real)
    d = fm - 2f0 + fp
    d < 0 ? clamp((fm - fp) / (2d), -1.0, 1.0) : 0.0
end

"""
    peak_position(image, grid) -> SVector{2,Float64}

The position [mas] of an image's maximum, refined to sub-pixel by a parabolic fit against its four
neighbours (no refinement at the map edge).
"""
function peak_position(image::AbstractMatrix{Float64}, grid::ImageGrid)
    xs, ys = grid_axes(grid)
    I = argmax(image)
    i, j = Tuple(I)
    dx = (1 < i < size(image, 1)) ? _subpixel(image[i - 1, j], image[i, j], image[i + 1, j]) : 0.0
    dy = (1 < j < size(image, 2)) ? _subpixel(image[i, j - 1], image[i, j], image[i, j + 1]) : 0.0
    SVector(xs[i] + dx * grid.pixel, ys[j] + dy * grid.pixel)
end


function _model_visibility(comps::AbstractVector, u::Real, v::Real)
    z = zero(ComplexF64)
    @inbounds for c in comps
        z += flux(c) * cis(2π * (u * coords(c)[1] + v * coords(c)[2]))
    end
    z
end

function _model_columns(ds::Dataset, models, floor::Real; who::AbstractString = "model_dataset")
    0 ≤ floor < 1 || error("$who: `floor` is a fraction of the model's total flux, got $floor")
    occupied = Set(ds.rows.source_ix)
    comps = map(eachindex(ds.sources.name)) do sid
        nm = ds.sources.name[sid]
        sid in occupied || return Point{Float64,Float64}[]
        haskey(models, nm) ||
            error("$who: no model for source $nm — every source the dataset carries rows for needs one")
        cs = collect(components(models[nm]))
        isempty(cs) &&
            error("$who: the model for source $nm has no components")
        cs
    end
    thr = map(cs -> floor * sum(abs ∘ flux, cs; init = 0.0), comps)
    all(t -> isfinite(t), thr) || error("$who: a model has a non-finite total flux")
    (comps, thr)
end

"""
    model_dataset(ds::Dataset, models; floor) -> Dataset

Divide each visibility by its source's model visibility at that datum's `(u, v, ν)`, LAZILY — the
per-baseline analogue of [`calibrated_dataset`](@ref)'s antenna-gain division, and the transform the
hybrid loop re-fringes against (a point-source fit on model-divided data is unbiased where a
point-source fit on the raw data is not). `models` maps source names to
`InterferometricModels` models (anything supporting `haskey`/`getindex`; every source the dataset
carries rows for must have one, and no model may be empty).

It REWEIGHTS: `V′ = V/V_model`, `w′ = w·|V_model|²` — the diagonal-gain weight pattern extended per
baseline. Rows where `|V_model|` falls below `floor` × the model's total flux are
FLAGGED (`w′ = 0`, value untouched) instead of divided: near a null of a strong two-component source
the division amplifies noise without bound, and un-guarded it would destabilize the hybrid loop on
exactly the sources it exists for.
"""
function model_dataset(ds::Dataset, models; floor)
    comps, thr = _model_columns(ds, models, floor)
    νs = ds.freq.ν; sids = ds.rows.source_ix; uvws = ds.rows.uvw
    vcol = ds.rows.visibility; wcol = ds.rows.weight
    vis = mapview(eachindex(vcol)) do j
        cs = comps[sids[j]]; t = thr[sids[j]]; uvw = uvws[j]; V = vcol[j]
        mapview(eachindex(νs)) do c
            uv = uv_of(uvw, νs[c])
            M = _model_visibility(cs, uv[1], uv[2])
            abs(M) < t ? V[c] : V[c] ./ M
        end
    end
    wgt = mapview(eachindex(wcol)) do j
        cs = comps[sids[j]]; t = thr[sids[j]]; uvw = uvws[j]; W = wcol[j]
        mapview(eachindex(νs)) do c
            uv = uv_of(uvw, νs[c])
            M = _model_visibility(cs, uv[1], uv[2])
            abs(M) < t ? zero(W[c]) : W[c] .* abs2(M)
        end
    end
    dsv = @set ds.rows.visibility = vis
    @set dsv.rows.weight = wgt
end


"""
    noise_scale(vis) -> Float64

The factor `k` for which `k·w` is `1/σ²` in the visibilities' own flux units.

Correlator weights are proportional to `1/σ²` but carry no `2·Δν·τ` factor, so after amplitude
calibration they are `1/σ²` only up to ONE global constant — enough for imaging (which uses relative
weights) but not for a χ² against the thermal noise, which the closure statistic is read against.

The constant is MEASURED from the data rather than assumed from the correlator's conventions: within a
`(source, baseline, channel)` series of one scan the sky is constant to well below the noise, so the
second differences `V₁ − 2V₂ + V₃` are pure noise with `E|d|² = 6σ²`. `k` is the median over such
series of `(1/σ̂²)/w̄`. Fails loud if no series is long enough to carry three samples.

The estimator is an upper bound on σ: anything else that varies from sample to sample — residual
atmospheric phase curvature within a scan or an antenna entering/leaving — enters it too. Thus an
absolute χ² normalized with it is conservative, while iteration-to-iteration comparisons are
invariant to the common scale.
"""
function noise_scale(vis)
    ks = Float64[]
    for (_, idx) in pairs(groupfind(k -> (vis.source[k], vis.baseline_ix[k], vis.chan[k], vis.scan[k]),
                                    eachindex(vis.t)))
        length(idx) ≥ 3 || continue
        ord = idx[sortperm(view(vis.t, idx))]
        Σd = 0.0; n = 0
        for m in 3:length(ord)
            d = vis.V[ord[m - 2]] - 2 * vis.V[ord[m - 1]] + vis.V[ord[m]]
            Σd += abs2(d); n += 1
        end
        σ² = Σd / (6n)
        w̄ = mean(view(vis.w, ord))
        (σ² > 0 && w̄ > 0) && push!(ks, 1 / (σ² * w̄))
    end
    isempty(ks) &&
        error("noise_scale: no (source, baseline, channel, scan) series has the three samples the second-difference estimator needs")
    median(ks)
end

"""
    map_sigma(vis; weight_scale, weights = vis.w) -> Float64

The noise σ of ONE PIXEL of the dirty map [Jy/beam] that `vis` and `weights` produce — the level an
absolute CLEAN threshold is set from ([`hogbom`](@ref)), and the honest denominator for "how deep did
this map go".

`weight_scale` is the `k` of [`noise_scale`](@ref), for which `k·w` is `1/σ²` in flux units. The map is
`Σ wt·Re[V·cis(…)] / Σ wt`, and taking a REAL part halves the variance of a circular complex noise, so

    σ_map² = Σ wt² · σ²/2 / (Σ wt)² = Σ (wt²/w) / (2k·(Σ wt)²)   →   1/√(2k·Σw)  for natural weighting.

Two conventions are chosen here deliberately, and both are stated because they set what a "5σ" depth
means:

* **The factor of 2 is kept.** `1/√(k·Σw)` — the same expression without it — is the σ of the COMPLEX
  map, √2 above the real one that is actually made. The pixel σ of the real map is the quantity a
  threshold and a residual rms are compared against, so that is what this returns.
* **`k` is taken as it comes, upper bound and all.** [`noise_scale`](@ref)'s σ absorbs everything that
  varies within a scan, not just thermal noise. Thus `map_sigma` is an upper bound on thermal map noise,
  and cleaning to its multiple is conservative.

Not the residual rms: that one is an OUTPUT of CLEAN, and a map cleaned to convergence should reach a
residual rms of a few × this.
"""
function map_sigma(vis; weight_scale::Real, weights = vis.w)
    weight_scale > 0 || error("map_sigma: weight_scale must be positive, got $weight_scale")
    Σwt = 0.0; Σq = 0.0
    for k in eachindex(weights)
        wt = weights[k]; w = vis.w[k]
        (wt == 0) && continue
        w > 0 || error("map_sigma: visibility $k has a non-positive natural weight $w but imaging weight $wt")
        Σwt += wt; Σq += wt^2 / w
    end
    Σwt > 0 || error("map_sigma: total weight is zero")
    sqrt(Σq / (2 * weight_scale)) / Σwt
end


"""
    closure_chi2(vis, model; weight_scale = 1.0) -> (; chi2, n)

The closure-phase statistic of one source: over every antenna triangle of every `(source, time,
channel)` of the Stokes-I table `vis`, the difference between the observed closure phase and the `model`'s,
normalized by the thermal closure noise `σ² = Σ 1/SNR²` and averaged —

    χ² = ⟨ wrap(φ_closure,data − φ_closure,model)² / σ² ⟩ .

Closure phases are invariant under any antenna gain, so this is the arbiter of image quality that the
self-calibration gauge cannot flatter: it must fall through the hybrid iterations and then plateau, and
a final value ≫ 1 is an honest "structure not captured" marker. `model = nothing` is the point-source
reference (model closure phases identically zero).

`weight_scale` is the `k` of [`noise_scale`](@ref): the weights are `1/σ²` only up to one global
constant, and χ² is read against the thermal noise, so the constant matters here (it does not for the
imaging itself). The default of 1 leaves χ² on the weights' own arbitrary scale — fine for comparing
iterations of one source, meaningless as an absolute number.

Two things make the absolute value CONSERVATIVE (too small) by a known factor, both harmless for the
iteration-to-iteration comparison the loop is judged on: `noise_scale`'s σ is an upper bound on the
thermal one, and the per-baseline `SNR` here is `|V|/σ_complex`, √2 below the amplitude SNR the
textbook `σ_φ² = Σ 1/SNR²` is written with — so the normalization is 2× the thermal closure variance.
Triangles are enumerated in full (not a maximal independent set): the mean of `Δφ²/σ²` is unbiased
either way, only its scatter is understated by the correlation between triangles.
"""
function closure_chi2(vis, model; weight_scale::Real = 1.0)
    comps = model === nothing ? nothing : collect(components(model))
    Σ = 0.0; n = 0
    for (_, ks) in pairs(groupfind(k -> (vis.source[k], vis.t[k], vis.chan[k]), eachindex(vis.t)))
        idx = Dictionary{Tuple{Int,Int}, Int}()
        for k in ks
            p, q = vis.baseline_ix[k].antennas
            set!(idx, p < q ? (p, q) : (q, p), k)
        end
        ants = sort!(unique(Iterators.flatten(keys(idx))))
        length(ants) ≥ 3 || continue
        for ia in 1:length(ants), ib in (ia + 1):length(ants), ic in (ib + 1):length(ants)
            p, q, r = ants[ia], ants[ib], ants[ic]
            k1 = get(idx, (p, q), 0); k2 = get(idx, (q, r), 0); k3 = get(idx, (p, r), 0)
            (k1 > 0 && k2 > 0 && k3 > 0) || continue
            V1 = _oriented(vis, k1, p, q); V2 = _oriented(vis, k2, q, r); V3 = _oriented(vis, k3, p, r)
            snr2 = 0.0
            for k in (k1, k2, k3)
                s = abs(vis.V[k]) * sqrt(weight_scale * vis.w[k])
                s > 0 || (snr2 = NaN; break)
                snr2 += 1 / s^2
            end
            isfinite(snr2) || continue
            φd = angle(V1 * V2 * conj(V3))
            φm = 0.0
            if comps !== nothing
                M1 = _oriented_model(comps, vis, k1, p, q)
                M2 = _oriented_model(comps, vis, k2, q, r)
                M3 = _oriented_model(comps, vis, k3, p, r)
                φm = angle(M1 * M2 * conj(M3))
            end
            Σ += rem(φd - φm, 2π, RoundNearest)^2 / snr2
            n += 1
        end
    end
    (; chi2 = n > 0 ? Σ / n : NaN, n)
end

"""
    closure_amp_chi2(vis, model; weight_scale, min_snr) -> (; chi2, rms, n)

The closure-AMPLITUDE statistic of one source — what [`closure_chi2`](@ref) is for the phases. Over
every antenna quadrangle of every `(source, time, channel)` of the Stokes-I table `vis`, the log
closure amplitude

    ln A(p,q,r,s) = ln|V_pq| + ln|V_rs| − ln|V_pr| − ln|V_qs|

is invariant under any antenna amplitude gain, exactly as a closure phase is under any antenna phase.
Returned are the χ² against the `model`'s own closure amplitudes, normalized by the thermal
`σ² = Σ 1/SNR²` of the four baselines, and the plain weighted-free rms of `ln A_data − ln A_model` —
a dimensionless "by what factor do the gain-invariant amplitude ratios disagree".

This is the arbiter that says whether a residual amplitude error is an antenna gain (which
[`AmpSelfCal`](@ref) can remove) or the source model being wrong (which it cannot): a model that fits
the closure amplitudes while the visibility amplitudes are off by a factor is a model fighting antenna
gains, and the reverse is a model that does not describe the sky.

`weight_scale` is the `k` of [`noise_scale`](@ref) (the weights are `1/σ²` only up to one global
constant, and χ² is read against the thermal noise). `min_snr` drops a quadrangle in which any
baseline falls below it — `SNR = |V|·√(k·w) = |V|/σ_complex`, so a datum's `ln|V|` has variance
`1/(2·SNR²)` and a bias of `1/(4·SNR²)` (≈ 1 % per baseline at `min_snr = 5`, partly cancelling
between numerator and denominator), which is the floor this statistic can reach.

Two conventions it shares with [`closure_chi2`](@ref), both making the absolute value CONSERVATIVE
(too small) by a known factor and both harmless for the comparisons the loop is judged on:
`noise_scale`'s σ is an upper bound on the thermal one, and the normalization `σ² = Σ 1/SNR²` is
written with the complex SNR, hence exactly 2× the thermal variance of `ln A`.

All three pairings of each 4-subset are used — `(pq·rs)/(pr·qs)`, `(pq·rs)/(ps·qr)`, `(pr·qs)/(ps·qr)`
— of which only two are independent (the third is their ratio); as for the triangles of
[`closure_chi2`](@ref) the mean is unbiased either way and only its scatter is understated.

`model = nothing` is the point-source reference (model closure amplitudes identically 1).

The per-datum `ln|V|`, `ln|V_model|` and SNR are computed ONCE (threaded) and the quadrangle loop then
only sums four table entries: a quadrangle enumeration touches each datum ~C(n−1,3)·3 times, so
re-evaluating a thousand-component model inside it costs three orders of magnitude more than the
transform it is checking (measured: 20–50 s per bright source, against well under one with the
precomputation).
"""
function closure_amp_chi2(vis, model; weight_scale::Real, min_snr::Real)
    weight_scale > 0 || error("closure_amp_chi2: weight_scale must be positive, got $weight_scale")
    min_snr > 0 || error("closure_amp_chi2: min_snr must be positive, got $min_snr")
    comps = model === nothing ? nothing : collect(components(model))
    N = length(vis.t)
    eachindex(vis.t) == Base.OneTo(N) ||
        error("closure_amp_chi2: the visibility table must be 1-based, got $(eachindex(vis.t))")
    lnV = zeros(N); lnM = zeros(N); isnr2 = zeros(N); ok = fill(false, N)
    _foreach_tile(N) do k
        a = abs(vis.V[k])
        snr = a * sqrt(weight_scale * vis.w[k])
        (a > 0 && snr ≥ min_snr) || return
        if comps !== nothing
            m = abs(_model_visibility(comps, vis.u[k], vis.v[k]))
            m > 0 || return
            @inbounds lnM[k] = log(m)
        end
        @inbounds lnV[k] = log(a); @inbounds isnr2[k] = 1 / snr^2; @inbounds ok[k] = true
    end
    nant = maximum(k -> maximum(vis.baseline_ix[k].antennas), 1:N; init = 0)
    idx = zeros(Int, nant, nant)
    ants = Int[]
    Σ = 0.0; Σr = 0.0; n = 0
    for (_, ks) in pairs(groupfind(k -> (vis.source[k], vis.t[k], vis.chan[k]), 1:N))
        empty!(ants)
        for k in ks
            ok[k] || continue
            p, q = minmax(vis.baseline_ix[k].antennas...)
            idx[p, q] = k
            push!(ants, p); push!(ants, q)
        end
        sort!(unique!(ants))
        na = length(ants)
        if na ≥ 4
            @inbounds for ia in 1:na, ib in (ia + 1):na, ic in (ib + 1):na, id in (ic + 1):na
                p, q, r, s = ants[ia], ants[ib], ants[ic], ants[id]
                pq = idx[p, q]; rs = idx[r, s]; pr = idx[p, r]; qs = idx[q, s]
                ps = idx[p, s]; qr = idx[q, r]
                for (k1, k2, k3, k4) in ((pq, rs, pr, qs), (pq, rs, ps, qr), (pr, qs, ps, qr))
                    (k1 > 0 && k2 > 0 && k3 > 0 && k4 > 0) || continue
                    d = (lnV[k1] + lnV[k2] - lnV[k3] - lnV[k4]) -
                        (lnM[k1] + lnM[k2] - lnM[k3] - lnM[k4])
                    σ2 = isnr2[k1] + isnr2[k2] + isnr2[k3] + isnr2[k4]
                    Σ += d^2 / σ2; Σr += d^2; n += 1
                end
            end
        end
        for k in ks
            ok[k] || continue
            @inbounds idx[minmax(vis.baseline_ix[k].antennas...)...] = 0
        end
    end
    (; chi2 = n > 0 ? Σ / n : NaN, rms = n > 0 ? sqrt(Σr / n) : NaN, n)
end

@inline _oriented(vis, k, p, q) = vis.baseline_ix[k].antennas == (p, q) ? vis.V[k] : conj(vis.V[k])
@inline function _oriented_model(comps, vis, k, p, q)
    M = _model_visibility(comps, vis.u[k], vis.v[k])
    vis.baseline_ix[k].antennas == (p, q) ? M : conj(M)
end
