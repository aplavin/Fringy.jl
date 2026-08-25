
export GlobalDelayFit, AstrometryFit, fit, positions, antenna_clocks, antenna_zwd,
       antenna_gradients, receptor_offsets, parameter_names

function _validate_receptor_slot(receptor_slot::Int, context)
    receptor_slot in 1:2 ||
        error("$context: reference_receptor_slot must be 1 or 2, got $(repr(receptor_slot))")
    receptor_slot
end


"""
    GlobalDelayFit(; reference_antenna, reference_receptor_slot, elevation_cutoff, min_snr,
                   clock_node, zwd_node, clock_rate_constraint, zwd_rate_constraint,
                   gradient_constraint, reweight, reject, outlier_nsigma,
                   robust_iterations, sigma_floor_init, min_baseline_obs, gn_iterations)

Immutable specification of the global delay solve. Every field is required — there is no material
default anywhere in this pipeline. `elevation_cutoff` is explicit because ionosphere and hydrostatic
mapping functions degrade at low elevation.

| field | unit | meaning |
|---|---|---|
| `reference_antenna` | — | antenna whose clock and per-receptor-slot offset are fixed at zero (the clock gauge) |
| `reference_receptor_slot` | — | structural receptor slot, `1` or `2`, whose per-antenna offset is zero |
| `elevation_cutoff` | rad | both antennas of an accepted observation must be above it |
| `min_snr` | — | fringe SNR acceptance |
| `clock_node`, `zwd_node` | s | PWL node spacings |
| `clock_rate_constraint` | s/s | σ of the clock-rate pseudo-observation (72 ps/h = 2e-14) |
| `zwd_rate_constraint` | s/s | σ of the ZWD-rate pseudo-observation (40 ps/h) |
| `gradient_constraint` | s | σ of the gradient prior (0.5 mm of path / c) |
| `reweight` | — | iterate the per-baseline σ_floor to reduced χ² ≈ 1 |
| `reject` | — | reject at `outlier_nsigma`, with restoration |
| `outlier_nsigma` | — | rejection pull |
| `robust_iterations` | — | reweight/reject passes before the final solve |
| `sigma_floor_init` | s | starting per-baseline additive noise |
| `min_baseline_obs` | — | a baseline with fewer accepted observations keeps the initial floor |
| `gn_iterations` | — | Gauss–Newton steps when a `relinearize` hook is supplied to [`fit`](@ref) |
"""
struct GlobalDelayFit
    reference_antenna::Symbol
    reference_receptor_slot::Int
    elevation_cutoff::Float64
    min_snr::Float64
    clock_node::Float64
    zwd_node::Float64
    clock_rate_constraint::Float64
    zwd_rate_constraint::Float64
    gradient_constraint::Float64
    reweight::Bool
    reject::Bool
    outlier_nsigma::Float64
    robust_iterations::Int
    sigma_floor_init::Float64
    min_baseline_obs::Int
    gn_iterations::Int

    function GlobalDelayFit(args...)
        length(args) == 16 || throw(MethodError(GlobalDelayFit, args))
        _validate_receptor_slot(args[2], "GlobalDelayFit")
        new(args...)
    end
end

function GlobalDelayFit(; reference_antenna, reference_receptor_slot, elevation_cutoff, min_snr, clock_node, zwd_node,
               clock_rate_constraint, zwd_rate_constraint, gradient_constraint, reweight, reject,
               outlier_nsigma, robust_iterations, sigma_floor_init, min_baseline_obs, gn_iterations)
    receptor_slot = _validate_receptor_slot(reference_receptor_slot, "GlobalDelayFit")
    GlobalDelayFit(Symbol(reference_antenna), receptor_slot, Float64(elevation_cutoff),
                   Float64(min_snr), Float64(clock_node), Float64(zwd_node),
                   Float64(clock_rate_constraint), Float64(zwd_rate_constraint),
                   Float64(gradient_constraint), Bool(reweight), Bool(reject),
                   Float64(outlier_nsigma), Int(robust_iterations), Float64(sigma_floor_init),
                   Int(min_baseline_obs), Int(gn_iterations))
end


"""
    ParameterLayout

Which column of the design matrix is which: the source list (2 columns each), the free antennas (all
but the reference) with their clock PWL nodes and per-receptor-slot offset, and every antenna's ZWD PWL nodes
and two gradient components. Node grids are built from the accepted observations' epoch range.
"""
struct ParameterLayout
    sources::Vector{Symbol}
    antennas::Vector{Symbol}
    free::Vector{Symbol}
    clock_nodes::Vector{Float64}
    zwd_nodes::Vector{Float64}
    i_src::Int
    i_clk::Int
    i_receptor::Int
    i_zwd::Int
    i_grad::Int
    n::Int
end

function ParameterLayout(spec::GlobalDelayFit, sources, antennas, t0::Float64, t1::Float64)
    reference = spec.reference_antenna
    reference in antennas ||
        error("GlobalDelayFit: reference antenna $reference is absent from the accepted observations $(antennas)")
    free = filter(!=(reference), antennas)
    nodes(step) = (k = max(ceil(Int, (t1 - t0) / step), 1) + 1; collect(t0 .+ step .* (0:k-1)))
    cn = nodes(spec.clock_node)
    zn = nodes(spec.zwd_node)
    n = 0
    i_src = n; n += 2 * length(sources)
    i_clk = n; n += length(free) * length(cn)
    i_receptor = n; n += length(free)
    i_zwd = n; n += length(antennas) * length(zn)
    i_grad = n; n += 2 * length(antennas)
    ParameterLayout(collect(sources), collect(antennas), free, cn, zn,
                    i_src, i_clk, i_receptor, i_zwd, i_grad, n)
end

src_col(L::ParameterLayout, s::Integer, which::Integer) = L.i_src + 2 * (s - 1) + which
clk_col(L::ParameterLayout, a::Integer, k::Integer) = L.i_clk + (a - 1) * length(L.clock_nodes) + k
receptor_col(L::ParameterLayout, a::Integer) = L.i_receptor + a
zwd_col(L::ParameterLayout, a::Integer, k::Integer) = L.i_zwd + (a - 1) * length(L.zwd_nodes) + k
grad_col(L::ParameterLayout, a::Integer, which::Integer) = L.i_grad + 2 * (a - 1) + which

"""
    parameter_names(L::ParameterLayout) -> Vector{String}

One human-readable name per column, in column order.
"""
function parameter_names(L::ParameterLayout)
    nm = fill("", L.n)
    for (s, src) in enumerate(L.sources)
        nm[src_col(L, s, 1)] = "Δα★[$src]"
        nm[src_col(L, s, 2)] = "Δδ[$src]"
    end
    for (a, st) in enumerate(L.free)
        for k in eachindex(L.clock_nodes)
            nm[clk_col(L, a, k)] = "clock[$st,$k]"
        end
        nm[receptor_col(L, a)] = "recofs[$st]"
    end
    for (a, st) in enumerate(L.antennas)
        for k in eachindex(L.zwd_nodes)
            nm[zwd_col(L, a, k)] = "zwd[$st,$k]"
        end
        nm[grad_col(L, a, 1)] = "gradN[$st]"
        nm[grad_col(L, a, 2)] = "gradE[$st]"
    end
    nm
end

function _pwl(nodes::Vector{Float64}, t::Float64)
    j = clamp(searchsortedlast(nodes, t), 1, length(nodes) - 1)
    u = (t - nodes[j]) / (nodes[j+1] - nodes[j])
    (j, 1 - u, u)
end


struct DesignRows
    ptr::Vector{Int}
    col::Vector{Int}
    val::Vector{Float64}
    npar::Int
end

Base.length(D::DesignRows) = length(D.ptr) - 1

@inline function row_dot(D::DesignRows, i::Integer, x::AbstractVector{Float64})
    s = 0.0
    @inbounds for k in D.ptr[i]:(D.ptr[i+1] - 1)
        s += D.val[k] * x[D.col[k]]
    end
    s
end

function accumulate_normal!(N::Matrix{Float64}, rhs::Vector{Float64}, D::DesignRows,
                            y::AbstractVector{Float64}, w::AbstractVector{Float64})
    @inbounds for i in 1:length(D)
        wi = w[i]
        wi > 0 || continue
        lo = D.ptr[i]; hi = D.ptr[i+1] - 1
        wy = wi * y[i]
        for k in lo:hi
            ck = D.col[k]; vk = D.val[k]
            rhs[ck] += wy * vk
            wv = wi * vk
            for l in lo:hi
                N[D.col[l], ck] += wv * D.val[l]
            end
        end
    end
    nothing
end

struct _RowBuilder
    ptr::Vector{Int}
    col::Vector{Int}
    val::Vector{Float64}
end
_RowBuilder() = _RowBuilder([1], Int[], Float64[])
@inline function push_entry!(b::_RowBuilder, c::Integer, v::Real)
    v == 0 && return nothing
    push!(b.col, Int(c)); push!(b.val, Float64(v)); nothing
end
end_row!(b::_RowBuilder) = (push!(b.ptr, length(b.col) + 1); nothing)
DesignRows(b::_RowBuilder, npar::Integer) = DesignRows(b.ptr, b.col, b.val, Int(npar))


"""
    AstrometryFit

The result of [`fit`](@ref): the estimated parameter vector `x` and its covariance `cov` (both in SI —
positions in radians, delays in seconds), the `layout` naming every column, per-observation
`residual`/`weight`/`accepted` over the **accepted-candidate rows** (`selected`, indices into the
input table), the per-baseline `sigma_floor`, the iteration `log`, and the summary `chi2`, `wrms`,
`ndof`.

`cov` is the inverse normal matrix of the weights actually used, unscaled: since the per-baseline
reweighting drives the reduced χ² to ≈ 1 the two conventions coincide, and `chi2` is reported so the
caller can rescale explicitly. It is a FORMAL covariance and, as the whole verification phase
concluded, must be reported alongside — never instead of — a systematic error budget.

Two conventions worth stating explicitly because they are read off these fields:

- **dead columns** — a parameter that no observation and no constraint touches (in practice only a
  relative receptor-slot offset of an antenna observed in the reference receptor slot alone) is *fixed* at zero rather
  than made singular, and its row/column of `cov` is zero: a fixed parameter, not an infinitely
  uncertain one. `factorization` names how the scaled normal matrix was inverted (`:cholesky`, or
  `:pseudoinverse` when exact degeneracies remain — e.g. every relative receptor-slot offset in a
  split solve with one receptor slot).
- **`chi2`** is `Σ w r²` over the accepted DATA rows divided by `ndof = n_accepted − n_parameters +
  n_constraints`, i.e. the constraint pseudo-observations enter as the count of effectively-frozen
  parameter directions but their own residuals are NOT in the numerator.
  `sessions/diagnostics/residuals.jl` reports the
  constraint block's own χ² separately, which is where prior-vs-data tension shows up.
"""
struct AstrometryFit
    spec::GlobalDelayFit
    layout::ParameterLayout
    x::Vector{Float64}
    cov::Matrix{Float64}
    selected::Vector{Int}
    residual::Vector{Float64}
    weight::Vector{Float64}
    accepted::BitVector
    sigma_floor::Dictionary{Tuple{Symbol,Symbol}, Float64}
    log::Vector{NamedTuple}
    gn_log::Vector{NamedTuple}
    chi2::Float64
    wrms::Float64
    ndof::Int
    factorization::Symbol
end

Base.show(io::IO, f::AstrometryFit) = print(io, "AstrometryFit(", length(f.selected), " observables, ",
    f.layout.n, " parameters, wrms ", round(1e12 * f.wrms; digits = 2), " ps, χ²/dof ",
    round(f.chi2; digits = 3), ", rejected ",
    round(100 * (1 - count(f.accepted) / length(f.accepted)); digits = 2), "%)")


function _solve_normal(N::Matrix{Float64}, rhs::Vector{Float64})
    n = size(N, 1)
    dead = [N[i, i] ≤ 0 for i in 1:n]
    if any(dead)
        for i in findall(dead)
            N[i, :] .= 0.0; N[:, i] .= 0.0; N[i, i] = 1.0; rhs[i] = 0.0
        end
    end
    sc = sqrt.(diag(N))
    Ns = Symmetric(N ./ (sc * sc'))
    local Ninv, how
    ch = cholesky(Ns; check = false)
    if issuccess(ch)
        Ninv = inv(ch)
        how = :cholesky
    else
        e = eigen(Ns)
        cutoff = 1e-10 * maximum(e.values)
        inv_λ = [λ > cutoff ? 1 / λ : 0.0 for λ in e.values]
        Ninv = e.vectors * Diagonal(inv_λ) * e.vectors'
        how = :pseudoinverse
    end
    x = (Ninv * (rhs ./ sc)) ./ sc
    cov = Ninv ./ (sc * sc')
    for i in findall(dead)
        x[i] = 0.0; cov[i, :] .= 0.0; cov[:, i] .= 0.0
    end
    (x, Matrix(cov), how)
end

function _floor_for_chi2(r::AbstractVector{Float64}, var0::AbstractVector{Float64}, scale::Float64)
    f(s2) = mean(r[k]^2 / (var0[k] + s2) for k in eachindex(r)) / scale - 1
    f(0.0) ≤ 0 && return 0.0
    lo = 0.0
    hi = mean(abs2, r) / max(scale, 1e-3)
    for _ in 1:20
        f(hi) > 0 || break
        hi *= 4
    end
    for _ in 1:100
        mid = (lo + hi) / 2
        f(mid) > 0 ? (lo = mid) : (hi = mid)
    end
    (lo + hi) / 2
end

"""
    fit(spec::GlobalDelayFit, observables; relinearize = nothing) -> AstrometryFit

Solve the global weighted least-squares problem on the observable table produced by
[`total_delays`](@ref).

Required columns: `τ_residual`, `σ_τ`, `σ_τ_ionosphere`, `snr`, `t`, `source`, `a1`, `a2`, `receptor_slots`, `el1`,
`el2`, `az1`, `az2`, `wet_mapping1`, `wet_mapping2`, `gradient_mapping1`, `gradient_mapping2`, `∂τ_∂α★`, `∂τ_∂δ`. `receptor_slots` is the
ordered structural receptor-slot pair. This astrometry estimator accepts only diagonal pairs `(1, 1)` and
`(2, 2)`; receptor labels are not inspected.

Acceptance requires finite `τ_residual`, `snr ≥ spec.min_snr`, and both elevations above
`spec.elevation_cutoff`. The `spec.robust_iterations` passes then alternate the solve with the
per-baseline σ_floor reweighting (to reduced χ² ≈ 1) and the `spec.outlier_nsigma` rejection **with
restoration** (every row is re-tested each pass), before a final solve on the converged weights.

`relinearize` is the Gauss–Newton hook: a function `(Δα★, Δδ) per source ⇒ new observable table`
(receiving a `Dictionary{Symbol, Tuple{Float64,Float64}}` of the accumulated position offsets in
radians). When supplied, `spec.gn_iterations` linearizations are performed and each step's position
update is recorded in the log. When it is `nothing`, the problem is solved once about the supplied
source positions.
"""
function fit(spec::GlobalDelayFit, observables; relinearize = nothing)
    obs = observables
    for i in eachindex(obs)
        _astrometric_receptor_slot(obs.receptor_slots[i], "GlobalDelayFit: row $i")
    end
    accept = map(eachindex(obs)) do i
        isfinite(obs.τ_residual[i]) && isfinite(obs.σ_τ[i]) && isfinite(obs.σ_τ_ionosphere[i]) &&
            isfinite(obs.snr[i]) && obs.snr[i] ≥ spec.min_snr &&
            obs.el1[i] ≥ spec.elevation_cutoff && obs.el2[i] ≥ spec.elevation_cutoff
    end
    sel = findall(accept)
    isempty(sel) && error("GlobalDelayFit: no observation passes acceptance (SNR ≥ $(spec.min_snr), elevation ≥ $(rad2deg(spec.elevation_cutoff))°)")
    sources = sort!(unique(@view obs.source[sel]))
    antennas = sort!(unique(vcat(collect(@view obs.a1[sel]), collect(@view obs.a2[sel]))))
    t0, t1 = extrema(@view obs.t[sel])
    L = ParameterLayout(spec, sources, antennas, Float64(t0), Float64(t1))

    offsets = Dictionary(sources, [(0.0, 0.0) for _ in sources])
    nsteps = isnothing(relinearize) ? 1 : max(spec.gn_iterations, 1)
    local result
    gnlog = NamedTuple[]
    for step in 1:nsteps
        cur = step == 1 ? obs : relinearize(offsets)
        length(cur) == length(obs) ||
            error("GlobalDelayFit: `relinearize` returned $(length(cur)) rows for a $(length(obs))-row observable table — the hook must rebuild the same rows in the same order")
        result = _fit_once(spec, cur, sel, L)
        upd = _position_update(result, L)
        push!(gnlog, (; gn_step = step, chi2 = result.chi2, wrms = result.wrms,
                      max_position_update_mas = isempty(upd) ? 0.0 : maximum(abs, upd) / _MAS))
        for (k, s) in enumerate(sources)
            offsets[s] = (offsets[s][1] + result.x[src_col(L, k, 1)],
                          offsets[s][2] + result.x[src_col(L, k, 2)])
        end
    end
    x = copy(result.x)
    for (k, s) in enumerate(sources)
        x[src_col(L, k, 1)] = offsets[s][1]
        x[src_col(L, k, 2)] = offsets[s][2]
    end
    AstrometryFit(spec, L, x, result.cov, sel, result.residual, result.weight, result.accepted,
                  result.sigma_floor, result.log, gnlog, result.chi2, result.wrms,
                  result.ndof, result.factorization)
end

_position_update(r, L::ParameterLayout) =
    [r.x[src_col(L, k, w)] for k in eachindex(L.sources), w in 1:2]

function _fit_once(spec::GlobalDelayFit, obs, sel::Vector{Int}, L::ParameterLayout)
    srcof = Dictionary(L.sources, eachindex(L.sources))
    freeof = Dictionary(L.free, eachindex(L.free))
    antof = Dictionary(L.antennas, eachindex(L.antennas))
    ncl = length(L.clock_nodes)
    nzw = length(L.zwd_nodes)

    b = _RowBuilder()
    for i in sel
        t = Float64(obs.t[i])
        s = srcof[obs.source[i]]
        push_entry!(b, src_col(L, s, 1), obs.∂τ_∂α★[i])
        push_entry!(b, src_col(L, s, 2), obs.∂τ_∂δ[i])
        jc, c0, c1 = _pwl(L.clock_nodes, t)
        jz, z0, z1 = _pwl(L.zwd_nodes, t)
        other_receptor = obs.receptor_slots[i][1] == spec.reference_receptor_slot ? 0.0 : 1.0
        for (ant, sgn, wet_map, gradient_map, az) in ((obs.a2[i], +1.0, obs.wet_mapping2[i], obs.gradient_mapping2[i], obs.az2[i]),
                                                       (obs.a1[i], -1.0, obs.wet_mapping1[i], obs.gradient_mapping1[i], obs.az1[i]))
            if haskey(freeof, ant)
                a = freeof[ant]
                push_entry!(b, clk_col(L, a, jc), sgn * c0)
                push_entry!(b, clk_col(L, a, jc + 1), sgn * c1)
                push_entry!(b, receptor_col(L, a), sgn * other_receptor)
            end
            a = antof[ant]
            push_entry!(b, zwd_col(L, a, jz), sgn * wet_map * z0)
            push_entry!(b, zwd_col(L, a, jz + 1), sgn * wet_map * z1)
            push_entry!(b, grad_col(L, a, 1), sgn * gradient_map * cos(az))
            push_entry!(b, grad_col(L, a, 2), sgn * gradient_map * sin(az))
        end
        end_row!(b)
    end
    D = DesignRows(b, L.n)

    cb = _RowBuilder()
    cσ = Float64[]
    for a in eachindex(L.free), j in 1:(ncl - 1)
        dt = L.clock_nodes[j+1] - L.clock_nodes[j]
        push_entry!(cb, clk_col(L, a, j), -1 / dt)
        push_entry!(cb, clk_col(L, a, j + 1), 1 / dt)
        end_row!(cb); push!(cσ, spec.clock_rate_constraint)
    end
    for a in eachindex(L.antennas)
        for j in 1:(nzw - 1)
            dt = L.zwd_nodes[j+1] - L.zwd_nodes[j]
            push_entry!(cb, zwd_col(L, a, j), -1 / dt)
            push_entry!(cb, zwd_col(L, a, j + 1), 1 / dt)
            end_row!(cb); push!(cσ, spec.zwd_rate_constraint)
        end
        for w in 1:2
            push_entry!(cb, grad_col(L, a, w), 1.0)
            end_row!(cb); push!(cσ, spec.gradient_constraint)
        end
    end
    C = DesignRows(cb, L.n)
    cw = 1 ./ cσ .^ 2
    cy = zeros(length(C))

    y = Float64[obs.τ_residual[i] for i in sel]
    var0 = Float64[obs.σ_τ[i]^2 + obs.σ_τ_ionosphere[i]^2 for i in sel]
    blkey = [minmax(obs.a1[i], obs.a2[i]) for i in sel]
    ublk = sort!(unique(blkey))
    blof = Dictionary(ublk, eachindex(ublk))
    blix = [blof[k] for k in blkey]
    floors = fill(spec.sigma_floor_init^2, length(ublk))
    keep = trues(length(sel))
    members = [Int[] for _ in ublk]
    for (i, k) in enumerate(blix)
        push!(members[k], i)
    end

    npass = (spec.reweight || spec.reject) ? spec.robust_iterations : 0
    itlog = NamedTuple[]
    nchanged = 0
    local x, cov, how, r, w, chi2, wrms, ndof
    for it in 0:npass
        w = [keep[i] ? 1 / (var0[i] + floors[blix[i]]) : 0.0 for i in eachindex(sel)]
        N = zeros(L.n, L.n); rhs = zeros(L.n)
        accumulate_normal!(N, rhs, D, y, w)
        accumulate_normal!(N, rhs, C, cy, cw)
        x, cov, how = _solve_normal(N, rhs)
        r = [y[i] - row_dot(D, i, x) for i in eachindex(y)]
        sw = sum(w)
        swr = sum(w[i] * r[i]^2 for i in eachindex(r))
        ndof = max(count(keep) - L.n + length(C), 1)
        chi2 = swr / ndof
        wrms = sqrt(swr / sw)
        push!(itlog, (; iteration = it, chi2, wrms, kept = count(keep), changed = nchanged,
                      floor_median_ps = 1e12 * sqrt(median(floors)), factorization = how))
        it == npass && break
        if spec.reweight
            scale = ndof / count(keep)
            for k in eachindex(ublk)
                idx = filter(i -> keep[i], members[k])
                length(idx) ≥ spec.min_baseline_obs || continue
                floors[k] = _floor_for_chi2(view(r, idx), view(var0, idx), scale)
            end
        end
        if spec.reject
            newkeep = BitVector(abs(r[i]) / sqrt(var0[i] + floors[blix[i]]) < spec.outlier_nsigma
                                for i in eachindex(r))
            nchanged = count(newkeep .⊻ keep)
            keep = newkeep
        end
    end
    (; x, cov, residual = r, weight = w, accepted = BitVector(keep),
       sigma_floor = Dictionary(ublk, sqrt.(floors)), log = convert(Vector{NamedTuple}, itlog),
       chi2, wrms, ndof, factorization = how)
end


const _MAS = deg2rad(1e-3 / 3600)

"""
    positions(f::AstrometryFit) -> StructArray

One row per source: `(; source, Δα★_mas, Δδ_mas, σ_Δα★_mas, σ_Δδ_mas, correlation)` — the
estimated offsets from the source positions used to build the original observable table, in
(Δα★ = cos(δ) Δα, Δδ), and their formal uncertainties.
"""
function positions(f::AstrometryFit)
    L = f.layout
    StructArray(map(eachindex(L.sources)) do k
        ia = src_col(L, k, 1); id = src_col(L, k, 2)
        σa = sqrt(max(f.cov[ia, ia], 0.0)); σd = sqrt(max(f.cov[id, id], 0.0))
        (; source = L.sources[k], Δα★_mas = f.x[ia] / _MAS, Δδ_mas = f.x[id] / _MAS,
           σ_Δα★_mas = σa / _MAS, σ_Δδ_mas = σd / _MAS,
           correlation = σa > 0 && σd > 0 ? f.cov[ia, id] / (σa * σd) : 0.0)
    end)
end

"""
    antenna_clocks(f::AstrometryFit) -> Dictionary{Symbol, Vector{Float64}}

The estimated residual clock [s] at each PWL node, per free antenna (the reference antenna is fixed at
zero and absent). Node epochs are `f.layout.clock_nodes`.
"""
antenna_clocks(f::AstrometryFit) = Dictionary(f.layout.free,
    [[f.x[clk_col(f.layout, a, k)] for k in eachindex(f.layout.clock_nodes)]
     for a in eachindex(f.layout.free)])

"""
    antenna_zwd(f::AstrometryFit) -> Dictionary{Symbol, Vector{Float64}}

The estimated zenith wet delay [s] at each PWL node, per antenna. Node epochs are
`f.layout.zwd_nodes`.
"""
antenna_zwd(f::AstrometryFit) = Dictionary(f.layout.antennas,
    [[f.x[zwd_col(f.layout, a, k)] for k in eachindex(f.layout.zwd_nodes)]
     for a in eachindex(f.layout.antennas)])

"""
    antenna_gradients(f::AstrometryFit) -> Dictionary{Symbol, NTuple{2,Float64}}

The estimated (north, east) tropospheric gradients [s of zenith-equivalent path] per antenna.
"""
antenna_gradients(f::AstrometryFit) = Dictionary(f.layout.antennas,
    [(f.x[grad_col(f.layout, a, 1)], f.x[grad_col(f.layout, a, 2)])
     for a in eachindex(f.layout.antennas)])

"""
    receptor_offsets(f::AstrometryFit) -> Dictionary{Symbol, Float64}

The estimated per-antenna clock offset [s] of the non-reference receptor slot.
"""
receptor_offsets(f::AstrometryFit) = Dictionary(f.layout.free,
    [f.x[receptor_col(f.layout, a)] for a in eachindex(f.layout.free)])
