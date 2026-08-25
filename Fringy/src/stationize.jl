
export stationize, stefcal!, SNRWeighted, CurvatureWeighted

@kwdef struct SNRWeighted
    iterations::Int
end
@kwdef struct CurvatureWeighted
    iterations::Int
end

node_index(p::Integer, i::Integer) = 2 * (p - 1) + i
node_antenna(n::Integer) = (n + 1) ÷ 2
node_receptor(n::Integer) = ((n - 1) % 2) + 1

function _node_components(node_pairs, nant::Integer)
    graph = SimpleGraph(2nant)
    present = falses(2nant)
    for (na, nb) in node_pairs
        present[na] = true; present[nb] = true
        add_edge!(graph, na, nb)
    end
    comps = [sort(filter(n -> present[n], c)) for c in connected_components(graph) if any(n -> present[n], c)]
    (; present, components = comps)
end

"""
    stefcal!(g::AbstractVector{ComplexF64}, edges; iterations) -> g

Classical simultaneous StEFCal over a node set. Each element of `edges` carries `(a, b, value, model,
weight)`; a directed `a→b` means `value ≈ g[a]·model·conj(g[b])`. Each edge contributes to BOTH its
endpoints' updates (the reverse orientation is the conjugate model). Every node is updated from the SAME
previous full vector (order-independent, deterministic). Salvini–Wijnholds relaxation: on even
iterations the new iterate is averaged with the previous one. Mutates and returns `g`; start it at ones.
"""
function stefcal!(g::AbstractVector{ComplexF64}, edges; iterations::Integer)
    n = length(g)
    num = zeros(ComplexF64, n)
    den = zeros(Float64, n)
    gprev = copy(g)
    for k in 1:iterations
        copyto!(gprev, g)
        fill!(num, zero(ComplexF64))
        fill!(den, 0.0)
        for e in edges
            za = e.model * conj(gprev[e.b])
            num[e.a] += e.weight * conj(za) * e.value
            den[e.a] += e.weight * abs2(za)
            zb = conj(e.model) * conj(gprev[e.a])
            num[e.b] += e.weight * conj(zb) * conj(e.value)
            den[e.b] += e.weight * abs2(zb)
        end
        for m in 1:n
            den[m] > 0 && (g[m] = num[m] / den[m])
        end
        iseven(k) && (g .= (g .+ gprev) ./ 2)
    end
    g
end

function _scalar_wls(nc::Integer, cedges, mfun, wfun, anchor::Integer)
    A = zeros(nc, nc)
    b = zeros(nc)
    for e in cedges
        w = wfun(e)
        m = mfun(e)
        a, c = e.la, e.lb
        A[a, a] += w; A[c, c] += w; A[a, c] -= w; A[c, a] -= w
        b[a] += w * m; b[c] -= w * m
    end
    keep = filter(!=(anchor), 1:nc)
    x = zeros(nc)
    x[keep] = cholesky(Symmetric(A[keep, keep])) \ b[keep]
    x
end

function _block_wls(nc::Integer, cedges, anchor::Integer, Δr::Real, Δτ::Real)
    Sinv = SDiagonal(float(Δr), float(Δτ))
    A = zeros(2nc, 2nc)
    b = zeros(2nc)
    rate_info = 0.0
    delay_info = 0.0
    for e in cedges
        Qp = Sinv * e.Q * Sinv
        mp = SVector(e.rate / Δr, e.delay / Δτ)
        rate_info += Qp[1, 1]
        delay_info += Qp[2, 2]
        a, c = e.la, e.lb
        ba, bc = (2a-1):2a, (2c-1):2c
        A[ba, ba] .+= Qp; A[bc, bc] .+= Qp
        A[ba, bc] .-= Qp; A[bc, ba] .-= Qp
        b[ba] .+= Qp * mp; b[bc] .-= Qp * mp
    end
    keep = filter(k -> !(k == 2anchor - 1 || k == 2anchor), 1:2nc)
    chol = cholesky(Symmetric(A[keep, keep]); check=false)
    issuccess(chol) || _fail_singular_block(rate_info, delay_info)
    xp = zeros(2nc)
    xp[keep] = chol \ b[keep]
    rate = zeros(nc)
    delay = zeros(nc)
    for l in 1:nc
        xn = Sinv * SVector(xp[2l-1], xp[2l])
        rate[l] = xn[1]
        delay[l] = xn[2]
    end
    (rate, delay)
end

function _fail_singular_block(rate_info, delay_info)
    mx = max(rate_info, delay_info)
    dir = delay_info ≤ 1e-10 * mx ? "delay (e.g. a single-channel tile — no delay curvature)" :
          rate_info ≤ 1e-10 * mx ? "rate (e.g. a single-integration tile — no rate curvature)" :
          "rate/delay"
    error("CurvatureWeighted block solve: the reduced normal system is singular — the $dir direction is unconstrained across all edges.")
end

function _phase_solve(nc::Integer, cedges, wfun, anchor::Integer, iterations::Integer)
    sedges = map(cedges) do e
        (; a = e.la, b = e.lb, value = e.coeff / abs(e.coeff), model = ComplexF64(1), weight = wfun(e))
    end
    g = ones(ComplexF64, nc)
    stefcal!(g, sedges; iterations)
    rot = conj(g[anchor]) / abs(g[anchor])
    map(x -> angle(rot * x), g)
end

_solve_delayrate(::SNRWeighted, nc, cedges, anchor, Δr, Δτ) =
    (_scalar_wls(nc, cedges, e -> e.rate, e -> e.snr, anchor),
     _scalar_wls(nc, cedges, e -> e.delay, e -> e.snr, anchor))
_solve_delayrate(::CurvatureWeighted, nc, cedges, anchor, Δr, Δτ) =
    _block_wls(nc, cedges, anchor, Δr, Δτ)

_phaseweight(::SNRWeighted) = e -> e.snr
_phaseweight(::CurvatureWeighted) = e -> e.q_ab

function _component_edges(edges, local_of)
    keep = filter(e -> haskey(local_of, node_index(e.p, e.i)) && haskey(local_of, node_index(e.q, e.j)), edges)
    map(keep) do e
        (; la = local_of[node_index(e.p, e.i)], lb = local_of[node_index(e.q, e.j)],
           e.delay, e.rate, e.coeff, e.snr, e.Q, e.q_ab)
    end
end

"""
    stationize(edges, nant; globalization, reference_antenna, rate_binwidth, delay_binwidth)
        -> (; delay, rate, phase, node_present, components)

Solve one tile's accepted correlation-product peaks into per-node (antenna, receptor slot)
delay/rate/phase. Nodes span `1:2·nant` (`node_index`); each edge connects
`(p,i)—(q,j)`, where `i` and `j` are structural receptor-slot indices. Receptor labels are not inspected.
Connected components of the node graph are solved independently. Per component the gauge node is
`(reference_antenna, its lowest present receptor slot)` when the reference antenna has a node in the
component, else the lowest-index node (`reference_antenna=0` always selects the latter). The gauge node's
delay/rate/phase are gauged to 0.

`globalization::SNRWeighted|CurvatureWeighted` selects the delay/rate solver (scalar SNR-weighted vs
2×2 curvature-block) and the phase edge weight (snr vs q_ab); `globalization.iterations` counts the
phase-StEFCal iterations. `rate_binwidth`/`delay_binwidth` nondimensionalize the block solve.

Returns length-`2·nant` `delay` [s], `rate` [Hz], `phase` [rad] (zero for absent nodes),
`node_present::Vector{Bool}` (nodes that appear on an edge), and `components` — a vector of
`(; nodes, anchor, anchored_at_reference)` diagnostics per solved connected component.
"""
function stationize(edges, nant::Integer; globalization, reference_antenna::Integer,
                    rate_binwidth::Real, delay_binwidth::Real)
    nnodes = 2nant
    gc = _node_components(((node_index(e.p, e.i), node_index(e.q, e.j)) for e in edges), nant)
    node_present = gc.present

    delay = zeros(nnodes)
    rate = zeros(nnodes)
    phase = zeros(nnodes)
    diags = map(gc.components) do compnodes
        local_of = Dictionary(compnodes, eachindex(compnodes))
        nc = length(compnodes)
        cedges = _component_edges(edges, local_of)
        refnodes = filter(n -> node_antenna(n) == reference_antenna, compnodes)
        anchor_node = isempty(refnodes) ? minimum(compnodes) : minimum(refnodes)
        anchor = local_of[anchor_node]

        rvals, dvals = _solve_delayrate(globalization, nc, cedges, anchor, rate_binwidth, delay_binwidth)
        pvals = _phase_solve(nc, cedges, _phaseweight(globalization), anchor, globalization.iterations)
        for (l, n) in enumerate(compnodes)
            rate[n] = rvals[l]
            delay[n] = dvals[l]
            phase[n] = pvals[l]
        end
        (; nodes = compnodes, anchor = anchor_node, anchored_at_reference = !isempty(refnodes))
    end
    (; delay, rate, phase, node_present, components = diags)
end
