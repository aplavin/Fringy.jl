
abstract type PartitionAxis end
struct TimeAxis <: PartitionAxis end
struct FreqAxis <: PartitionAxis end

"""
    Partition{AX<:PartitionAxis, Q<:Quantity}

Resolved division of a dataset axis. `cells` is a `StructArray` with a `support::ClosedInterval{Q}`
(the cell's physical envelope) and a `reference::Q` (its support midpoint, deterministic — may lie in
a gap, e.g. a frequency reference between two IFs). `lookup[i]` maps elementary item `i` (row for
time, channel for frequency) to its cell index. Time supports/references are seconds relative to
`ds.time0`; frequency ones are Hz.
"""
struct Partition{AX<:PartitionAxis, Q<:Quantity, C<:StructVector}
    cells::C
    lookup::Vector{Int32}
end

const TimePartition = Partition{TimeAxis, typeof(1.0u"s")}
const FreqPartition = Partition{FreqAxis, typeof(1.0u"Hz")}

_midpoint(iv::ClosedInterval) = (leftendpoint(iv) + rightendpoint(iv)) / 2

function _make_partition(::Type{AX}, supports::AbstractVector{<:ClosedInterval}, lookup) where {AX<:PartitionAxis}
    refs = map(_midpoint, supports)
    cells = StructArray((; support = supports, reference = refs))
    Partition{AX, eltype(refs), typeof(cells)}(cells, collect(Int32, lookup))
end

"""
    ncells(part) -> Int

Number of cells.
"""
ncells(part::Partition) = length(part.cells)

"""
    supports(part) -> Vector{ClosedInterval}

Per-cell physical support envelopes.
"""
supports(part::Partition) = part.cells.support

"""
    references(part) -> Vector{Quantity}

Per-cell references (support midpoints), the deterministic coordinate a component's slope terms
measure against.
"""
references(part::Partition) = part.cells.reference

"""
    cellof(part, x::Quantity) -> Int

Cell whose support envelope contains `x` (interval search). Fails loud when `x` is in no cell.
"""
function cellof(part::Partition, x::Quantity)
    i = findfirst(s -> x in s, supports(part))
    isnothing(i) && error("value $x is in no cell of the partition (covered $(minimum(leftendpoint, supports(part))..maximum(rightendpoint, supports(part))))")
    i
end

"""
    cell_of_row(part::TimePartition, i) -> Int

Cell of root row `i` via the compact lookup.
"""
cell_of_row(part::TimePartition, i::Integer) = @inbounds Int(part.lookup[i])

"""
    cell_of_channel(part::FreqPartition, c) -> Int

Cell of root channel `c` via the compact lookup.
"""
cell_of_channel(part::FreqPartition, c::Integer) = @inbounds Int(part.lookup[c])

_assert_root(ds::Dataset) =
    (parentrows(ds) isa Base.OneTo && parentchannels(ds) isa Base.OneTo) ||
        error("partition() binds to the ROOT dataset only; got a subset (non-identity parent indices)")


struct WholeObservation end
struct ByScan end
struct ByDuration{Q<:Quantity}; d::Q; end
struct ByIF end
struct ByChannel end
struct TimeBoundaries; edges::Vector{DateTime}; end

"""
    partition(ds::Dataset, strategy; within=nothing) -> Partition

Resolve a partition of `ds` under a strategy. Time strategies key off row `t`/`scan`/`datetime` only
(no source/baseline awareness); `ByDuration` requires a `within` containing partition. Fails loud on
a subset dataset.
"""
function partition end

function partition(ds::Dataset, ::WholeObservation; within=nothing)
    _assert_root(ds)
    lo, hi = extrema(ds.rows.t)
    _make_partition(TimeAxis, [(lo * u"s")..(hi * u"s")], ones(Int32, nrows(ds)))
end

function partition(ds::Dataset, ::ByScan; within=nothing)
    _assert_root(ds)
    scans = ds.rows.scan
    uscans = sort!(unique(scans))
    remap = Dictionary(uscans, eachindex(uscans))
    supports = map(uscans) do s
        lo, hi = extrema(@view ds.rows.t[findall(==(s), scans)])
        (lo * u"s")..(hi * u"s")
    end
    _make_partition(TimeAxis, supports, map(s -> remap[s], scans))
end

function partition(ds::Dataset, strat::ByDuration; within=nothing)
    _assert_root(ds)
    within isa TimePartition || error("ByDuration requires `within` to be a resolved time partition")
    length(within.lookup) == nrows(ds) || error("`within` was built on a different dataset")
    d = strat.d
    contsup = supports(within)
    ns = map(iv -> max(round(Int, NoUnits((rightendpoint(iv) - leftendpoint(iv)) / d)), 1), contsup)
    base = [0; cumsum(ns)]
    supports_out = flatmap(1:ncells(within)) do c
        iv = contsup[c]; lo = leftendpoint(iv); w = (rightendpoint(iv) - lo) / ns[c]
        [(lo + (k - 1) * w)..(lo + k * w) for k in 1:ns[c]]
    end
    lookup = map(eachindex(ds.rows.t)) do r
        c = Int(within.lookup[r]); iv = contsup[c]; lo = leftendpoint(iv)
        w = (rightendpoint(iv) - lo) / ns[c]
        sub = clamp(floor(Int, NoUnits((ds.rows.t[r] * u"s" - lo) / w)) + 1, 1, ns[c])
        base[c] + sub
    end
    _make_partition(TimeAxis, supports_out, lookup)
end

function partition(ds::Dataset, strat::TimeBoundaries; within=nothing)
    _assert_root(ds)
    edges = strat.edges
    length(edges) ≥ 2 || error("TimeBoundaries needs at least two edges")
    issorted(edges) || error("TimeBoundaries edges must be strictly increasing")
    esec = map(e -> float(ustrip(u"s", e - ds.time0)), edges)
    supports = map(k -> (esec[k] * u"s")..(esec[k + 1] * u"s"), 1:length(esec) - 1)
    lookup = map(ds.rows.t) do t
        esec[1] ≤ t ≤ esec[end] || error("row at t=$(t) s is outside TimeBoundaries [$(esec[1]), $(esec[end])] s")
        clamp(searchsortedlast(esec, t), 1, length(esec) - 1)
    end
    _make_partition(TimeAxis, supports, lookup)
end

function partition(ds::Dataset, ::ByIF; within=nothing)
    _assert_root(ds)
    if_of = ds.freq.if_of
    uifs = sort!(unique(if_of))
    remap = Dictionary(uifs, eachindex(uifs))
    supports = map(uifs) do a
        lo, hi = extrema(@view ds.freq.ν[findall(==(a), if_of)])
        (lo * u"Hz")..(hi * u"Hz")
    end
    _make_partition(FreqAxis, supports, map(a -> remap[a], if_of))
end

function partition(ds::Dataset, ::ByChannel; within=nothing)
    _assert_root(ds)
    half = ds.freq.Δν / 2 * u"Hz"
    supports = map(ν -> (ν * u"Hz") ± half, ds.freq.ν)
    _make_partition(FreqAxis, supports, 1:nchannels(ds))
end


function _refine_scan(fine::Partition{AX}, coarse::Partition{AX}) where {AX}
    out = zeros(Int32, ncells(fine))
    for i in eachindex(fine.lookup)
        f = fine.lookup[i]; c = coarse.lookup[i]
        if out[f] == 0
            out[f] = c
        elseif out[f] != c
            return (; mapping = nothing, conflict = (; fine = f, coarse = (out[f], c)))
        end
    end
    (; mapping = out, conflict = nothing)
end

"""
    refines(fine, coarse) -> Bool

Whether every `fine` cell lies wholly within one `coarse` cell (structural check over the lookups).
"""
refines(fine::Partition{AX}, coarse::Partition{AX}) where {AX} =
    length(fine.lookup) == length(coarse.lookup) && _refine_scan(fine, coarse).conflict === nothing

function _containing(fine::Partition{AX}, coarse::Partition{AX}) where {AX}
    length(fine.lookup) == length(coarse.lookup) ||
        error("partitions cover different item counts ($(length(fine.lookup)) vs $(length(coarse.lookup)))")
    scan = _refine_scan(fine, coarse)
    if scan.conflict !== nothing
        cf = scan.conflict
        error("fine cell $(cf.fine) straddles coarse cells $(cf.coarse[1]) and $(cf.coarse[2]) — not a refinement")
    end
    all(!=(0), scan.mapping) || error("fine partition has an empty cell — cannot map to the coarse partition")
    scan.mapping
end

"""
    assert_refines(fine, coarse)

Fail loud (naming the straddling cell) unless `fine` refines `coarse`.
"""
assert_refines(fine::Partition, coarse::Partition) = (_containing(fine, coarse); nothing)


"""
    coarsen(fine::TimePartition, n::Integer; within) -> TimePartition

Merge consecutive `fine` cells in blocks of `n` within each `within` container (the last block may be
shorter, never crossing a container). `fine` must refine `within`; the result is validated to still
refine it.
"""
function coarsen(fine::TimePartition, n::Integer; within)
    within isa TimePartition || error("coarsen requires `within` to be a resolved time partition")
    contain = _containing(fine, within)
    newof = zeros(Int32, ncells(fine))
    finesup = supports(fine)
    supports_out = ClosedInterval{typeof(1.0u"s")}[]
    g = 0
    for c in 1:ncells(within)
        members = findall(==(c), contain)
        for block in Iterators.partition(members, n)
            g += 1
            newof[block] .= g
            push!(supports_out, minimum(leftendpoint, finesup[block])..maximum(rightendpoint, finesup[block]))
        end
    end
    result = _make_partition(TimeAxis, supports_out, map(fc -> newof[fc], fine.lookup))
    assert_refines(result, within)
    result
end

"""
    group(per_if::FreqPartition, groups::Tuple) -> FreqPartition

Regroup IF cells positionally against the declared IF order, e.g. `group(per_if, (1:4, 5:8))`. Each
group's support is the envelope of its member channels (may straddle an inter-IF gap). Every original
cell must appear in exactly one group (fail loud otherwise).
"""
function group(per_if::FreqPartition, groups::Tuple)
    allcells = collect(Iterators.flatten(groups))
    (issetequal(allcells, 1:ncells(per_if)) && length(allcells) == ncells(per_if)) ||
        error("group(): every IF cell must appear in exactly one group (got $(sort(allcells)) vs 1:$(ncells(per_if)))")
    cellof_if = zeros(Int32, ncells(per_if))
    for (g, grp) in enumerate(groups), c in grp
        cellof_if[c] = g
    end
    ifsup = supports(per_if)
    supports_out = map(groups) do grp
        ivs = ifsup[collect(grp)]
        minimum(leftendpoint, ivs)..maximum(rightendpoint, ivs)
    end
    _make_partition(FreqAxis, collect(supports_out), map(c -> cellof_if[c], per_if.lookup))
end
