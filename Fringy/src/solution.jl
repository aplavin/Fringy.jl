
const COMPONENT_KINDS = (PhaseOffset, Delay, Rate, PhaseBandpass, LogAmplitudeBandpass)

"""
    Solution(ds::Dataset; components::Tuple)

Calibration solution over the ROOT dataset `ds`: one `ComponentState` per requested component
definition (≤1 per kind), zero-initialized (identity gain), stored canonically ordered
(`PhaseOffset, Delay, Rate, PhaseBandpass, LogAmplitudeBandpass`). All definitions' partitions must
be built on `ds` itself. Access states as `sol[Delay]`; update immutably via
`@set sol.components.Delay = newstate`.
"""
struct Solution{DS<:Dataset, C<:NamedTuple}
    dataset::DS
    components::C
end

function Solution(ds::Dataset; components::Tuple)
    _assert_root(ds)
    foreach(components) do def
        def isa ComponentKind || error("Solution components must be component definitions, got $(typeof(def))")
        length(def.time.lookup) == nrows(ds) ||
            error("$(nameof(kindof(def))) time partition covers $(length(def.time.lookup)) rows, dataset has $(nrows(ds)) — built on a different dataset?")
        length(def.frequency.lookup) == nchannels(ds) ||
            error("$(nameof(kindof(def))) frequency partition covers $(length(def.frequency.lookup)) channels, dataset has $(nchannels(ds)) — built on a different dataset?")
    end
    kinds = map(kindof, components)
    allunique(kinds) || error("at most one component per kind; got $(map(nameof, kinds))")
    ordered = filter(K -> K in kinds, COMPONENT_KINDS)
    states = map(K -> zerostate(components[findfirst(k -> k === K, kinds)], nantennas(ds)), ordered)
    Solution(ds, NamedTuple{map(nameof, ordered)}(states))
end

"""
    sol[Kind] -> ComponentState

The state of a component kind, e.g. `sol[Delay]`. Type-stable; fails loud when the kind is absent.
"""
@generated function Base.getindex(sol::Solution{DS,C}, ::Type{K}) where {DS, C, K<:ComponentKind}
    name = nameof(K)
    name in fieldnames(C) || return :(error("solution has no ", $(QuoteNode(name)), " component"))
    :(getfield(getfield(sol, :components), $(QuoteNode(name))))
end

"""
    haskind(sol, Kind) -> Bool

Whether the solution holds a component of this kind.
"""
@generated haskind(::Solution{DS,C}, ::Type{K}) where {DS, C, K<:ComponentKind} =
    nameof(K) in fieldnames(C)

"""
    dataset(sol) -> Dataset

The root dataset the solution is bound to.
"""
dataset(sol::Solution) = sol.dataset

"""
    jones_terms(sol) -> Tuple

The present component kinds in canonical order — the full-chain `terms` tuple for
`calibrated_dataset(sol, ds; terms=jones_terms(sol))`.
"""
jones_terms(sol::Solution) = map(kindof, values(sol.components))

gain(sol::Solution, p::Integer, i::Integer, row::Integer, chan::Integer, t::Real, ν::Real) =
    gain(sol.components, p, i, row, chan, t, ν)
