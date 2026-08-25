
abstract type ComponentKind end

"""
    PhaseOffset(; time, frequency)

Per-cell phase offset [rad]: `φ += values[p,i,tc,fc]`.
"""
@kwdef struct PhaseOffset{T<:TimePartition, F<:FreqPartition} <: ComponentKind
    time::T
    frequency::F
end

"""
    Delay(; time, frequency)

Per-cell delay [s]: `φ += 2π·values[p,i,tc,fc]·(ν − ν_ref)` with `ν_ref` the frequency cell's
reference (stored corruption convention: delay = +(1/2π)∂φ/∂ν = −group delay).
"""
@kwdef struct Delay{T<:TimePartition, F<:FreqPartition} <: ComponentKind
    time::T
    frequency::F
end

"""
    Rate(; time, frequency)

Per-cell fringe rate [Hz = cycles/s]: `φ += 2π·values[p,i,tc,fc]·(t − t_ref)` with `t_ref` the time
cell's reference.
"""
@kwdef struct Rate{T<:TimePartition, F<:FreqPartition} <: ComponentKind
    time::T
    frequency::F
end

"""
    PhaseBandpass(; time, frequency)

Per-cell bandpass phase [rad]: `φ += values[p,i,tc,fc]`.
"""
@kwdef struct PhaseBandpass{T<:TimePartition, F<:FreqPartition} <: ComponentKind
    time::T
    frequency::F
end

"""
    LogAmplitudeBandpass(; time, frequency)

Per-cell bandpass log-amplitude [dimensionless]: `a += values[p,i,tc,fc]`.
"""
@kwdef struct LogAmplitudeBandpass{T<:TimePartition, F<:FreqPartition} <: ComponentKind
    time::T
    frequency::F
end

Unitful.unit(::Type{<:PhaseOffset})          = u"rad"
Unitful.unit(::Type{<:Delay})                = u"s"
Unitful.unit(::Type{<:Rate})                 = u"Hz"
Unitful.unit(::Type{<:PhaseBandpass})        = u"rad"
Unitful.unit(::Type{<:LogAmplitudeBandpass}) = NoUnits

isphase(::Type{<:PhaseOffset})          = true
isphase(::Type{<:Delay})                = true
isphase(::Type{<:Rate})                 = true
isphase(::Type{<:PhaseBandpass})        = true
isphase(::Type{<:LogAmplitudeBandpass}) = false

"""
    ComponentState{D}

A component's solved state: the `definition` (kind struct with its partitions) + dense
`values::Array{Float64,4}` `[antenna, receptor slot, tcell, fcell]` in the kind's physical unit
(`unit(kindof(state))`). Zeros = identity gain.
"""
struct ComponentState{D<:ComponentKind}
    definition::D
    values::Array{Float64,4}
end

"""
    kindof(x) -> Type

The component kind (e.g. `Delay`) of a definition or state.
"""
kindof(::PhaseOffset)          = PhaseOffset
kindof(::Delay)                = Delay
kindof(::Rate)                 = Rate
kindof(::PhaseBandpass)        = PhaseBandpass
kindof(::LogAmplitudeBandpass) = LogAmplitudeBandpass
kindof(s::ComponentState)      = kindof(s.definition)

zerostate(def::ComponentKind, nant::Integer) =
    ComponentState(def, zeros(nant, 2, ncells(def.time), ncells(def.frequency)))

"""
    response(state, p, i, tcell, fcell, t, ν) -> Float64

One component's contribution at antenna `p`, receptor slot `i`, its own cells `(tcell, fcell)`, and
physical coordinates `t` (Float64 seconds from `ds.time0`) / `ν` (Float64 Hz). Added into φ for
phase-family kinds, into the log-amplitude `a` for `LogAmplitudeBandpass` (see `isphase`).
"""
response(s::ComponentState{<:Union{PhaseOffset,PhaseBandpass,LogAmplitudeBandpass}}, p, i, tc, fc, t, ν) =
    @inbounds s.values[p, i, tc, fc]
response(s::ComponentState{<:Delay}, p, i, tc, fc, t, ν) =
    @inbounds 2π * s.values[p, i, tc, fc] * (ν - ustrip(u"Hz", references(s.definition.frequency)[fc]))
response(s::ComponentState{<:Rate}, p, i, tc, fc, t, ν) =
    @inbounds 2π * s.values[p, i, tc, fc] * (t - ustrip(u"s", references(s.definition.time)[tc]))

"""
    gain(states, p, i, rowindex, channel, t, ν) -> ComplexF64

Forward corruption gain `exp(a + iφ)` for antenna `p`, receptor slot `i`, accumulated over all present
component states (a Tuple/NamedTuple of `ComponentState`s, or a `Solution`). `rowindex`/`channel` are
ROOT dataset indices routed through each component's own partition lookups; `t`/`ν` are the datum's
physical coordinates (Float64 seconds from `time0` / Hz). Type-stable and allocation-free.
"""
function gain(states::Tuple, p::Integer, i::Integer, row::Integer, chan::Integer, t::Real, ν::Real)
    a, φ = _accum(states, p, i, row, chan, t, ν)
    exp(complex(a, φ))
end
gain(states::NamedTuple, args...) = gain(values(states), args...)

_accum(::Tuple{}, p, i, row, chan, t, ν) = (0.0, 0.0)
function _accum(states::Tuple, p, i, row, chan, t, ν)
    a, φ = _accum(Base.tail(states), p, i, row, chan, t, ν)
    s = first(states)
    tc = cell_of_row(s.definition.time, row)
    fc = cell_of_channel(s.definition.frequency, chan)
    r = response(s, p, i, tc, fc, t, ν)
    isphase(kindof(s)) ? (a, φ + r) : (a + r, φ)
end
