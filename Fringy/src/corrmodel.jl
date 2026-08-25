
struct ModelIntervals{N,G<:SVector}
    t0::Vector{Float64}
    duration::Vector{Float64}
    gdelay::NTuple{N,Vector{G}}
    clock::NTuple{N,Vector{Float64}}
    dclock::NTuple{N,Vector{Float64}}
    atmos::Vector{Float64}
    datmos::Vector{Float64}
end

"""
    CorrelatorDelayModel(uv::UVData; time0::DateTime)
    CorrelatorDelayModel(uv::UVData, ds::Dataset)

The correlator model of one file, keyed by `(antenna name, source name)`, with every epoch
expressed in canonical Float64 seconds after `time0` — pass the `Dataset`'s own `time0` (the second
form does) so that polynomial and model evaluations share one epoch scale.

Query it with [`τ_correlator`](@ref) (the GDELAY polynomial), [`τ_correlator_clock`](@ref) (the
correlator's linear CLOCK, about which the estimator solves a residual), and
[`τ_correlator_troposphere`](@ref) (the correlator troposphere, used for diagnostics only).

`eop` keeps the CALC-table EOP rows the correlator actually applied, for the standing
a-priori-minus-correlator diagnostic; [`AstrometryModel`](@ref) uses IERS finals instead.
"""
struct CorrelatorDelayModel{N,G<:SVector,E}
    time0::DateTime
    intervals::Dictionary{Tuple{Symbol,Symbol},ModelIntervals{N,G}}
    eop::E
end

function CorrelatorDelayModel(uv; time0::DateTime)
    imod = VLBIFiles.interferometer_model(uv)
    mc = VLBIFiles.model_comps(uv)
    calc = VLBIFiles.calc_table(uv)
    antennas = only(uv.ant_arrays).antennas
    srctable = VLBIFiles.sources(uv)
    _build_corrmodel(imod, mc, calc, antennas, srctable, time0)
end

CorrelatorDelayModel(uv, ds::Dataset) = CorrelatorDelayModel(uv; time0 = ds.time0)

function _build_corrmodel(imod, mc, calc, antennas, srctable, time0::DateTime)
    length(imod) == length(mc) ||
        error("INTERFEROMETER_MODEL has $(length(imod)) rows, MODEL_COMPS $(length(mc)) — not the row-by-row pair this model assumes")
    (imod.time == mc.time && imod.antenna_no == mc.antenna_no && imod.source_rawid == mc.source_rawid) ||
        error("INTERFEROMETER_MODEL and MODEL_COMPS rows are not aligned in (time, antenna, source)")

    n_imod = hasproperty(imod, :gdelay_2) ? 2 : 1
    n_clock = hasproperty(mc, :clock_2) ? 2 : 1
    n_dclock = hasproperty(mc, :dclock_2) ? 2 : 1
    n_clock == n_dclock ||
        error("MODEL_COMPS CLOCK has $(_receptor_slot_count(n_clock)), DCLOCK has $(_receptor_slot_count(n_dclock))")
    n_imod == n_clock ||
        error("INTERFEROMETER_MODEL has $(_receptor_slot_count(n_imod)); MODEL_COMPS has $(_receptor_slot_count(n_clock))")
    _build_corrmodel(imod, mc, calc, antennas, srctable, time0, Val(n_imod))
end

function _build_corrmodel(imod, mc, calc, antennas, srctable, time0::DateTime, ::Val{N}) where {N}
    key = map(eachindex(imod)) do i
        (antennas[imod.antenna_no[i]].name, srctable[imod.source_rawid[i]].name)
    end
    G = eltype(imod.gdelay_1)
    groups = groupfind(identity, key)
    intervals = map(collect(keys(groups))) do k
        ix = groups[k]
        t = Float64[Dates.value(imod.time[i] - time0) / 1000 for i in ix]
        o = sortperm(t)
        ix = ix[o]
        ModelIntervals{N,G}(
            t[o],
            Float64[ustrip(u"s", imod.interval[i]) for i in ix],
            ntuple(receptor_slot -> G[getproperty(imod, Symbol(:gdelay_, receptor_slot))[i] for i in ix], N),
            ntuple(receptor_slot -> Float64[ustrip(u"s", getproperty(mc, Symbol(:clock_, receptor_slot))[i]) for i in ix], N),
            ntuple(receptor_slot -> Float64[getproperty(mc, Symbol(:dclock_, receptor_slot))[i] for i in ix], N),
            Float64[ustrip(u"s", mc.atmos[i]) for i in ix], Float64[mc.datmos[i] for i in ix],
        )
    end
    CorrelatorDelayModel{N,G,typeof(calc)}(time0, Dictionary(collect(keys(groups)), intervals), calc)
end

Base.show(io::IO, correlator_model::CorrelatorDelayModel) = print(io, "CorrelatorDelayModel(", length(correlator_model.intervals),
    " (antenna, source) series, ", _receptor_slot_count(_n_receptor_slots(correlator_model)),
    ", time0 = ", correlator_model.time0, ")")

_ant_name(a::Symbol) = a
_ant_name(a::Antenna) = a.name

function _interval_index(correlator_model::CorrelatorDelayModel, ant::Symbol, source::Symbol, t::Float64)
    haskey(correlator_model.intervals, (ant, source)) ||
        error("correlator model has no rows for antenna $ant on source $source")
    iv = correlator_model.intervals[(ant, source)]
    j = searchsortedlast(iv.t0, t)
    (j ≥ 1 && t - iv.t0[j] < iv.duration[j]) ||
        error("epoch $t s is in no model interval of antenna $ant on source $source " *
              "(covered: $(iv.t0[1]) .. $(iv.t0[end] + iv.duration[end]) s)")
    (iv, j)
end

"""
    model_covers(correlator_model, ant, source, t) -> Bool

Whether the correlator model tabulates an interval containing the canonical epoch `t` for this
(antenna, source). The negative answer is a genuine data condition — an antenna slewing onto the
source part-way through a scan can have its first model interval *after* the scan's reference epoch — so
consumers that must tolerate it (`total_delays`) ask first rather than catching the loud error that
[`τ_correlator`](@ref) raises, which would also swallow real configuration bugs.
"""
function model_covers(correlator_model::CorrelatorDelayModel, ant, source::Symbol, t::Real)
    haskey(correlator_model.intervals, (_ant_name(ant), source)) || return false
    iv = correlator_model.intervals[(_ant_name(ant), source)]
    j = searchsortedlast(iv.t0, Float64(t))
    j ≥ 1 && Float64(t) - iv.t0[j] < iv.duration[j]
end

"""
    τ_correlator(correlator_model, ant, source, t; receptor_slot::Int) -> Float64

The correlator delay [s] of one antenna on one source at the canonical epoch `t`
(Float64 seconds after `correlator_model.time0`): the GDELAY polynomial of the covering interval evaluated by
Horner at `dt = t − interval start`.

`ant` is an antenna name (`Symbol`) or `Antenna`, `source` a source name (`Symbol`). Fails loud
when `t` falls in no interval of that (antenna, source) — an epoch outside the model grid is a bug,
never a zero.

`receptor_slot` selects the structural FITS-IDI receptor slot `GDELAY_1` or `GDELAY_2`.
It is always required; receptor labels are not inspected.
"""
function τ_correlator(correlator_model::CorrelatorDelayModel, ant, source::Symbol, t::Real; receptor_slot::Int)
    _check_receptor_slot(correlator_model, receptor_slot)
    iv, j = _interval_index(correlator_model, _ant_name(ant), source, Float64(t))
    dt = Float64(t) - iv.t0[j]
    evalpoly(dt, Tuple(iv.gdelay[receptor_slot][j]))
end

"""
    τ_correlator(correlator_model, bl::Baseline, source, t; receptor_slots::NTuple{2,Int}) -> Float64

The baseline correlator delay [s]: `τ(second antenna) − τ(first antenna)`, i.e. the file's
`ant2 − ant1` convention when the baseline is ordered as the data carry it. Both antennas must have
a model interval covering `t` on that source. The ordered pair selects the structural receptor slot
independently at the first and second endpoint.
"""
function τ_correlator(correlator_model::CorrelatorDelayModel, bl::Baseline, source::Symbol, t::Real;
                   receptor_slots::NTuple{2,Int})
    _check_receptor_slots(correlator_model, receptor_slots)
    τ_correlator(correlator_model, bl.antennas[2], source, t; receptor_slot = receptor_slots[2]) -
        τ_correlator(correlator_model, bl.antennas[1], source, t; receptor_slot = receptor_slots[1])
end

"""
    τ_correlator_clock(correlator_model, ant, source, t; receptor_slot::Int) -> Float64

The correlator's CLOCK model [s] of one antenna at epoch `t`: `CLOCK + DCLOCK·dt`, the linear
antenna clock from MODEL_COMPS. This is bookkeeping the correlator applied, not CALC physics, and
the estimator adopts it as the correlator clock, solving only a piecewise-linear residual on top
so the fitted clock remains a residual about that polynomial.

Sign convention: the clock SUBTRACTS from the correlator's total delay
(`GDELAY = geometry + ATMOS − CLOCK`), so this value is what was subtracted.
"""
function τ_correlator_clock(correlator_model::CorrelatorDelayModel, ant, source::Symbol, t::Real; receptor_slot::Int)
    _check_receptor_slot(correlator_model, receptor_slot)
    iv, j = _interval_index(correlator_model, _ant_name(ant), source, Float64(t))
    dt = Float64(t) - iv.t0[j]
    iv.clock[receptor_slot][j] + iv.dclock[receptor_slot][j] * dt
end

"""
    τ_correlator_clock(correlator_model, bl::Baseline, source, t; receptor_slots::NTuple{2,Int}) -> Float64

The baseline correlator clock [s], second endpoint minus first, with the ordered receptor-slot pair
selecting each endpoint independently.
"""
function τ_correlator_clock(correlator_model::CorrelatorDelayModel, bl::Baseline, source::Symbol, t::Real;
                       receptor_slots::NTuple{2,Int})
    _check_receptor_slots(correlator_model, receptor_slots)
    τ_correlator_clock(correlator_model, bl.antennas[2], source, t; receptor_slot = receptor_slots[2]) -
        τ_correlator_clock(correlator_model, bl.antennas[1], source, t; receptor_slot = receptor_slots[1])
end

"""
    τ_correlator_troposphere(correlator_model, ant, source, t) -> Float64

The correlator's tropospheric model [s] at epoch `t` (`ATMOS + DATMOS·dt`). Diagnostic only: the
a priori model supplies the troposphere; this is here for the standing
a-priori-minus-correlator cross-check.
"""
function τ_correlator_troposphere(correlator_model::CorrelatorDelayModel, ant, source::Symbol, t::Real)
    iv, j = _interval_index(correlator_model, _ant_name(ant), source, Float64(t))
    iv.atmos[j] + iv.datmos[j] * (Float64(t) - iv.t0[j])
end

_n_receptor_slots(::CorrelatorDelayModel{N}) where {N} = N
_receptor_slot_count(n) = "$n receptor slot" * (n == 1 ? "" : "s")

function _check_receptor_slot(correlator_model::CorrelatorDelayModel, receptor_slot::Int)
    receptor_slot in 1:2 ||
        error("receptor_slot must select structural receptor slot 1 or 2, got $receptor_slot")
    receptor_slot ≤ _n_receptor_slots(correlator_model) ||
        error("correlator model has $(_receptor_slot_count(_n_receptor_slots(correlator_model))); " *
              "receptor slot $receptor_slot is unavailable")
    receptor_slot
end

_check_receptor_slots(correlator_model::CorrelatorDelayModel, receptor_slots::NTuple{2,Int}) =
    map(receptor_slot -> _check_receptor_slot(correlator_model, receptor_slot), receptor_slots)

"""
    model_epochs(correlator_model, ant, source) -> Vector{Float64}

The interval start epochs [s after `correlator_model.time0`] tabulated for one (antenna, source) — the model grid
itself, e.g. to assert that an observable's epoch is covered before evaluating.
"""
model_epochs(correlator_model::CorrelatorDelayModel, ant, source::Symbol) =
    correlator_model.intervals[(_ant_name(ant), source)].t0
