
export CapturedAppliedPhase, AstrometryModel, total_delays, applied_phase_delay, antenna_delay_terms


"""
    CapturedAppliedPhase(sol::Solution)
    CapturedAppliedPhase(; ν, if_of, νref, delay, phase, fcell = …)

The phase `φ(antenna, receptor slot, time cell, ν)` that instrumental calibration divided out of
the visibilities, in the form [`applied_phase_delay`](@ref) needs: per-channel frequencies plus the
`Delay`/`PhaseOffset` states that generate it,

    φ = phase[p, i, tc, fc] + 2π · delay[p, i, tc, fc] · (ν − νref[fc])

exactly `Fringy.response` for those two kinds. Built either from the Solution passed **into** the
astrometric `FringeFit` (never the returned one, whose states include the captured applied phase plus the
fringe install, which would double-count the measurement) or from the small arrays a cached fringe
product carries, so the observable stage never reopens the visibility file.

`fcell` maps a root channel to its frequency cell; the keyword form defaults to the `ByIF` mapping
(cells in ascending IF order), which is the astrometric configuration's partition.
"""
struct CapturedAppliedPhase
    ν::Vector{Float64}
    if_of::Vector{Int}
    fcell::Vector{Int}
    νref::Vector{Float64}
    delay::Array{Float64,4}
    phase::Array{Float64,4}
end

_byif_cells(if_of) = (u = sort!(unique(if_of)); map(a -> findfirst(==(a), u)::Int, if_of))

function CapturedAppliedPhase(; ν, if_of, νref, delay, phase, fcell = _byif_cells(if_of))
    length(ν) == length(if_of) == length(fcell) ||
        error("CapturedAppliedPhase: ν, if_of and fcell must have one entry per channel")
    size(delay) == size(phase) ||
        error("CapturedAppliedPhase: delay and phase state arrays differ in shape, $(size(delay)) vs $(size(phase))")
    maximum(fcell) ≤ length(νref) == size(delay, 4) ||
        error("CapturedAppliedPhase: $(length(νref)) frequency-cell references for $(size(delay, 4)) state cells")
    CapturedAppliedPhase(collect(Float64, ν), collect(Int, if_of), collect(Int, fcell),
                        collect(Float64, νref), Array{Float64,4}(delay), Array{Float64,4}(phase))
end

function CapturedAppliedPhase(sol::Solution)
    (haskind(sol, Delay) && haskind(sol, PhaseOffset)) ||
        error("CapturedAppliedPhase: the captured-phase solution needs both Delay and PhaseOffset states")
    d = sol[Delay]; p = sol[PhaseOffset]
    (d.definition.time.lookup == p.definition.time.lookup &&
     d.definition.frequency.lookup == p.definition.frequency.lookup) ||
        error("CapturedAppliedPhase: the Delay and PhaseOffset components must share their partitions")
    ds = dataset(sol)
    CapturedAppliedPhase(; ν = ds.freq.ν, if_of = ds.freq.if_of,
                        fcell = Int.(d.definition.frequency.lookup),
                        νref = ustrip.(u"Hz", references(d.definition.frequency)),
                        delay = d.values, phase = p.values)
end

Base.show(io::IO, ai::CapturedAppliedPhase) = print(io, "CapturedAppliedPhase(",
    length(ai.ν), " channels, ", size(ai.delay, 1), " antennas × ", size(ai.delay, 3),
    " time cells × ", size(ai.delay, 4), " frequency cells)")

"""
    applied_phase(ai::CapturedAppliedPhase, p, i, tc, c) -> Float64

The calibration phase [rad] applied to antenna `p`, structural receptor slot `i`, in time cell `tc`,
at root channel `c`.
"""
@inline function applied_phase(ai::CapturedAppliedPhase, p::Integer, i::Integer, tc::Integer, c::Integer)
    fc = @inbounds ai.fcell[c]
    @inbounds ai.phase[p, i, tc, fc] + 2π * ai.delay[p, i, tc, fc] * (ai.ν[c] - ai.νref[fc])
end

"""
    applied_phase_delay(ai, p, q, receptor_slots, tcell, ifs, weights) -> Float64

The instrumental delay [s] to add back to a fringe measurement made on calibrated data of baseline
`(p, q)` (antenna indices, in the data's own `ant1, ant2` order), explicit diagonal structural
receptor-slot pair `(1, 1)` or `(2, 2)`, in time cell `tcell`. No receptor label is inspected.

It is the delay the calibration *applied at the tile's own frequency support*: not an average of the
per-IF captured delays, but the **weighted least-squares slope of the applied phase across the tile's
channels**, using the per-IF accumulated weights `weights` of IFs `ifs` that the fringe sum itself
used, captured in the `Fringes` product. To first order in the applied phase that
slope is exactly the shift the maximum-likelihood delay estimate suffers, whatever the applied gain
chain contains — per-IF delays, per-IF phase steps, or any other installed term.

Sign: the corruption `g` is divided out as `V′ = V/(g_p·conj(g_q))`, so calibration removes
`φ_p − φ_q` from the baseline phase and hence `D_p − D_q` from the measured delay; the add-back is
therefore `D_q − D_p`, the same `ant2 − ant1` ordering as the correlator polynomial. The
round-trip test in `test/test_totals.jl` pins this sign.
"""
function _astrometric_receptor_slot(receptor_slots::NTuple{2,Int}, context)
    receptor_slots in ((1, 1), (2, 2)) ||
        error("$context: astrometry accepts only diagonal pairs of receptor slots (1, 1) and (2, 2), got $receptor_slots")
    first(receptor_slots)
end

function applied_phase_delay(ai::CapturedAppliedPhase, p::Integer, q::Integer,
                      receptor_slots::NTuple{2,Int}, tcell::Integer,
                      ifs::AbstractVector{<:Integer}, weights::AbstractVector{<:Real})
    receptor_slot = _astrometric_receptor_slot(receptor_slots, "applied_phase_delay")
    length(ifs) == length(weights) ||
        error("applied_phase_delay: $(length(ifs)) IFs but $(length(weights)) captured weights")
    checkbounds(Bool, ai.delay, p, receptor_slot, tcell, 1) &&
        checkbounds(Bool, ai.delay, q, receptor_slot, tcell, 1) ||
        error("applied_phase_delay: (antenna $p / $q, receptor slots $receptor_slots, time cell $tcell) is outside the captured applied-phase states $(size(ai.delay)) — the fringe tiles and the captured-phase partitions disagree")
    a = _SlopeAccum()
    @inbounds for c in eachindex(ai.ν)
        k = findfirst(==(ai.if_of[c]), ifs)
        isnothing(k) && continue
        w = Float64(weights[k])
        w > 0 || continue
        φ = applied_phase(ai, q, receptor_slot, tcell, c) -
            applied_phase(ai, p, receptor_slot, tcell, c)
        _accumulate!(a, ai.ν[c], φ, w)
    end
    s = _slope(a)
    isnan(s) ? 0.0 : s / 2π
end


"""
    AstrometryModel(; antenna_geometry, sources, eop, ionex, pressure, terms, time0, doy, ν_eff,
                 iono_primary, iono_inflation)

Inputs for the combined a priori model evaluated at one canonical epoch: antenna positions and geophysical
metadata, source coordinates, IERS Earth-orientation data, IONEX products, pressure series, selected
[`GeometryTerms`](@ref), canonical `time0`, day of year, and ionosphere configuration.

Every field is required — this is analysis input data, and the package supplies no defaults for it.
"""
struct AstrometryModel{P<:NamedTuple, S, PR}
    antenna_geometry::Dictionary{Symbol, AntennaGeometry}
    sources::S
    eop::EOPSeries
    ionex::P
    pressure::PR
    terms::GeometryTerms
    time0::DateTime
    doy::Int
    ν_eff::Float64
    iono_primary::Symbol
    iono_inflation::Float64
end

AstrometryModel(; antenna_geometry, sources, eop, ionex, pressure, terms, time0, doy, ν_eff,
             iono_primary, iono_inflation) =
    AstrometryModel{typeof(ionex), typeof(sources), typeof(pressure)}(
        antenna_geometry, sources, eop, ionex, pressure, terms, DateTime(time0), Int(doy),
        Float64(ν_eff), Symbol(iono_primary), Float64(iono_inflation))

Base.show(io::IO, astrometry_model::AstrometryModel) = print(io, "AstrometryModel(", length(astrometry_model.antenna_geometry), " antennas, ",
    length(astrometry_model.sources), " sources, τ_ionosphere ", keys(astrometry_model.ionex), " primary ", astrometry_model.iono_primary, ")")

"""
    antenna_delay_terms(astrometry_model::AstrometryModel, antenna::Symbol, ra, dec, t) -> NamedTuple

The complete a priori model for one antenna towards `(ra, dec)` [rad] at canonical epoch `t` [s
after `astrometry_model.time0`].

`(; τ_geometric, τ_geometric_hydrostatic, el, az, τ_zenith_hydrostatic, hydrostatic_mapping, wet_mapping, gradient_mapping, ∂τ_∂α★, ∂τ_∂δ, τ_ionosphere, mjd)` contains the geocentre→antenna
geometric delay [s], the same plus hydrostatic troposphere, elevation and azimuth [rad], zenith
hydrostatic delay [s], dimensionless mapping factors, source partials [s/rad], per-product ionospheric delays [s],
and UTC MJD.
"""
function antenna_delay_terms(astrometry_model::AstrometryModel, antenna::Symbol, ra::Real, dec::Real, t::Real)
    st = astrometry_model.antenna_geometry[antenna]
    ep = UTCEpoch(astrometry_model.time0, t)
    mjd = utc_mjd(ep)
    g = geometric_delay(st, Float64(ra), Float64(dec), ep, astrometry_model.eop, astrometry_model.terms)
    press = surface_pressure(astrometry_model.pressure[antenna], t, st.height)
    τ_zenith_hydrostatic = zenith_hydrostatic_path(press, st.lat, st.height) / C_LIGHT_M_S
    hydrostatic_map = hydrostatic_mapping(g.el, st.lat, st.height, astrometry_model.doy)
    wet_map = wet_mapping(g.el, st.lat)
    gradient_map = gradient_mapping(g.el)
    τ_ionosphere = iono_delays(astrometry_model.ionex, st, mjd, g.az, g.el; ν_eff = astrometry_model.ν_eff)
    (; g.τ_geometric, τ_geometric_hydrostatic = g.τ_geometric + τ_zenith_hydrostatic * hydrostatic_map,
       g.el, g.az, τ_zenith_hydrostatic, hydrostatic_mapping = hydrostatic_map,
       wet_mapping = wet_map, gradient_mapping = gradient_map,
       ∂τ_∂α★ = g.∂τ_∂α★, ∂τ_∂δ = g.∂τ_∂δ, τ_ionosphere, mjd)
end


function _correlator_or_nan(f, correlator_model::CorrelatorDelayModel, ant::Symbol, source::Symbol, t::Float64;
                         receptor_slot::Int)
    _check_receptor_slot(correlator_model, receptor_slot)
    model_covers(correlator_model, ant, source, t) ? f(correlator_model, ant, source, t; receptor_slot) : NaN
end

_structure_comb(ai::CapturedAppliedPhase, ifs) =
    (p = sortperm(ai.ν); StructureComb(ai.ν[p], ai.if_of[p], ifs))

"""
    total_delays(obs, captured_phase_solution::Solution, correlator_model, astrometry_model; kwargs...) -> StructArray
    total_delays(obs, ai::CapturedAppliedPhase, correlator_model::CorrelatorDelayModel, astrometry_model::AstrometryModel;
                 captured_phase_mode, source_offsets = (), epoch_shift = 0.0) -> StructArray

Form the astrometric observables: one row per selected `(scan, baseline, diagonal pair of receptor slots)` of the
captured fringe table `obs`, carrying the total delay, a priori model, ionospheric correction
and everything the estimator needs.

`obs` is the tidy table per selected `(scan, baseline, diagonal pair of receptor slots)` from the fringe pass: columns `source`,
`source_ix`, `baseline` (a `Baseline{Symbol}`), `baseline_ix` (antenna indices), `tcell` (the fringe
tile's time cell, which is also the captured-phase states' time cell), `t` (the canonical Float64 epoch),
`datetime`, `receptor_slots`, `delay`, `rate`, `snr`, `sigma_delay`, and `ifweights`.
`receptor_slots` is the ordered pair of structural receptor slots at the two baseline endpoints.
`ifweights` stores the per-IF Σw′ in ascending IF identity.

`captured_phase_solution` is the `Solution` passed **into** the astrometric `FringeFit`, never the returned one.

Required keyword `captured_phase_mode` (`:add_back` | `:retain`) decides whether this table includes the
captured applied-phase slope. `:retain` leaves it out; `:add_back` adds it. The latter does not undo
IF alignment or other uncaptured calibration terms, and it does not rerun or refit the fringe pass.
The `τ_applied_phase` column is reported in both, so this table-column transform is exact and auditable.

Keyword `source_offsets` shifts the model source positions by local `(Δα★, Δδ)` [rad] per source name,
where `Δα★ = cos(δ) Δα`.
(anything supporting `haskey`/`getindex`) — the Gauss–Newton relinearization hook. `epoch_shift`
[s] moves the canonical epoch of every row consistently through every evaluation.

Keyword `structure` (a [`StructureModels`](@ref)) turns the point-source
observable into the REGISTRATION one: each row gains the structure delay of its source's image model,
computed on this row's own channel comb and captured per-IF weights ([`structure_delay`](@ref)), and
`τ_corrected` has it subtracted alongside the ionosphere. The fitted position is then the absolute position
of the model's ORIGIN rather than of the structure-delay-weighted centroid. Without it every
`τ_structure` is 0.0 and the table is bit-identical to the point-source one. Rows whose wrap guard trips
carry `τ_structure = NaN`, hence a non-finite `τ_residual`, and the estimator drops them; they are
counted per source in the warning.

Columns of the result:

| column | meaning |
|---|---|
| `scan, source, source_ix, baseline, a1, a2, receptor_slots` | identity; ordered structural receptor-slot pair at the two baseline endpoints |
| `ra, dec` | the source position at which the a priori model was evaluated [rad] |
| `t, datetime, mjd` | the canonical epoch (Float64 s after `time0`), for display, and as UTC MJD |
| `τ_correlator` | correlator GDELAY polynomial, ant2 − ant1 [s] |
| `τ_applied_phase` | the instrumental delay add-back [s] ([`applied_phase_delay`](@ref)) — reported whether or not `captured_phase_mode` includes it in `τ_total` |
| `delay, rate, snr, σ_τ` | the fringe measurement and its thermal precision |
| `τ_total` | `τ_correlator + τ_applied_phase − delay` (`:add_back`) or `τ_correlator − delay` (`:retain`) |
| `τ_ionosphere, σ_τ_ionosphere` | baseline ionospheric correction and its 3-product uncertainty [s] |
| `τ_structure` | the source-structure delay relative to the image model's origin [s]; 0.0 without `structure`, NaN where the wrap guard tripped |
| `τ_corrected` | `τ_total − τ_ionosphere − τ_structure` |
| `τ_geometric_hydrostatic` | independent non-dispersive model, ant2 − ant1 [s] |
| `τ_correlator_clock` | the correlator's linear CLOCK, ant2 − ant1 [s] |
| `τ_residual` | `τ_corrected − τ_geometric_hydrostatic + τ_correlator_clock` — what the estimator fits |
| `el1, el2, az1, az2` | elevations and azimuths [rad] |
| `wet_mapping1, wet_mapping2, gradient_mapping1, gradient_mapping2` | dimensionless wet and gradient mapping factors |
| `∂τ_∂α★, ∂τ_∂δ` | baseline source partials [s/rad] w.r.t. local `(Δα★ = cos(δ) Δα, Δδ)` |
"""
function total_delays(obs, ai::CapturedAppliedPhase, correlator_model::CorrelatorDelayModel, astrometry_model::AstrometryModel;
                      captured_phase_mode::Symbol, source_offsets = (), epoch_shift::Real = 0.0,
                      structure::Union{Nothing,StructureModels} = nothing)
    captured_phase_mode in (:retain, :add_back) ||
        error("total_delays: `captured_phase_mode` must be :add_back (algebraically add the captured applied-phase slope) or :retain (leave it out), got $captured_phase_mode")
    add_back_scale = captured_phase_mode === :add_back ? 1.0 : 0.0
    correlator_model.time0 == astrometry_model.time0 ||
        error("total_delays: the correlator model and the a priori model use different epoch origins ($(correlator_model.time0) vs $(astrometry_model.time0)); every row requires one canonical epoch")
    ifs = sort!(unique(ai.if_of))
    Δt = Float64(epoch_shift)

    comb = isnothing(structure) ? nothing : _structure_comb(ai, ifs)
    nostruct = Set{Symbol}()
    modelof = Dict{Symbol, Any}()
    function structure_model(src::Symbol)
        get!(modelof, src) do
            haskey(structure.models, src) ? structure.models[src] :
                (push!(nostruct, src); MultiComponentModel(Point{Float64,Float64}[]))
        end
    end
    nwrapped = 0

    cache = Dict{Tuple{Float64,Symbol,Symbol}, Any}()
    function radec(src::Symbol)
        c = astrometry_model.sources[src]
        ra, dec = Float64(c.ra), Float64(c.dec)
        if !isempty(source_offsets) && haskey(source_offsets, src)
            dα, dδ = source_offsets[src]
            dec += dδ
            ra += dα / cos(dec)
        end
        (ra, dec)
    end
    function stmodel(ant::Symbol, src::Symbol, t::Float64)
        get!(cache, (t, src, ant)) do
            ra, dec = radec(src)
            antenna_delay_terms(astrometry_model, ant, ra, dec, t)
        end
    end

    nmissing = 0
    rows = map(eachindex(obs)) do n
        src = obs.source[n]
        t = Float64(obs.t[n]) + Δt
        tc = Int(obs.tcell[n])
        p, q = obs.baseline_ix[n].antennas
        n1, n2 = obs.baseline[n].antennas
        receptor_slots = obs.receptor_slots[n]
        _astrometric_receptor_slot(receptor_slots, "total_delays: row $n")

        τ1 = _correlator_or_nan(τ_correlator, correlator_model, n1, src, t; receptor_slot = receptor_slots[1])
        τ2 = _correlator_or_nan(τ_correlator, correlator_model, n2, src, t; receptor_slot = receptor_slots[2])
        c1 = _correlator_or_nan(τ_correlator_clock, correlator_model, n1, src, t; receptor_slot = receptor_slots[1])
        c2 = _correlator_or_nan(τ_correlator_clock, correlator_model, n2, src, t; receptor_slot = receptor_slots[2])
        (isnan(τ1) || isnan(τ2)) && (nmissing += 1)

        τp = applied_phase_delay(ai, p, q, receptor_slots, tc, ifs, obs.ifweights[n])

        m1 = stmodel(n1, src, t)
        m2 = stmodel(n2, src, t)
        ic = iono_baseline_correction(m1.τ_ionosphere, m2.τ_ionosphere;
                                      primary = astrometry_model.iono_primary, inflation = astrometry_model.iono_inflation)

        ∂τ_∂α★ = m2.∂τ_∂α★ - m1.∂τ_∂α★
        ∂τ_∂δ = m2.∂τ_∂δ - m1.∂τ_∂δ
        τ_st = if isnothing(structure)
            0.0
        else
            v = structure_delay(structure_model(src), ∂τ_∂α★, ∂τ_∂δ, comb, obs.ifweights[n];
                                wrap_threshold = structure.wrap_threshold)
            isnan(v) && (nwrapped += 1)
            v
        end

        τ_ap = τ2 - τ1
        τ_tot = τ_ap + add_back_scale * τp - obs.delay[n]
        τ_mod = m2.τ_geometric_hydrostatic - m1.τ_geometric_hydrostatic
        clk = c2 - c1
        τ_corrected = τ_tot - ic.correction - τ_st
        ra, dec = radec(src)
        (; scan = tc, source = src, source_ix = obs.source_ix[n], baseline = obs.baseline[n],
           a1 = n1, a2 = n2, receptor_slots, ra, dec,
           t, datetime = obs.datetime[n], mjd = m1.mjd,
           τ_correlator = τ_ap, τ_applied_phase = τp, delay = obs.delay[n], rate = obs.rate[n],
           snr = obs.snr[n], σ_τ = obs.sigma_delay[n],
           τ_total = τ_tot, τ_ionosphere = ic.correction, σ_τ_ionosphere = ic.σ, τ_structure = τ_st, τ_corrected,
           τ_geometric_hydrostatic = τ_mod, τ_correlator_clock = clk, τ_residual = τ_corrected - τ_mod + clk,
           el1 = m1.el, el2 = m2.el, az1 = m1.az, az2 = m2.az,
           wet_mapping1 = m1.wet_mapping, wet_mapping2 = m2.wet_mapping,
           gradient_mapping1 = m1.gradient_mapping, gradient_mapping2 = m2.gradient_mapping,
           ∂τ_∂α★, ∂τ_∂δ)
    end
    nmissing > 0 && @warn "total_delays: $nmissing of $(length(obs)) rows have no correlator model interval covering their epoch (an antenna joined the scan late) — their τ_total is NaN and the estimator's acceptance drops them"
    nwrapped > 0 && @warn "total_delays: the τ_structure wrap guard tripped on $nwrapped of $(length(obs)) rows (in-band nulls of resolved structure) — those rows carry NaN and the estimator's acceptance drops them"
    (isempty(nostruct) || isempty(structure.models)) ||
        @warn "total_delays: no structure model for $(length(nostruct)) source(s) — they keep τ_structure = 0.0 and are registered on their point-source position: $(join(sort!(collect(nostruct)), ", "))"
    StructArray(identity.(rows))
end

total_delays(obs, captured_phase_solution::Solution, correlator_model::CorrelatorDelayModel, astrometry_model::AstrometryModel; kwargs...) =
    total_delays(obs, CapturedAppliedPhase(captured_phase_solution), correlator_model, astrometry_model; kwargs...)
