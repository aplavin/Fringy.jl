
export SessionFacts, session_facts, load_session

"""
    SessionFacts

Five facts derived from the data rather than stored as configuration,
plus the channel comb they are read from.

| field | meaning |
|---|---|
| `name` | the project code, FITS-IDI `OBSCODE`. Every product header states it |
| `date` | the observing date, the UT day the first visibility falls in |
| `start`, `stop` | first and last visibility epoch — the observation, as recorded |
| `epoch_obs` | decimal year of the MID-OBSERVATION epoch; ITRF positions are propagated to it |
| `doy` | day of year of `start`, the seasonal argument of the Niell hydrostatic mapping |
| `ν`, `if_of`, `Δν` | the channel comb [Hz], its per-channel IF index, and the channel width |
| `ν_eff` | the group-delay effective frequency of that comb [Hz]; one TECU ↦ K/ν_eff² |

`epoch_obs` uses the mid-observation epoch under the rule documented by [`session_facts`](@ref).
"""
struct SessionFacts
    name::String
    date::Date
    start::DateTime
    stop::DateTime
    epoch_obs::Float64
    doy::Int
    ν::Vector{Float64}
    if_of::Vector{Int}
    Δν::Float64
    ν_eff::Float64
end

Base.show(io::IO, f::SessionFacts) = print(io,
    "SessionFacts($(f.name), $(f.date), $(round(f.epoch_obs, digits = 7)), doy $(f.doy), ",
    "$(length(f.ν)) channels over $(maximum(f.if_of)) IFs, ν_eff $(round(f.ν_eff / 1e9, digits = 6)) GHz)")

"""
    decimal_year(t::DateTime) -> Float64

`t` as a decimal year: `year + (dayofyear − 1 + fraction of the day) / days in that year`. The one
rule — the year's own length, never a 365.25 constant — so no session needs a hand-chosen day count.
"""
decimal_year(t::DateTime) =
    year(t) + (dayofyear(t) - 1 + Dates.value(Millisecond(t - DateTime(Date(t)))) / 86_400_000) /
              daysinyear(year(t))

"""
    session_facts(; name, start, stop, ν, if_of, Δν) -> SessionFacts

The five derived facts, from values alone (no file): the identity string, the observation's own time
span and its channel comb.

* `date` is the UT day of `start` and `doy` its day of year — the IONEX index and the header date.
* `epoch_obs` is [`decimal_year`](@ref) at the MID-OBSERVATION epoch `start + (stop − start)/2`:
  ITRF velocities are propagated to the epoch the geometry is measured at, and for a 24 h session the
  mid-point is that epoch. (A session spanning a year-boundary is handled by construction — the
  fraction is taken in the year the mid-point falls in.)
* `ν_eff` is [`group_delay_ν_eff`](@ref) of the comb.
"""
function session_facts(; name, start::DateTime, stop::DateTime, ν, if_of, Δν)
    stop ≥ start || error("session_facts: the observation ends before it starts ($start .. $stop)")
    mid = start + Millisecond(Dates.value(Millisecond(stop - start)) ÷ 2)
    SessionFacts(String(name), Date(start), start, stop, decimal_year(mid), dayofyear(start),
                 collect(Float64, ν), collect(Int, if_of), Float64(Δν), group_delay_ν_eff(ν))
end

"""
    load_session(cfg) -> (; ds, antennas, flags, pcal, tsys, gain, byscan, byif, wholeband, selection, facts)

Open the session and return all loaded inputs and derived facts.

This is the ONLY function in the pipeline that opens `UV_DATA`, and it reads exactly one file,
`cfg.data`. Everything the later steps need comes back as a value:

| returned | what it is |
|---|---|
| `ds` | the `Dataset` — mmap-backed visibilities under `cfg.scans` |
| `antennas` | FITS antenna number ⇒ antenna name, the map every table reader is keyed by |
| `flags`, `pcal`, `tsys`, `gain` | the FLAG, PHASE-CAL, SYSTEM_TEMPERATURE and GAIN_CURVE tables |
| `byscan`, `byif`, `wholeband` | the three partitions every step is tiled on |
| `selection` | the FLAG-table row predicate, applied ONCE |
| `facts` | [`SessionFacts`](@ref) |

`selection` is built once and shared by the whole-band, per-IF, and imaging paths.

The result holds a memory-mapped `Dataset`, so it is not a serializable product and the runner does
not persist it (`facts` is what rides in every product's metadata instead).
"""
function load_session(cfg)
    uv = VLBIFiles.VLBI.load(cfg.data)
    antennas = map(a -> a.name, only(uv.ant_arrays).antennas)
    flags = VLBIFiles.flags(uv)
    pcal = VLBIFiles.phase_cal(uv)
    tsys = VLBIFiles.system_temperature(uv)
    gain = VLBIFiles.gain_curve(uv)
    ds = load_dataset(cfg.data; scans = cfg.scans)

    byscan = partition(ds, ByScan())
    byif = partition(ds, ByIF())
    wholeband = group(byif, (Tuple(1:maximum(ds.freq.if_of)),))
    selection = flag_selection(flags; antennas)

    facts = session_facts(; name = _obscode(uv),
                          start = minimum(ds.rows.datetime), stop = maximum(ds.rows.datetime),
                          ds.freq.ν, ds.freq.if_of, ds.freq.Δν)
    (; ds, antennas, flags, pcal, tsys, gain, byscan, byif, wholeband, selection, facts)
end

function _obscode(uv)
    h = uv.header.fits
    for k in ("OBSCODE", "OBS_CODE")
        haskey(h, k) && !isempty(strip(string(h[k]))) && return strip(string(h[k]))
    end
    error("load_session: $(uv.path) declares no OBSCODE — the session cannot name itself")
end
