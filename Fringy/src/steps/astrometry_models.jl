
export astrometry_models

"""
    astrometry_models(cfg, facts::SessionFacts; time0::DateTime) -> (; uv, correlator_model, astrometry_model, antennas, sources)
    astrometry_models(cfg, ld) -> (; uv, correlator_model, astrometry_model, antennas, sources)

The two model layers a delay is formed against, both on the canonical `time0` of the product they
will be joined with:

* `correlator_model` — the CORRELATOR's own model: the exact delay polynomial it correlated with, and the linear
  CLOCK a priori it removed. What the observed delay is a residual to.
* `astrometry_model` — the combined [`AstrometryModel`](@ref): ITRF2020 antennas propagated to `facts.epoch_obs`,
  IERS finals EOP including celestial-pole offsets, WEATHER pressure (US standard atmosphere where
  an antenna reports none), selected geometry terms, and IONEX products evaluated at `facts.ν_eff`.

Every value that is a FACT of the session — the propagation epoch, the day of year the hydrostatic
mapping is seasonal in, the effective frequency the TEC is converted at — comes from `facts`, i.e.
from the data. Everything that is a CHOICE — which reference files, which geometric terms, which
ionospheric product is primary and how its spread is inflated — comes from `cfg`.

The second form is the one the runner calls, `ld` being [`load_session`](@ref)'s result; the first is
for a caller that has a cached product's `facts` and `time0` but no open dataset.
"""
function astrometry_models(cfg, facts::SessionFacts; time0::DateTime)
    uv = VLBIFiles.VLBI.load(cfg.data)
    antennas = only(uv.ant_arrays).antennas
    correlator_model = CorrelatorDelayModel(uv; time0)
    srctable = collect(values(VLBIFiles.sources(uv)))
    sources = Dictionary(map(r -> r.name, srctable),
                         map(r -> (ra = Float64(r.coords.ra), dec = Float64(r.coords.dec)), srctable))
    weather = VLBIFiles.weather(uv)
    pressure = Dictionary(map(a -> a.name, antennas),
                          map(i -> pressure_series(weather, i, time0), eachindex(antennas)))
    astrometry_model = AstrometryModel(; antenna_geometry = load_antenna_geometry(cfg.antenna_geometry; epoch = facts.epoch_obs), sources,
                      eop = read_finals2000A(cfg.eop),
                      ionex = map(v -> IonexSeries(joinpath.(cfg.ionex_dir, v)), cfg.ionex_files),
                      pressure, terms = cfg.geometry, time0, doy = facts.doy, ν_eff = facts.ν_eff,
                      iono_primary = cfg.iono_primary, iono_inflation = cfg.iono_inflation)
    (; uv, correlator_model, astrometry_model, antennas, sources)
end

astrometry_models(cfg, ld) = astrometry_models(cfg, ld.facts; time0 = ld.ds.time0)
