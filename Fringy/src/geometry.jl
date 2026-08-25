
const C_LIGHT_M_S = 299792458.0
const AU_M = 1.49597870700e11
const DAY_S = 86400.0
const GM_SUN = 1.32712440041e20
const GM_EARTH = 3.986004418e14
const GM_MOON = 4.9028000e12
const PPN_GAMMA = 1.0
const OMEGA_EARTH = 7.292115146706979e-5
const R_EARTH_EQ = 6378137.0
const ARCSEC = deg2rad(1 / 3600)
const MAS = deg2rad(1e-3 / 3600)
const JD_MJD0 = 2400000.5

const GRAV_PLANETS = ((1.26686534e17, 5),
                      (3.7931207e16, 6),
                      (3.24858592e14, 2),
                      (4.282837e13, 4),
                      (5.7939513e15, 7),
                      (6.836153e15, 8))

const LEAP_MJD = [41317.0, 41499.0, 41683.0, 42048.0, 42413.0, 42778.0, 43144.0, 43509.0, 43874.0,
                  44239.0, 44786.0, 45151.0, 45516.0, 46247.0, 47161.0, 47892.0, 48257.0, 48804.0,
                  49169.0, 49534.0, 50083.0, 50630.0, 51179.0, 53736.0, 54832.0, 56109.0, 57204.0,
                  57754.0]
const LEAP_TAI_UTC = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0,
                      23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0,
                      36.0, 37.0]

const MJD_EPOCH_DATE = Date(1858, 11, 17)

"""
    UTCEpoch(mjd_utc)
    UTCEpoch(time0::DateTime, t::Real)

An instant in UTC as a two-part MJD: an integer `day` plus a day `frac`tion. The split is not
cosmetic — a single Float64 MJD of 2026 resolves only to ~0.6 µs, which is ~0.5 ps of antenna delay
and would be material at picosecond precision. Keeping the day integral makes the
Earth-rotation angle exact to femtoseconds.

The second form is the canonical one for the pipeline: `t` is the observable's canonical Float64
epoch, seconds after the `Dataset`'s `time0` (DateTime is display-only). The first splits an
existing Float64 MJD into its integral day and fraction, which is exact and hence loses nothing.

Only the sum `day + frac` is physical; the split controls precision. `UTCEpoch(0.0, mjd)` therefore
evaluates a single-Float64 MJD with exactly the arguments the python reference passes to ERFA
(UT1−UTC added to a ~61000-day number), which is what the reference gate compares against; the
proper split is ~0.3 ps more accurate.
"""
struct UTCEpoch
    day::Float64
    frac::Float64
end

UTCEpoch(mjd_utc::Real) = (d = floor(Float64(mjd_utc)); UTCEpoch(d, Float64(mjd_utc) - d))
UTCEpoch(time0::DateTime, t::Real) =
    UTCEpoch(Float64(Dates.value(Date(time0) - MJD_EPOCH_DATE)),
             (Dates.value(time0 - DateTime(Date(time0))) / 1000 + Float64(t)) / DAY_S)

"""
    utc_mjd(ep::UTCEpoch) -> Float64
    utc_mjd(time0::DateTime, t::Real) -> Float64

The UTC MJD as one Float64 — for the model terms that are insensitive to its ~0.6 µs granularity
(tides, ocean loading) and for reporting. The delay itself is evaluated from the two-part
[`UTCEpoch`](@ref).
"""
utc_mjd(ep::UTCEpoch) = ep.day + ep.frac
utc_mjd(time0::DateTime, t::Real) = utc_mjd(UTCEpoch(time0, t))

"""
    tai_minus_utc(mjd_utc) -> Float64

TAI−UTC in seconds at a UTC MJD, from the IERS leap-second table (37 s since 2017-01-01).
Fails loud before 1972-01-01, where UTC was rate-adjusted and this table does not apply.
"""
function tai_minus_utc(mjd_utc::Real)
    mjd_utc ≥ LEAP_MJD[1] || error("tai_minus_utc: MJD $mjd_utc precedes 1972-01-01, when UTC was rate-adjusted")
    LEAP_TAI_UTC[searchsortedlast(LEAP_MJD, Float64(mjd_utc))]
end

function _time_scales(ep::UTCEpoch, ut1_utc::Float64)
    jd1 = JD_MJD0 + ep.day
    ut12 = ep.frac + ut1_utc / DAY_S
    tt2 = ep.frac + (tai_minus_utc(utc_mjd(ep)) + 32.184) / DAY_S
    (jd1, ut12, jd1, tt2)
end



"""
    AntennaGeometry

One antenna of the a priori delay model, at the observing epoch: ITRF position (metres, tides NOT
applied), its derived geodetic coordinates (WGS84 `lon`, `lat` in radians, `height` in metres), the
alt-azimuth `axis_offset` in metres, and the ocean-loading BLQ coefficients — `otl_amplitude`
[metres] and `otl_phase` [degrees], rows radial/west/south, columns in `OTL_TIDES` order.

Constructed by [`load_antenna_geometry`](@ref) from a configuration file: antenna coordinates are input
data of the analysis, never package constants.
"""
struct AntennaGeometry
    code::Symbol
    xyz::SVector{3,Float64}
    lon::Float64
    lat::Float64
    height::Float64
    axis_offset::Float64
    otl_amplitude::SMatrix{3,11,Float64,33}
    otl_phase::SMatrix{3,11,Float64,33}
end

const OTL_TIDES = (:M2, :S2, :N2, :K2, :K1, :O1, :P1, :Q1, :MF, :MM, :SSA)
const OTL_DOODSON = SMatrix{6,11,Int}(
    2, 0, 0, 0, 0, 0,
    2, 2, -2, 0, 0, 0,
    2, -1, 0, 1, 0, 0,
    2, 2, 0, 0, 0, 0,
    1, 1, 0, 0, 0, 0,
    1, -1, 0, 0, 0, 0,
    1, 1, -2, 0, 0, 0,
    1, -2, 0, 1, 0, 0,
    0, 2, 0, 0, 0, 0,
    0, 1, 0, -1, 0, 0,
    0, 0, 2, 0, 0, 0)

"""
    load_antenna_geometry(path; epoch::Real) -> Dictionary{Symbol,AntennaGeometry}

Read the antenna-geometry configuration (TOML, e.g. `sessions/refdata/antenna_geometry.toml`) and propagate every
position from its own reference epoch to the observing `epoch` (a decimal year) with the tabulated
linear velocities.

The file gives, per antenna: `xyz` [m] and `velocity` [m/yr] at `epoch` (ITRF2020), `axis_offset`
[m], `mount` (only `"altaz"` is modelled) and the BLQ `ocean_loading_amplitude` [m] /
`ocean_loading_phase` [deg] as three rows (radial, west, south) over the tide columns listed in the
top-level `tides` key, which must be the canonical `OTL_TIDES` order.
"""
function load_antenna_geometry(path::AbstractString; epoch::Real)
    cfg = TOML.parsefile(path)
    tides = Symbol.(cfg["tides"])
    Tuple(tides) == OTL_TIDES ||
        error("$path lists ocean-loading tides $(tides), expected the canonical order $(collect(OTL_TIDES))")
    entries = cfg["antenna"]
    pairs = map(sort!(collect(keys(entries)))) do code
        e = entries[code]
        e["mount"] == "altaz" || error("antenna $code has mount $(e["mount"]); only alt-azimuth mounts are modelled")
        dt = Float64(epoch) - Float64(e["epoch"])
        xyz = SVector{3,Float64}(e["xyz"]) + dt * SVector{3,Float64}(e["velocity"])
        amp = _otl_matrix(e["ocean_loading_amplitude"], code, "amplitude")
        pha = _otl_matrix(e["ocean_loading_phase"], code, "phase")
        lon, lat, height = ERFA.gc2gd(ERFA.WGS84, xyz)
        Symbol(code) => AntennaGeometry(Symbol(code), xyz, lon, lat, height, Float64(e["axis_offset"]), amp, pha)
    end
    Dictionary(first.(pairs), last.(pairs))
end

function _otl_matrix(rows, code, what)
    length(rows) == 3 && all(r -> length(r) == 11, rows) ||
        error("antenna $code ocean-loading $what is not 3 rows × 11 tides")
    SMatrix{3,11,Float64}(reduce(vcat, permutedims.(Float64.(r) for r in rows)))
end



"""
    EOPSeries

Daily Earth-orientation parameters: `mjd` (UTC), `ut1_utc` [s], `xp`/`yp` polar motion [arcsec] and
the celestial-pole offsets `dX`/`dY` [rad] of the IAU2000A/2006 convention. Read from a committed
IERS file by [`read_finals2000A`](@ref) — no network access, so an analysis is reproducible.
"""
struct EOPSeries
    mjd::Vector{Float64}
    ut1_utc::Vector{Float64}
    xp::Vector{Float64}
    yp::Vector{Float64}
    dX::Vector{Float64}
    dY::Vector{Float64}
end

Base.length(e::EOPSeries) = length(e.mjd)
Base.extrema(e::EOPSeries) = extrema(e.mjd)

const _FINALS_REGIONS = (mjd = 8:15, xp = 18:36, yp = 37:55, ut1_utc = 59:78, nutation = 98:133)

"""
    read_finals2000A(path) -> EOPSeries

Parse an IERS `finals2000A.all`-format file (Bulletin-A values). Rows whose polar-motion, UT1 or
nutation fields are blank (predictions not yet filled in) are skipped, so the returned series covers
only the days the file has values for; `dX`/`dY` are converted from milliarcseconds to radians.

Deliberately a small parser over a committed file rather than a download-on-demand dependency
so an analysis remains reproducible offline.
"""
function read_finals2000A(path::AbstractString)
    mjd, ut1, xp, yp, dX, dY = (Float64[] for _ in 1:6)
    for line in eachline(path)
        length(line) ≥ last(_FINALS_REGIONS.nutation) || continue
        region(r) = split(line[r])
        pm_x, pm_y, ut = region(_FINALS_REGIONS.xp), region(_FINALS_REGIONS.yp), region(_FINALS_REGIONS.ut1_utc)
        nut = region(_FINALS_REGIONS.nutation)
        (length(pm_x) ≥ 1 && length(pm_y) ≥ 1 && length(ut) ≥ 1 && length(nut) ≥ 3) || continue
        push!(mjd, parse(Float64, strip(line[_FINALS_REGIONS.mjd])))
        push!(xp, parse(Float64, pm_x[1]))
        push!(yp, parse(Float64, pm_y[1]))
        push!(ut1, parse(Float64, ut[1]))
        push!(dX, parse(Float64, nut[1]) * MAS)
        push!(dY, parse(Float64, nut[3]) * MAS)
    end
    isempty(mjd) && error("no usable EOP rows parsed from $path — is it a finals2000A file?")
    issorted(mjd) || error("EOP rows in $path are not in MJD order")
    EOPSeries(mjd, ut1, xp, yp, dX, dY)
end

"""
    eop_at(eop::EOPSeries, mjd_utc) -> (; ut1_utc, xp, yp, dX, dY)

Earth-orientation parameters interpolated to a UTC MJD: UT1−UTC and polar motion by cubic Lagrange
interpolation through the four bracketing daily samples (holding UT1−UTC at its RDATE value instead
costs ~280 ps rms — verified), the celestial-pole offsets linearly (they move by 0.004 mas across a
day). Units as in [`EOPSeries`](@ref). Fails loud outside the tabulated range.
"""
function eop_at(eop::EOPSeries, mjd_utc::Real)
    t = Float64(mjd_utc)
    lo, hi = first(eop.mjd), last(eop.mjd)
    lo ≤ t ≤ hi || error("eop_at: MJD $t outside the tabulated EOP range $lo..$hi")
    (ut1_utc = _lagrange4(eop.mjd, eop.ut1_utc, t),
     xp = _lagrange4(eop.mjd, eop.xp, t),
     yp = _lagrange4(eop.mjd, eop.yp, t),
     dX = _linterp(eop.mjd, eop.dX, t),
     dY = _linterp(eop.mjd, eop.dY, t))
end

function _lagrange4(x::Vector{Float64}, y::Vector{Float64}, v::Float64)
    n = length(x)
    n ≥ 4 || error("_lagrange4 needs at least 4 samples, got $n")
    j = clamp(searchsortedfirst(x, v) - 2, 1, n - 3)
    s = 0.0
    for a in 0:3
        L = 1.0
        for b in 0:3
            a == b && continue
            L *= (v - x[j+b]) / (x[j+a] - x[j+b])
        end
        s += y[j+a] * L
    end
    s
end

function _linterp(x::Vector{Float64}, y::Vector{Float64}, v::Float64)
    j = clamp(searchsortedlast(x, v), 1, length(x) - 1)
    w = clamp((v - x[j]) / (x[j+1] - x[j]), 0.0, 1.0)
    (1 - w) * y[j] + w * y[j+1]
end

"""
    c2t_matrix(tt1, tt2, ut11, ut12, xp_as, yp_as, dX, dY) -> SMatrix{3,3}

Celestial (GCRS) → terrestrial (ITRS) rotation: IAU2006 precession / IAU2000A nutation, CIO based,
with the IERS celestial-pole offsets `dX`, `dY` [rad] added to the CIP coordinates (no ERFA
convenience routine applies them), Earth rotation angle from UT1 and the polar-motion matrix from
`xp_as`, `yp_as` [arcsec] with TIO locator s′.

Passing `dX = dY = 0` is exactly `ERFA.c2t06a`; the correlator used no pole offsets, we do (0.4 mas
≈ 40 ps of antenna delay).
"""
function c2t_matrix(tt1::Float64, tt2::Float64, ut11::Float64, ut12::Float64,
                    xp_as::Float64, yp_as::Float64, dX::Float64, dY::Float64)
    x, y, s = ERFA.xys06a(tt1, tt2)
    rc2i = ERFA.c2ixys(x + dX, y + dY, s)
    era = ERFA.era00(ut11, ut12)
    rpom = ERFA.pom00(xp_as * ARCSEC, yp_as * ARCSEC, ERFA.sp00(tt1, tt2))
    SMatrix{3,3,Float64}(ERFA.c2tcio(rc2i, era, rpom))
end



"""
    solid_tide_displacement(xyz, r_sun_itrf, r_moon_itrf, mjd_utc; step2::Bool) -> SVector{3}

Solid Earth tide displacement of an antenna [m, ITRF], IERS Conventions (2010) `DEHANTTIDEINEL`:
step 1 (degree 2 and 3 in-phase with latitude-dependent Love/Shida numbers, plus the out-of-phase
diurnal/semidiurnal and the l⁽¹⁾ latitude terms) and, with `step2 = true`, the frequency-dependent
corrections of Tables 7.3a/7.3b. Sun and Moon positions must be geocentric IN THE SAME Earth-fixed
frame as `xyz`.

The permanent (zero-frequency) deformation is INCLUDED, matching the official routine and the ITRF
"conventional tide free" convention: `X(t) = X_ITRF + solid_tide_displacement(...)`.
"""
function solid_tide_displacement(xyz::SVector{3,Float64}, xsun::SVector{3,Float64},
                                 xmon::SVector{3,Float64}, mjd_utc::Float64; step2::Bool)
    h20, l20, h3, l3 = 0.6078, 0.0847, 0.292, 0.015
    mass_ratio_sun, mass_ratio_moon = 332946.0482, 0.0123000371
    re = 6378136.6

    rsta, rsun, rmon = norm(xyz), norm(xsun), norm(xmon)
    scsun = dot(xyz, xsun) / (rsta * rsun)
    scmon = dot(xyz, xmon) / (rsta * rmon)
    cosphi = hypot(xyz[1], xyz[2]) / rsta
    h2 = h20 - 0.0006 * (1 - 1.5 * cosphi^2)
    l2 = l20 + 0.0002 * (1 - 1.5 * cosphi^2)

    p2sun = 3 * (h2 / 2 - l2) * scsun^2 - h2 / 2
    p2mon = 3 * (h2 / 2 - l2) * scmon^2 - h2 / 2
    p3sun = 2.5 * (h3 - 3l3) * scsun^3 + 1.5 * (l3 - h3) * scsun
    p3mon = 2.5 * (h3 - 3l3) * scmon^3 + 1.5 * (l3 - h3) * scmon
    x2sun, x2mon = 3 * l2 * scsun, 3 * l2 * scmon
    x3sun = 3 * l3 / 2 * (5 * scsun^2 - 1)
    x3mon = 3 * l3 / 2 * (5 * scmon^2 - 1)

    fac2sun = mass_ratio_sun * re * (re / rsun)^3
    fac2mon = mass_ratio_moon * re * (re / rmon)^3
    fac3sun = fac2sun * (re / rsun)
    fac3mon = fac2mon * (re / rmon)

    d = fac2sun * (x2sun * xsun / rsun + p2sun * xyz / rsta) +
        fac2mon * (x2mon * xmon / rmon + p2mon * xyz / rsta) +
        fac3sun * (x3sun * xsun / rsun + p3sun * xyz / rsta) +
        fac3mon * (x3mon * xmon / rmon + p3mon * xyz / rsta)

    d += _st1idiu(xyz, xsun, xmon, fac2sun, fac2mon)
    d += _st1isem(xyz, xsun, xmon, fac2sun, fac2mon)
    d += _st1l1(xyz, xsun, xmon, fac2sun, fac2mon)

    if step2
        day = floor(mjd_utc)
        fhr = (mjd_utc - day) * 24
        t = (mjd_utc + (JD_MJD0 - 2451545.0)) / 36525.0 +
            (tai_minus_utc(mjd_utc) + 32.184) / (3600 * 24 * 36525)
        d += _step2diu(xyz, fhr, t) + _step2lon(xyz, t)
    end
    d
end

function _antenna_frame(x::SVector{3,Float64})
    r = norm(x)
    sinphi = x[3] / r
    cosphi = hypot(x[1], x[2]) / r
    (r, sinphi, cosphi, x[2] / cosphi / r, x[1] / cosphi / r)
end

_rne_to_xyz(dr, dn, de, sinphi, cosphi, sinla, cosla) = SVector(
    dr * cosla * cosphi - de * sinla - dn * sinphi * cosla,
    dr * sinla * cosphi + de * cosla - dn * sinphi * sinla,
    dr * sinphi + dn * cosphi)

function _st1idiu(xsta, xsun, xmon, fac2sun, fac2mon)
    dhi, dli = -0.0025, -0.0007
    _, sinphi, cosphi, sinla, cosla = _antenna_frame(xsta)
    cos2phi = cosphi^2 - sinphi^2
    terms(x, r, fac) = let a = x[3] * (x[1] * sinla - x[2] * cosla) / r^2,
                           b = x[3] * (x[1] * cosla + x[2] * sinla) / r^2
        (-3 * dhi * sinphi * cosphi * fac * a, -3 * dli * cos2phi * fac * a, -3 * dli * sinphi * fac * b)
    end
    drs, dns, des = terms(xsun, norm(xsun), fac2sun)
    drm, dnm, dem = terms(xmon, norm(xmon), fac2mon)
    _rne_to_xyz(drs + drm, dns + dnm, des + dem, sinphi, cosphi, sinla, cosla)
end

function _st1isem(xsta, xsun, xmon, fac2sun, fac2mon)
    dhi, dli = -0.0022, -0.0007
    _, sinphi, cosphi, sinla, cosla = _antenna_frame(xsta)
    costwola = cosla^2 - sinla^2
    sintwola = 2 * cosla * sinla
    terms(x, r, fac) = let dd = x[1]^2 - x[2]^2, c = 2 * x[1] * x[2],
                           a = (dd * sintwola - c * costwola) / r^2,
                           b = (dd * costwola + c * sintwola) / r^2
        (-3 / 4 * dhi * cosphi^2 * fac * a, 3 / 2 * dli * sinphi * cosphi * fac * a,
         -3 / 2 * dli * cosphi * fac * b)
    end
    drs, dns, des = terms(xsun, norm(xsun), fac2sun)
    drm, dnm, dem = terms(xmon, norm(xmon), fac2mon)
    _rne_to_xyz(drs + drm, dns + dnm, des + dem, sinphi, cosphi, sinla, cosla)
end

function _st1l1(xsta, xsun, xmon, fac2sun, fac2mon)
    l1d, l1sd = 0.0012, 0.0024
    _, sinphi, cosphi, sinla, cosla = _antenna_frame(xsta)
    diu(x, r, fac) = (-l1d * sinphi^2 * fac * x[3] * (x[1] * cosla + x[2] * sinla) / r^2,
                      l1d * sinphi * (cosphi^2 - sinphi^2) * fac * x[3] * (x[1] * sinla - x[2] * cosla) / r^2)
    dns, des = diu(xsun, norm(xsun), fac2sun)
    dnm, dem = diu(xmon, norm(xmon), fac2mon)
    out = _rne_to_xyz(0.0, 3 * (dns + dnm), 3 * (des + dem), sinphi, cosphi, sinla, cosla)

    costwola = cosla^2 - sinla^2
    sintwola = 2 * cosla * sinla
    sem(x, r, fac) = let dd = x[1]^2 - x[2]^2, c = 2 * x[1] * x[2]
        (-l1sd / 2 * sinphi * cosphi * fac * (dd * costwola + c * sintwola) / r^2,
         -l1sd / 2 * sinphi^2 * cosphi * fac * (dd * sintwola - c * costwola) / r^2)
    end
    dns, des = sem(xsun, norm(xsun), fac2sun)
    dnm, dem = sem(xmon, norm(xmon), fac2mon)
    out + _rne_to_xyz(0.0, 3 * (dns + dnm), 3 * (des + dem), sinphi, cosphi, sinla, cosla)
end

function _fundamental_args(t::Float64)
    s = 218.31664563 + (481267.88194 + (-0.0014663889 + 0.00000185139t) * t) * t
    pr = (1.396971278 + (0.000308889 + (0.000000021 + 0.000000007t) * t) * t) * t
    h = 280.46645 + (36000.7697489 + (0.00030322222 + (0.000000020 - 0.00000000654t) * t) * t) * t
    p = 83.35324312 + (4069.01363525 + (-0.01032172222 + (-0.0000124991 + 0.00000005263t) * t) * t) * t
    zns = 234.95544499 + (1934.13626197 + (-0.00207561111 + (-0.00000213944 + 0.00000001650t) * t) * t) * t
    ps = 282.93734098 + (1.71945766667 + (0.00045688889 + (-0.00000001778 - 0.00000000334t) * t) * t) * t
    (s, pr, h, p, zns, ps)
end

const _DATDI_DIU = SMatrix{9,31,Float64}(
    -3, 0, 2, 0, 0, -0.01, 0.00, 0.00, 0.00,
    -3, 2, 0, 0, 0, -0.01, 0.00, 0.00, 0.00,
    -2, 0, 1, -1, 0, -0.02, 0.00, 0.00, 0.00,
    -2, 0, 1, 0, 0, -0.08, 0.00, -0.01, 0.01,
    -2, 2, -1, 0, 0, -0.02, 0.00, 0.00, 0.00,
    -1, 0, 0, -1, 0, -0.10, 0.00, 0.00, 0.00,
    -1, 0, 0, 0, 0, -0.51, 0.00, -0.02, 0.03,
    -1, 2, 0, 0, 0, 0.01, 0.00, 0.00, 0.00,
    0, -2, 1, 0, 0, 0.01, 0.00, 0.00, 0.00,
    0, 0, -1, 0, 0, 0.02, 0.00, 0.00, 0.00,
    0, 0, 1, 0, 0, 0.06, 0.00, 0.00, 0.00,
    0, 0, 1, 1, 0, 0.01, 0.00, 0.00, 0.00,
    0, 2, -1, 0, 0, 0.01, 0.00, 0.00, 0.00,
    1, -3, 0, 0, 1, -0.06, 0.00, 0.00, 0.00,
    1, -2, 0, -1, 0, 0.01, 0.00, 0.00, 0.00,
    1, -2, 0, 0, 0, -1.23, -0.07, 0.06, 0.01,
    1, -1, 0, 0, -1, 0.02, 0.00, 0.00, 0.00,
    1, -1, 0, 0, 1, 0.04, 0.00, 0.00, 0.00,
    1, 0, 0, -1, 0, -0.22, 0.01, 0.01, 0.00,
    1, 0, 0, 0, 0, 12.00, -0.80, -0.67, -0.03,
    1, 0, 0, 1, 0, 1.73, -0.12, -0.10, 0.00,
    1, 0, 0, 2, 0, -0.04, 0.00, 0.00, 0.00,
    1, 1, 0, 0, -1, -0.50, -0.01, 0.03, 0.00,
    1, 1, 0, 0, 1, 0.01, 0.00, 0.00, 0.00,
    0, 1, 0, 1, -1, -0.01, 0.00, 0.00, 0.00,
    1, 2, -2, 0, 0, -0.01, 0.00, 0.00, 0.00,
    1, 2, 0, 0, 0, -0.11, 0.01, 0.01, 0.00,
    2, -2, 1, 0, 0, -0.01, 0.00, 0.00, 0.00,
    2, 0, -1, 0, 0, -0.02, 0.00, 0.00, 0.00,
    3, 0, 0, 0, 0, 0.00, 0.00, 0.00, 0.00,
    3, 0, 0, 1, 0, 0.00, 0.00, 0.00, 0.00)

const _DATDI_LON = SMatrix{9,5,Float64}(
    0, 0, 0, 1, 0, 0.47, 0.23, 0.16, 0.07,
    0, 2, 0, 0, 0, -0.20, -0.12, -0.11, -0.05,
    1, 0, -1, 0, 0, -0.11, -0.08, -0.09, -0.04,
    2, 0, 0, 0, 0, -0.13, -0.11, -0.15, -0.07,
    2, 0, 0, 1, 0, -0.05, -0.05, -0.06, -0.03)

function _step2diu(xsta::SVector{3,Float64}, fhr::Float64, t::Float64)
    s, pr, h, p, zns, ps = _fundamental_args(t)
    tau = fhr * 15 + 280.4606184 + (36000.7700536 + (0.00038793 - 0.0000000258t) * t) * t - s
    s += pr
    s, tau = rem(s, 360.0), rem(tau, 360.0)
    h, p = rem(h, 360.0), rem(p, 360.0)
    zns, ps = rem(zns, 360.0), rem(ps, 360.0)
    _, sinphi, cosphi, sinla, cosla = _antenna_frame(xsta)
    zla = atan(xsta[2], xsta[1])
    dr = dn = de = 0.0
    for j in axes(_DATDI_DIU, 2)
        thetaf = deg2rad(tau + _DATDI_DIU[1, j] * s + _DATDI_DIU[2, j] * h + _DATDI_DIU[3, j] * p +
                         _DATDI_DIU[4, j] * zns + _DATDI_DIU[5, j] * ps)
        sa, ca = sincos(thetaf + zla)
        dr += (_DATDI_DIU[6, j] * sa + _DATDI_DIU[7, j] * ca) * 2 * sinphi * cosphi
        dn += (_DATDI_DIU[8, j] * sa + _DATDI_DIU[9, j] * ca) * (cosphi^2 - sinphi^2)
        de += (_DATDI_DIU[8, j] * ca - _DATDI_DIU[9, j] * sa) * sinphi
    end
    _rne_to_xyz(dr, dn, de, sinphi, cosphi, sinla, cosla) / 1000
end

function _step2lon(xsta::SVector{3,Float64}, t::Float64)
    s, pr, h, p, zns, ps = _fundamental_args(t)
    s += pr
    s, h, p = rem(s, 360.0), rem(h, 360.0), rem(p, 360.0)
    zns, ps = rem(zns, 360.0), rem(ps, 360.0)
    _, sinphi, cosphi, sinla, cosla = _antenna_frame(xsta)
    dr = dn = 0.0
    for j in axes(_DATDI_LON, 2)
        thetaf = deg2rad(_DATDI_LON[1, j] * s + _DATDI_LON[2, j] * h + _DATDI_LON[3, j] * p +
                         _DATDI_LON[4, j] * zns + _DATDI_LON[5, j] * ps)
        st, ct = sincos(thetaf)
        dr += (_DATDI_LON[6, j] * ct + _DATDI_LON[8, j] * st) * (3 * sinphi^2 - 1) / 2
        dn += (_DATDI_LON[7, j] * ct + _DATDI_LON[9, j] * st) * (2 * cosphi * sinphi)
    end
    _rne_to_xyz(dr, dn, 0.0, sinphi, cosphi, sinla, cosla) / 1000
end

"""
    mean_pole(mjd_utc) -> (xs, ys)

IERS 2010 (2018 update) secular pole model, arcseconds — the reference the pole tide is measured
against.
"""
function mean_pole(mjd_utc::Real)
    t = 2000.0 + (Float64(mjd_utc) - 51544.5) / 365.25
    ((55.0 + 1.677 * (t - 2000)) * 1e-3, (320.5 + 3.460 * (t - 2000)) * 1e-3)
end

"""
    pole_tide_displacement(antenna, xp_as, yp_as, mjd_utc) -> SVector{3}

Solid-Earth pole tide displacement [m, ITRF] (IERS 2010 eq. 7.26) from the offset of the
instantaneous pole (`xp_as`, `yp_as` in arcseconds) from the secular [`mean_pole`](@ref).
"""
function pole_tide_displacement(st::AntennaGeometry, xp_as::Float64, yp_as::Float64, mjd_utc::Float64)
    xm, ym = mean_pole(mjd_utc)
    m1 = xp_as - xm
    m2 = -(yp_as - ym)
    θ = π / 2 - st.lat
    st_, ct = sincos(θ)
    λ = st.lon
    sl, cl = sincos(λ)
    dr = -32e-3 * st_ * ct * (m1 * cl + m2 * sl)
    dθ = -9e-3 * ct * (m1 * cl + m2 * sl)
    dλ = 9e-3 * ct * (m1 * sl - m2 * cl)
    up = SVector(st_ * cl, st_ * sl, ct)
    south = SVector(ct * cl, ct * sl, -st_)
    east = SVector(-sl, cl, 0.0)
    dr * up + dθ * south + dλ * east
end

function _doodson_args(mjd_utc::Float64, gmst_deg::Float64)
    T = (mjd_utc - 51544.5) / 36525.0
    s = 218.31664563 + 481267.88194T - 0.0014663889T^2
    h = 280.46645 + 36000.7697489T + 0.00030322222T^2
    p = 83.35324312 + 4069.01363525T - 0.01032172222T^2
    Np = -(125.04455501 - 1934.13626197T + 0.00207561111T^2)
    ps = 282.93734098 + 1.71945766667T
    (gmst_deg + 180.0 - s, s, h, p, Np, ps)
end

"""
    ocean_loading_displacement(antenna, mjd_utc, gmst_deg) -> SVector{3}

Ocean tidal loading displacement [m, ITRF] from the antenna's BLQ coefficients (TPXO.7.2), summed
over the 11 tabulated constituents with no minor-tide admittance (good to a few percent of the
loading signal, itself ~20 ps of delay).

Arguments follow IERS `HARDISP`/`ADMINT`: the Doodson combination plus a species-dependent offset
(+180° long period, +90° diurnal, 0° semidiurnal), and the BLQ phases are lags; the BLQ rows
radial/west/south are returned as an up/east/north-composed Cartesian vector.
"""
function ocean_loading_displacement(st::AntennaGeometry, mjd_utc::Float64, gmst_deg::Float64)
    args = _doodson_args(mjd_utc, gmst_deg)
    up = SVector(cos(st.lat) * cos(st.lon), cos(st.lat) * sin(st.lon), sin(st.lat))
    east = SVector(-sin(st.lon), cos(st.lon), 0.0)
    north = cross(up, east)
    dU = dW = dS = 0.0
    for j in 1:11
        species = OTL_DOODSON[1, j]
        off = species == 0 ? 180.0 : species == 1 ? 90.0 : 0.0
        θ = deg2rad(sum(OTL_DOODSON[i, j] * args[i] for i in 1:6) + off)
        dU += st.otl_amplitude[1, j] * cos(θ - deg2rad(st.otl_phase[1, j]))
        dW += st.otl_amplitude[2, j] * cos(θ - deg2rad(st.otl_phase[2, j]))
        dS += st.otl_amplitude[3, j] * cos(θ - deg2rad(st.otl_phase[3, j]))
    end
    dU * up - dW * east - dS * north
end



"""
    GeometryTerms(; solid_tides, pole_tide, ocean_loading, axis_offset, gravitation, aberration)

Which terms [`geometric_delay`](@ref) includes. Every field is required — the full model is
[`FULL_GEOMETRY`](@ref); switching terms off is what makes the term-by-term validation gate
possible. Reproducible offline evaluation instead comes from the pinned local Earth-orientation,
antenna, loading, and source inputs consumed by the full model.
"""
@kwdef struct GeometryTerms
    solid_tides::Bool
    pole_tide::Bool
    ocean_loading::Bool
    axis_offset::Bool
    gravitation::Bool
    aberration::Bool
end

"""
    FULL_GEOMETRY

All a priori geometry terms on: solid Earth tides (step 1 + 2), solid pole tide, ocean
loading, axis offset, gravitational delay (Sun, Moon, Earth, six planets) and the aberration /
retarded-baseline terms of the consensus model.
"""
const FULL_GEOMETRY = GeometryTerms(solid_tides = true, pole_tide = true, ocean_loading = true,
                                    axis_offset = true, gravitation = true, aberration = true)

"""
    geometric_delay(st::AntennaGeometry, ra, dec, epoch, eop::EOPSeries, terms::GeometryTerms) -> NamedTuple

The a priori geocentre→antenna delay for a plane wave from ICRS `(ra, dec)` [rad] arriving at the
geocentre at `epoch` — a [`UTCEpoch`](@ref) (the canonical form: `UTCEpoch(ds.time0, t)`) or a plain
UTC MJD.

Returns `(; τ_geometric, τ_geometric_without_axis_offset, τ_axis_offset, el, az, ∂τ_∂α★, ∂τ_∂δ, r_cel, k, rc2t)`:
- `τ_geometric` [s] — the modelled delay `t_antenna − t_geocentre`, **negative when the source is above the
  antenna**; `τ_geometric_without_axis_offset` is it without the axis offset `τ_axis_offset`.
- `el`, `az` [rad] — geodetic elevation and azimuth (from north, east positive).
- `∂τ_∂α★`, `∂τ_∂δ` [s/rad] — analytic partials of `τ_geometric_without_axis_offset` with respect to
  local offsets `(Δα★ = cos(δ) Δα, Δδ)`, the
  estimator's source partials, tested against finite differences of `τ_geometric_without_axis_offset`. They
  keep the two O(v/c) terms of the consensus expression and omit the elevation dependence of the
  axis offset (≤ 2e-6 relative — the same choice the verified reference makes).
- `r_cel` [m] — geocentre→antenna vector in GCRS axes (tides/loading applied), `k` the source unit
  vector, `rc2t` the GCRS→ITRS matrix — the ingredients downstream terms (ionosphere piercing
  points, frame diagnostics) need without recomputing them.

`terms` selects which physical contributions enter; use [`FULL_GEOMETRY`](@ref) for the model.
"""
geometric_delay(st::AntennaGeometry, ra::Real, dec::Real, mjd_utc::Real, eop::EOPSeries, terms::GeometryTerms) =
    geometric_delay(st, Float64(ra), Float64(dec), UTCEpoch(mjd_utc), eop, terms)

function geometric_delay(st::AntennaGeometry, ra::Float64, dec::Float64, ep::UTCEpoch,
                         eop::EOPSeries, terms::GeometryTerms)
    mjd_utc = utc_mjd(ep)
    e = eop_at(eop, mjd_utc)
    ut11, ut12, tt1, tt2 = _time_scales(ep, e.ut1_utc)
    rc2t = c2t_matrix(tt1, tt2, ut11, ut12, e.xp, e.yp, e.dX, e.dY)

    pvh, pvb = ERFA.epv00(tt1, tt2)
    r_earth = SVector{3}(pvb[1]) * AU_M
    v_earth = terms.aberration ? SVector{3}(pvb[2]) * (AU_M / DAY_S) : zero(SVector{3,Float64})
    r_sun_cel = -SVector{3}(pvh[1]) * AU_M
    r_moon_cel = SVector{3}(ERFA.moon98(tt1, tt2)[1]) * AU_M

    dxyz = zero(SVector{3,Float64})
    if terms.solid_tides
        dxyz += solid_tide_displacement(st.xyz, rc2t * r_sun_cel, rc2t * r_moon_cel, mjd_utc; step2 = true)
    end
    if terms.pole_tide
        dxyz += pole_tide_displacement(st, e.xp, e.yp, mjd_utc)
    end
    if terms.ocean_loading
        dxyz += ocean_loading_displacement(st, mjd_utc, rad2deg(ERFA.gmst06(ut11, ut12, tt1, tt2)))
    end

    r_cel = rc2t' * (st.xyz + dxyz)
    w2 = cross(SVector(0.0, 0.0, OMEGA_EARTH), r_cel)

    sd, cd = sincos(dec)
    sa, ca = sincos(ra)
    k = SVector(cd * ca, cd * sa, sd)

    b = r_cel
    kb, kv, kw = dot(k, b), dot(k, v_earth), dot(k, w2)
    vb, vw, v2 = dot(v_earth, b), dot(v_earth, w2), dot(v_earth, v_earth)

    U = GM_SUN / norm(r_sun_cel)

    dgrav = 0.0
    if terms.gravitation
        r1 = r_earth
        r2 = r_earth + b
        dgrav += _grav_delay(k, r1, r2, r_earth + r_sun_cel, GM_SUN)
        dgrav += _grav_delay(k, r1, r2, r_earth + r_moon_cel, GM_MOON)
        for (gm, body) in GRAV_PLANETS
            pb = (r_earth + r_sun_cel) + SVector{3}(ERFA.plan94(tt1, tt2, body)[1]) * AU_M
            dgrav += _grav_delay(k, r1, r2, pb, gm)
        end
        dgrav += (2 * GM_EARTH / C_LIGHT_M_S^3) * log(2 * R_EARTH_EQ / (norm(b) + kb))
    end

    num = dgrav - (kb / C_LIGHT_M_S) * (1 - (1 + PPN_GAMMA) * U / C_LIGHT_M_S^2 -
                                        v2 / (2 * C_LIGHT_M_S^2) - vw / C_LIGHT_M_S^2) -
          (vb / C_LIGHT_M_S^2) * (1 + kv / (2 * C_LIGHT_M_S))
    den = 1 + (kv + kw) / C_LIGHT_M_S
    τ_geometric_without_axis_offset = num / den

    k_itrf = rc2t * k
    up = SVector(cos(st.lat) * cos(st.lon), cos(st.lat) * sin(st.lon), sin(st.lat))
    east = SVector(-sin(st.lon), cos(st.lon), 0.0)
    north = cross(up, east)
    el = asin(clamp(dot(k_itrf, up), -1, 1))
    az = atan(dot(k_itrf, east), dot(k_itrf, north))

    τ_axis_offset = terms.axis_offset ? -st.axis_offset * cos(el) / C_LIGHT_M_S : 0.0

    vwtot = v_earth + w2
    dtau_dk = (-b / C_LIGHT_M_S - (τ_geometric_without_axis_offset / den) * vwtot / C_LIGHT_M_S) / den
    dka = SVector(-sa, ca, 0.0)
    dkd = SVector(-sd * ca, -sd * sa, cd)

    (τ_geometric = τ_geometric_without_axis_offset + τ_axis_offset, τ_geometric_without_axis_offset, τ_axis_offset, el, az,
     ∂τ_∂α★ = dot(dtau_dk, dka), ∂τ_∂δ = dot(dtau_dk, dkd),
     r_cel, k, rc2t)
end

function _grav_delay(k, r1, r2, body, gm)
    R1 = r1 - body
    R2 = r2 - body
    (2 * gm / C_LIGHT_M_S^3) * log((norm(R1) + dot(k, R1)) / (norm(R2) + dot(k, R2)))
end

"""
    azel(st::AntennaGeometry, k_itrf) -> (az, el)

Azimuth (from north, east positive) and elevation [rad] of a source direction given as a unit
vector in ITRF, using the antenna's geodetic vertical. `geometric_delay` already returns both; this
is for consumers holding only the rotated direction.
"""
function azel(st::AntennaGeometry, k_itrf::SVector{3,Float64})
    up = SVector(cos(st.lat) * cos(st.lon), cos(st.lat) * sin(st.lon), sin(st.lat))
    east = SVector(-sin(st.lon), cos(st.lon), 0.0)
    north = cross(up, east)
    (atan(dot(k_itrf, east), dot(k_itrf, north)), asin(clamp(dot(k_itrf, up), -1, 1)))
end
