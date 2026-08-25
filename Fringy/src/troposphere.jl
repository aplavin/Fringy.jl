
const _NMF_LAT = SVector(15.0, 30.0, 45.0, 60.0, 75.0)
const _NMF_H_AVG = SMatrix{3,5,Float64}(
    1.2769934e-3, 2.9153695e-3, 62.610505e-3,
    1.2683230e-3, 2.9152299e-3, 62.837393e-3,
    1.2465397e-3, 2.9288445e-3, 63.721774e-3,
    1.2196049e-3, 2.9022565e-3, 63.824265e-3,
    1.2045996e-3, 2.9024912e-3, 64.258455e-3)
const _NMF_H_AMP = SMatrix{3,5,Float64}(
    0.0, 0.0, 0.0,
    1.2709626e-5, 2.1414979e-5, 9.0128400e-5,
    2.6523662e-5, 3.0160779e-5, 4.3497037e-5,
    3.4000452e-5, 7.2562722e-5, 84.795348e-5,
    4.1202191e-5, 11.723375e-5, 170.37206e-5)
const _NMF_H_HT = SVector(2.53e-5, 5.49e-3, 1.14e-3)
const _NMF_W_AVG = SMatrix{3,5,Float64}(
    5.8021897e-4, 1.4275268e-3, 4.3472961e-2,
    5.6794847e-4, 1.5138625e-3, 4.6729510e-2,
    5.8118019e-4, 1.4572752e-3, 4.3908931e-2,
    5.9727542e-4, 1.5007428e-3, 4.4626982e-2,
    6.1641693e-4, 1.7599082e-3, 5.4736038e-2)

_cf3(sinE, a, b, c) = (1 + a / (1 + b / (1 + c))) / (sinE + a / (sinE + b / (sinE + c)))

function _lat_interp(tbl::SMatrix{3,5,Float64}, row::Int, lat_deg::Float64)
    x = clamp(lat_deg, _NMF_LAT[1], _NMF_LAT[5])
    j = clamp(searchsortedlast(_NMF_LAT, x), 1, 4)
    w = (x - _NMF_LAT[j]) / (_NMF_LAT[j+1] - _NMF_LAT[j])
    (1 - w) * tbl[row, j] + w * tbl[row, j+1]
end

"""
    hydrostatic_mapping(el, lat, height, doy) -> Float64

Dimensionless Niell (1996) hydrostatic mapping function at elevation `el` [rad] for an antenna at geodetic
latitude `lat` [rad] and `height` [m], on day of year `doy` (the seasonal argument; southern
hemisphere is offset by half a year, as in the original).

This is the mapping the verified python cross-check used; it reproduces the correlator's ATMOS to a
few ps, which is why the a priori model uses it as the hydrostatic mapping.
"""
function hydrostatic_mapping(el::Float64, lat::Float64, height::Float64, doy::Real)
    lat_deg = rad2deg(lat)
    d = lat_deg < 0 ? Float64(doy) + 182.625 : Float64(doy)
    yr = cos(2π * (d - 28.0) / 365.25)
    a = _lat_interp(_NMF_H_AVG, 1, abs(lat_deg)) - _lat_interp(_NMF_H_AMP, 1, abs(lat_deg)) * yr
    b = _lat_interp(_NMF_H_AVG, 2, abs(lat_deg)) - _lat_interp(_NMF_H_AMP, 2, abs(lat_deg)) * yr
    c = _lat_interp(_NMF_H_AVG, 3, abs(lat_deg)) - _lat_interp(_NMF_H_AMP, 3, abs(lat_deg)) * yr
    sinE = sin(el)
    m = _cf3(sinE, a, b, c)
    dm = (1 / sinE - _cf3(sinE, _NMF_H_HT...)) * (height / 1000)
    m + dm
end

"""
    wet_mapping(el, lat) -> Float64

Dimensionless Niell (1996) wet mapping function at elevation `el` [rad] and geodetic latitude `lat` [rad] — the
partial of the delay with respect to the estimated zenith wet delay; no height correction and
no seasonal term, as published.
"""
function wet_mapping(el::Float64, lat::Float64)
    lat_deg = abs(rad2deg(lat))
    _cf3(sin(el), _lat_interp(_NMF_W_AVG, 1, lat_deg), _lat_interp(_NMF_W_AVG, 2, lat_deg),
         _lat_interp(_NMF_W_AVG, 3, lat_deg))
end

"""
    gradient_mapping(el) -> Float64

Dimensionless Chen & Herring (1997) gradient mapping function `1/(sin e·tan e + 0.0032)`: the partial of the
delay w.r.t. the estimated troposphere gradients, which enter as
`gradient_mapping(el) · (G_N cos A + G_E sin A)`.
"""
gradient_mapping(el::Float64) = 1 / (sin(el) * tan(el) + 0.0032)

"""
    zenith_hydrostatic_path(pressure_hpa, lat, height) -> Float64

Saastamoinen zenith hydrostatic path [m], from the surface `pressure_hpa` [hPa]
at an antenna of geodetic latitude `lat` [rad] and `height` [m] (Davis et al. 1985 form). Divide by
the speed of light for seconds.
"""
zenith_hydrostatic_path(pressure_hpa::Real, lat::Real, height::Real) =
    0.0022768 * pressure_hpa / (1 - 0.00266 * cos(2lat) - 0.00028 * height / 1000)

"""
    standard_pressure(height) -> Float64

US Standard Atmosphere (1976) pressure [hPa] at geometric `height` [m]. This is the MK fallback:
that antenna records no meteorology at all (all-zero WEATHER rows), so its hydrostatic path comes
from the standard atmosphere at its height.
"""
standard_pressure(height::Real) = 1013.25 * (1 - 2.25577e-5 * height)^5.2559

"""
    PressureSeries(t, pressure_hpa)

Surface pressure of one antenna as a time series: `t` in canonical Float64 seconds (same origin as
the `Dataset`), `pressure_hpa` in hPa, both sorted by `t`. An empty series means "no meteorology",
and [`surface_pressure`](@ref) then falls back to the standard atmosphere.
"""
struct PressureSeries
    t::Vector{Float64}
    pressure_hpa::Vector{Float64}
end

"""
    pressure_series(weather, antenna_no, time0; min_pressure_hpa = 100.0) -> PressureSeries

Extract one antenna's usable surface pressure from the `weather(uv)` table (VLBIFiles), converting
its `DateTime` tags to canonical seconds after `time0` and dropping records below
`min_pressure_hpa` — the all-zero rows of an antenna with no working sensors, which the reader
deliberately leaves in place for the consumer to judge.

Called behind a function barrier: the VLBIFiles reader's return type is not inferable, so nothing of it
escapes into the model's hot path.
"""
function pressure_series(weather, antenna_no::Integer, time0::DateTime; min_pressure_hpa::Real = 100.0)
    t = Float64[]
    p = Float64[]
    for r in weather
        r.antenna_no == antenna_no || continue
        press = ustrip(u"mbar", r.pressure)
        press > min_pressure_hpa || continue
        push!(t, Dates.value(r.time - time0) / 1000)
        push!(p, press)
    end
    o = sortperm(t)
    PressureSeries(t[o], p[o])
end

"""
    surface_pressure(ps::PressureSeries, t, height) -> Float64

Surface pressure [hPa] at canonical epoch `t`, linearly interpolated within the series and held
constant outside it; when the series is empty (no meteorology recorded, e.g. MK) the US Standard
Atmosphere at `height` is returned instead.
"""
function surface_pressure(ps::PressureSeries, t::Real, height::Real)
    isempty(ps.t) && return standard_pressure(height)
    _linterp(ps.t, ps.pressure_hpa, Float64(t))
end
