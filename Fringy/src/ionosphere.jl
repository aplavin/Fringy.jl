
const K_ION = 40.3082
const TECU = 1.0e16
const PETROV_DH = 56.7e3
const PETROV_ALPHA = 0.9782
const PETROV_K = 0.85

"""
    IonexSeries

One global-ionosphere-map product: `mjd` map epochs (UTC), the `lat` [deg, geocentric] × `lon`
[deg] grid, `tec[epoch, lat, lon]` VTEC in TECU (NaN where the file flags 9999) and `rms` likewise,
plus the shell geometry `base_radius` and `height` [m].

Built by [`read_ionex`](@ref) from one or several daily files (`IonexSeries(paths)`), whose shared
boundary epoch is de-duplicated. Latitudes are stored ascending regardless of the file's order.
"""
struct IonexSeries
    label::String
    mjd::Vector{Float64}
    lat::Vector{Float64}
    lon::Vector{Float64}
    tec::Array{Float64,3}
    rms::Array{Float64,3}
    base_radius::Float64
    height::Float64
end

Base.show(io::IO, s::IonexSeries) = print(io, "IonexSeries(", s.label, ", ", length(s.mjd),
    " maps, ", length(s.lat), "×", length(s.lon), " grid, shell ", round(s.height / 1e3), " km)")

function _ionex_lines(path::AbstractString)
    isfile(path) || error("IONEX file not found: $path")
    if endswith(path, ".gz") || endswith(path, ".Z")
        split(read(pipeline(`gzip -dc $path`), String), '\n')
    elseif endswith(path, ".xz")
        split(read(pipeline(`xz -dc $path`), String), '\n')
    else
        readlines(path)
    end
end

_ionex_label(line) = length(line) ≥ 61 ? strip(line[61:min(end, 80)]) : ""
_ionex_value(line) = length(line) ≥ 60 ? line[1:60] : line

function _ymdhms_mjd(y, m, d, hh, mm, ss)
    a = fld(14 - m, 12)
    y2 = y + 4800 - a
    m2 = m + 12a - 3
    jdn = d + fld(153m2 + 2, 5) + 365y2 + fld(y2, 4) - fld(y2, 100) + fld(y2, 400) - 32045
    (jdn - 2400001) + (hh + mm / 60 + ss / 3600) / 24
end

"""
    read_ionex(path) -> IonexSeries

Parse one IONEX 1.0 file (optionally `.gz`/`.Z`/`.xz` compressed). TEC values are fixed-width I5
fields scaled by the header `EXPONENT`, with 9999 mapped to `NaN`; RMS maps are read when present.

Fixed-width parsing is deliberate: whitespace splitting is unsafe once a 5-digit value abuts the
next field.
"""
function read_ionex(path::AbstractString)
    lines = _ionex_lines(path)
    hdr = Dict{String,String}()
    i = 1
    while i ≤ length(lines)
        lbl = _ionex_label(lines[i])
        i += 1
        lbl == "END OF HEADER" && break
        isempty(lbl) || (hdr[lbl] = _ionex_value(lines[i-1]))
    end
    exponent = parse(Int, first(split(get(hdr, "EXPONENT", "   -1"))))
    base_radius = parse(Float64, first(split(hdr["BASE RADIUS"]))) * 1e3
    h1, h2, _ = parse.(Float64, split(hdr["HGT1 / HGT2 / DHGT"]))
    height = 0.5 * (h1 + h2) * 1e3
    lat1, lat2, dlat = parse.(Float64, split(hdr["LAT1 / LAT2 / DLAT"]))
    lon1, lon2, dlon = parse.(Float64, split(hdr["LON1 / LON2 / DLON"]))
    nlat = round(Int, (lat2 - lat1) / dlat) + 1
    nlon = round(Int, (lon2 - lon1) / dlon) + 1
    lat = lat1 .+ dlat .* (0:nlat-1)
    lon = lon1 .+ dlon .* (0:nlon-1)

    tecmaps = Dict{Float64,Matrix{Float64}}()
    rmsmaps = Dict{Float64,Matrix{Float64}}()
    kind = :none
    epoch = NaN
    grid = fill(NaN, nlat, nlon)
    while i ≤ length(lines)
        line = lines[i]
        lbl = _ionex_label(line)
        if lbl == "START OF TEC MAP" || lbl == "START OF RMS MAP"
            kind = occursin("TEC", lbl) ? :tec : :rms
            grid = fill(NaN, nlat, nlon)
        elseif lbl == "END OF TEC MAP" || lbl == "END OF RMS MAP"
            (kind === :tec ? tecmaps : rmsmaps)[epoch] = grid
            kind = :none
        elseif lbl == "EPOCH OF CURRENT MAP"
            v = parse.(Int, split(_ionex_value(line)))
            epoch = _ymdhms_mjd(v[1], v[2], v[3], v[4], v[5], v[6])
        elseif lbl == "LAT/LON1/LON2/DLON/H" && kind !== :none
            rlat = parse(Float64, line[3:8])
            row = round(Int, (rlat - lat1) / dlat) + 1
            nread = 0
            while nread < nlon
                i += 1
                dat = lines[i]
                for c in 1:5:min(length(dat), 80)
                    fld = strip(dat[c:min(c + 4, length(dat))])
                    isempty(fld) && continue
                    v = parse(Int, fld)
                    nread += 1
                    grid[row, nread] = v == 9999 ? NaN : v * 10.0^exponent
                    nread == nlon && break
                end
            end
        end
        i += 1
    end
    isempty(tecmaps) && error("no TEC maps found in $path")

    ep = sort!(collect(keys(tecmaps)))
    tec = Array{Float64,3}(undef, length(ep), nlat, nlon)
    rms = fill(NaN, length(ep), nlat, nlon)
    for (n, e) in enumerate(ep)
        tec[n, :, :] = tecmaps[e]
        haskey(rmsmaps, e) && (rms[n, :, :] = rmsmaps[e])
    end
    _ionex_series(basename(path), ep, collect(lat), collect(lon), tec, rms, base_radius, height)
end

function _ionex_series(label, mjd, lat, lon, tec, rms, base_radius, height)
    if length(lat) > 1 && lat[1] > lat[end]
        lat = reverse(lat)
        tec = tec[:, end:-1:1, :]
        rms = rms[:, end:-1:1, :]
    end
    IonexSeries(label, mjd, lat, lon, tec, rms, base_radius, height)
end

"""
    IonexSeries(paths::AbstractVector; label = "")

Concatenate consecutive daily IONEX files into one series, de-duplicating the epoch daily files
share at their boundary. Fails loud if the files' grids or shell geometries differ.
"""
function IonexSeries(paths::AbstractVector{<:AbstractString}; label::AbstractString = "")
    parts = map(read_ionex, paths)
    p1 = first(parts)
    for p in parts
        (p.lat == p1.lat && p.lon == p1.lon) || error("IONEX files have different grids: $(p.label) vs $(p1.label)")
        (p.base_radius == p1.base_radius && p.height == p1.height) ||
            error("IONEX files have different shell geometry: $(p.label) vs $(p1.label)")
    end
    mjd = reduce(vcat, (p.mjd for p in parts))
    tec = reduce(vcat, (p.tec for p in parts))
    rms = reduce(vcat, (p.rms for p in parts))
    o = sortperm(mjd)
    mjd, tec, rms = mjd[o], tec[o, :, :], rms[o, :, :]
    keep = [n == 1 || mjd[n] - mjd[n-1] > 1e-9 for n in eachindex(mjd)]
    IonexSeries(isempty(label) ? p1.label : label, mjd[keep], p1.lat, p1.lon,
                tec[keep, :, :], rms[keep, :, :], p1.base_radius, p1.height)
end

function _bilinear(cube::Array{Float64,3}, it::Int, la::Vector{Float64}, lo::Vector{Float64},
                   lat::Float64, lon::Float64)
    lat = clamp(lat, la[1], la[end])
    lon = clamp(lo[1] + mod(lon - lo[1], 360.0), lo[1], lo[end])
    j = clamp(searchsortedfirst(la, lat) - 1, 1, length(la) - 1)
    k = clamp(searchsortedfirst(lo, lon) - 1, 1, length(lo) - 1)
    u = (lat - la[j]) / (la[j+1] - la[j])
    v = (lon - lo[k]) / (lo[k+1] - lo[k])
    (1 - u) * (1 - v) * cube[it, j, k] + u * (1 - v) * cube[it, j+1, k] +
    (1 - u) * v * cube[it, j, k+1] + u * v * cube[it, j+1, k+1]
end

"""
    vtec(s::IonexSeries, mjd, lat_deg, lon_deg) -> Float64

Vertical TEC [TECU] at a UTC MJD and geocentric `(lat_deg, lon_deg)` [deg], using the
IONEX-recommended **rotating** time interpolation (spec method 3): each bracketing map is evaluated
at the longitude the point had at that map's epoch in the sun-fixed frame, then the two are
combined linearly in time. Epochs outside the series are clamped to its ends.
"""
function vtec(s::IonexSeries, mjd::Real, lat_deg::Real, lon_deg::Real)
    it, w = _time_bracket(s.mjd, Float64(mjd))
    lon0 = lon_deg + 360 * (mjd - s.mjd[it])
    lon1 = lon_deg + 360 * (mjd - s.mjd[it+1])
    (1 - w) * _bilinear(s.tec, it, s.lat, s.lon, Float64(lat_deg), Float64(lon0)) +
    w * _bilinear(s.tec, it + 1, s.lat, s.lon, Float64(lat_deg), Float64(lon1))
end

"""
    vtec_rms(s::IonexSeries, mjd, lat_deg, lon_deg) -> Float64

The product's own formal VTEC uncertainty [TECU] at geocentric `(lat_deg, lon_deg)` [deg],
interpolated in time without the sun-fixed rotation (the RMS maps are smooth). `NaN` for products
that carry no RMS maps.
"""
function vtec_rms(s::IonexSeries, mjd::Real, lat_deg::Real, lon_deg::Real)
    it, w = _time_bracket(s.mjd, Float64(mjd))
    (1 - w) * _bilinear(s.rms, it, s.lat, s.lon, Float64(lat_deg), Float64(lon_deg)) +
    w * _bilinear(s.rms, it + 1, s.lat, s.lon, Float64(lat_deg), Float64(lon_deg))
end

function _time_bracket(t::Vector{Float64}, mjd::Float64)
    length(t) ≥ 2 || error("IONEX series has $(length(t)) maps, need at least 2 to interpolate")
    it = clamp(searchsortedfirst(t, mjd) - 1, 1, length(t) - 1)
    (it, clamp((mjd - t[it]) / (t[it+1] - t[it]), 0.0, 1.0))
end

"""
    pierce_point(xyz, az, el, height, base_radius) -> (lat_deg, lon_deg, el_geocentric_deg, range)

Ionospheric piercing point of the line of sight from ITRF `xyz` [m] towards geodetic azimuth `az`
and elevation `el` [rad], on the shell of radius `base_radius + height` [m].

Returned latitude/longitude are **geocentric degrees** because the straight line of sight is
intersected with a sphere in ECEF; `el_gc` [deg] is
the geocentric elevation the Petrov mapping needs, and `range` is the slant distance [m].
"""
function pierce_point(xyz::SVector{3,Float64}, az::Real, el::Real, height::Real, base_radius::Real)
    k = _los_unit_ecef(xyz, Float64(az), Float64(el))
    rn = norm(xyz)
    rk = dot(xyz, k)
    rs = base_radius + height
    s = -rk + sqrt(max(rk^2 - rn^2 + rs^2, 0.0))
    p = xyz + s * k
    pn = norm(p)
    (rad2deg(asin(p[3] / pn)), rad2deg(atan(p[2], p[1])), rad2deg(asin(clamp(rk / rn, -1, 1))), s)
end

function _los_unit_ecef(xyz::SVector{3,Float64}, az::Float64, el::Float64)
    lon, lat, _ = ERFA.gc2gd(ERFA.WGS84, xyz)
    sla, cla = sincos(lat)
    slo, clo = sincos(lon)
    saz, caz = sincos(az)
    sel, cel = sincos(el)
    e = cel * saz
    n = cel * caz
    u = sel
    SVector(-slo * e - sla * clo * n + cla * clo * u,
            clo * e - sla * slo * n + cla * slo * u,
            cla * n + sla * u)
end

"""
    iono_mapping_petrov(el_gc_deg, height, base_radius) -> Float64

Petrov (2023, AJ 165, 183) modified single-layer mapping function
`k / sqrt(1 − (R/(R + H + ΔH))² cos²(α·e_gc))` at the **geocentric** elevation, with ΔH = 56.7 km,
α = 0.9782, k = 0.85 — the mapping validated against a numerically integrated thick
shell, and the one the model uses.
"""
function iono_mapping_petrov(el_gc_deg::Real, height::Real, base_radius::Real)
    el = deg2rad(Float64(el_gc_deg)) * PETROV_ALPHA
    ratio = base_radius / (base_radius + height + PETROV_DH)
    PETROV_K / sqrt(1 - (ratio * cos(el))^2)
end

"""
    iono_mapping_slm(el_deg, height, base_radius) -> Float64

Standard single-layer mapping function at the geodetic elevation (how CODE builds its GIM). Kept
for comparison; the model uses [`iono_mapping_petrov`](@ref).
"""
function iono_mapping_slm(el_deg::Real, height::Real, base_radius::Real)
    ratio = base_radius / (base_radius + height)
    1 / sqrt(1 - (ratio * cos(deg2rad(Float64(el_deg))))^2)
end

"""
    tec_to_delay(stec_tecu, ν_eff) -> Float64

Ionospheric **group** delay [s] of a slant TEC [TECU] at the effective frequency `ν_eff` [Hz]:
`K/c · TEC / ν²`. At 15.3667 GHz (the group-delay effective frequency of this 512 MHz comb) one
TECU is 5.693 ps.
"""
tec_to_delay(stec_tecu::Real, ν_eff::Real) = K_ION / C_LIGHT_M_S * stec_tecu * TECU / ν_eff^2

"""
    group_delay_ν_eff(ν) -> Float64

The group-delay effective frequency [Hz] of a channel comb `ν` [Hz], uniformly weighted — the ν_eff
that [`tec_to_delay`](@ref) takes.

A group delay is the least-squares slope of phase against frequency, and the ionospheric phase goes
as 1/ν, so the delay a given TEC produces in a group-delay observable is the one that satisfies
`1/ν_eff² = −cov(ν, 1/ν)/var(ν)` over the channel set actually observed. It is therefore a property
of the COMB, not of the band centre: a different IF layout, a dropped IF or a different channel width
gives a different number, which is why it is derived from the data (`SessionFacts`) rather than
stated.

For a contiguous, uniformly weighted comb it has the closed form `ν₀(1 − 3B²/40ν₀²)`; the two agree
to better than 100 ppm exactly when the comb really is contiguous, which is what
`sessions/acquire/preflight.jl` prints side by side on a new file.
"""
function group_delay_ν_eff(ν)
    f = collect(Float64, ν)
    fb = f .- mean(f)
    sqrt(-mean(abs2, fb) / mean(fb .* (1 ./ f .- mean(1 ./ f))))
end

"""
    slant_iono_delay(s::IonexSeries, st::AntennaGeometry, mjd, az, el; ν_eff) -> (; delay, stec, vtec, mapping, lat, lon, el_gc)

Ionospheric group delay [s] for one (antenna, epoch, direction): piercing point on the product's own
shell, VTEC by rotating interpolation, Petrov mapping at the geocentric elevation, converted at
`ν_eff` [Hz].

`az`, `el` are the geodetic azimuth/elevation [rad] of the source — `geometric_delay` returns
exactly those.
"""
function slant_iono_delay(s::IonexSeries, st::AntennaGeometry, mjd::Real, az::Real, el::Real; ν_eff::Real)
    lat, lon, el_gc, _ = pierce_point(st.xyz, az, el, s.height, s.base_radius)
    v = vtec(s, mjd, lat, lon)
    m = iono_mapping_petrov(el_gc, s.height, s.base_radius)
    (delay = tec_to_delay(v * m, ν_eff), stec = v * m, vtec = v, mapping = m, lat, lon, el_gc)
end

"""
    iono_delays(products::NamedTuple, st::AntennaGeometry, mjd, az, el; ν_eff) -> NamedTuple

The slant ionospheric delay [s] of every product in `products` (e.g.
`(COD = …, ESA = …, UQR = …)` of `IonexSeries`) for one (antenna, epoch, direction). Type-stable:
the result is a `NamedTuple` with the same keys and `Float64` values.
"""
iono_delays(products::NamedTuple, st::AntennaGeometry, mjd::Real, az::Real, el::Real; ν_eff::Real) =
    map(s -> slant_iono_delay(s, st, mjd, az, el; ν_eff).delay, products)

"""
    iono_baseline_correction(d1::NamedTuple, d2::NamedTuple; primary::Symbol, inflation::Real) -> (; correction, σ)

The baseline ionospheric correction and its uncertainty from the per-antenna, per-product delays
`d1`, `d2` (as returned by [`iono_delays`](@ref)) of the two antennas of a baseline.

`correction = d2[primary] − d1[primary]` (the antenna order of the observable) and
`σ = inflation × std(d2 − d1 over products)` — the spread is formed on the **differential**, which
is what the observable sees; per-antenna spreads added in quadrature would double-count the large
common part. `inflation` is an explicit required configuration value.
"""
function iono_baseline_correction(d1::NamedTuple, d2::NamedTuple; primary::Symbol, inflation::Real)
    keys(d1) == keys(d2) || error("iono_baseline_correction: product sets differ, $(keys(d1)) vs $(keys(d2))")
    primary in keys(d1) || error("iono_baseline_correction: no product $primary in $(keys(d1))")
    diffs = map(-, values(d2), values(d1))
    length(diffs) ≥ 2 ||
        error("iono_baseline_correction: σ needs at least two products, got $(keys(d1))")
    (correction = d2[primary] - d1[primary], σ = inflation * std(diffs; corrected = true))
end
