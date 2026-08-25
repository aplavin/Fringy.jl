
export write_table, ra_sexagesimal, dec_sexagesimal, read_source_names, validate_source_names,
       rfc_reference_map, rfc_row,
       rotation_fit, error_model, error_model_fixed, pyfloat


_fmt(x::AbstractFloat) = isfinite(x) ? string(float(x)) : (isnan(x) ? "nan" : x > 0 ? "inf" : "-inf")
_fmt(x::Integer) = string(x)
_fmt(x) = (s = string(x); occursin(',', s) || occursin('"', s) ? '"' * replace(s, '"' => "\"\"") * '"' : s)

"""
    write_table(path, rows::AbstractVector{<:NamedTuple}) -> path

Write a CSV whose header is the first row's keys. Rounding belongs in the presentation, not in the
data, so a float is written as the shortest decimal that round-trips to it — the value, exactly — and
the file is the interface every consumer reads.

An empty table is an error: a product with no rows is a run that produced nothing, and writing a bare
header would hide it.
"""
function write_table(path, rows::AbstractVector)
    isempty(rows) && error("write_table: nothing to write to $path")
    ks = keys(first(rows))
    open(path, "w") do io
        println(io, join(string.(ks), ","))
        for r in rows
            println(io, join((_fmt(getproperty(r, k)) for k in ks), ","))
        end
    end
    path
end


"Right ascension [rad] as sexagesimal hh mm ss.ssssss."
function ra_sexagesimal(ra::Real)
    h = mod(rad2deg(ra), 360) / 15
    hh = floor(Int, h); m = (h - hh) * 60
    mm = floor(Int, m); s = (m - mm) * 60
    @sprintf("%02d %02d %09.6f", hh, mm, s)
end

"Declination [rad] as sexagesimal ±dd mm ss.sssss."
function dec_sexagesimal(dec::Real)
    d = rad2deg(dec)
    sgn = d < 0 ? "-" : "+"
    d = abs(d)
    dd = floor(Int, d); m = (d - dd) * 60
    mm = floor(Int, m); s = (m - mm) * 60
    @sprintf("%s%02d %02d %08.5f", sgn, dd, mm, s)
end

"""
    pyfloat(x) -> String

`str(x)` of CPython 3: the SHORTEST decimal that round-trips to `x`, laid out fixed or scientific by
CPython's rule (scientific iff the decimal point falls at or before position −3, or past position 16).

It exists for one reason: the registered FITS headers already shipped, written by `astropy.io.fits`, and
the port is accepted on reproducing them card for card — so the card's float text is part of the
specification and not a writer's private business. Julia's own `string(::Float64)` produces the same
shortest-round-trip digit string (there is only one shortest correctly-rounding decimal), so this
takes Julia's digits and re-lays them out rather than re-deriving them.
"""
function pyfloat(x::Real)
    v = Float64(x)
    isnan(v) && return "nan"
    isinf(v) && return v > 0 ? "inf" : "-inf"
    neg = signbit(v)
    v = abs(v)
    v == 0 && return neg ? "-0.0" : "0.0"
    s = string(v)
    i = findfirst('e', s)
    mant = isnothing(i) ? s : s[1:i-1]
    ex = isnothing(i) ? 0 : parse(Int, s[i+1:end])
    j = findfirst('.', mant)::Int
    digits = mant[1:j-1] * mant[j+1:end]
    decpt = (j - 1) + ex
    k = 1
    while k < length(digits) && digits[k] == '0'
        k += 1; decpt -= 1
    end
    digits = digits[k:end]
    n = length(digits)
    while n > 1 && digits[n] == '0'
        n -= 1
    end
    digits = digits[1:n]
    body = if decpt <= -4 || decpt > 16
        m = n == 1 ? digits : digits[1:1] * "." * digits[2:end]
        e = decpt - 1
        @sprintf("%se%s%02d", m, e < 0 ? "-" : "+", abs(e))
    elseif decpt <= 0
        "0." * "0"^(-decpt) * digits
    elseif decpt >= n
        digits * "0"^(decpt - n) * ".0"
    else
        digits[1:decpt] * "." * digits[decpt+1:end]
    end
    neg ? "-" * body : body
end


"""
    read_source_names(path) -> Dictionary

Read the application's canonical source-name table. RFC positions and uncertainties always come from
the application-supplied RFC catalogue; this table stores identity only.

The TSV header is exactly `source_name<TAB>rfc_jname`, followed by one source and one RFC J2000 name
per row. Source names must be unique and sorted; several aliases may name the same RFC identity.
"""
function read_source_names(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && error("read_source_names: empty file $path")
    chomp(first(lines)) == "source_name\trfc_jname" ||
        error("read_source_names: expected header source_name\\trfc_jname in $path")
    out = Dictionary{Symbol, String}()
    source_names = String[]
    for (offset, line) in enumerate(Iterators.drop(lines, 1))
        lineno = offset + 1
        isempty(line) && error("read_source_names: blank row $lineno in $path")
        fields = split(chomp(line), '\t'; keepempty = true)
        length(fields) == 2 || error("read_source_names: expected two TSV fields at $path:$lineno")
        source, jname = fields
        occursin(r"^\S+$", source) ||
            error("read_source_names: invalid source name at $path:$lineno")
        occursin(r"^J\d{4}[+-]\d{3}[0-9A-Z]$", jname) ||
            error("read_source_names: invalid RFC J2000 name $jname at $path:$lineno")
        source_name = Symbol(source)
        haskey(out, source_name) && error("read_source_names: duplicate source $source at $path:$lineno")
        insert!(out, source_name, jname)
        push!(source_names, source)
    end
    isempty(out) && error("read_source_names: no source identities in $path")
    issorted(source_names) || error("read_source_names: source names are not sorted in $path")
    out
end

"""
    validate_source_names(source_names, sources) -> nothing
    validate_source_names(path, sources) -> nothing

Require every source in one session to have exactly one canonical identity. Aliases may share an RFC
identity globally, but two aliases for the same identity may not occur in one session.
"""
function validate_source_names(source_names, sources)
    _selected_source_names(source_names, sources)
    nothing
end

validate_source_names(path::AbstractString, sources) =
    validate_source_names(read_source_names(path), sources)

"""
    rfc_reference_map(source_names, catalogue; sources) -> Dictionary

Resolve canonical names against an RFC `catalogue` — any iterable of rows carrying `names.J2000`
and an uncertain `coords` value. Each returned row retains the catalogue's `coords` intact,
including its full correlated tangent-plane covariance.
No RFC text columns or uncertainty convention are interpreted here.

Every selected J2000 name must occur exactly once in the catalogue. Missing or duplicate catalogue
identities are fatal because silently substituting or dropping an external reference changes the
session-wide uncertainty floor.
"""
function _selected_source_names(source_names, sources)
    selected = collect(sources)
    length(unique(selected)) == length(selected) || error("rfc_reference_map: duplicate requested source")
    selected_jnames = Set{String}()
    for source in selected
        haskey(source_names, source) ||
            error("rfc_reference_map: source $source is absent from the canonical source-name table")
        jname = source_names[source]
        jname in selected_jnames &&
            error("rfc_reference_map: multiple requested sources resolve to RFC J2000 name $jname")
        push!(selected_jnames, jname)
    end
    selected
end

function rfc_reference_map(source_names, catalogue; sources = keys(source_names))
    selected = _selected_source_names(source_names, sources)
    byname = Dict{String, eltype(catalogue)}()
    for r in catalogue
        name = r.names.J2000
        haskey(byname, name) && error("rfc_reference_map: duplicate RFC J2000 name $name")
        byname[name] = r
    end

    out = Dictionary{Symbol, NamedTuple}()
    for source in selected
        jname = source_names[source]
        haskey(byname, jname) ||
            error("rfc_reference_map: $jname, mapped from $source, is absent from the RFC catalogue")
        set!(out, source, (; jname, coords = byname[jname].coords))
    end
    out
end

"""
    rfc_reference_map(path, catalogue; sources) -> Dictionary

Resolve canonical source identities from the source-name table at `path` against `catalogue`.
The catalogue is supplied by the application — e.g. `AstrodataRFC.table(AstrodataRFC.RFC(version))`;
this package does not construct it.
"""
function rfc_reference_map(path::AbstractString, catalogue; sources = nothing)
    source_names = read_source_names(path)
    rfc_reference_map(source_names, catalogue; sources = something(sources, keys(source_names)))
end

"""
    rfc_row(map, s) -> NamedTuple

`map[s]`, requiring the map to carry the source. Acquisition refuses unresolved matches, so accepting
an absent identity here would invalidate the shared uncertainty-floor fit rather than degrade one
optional output column.
"""
function rfc_row(map, s)
    haskey(map, s) || error("rfc_row: $s is absent from the canonical source-name table")
    map[s]
end


"""
    rotation_fit(sources, radec, dα, dδ) -> (; rot, res_α, res_δ, rms_before, rms_after)

Least-squares rigid rotation [mas] of a set of position offsets [mas], and the residuals
after removing it — the standard way of comparing two celestial frames tied by different EOP/TRF. For
a single session with EOP and the TRF held fixed, such a rotation is the expected difference against
an external catalogue, and it is REPORTED rather than removed.
"""
function rotation_fit(sources, radec, dα, dδ)
    n = length(sources)
    A = zeros(2n, 3); y = zeros(2n)
    for k in 1:n
        α, δ = radec[sources[k]]
        A[2k-1, :] = [-sin(δ)cos(α), -sin(δ)sin(α), cos(δ)]
        A[2k, :] = [sin(α), -cos(α), 0.0]
        y[2k-1] = dα[k]; y[2k] = dδ[k]
    end
    rot = A \ y
    res = y - A * rot
    (; rot, res_α = res[1:2:end], res_δ = res[2:2:end],
       rms_before = (sqrt(mean(abs2, dα)), sqrt(mean(abs2, dδ))),
       rms_after = (sqrt(mean(abs2, res[1:2:end])), sqrt(mean(abs2, res[2:2:end]))))
end

"""
    error_model(d, σf, σext; sgrid, fmax, ndof_lost) -> (; s, F, chi2)

Maximum-likelihood fit of the uncertainty model σ_final² = (s·σ_formal)² + F² to the residuals `d`
(per source, one coordinate) against known external variances `σext²`. The scale `s` is bounded below
by 1 — a formal error is a lower bound on the truth, and a fit that "improves" it is fitting the
tens-of-sources noise, not the error model — and with a few dozen sources spanning a narrow range of
σ_formal the (s, F) pair is only weakly separable, which is why [`error_model_fixed`](@ref) (s pinned
at the geodetic-practice 1.5) is the adopted form and this is quoted beside it.
"""
function error_model(d, σf, σext; sgrid = 1.0:0.02:12.0, fmax = 1.5, ndof_lost = 3)
    best = (; s = NaN, F = NaN, nll = Inf)
    for s in sgrid, Fm in 0.0:0.005:fmax
        V = (s .* σf) .^ 2 .+ Fm^2 .+ σext .^ 2
        nll = sum(d .^ 2 ./ V .+ log.(V))
        nll < best.nll && (best = (; s, F = Fm, nll))
    end
    V = (best.s .* σf) .^ 2 .+ best.F^2 .+ σext .^ 2
    (; best.s, best.F, chi2 = sum(d .^ 2 ./ V) / (length(d) - ndof_lost))
end

"The same with `s` held fixed (only the floor `F` is fitted) — the adopted form."
error_model_fixed(d, σf, σext; s = 1.5, kwargs...) =
    error_model(d, σf, σext; sgrid = s:s, kwargs...)
