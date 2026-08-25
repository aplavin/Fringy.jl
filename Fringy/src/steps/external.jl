
export brightest_pixel, file_provenance, external_import, external_model_sets

"""
    file_provenance(path) -> (; path, bytes, mtime, sha256)

What identifies the bytes a product was computed from. Read once, hashed once: the file is small
(a `.mod` is kilobytes, a restored map a few MB) and the record travels with the model set into every
product header.
"""
function file_provenance(path)
    bytes = read(path)
    (; path = abspath(path), bytes = length(bytes), mtime = unix2datetime(mtime(path)),
       sha256 = bytes2hex(SHA.sha256(bytes)))
end

"""
    brightest_pixel(A, xs, ys) -> (; xy, value, index)

The brightest pixel of a map. It is distinct from the model origin, i.e. the file's reference pixel,
and the registration's deliverable is an absolute
WCS for that image. The peak is the brightness statistic the bias comparison and the gallery arrows
are computed from: an analysis lookup in the registered image.

`argmax` returns the first maximal element in Julia's column-major scan order, so exactly-equal pixels
resolve deterministically on a committed file.
"""
function brightest_pixel(A::AbstractMatrix{Float64}, xs, ys)
    I = argmax(A)
    (; xy = SVector(xs[I[1]], ys[I[2]]), value = A[I], index = I)
end

"""
    external_import(m::ExternalModels, src; date = nothing, verbose = true)
        -> nothing | NamedTuple

One source's external release: the full published component model in its own frame, its brightness
statistics, and the provenance of every file read.

Returns `nothing` when the set holds no usable model for `src` — an absent release, or a model file
with no components. That is a normal outcome, not an error: `StructureModels` gives an absent source
`τ_structure = 0` and the registration simply has no row for it in this set.

When `date` is supplied, the image header's `DATE-OBS` is checked against the session date. A missing,
unparsable, or different date emits a warning but does not alter the imported model.
"""
function external_import(m::ExternalModels, src; date = nothing, verbose = true)
    name = String(Symbol(src))
    modpath = model_path(m, name)
    if !_nonempty_file(modpath)
        verbose && @printf("  MISSING  %-10s no usable model file %s\n", name, modpath)
        return nothing
    end
    haveimage = !isnothing(m.image_pattern)
    imgpath = haveimage ? image_path(m, name) : nothing
    if haveimage && !_nonempty_file(imgpath)
        verbose && @printf("  MISSING  %-10s no usable image beside the model (%s)\n", name, imgpath)
        return nothing
    end

    model = ustrip(VLBIFiles.load(MultiComponentModel, modpath))
    cs = collect(components(model))
    if isempty(cs)
        verbose && @printf("  MISSING  %-10s %s has no components\n", name, modpath)
        return nothing
    end

    if haveimage
        fits = VLBIFiles.load(imgpath)
        ks = VLBIFiles.axiskeys(fits.data)
        xs = Float64.(ustrip.(ks[1]))
        ys = Float64.(ustrip.(ks[2]))
        A = Float64.(collect(fits.data))
        pk = brightest_pixel(A, xs, ys)
        peak_xy, peak, peak_from = pk.xy, pk.value, :pixel
        pixel_mas = abs(xs[2] - xs[1])
        npix = size(A, 1)
        crpix = (get(fits.header, "CRPIX1", NaN), get(fits.header, "CRPIX2", NaN))
        crval = (get(fits.header, "CRVAL1", NaN), get(fits.header, "CRVAL2", NaN))
        _check_date_obs(fits.header, date; source = Symbol(name), image = imgpath)
    else
        k = argmax(i -> flux(cs[i]), eachindex(cs))
        peak_xy, peak, peak_from = SVector{2, Float64}(coords(cs[k])), flux(cs[k]), :component
        pixel_mas, npix, crpix, crval = NaN, 0, (NaN, NaN), (NaN, NaN)
    end

    r = map(c -> hypot((coords(c) .- peak_xy)...), cs)
    prov = (; model = file_provenance(modpath),
              image = haveimage ? file_provenance(imgpath) : nothing)
    (; source = Symbol(name), model,
       peak_xy, peak, peak_from, pixel_mas, npix, path = modpath, image_path = imgpath,
       crpix, crval,
       n_components = length(cs), n_negative = count(c -> flux(c) < 0, cs),
       flux_total = sum(flux, cs; init = 0.0),
       compactness = (t = sum(flux, cs; init = 0.0); t == 0 ? NaN : maximum(flux, cs) / t),
       r_max = isempty(r) ? 0.0 : maximum(r),
       provenance = prov)
end

function _check_date_obs(header, date; source, image)
    isnothing(date) && return nothing
    raw = strip(string(get(header, "DATE-OBS", "")))
    if isempty(raw)
        @warn "external image DATE-OBS is missing" source image session_date = Date(date)
        return nothing
    end
    image_date = tryparse(Date, first(raw, 10))
    if isnothing(image_date)
        @warn "external image DATE-OBS is unparsable" source image date_obs = raw session_date = Date(date)
    elseif image_date != Date(date)
        @warn "external image DATE-OBS differs from the session date" source image date_obs = image_date session_date = Date(date)
    end
    nothing
end

"""
    external_set_id(m::ExternalModels) -> String

The set's identifier inside a model-set label: `"<archive>_<epoch>"` when both are stated, otherwise
the model directory's own name. It is a DATA VALUE — it appears in a label, a catalogue cell and a
directory name, and nowhere in the code.
"""
external_set_id(m::ExternalModels) =
    (isempty(m.archive) || isempty(m.epoch)) ? basename(rstrip(m.dir, '/')) : "$(m.archive)_$(m.epoch)"

"""
    external_model_sets(m::ExternalModels, sources; date = nothing, verbose = true)
        -> Dictionary{String, ModelSet}

The single full published model set one `ExternalModels` member produces. All components enter the
structure model without being moved or filtered.

Sources with no usable model are simply absent from the set;
`sources` is the session's own source list, so the absence is per source and is reported by the
returned set's `n_missing`.
"""
function external_model_sets(m::ExternalModels, sources; date = nothing, verbose = true)
    id = external_set_id(m)
    full = Dictionary{Symbol, Any}()
    images = Dictionary{Symbol, Any}()
    info = Dictionary{Symbol, NamedTuple}()
    files = Dictionary{Symbol, NamedTuple}()
    missing_sources = Symbol[]
    for s in sources
        rec = external_import(m, s; date, verbose)
        if isnothing(rec)
            push!(missing_sources, Symbol(s))
            continue
        end
        set!(full, rec.source, rec.model)
        set!(files, rec.source, rec.provenance)
        isnothing(rec.image_path) || set!(images, rec.source,
            (; path = rec.image_path, rec.npix, rec.pixel_mas, rec.crpix, rec.crval,
               provenance = rec.provenance.image))
        set!(info, rec.source,
             (; n_components = rec.n_components, n_components_total = rec.n_components,
                n_negative = rec.n_negative, flux_model_jy = rec.flux_total,
                flux_total_jy = rec.flux_total, compactness = rec.compactness,
                r_max_mas = rec.r_max, peak_x_mas = rec.peak_xy[1], peak_y_mas = rec.peak_xy[2],
                peak = rec.peak, peak_from = rec.peak_from,
                npix = rec.npix, pixel_mas = rec.pixel_mas, crpix = rec.crpix, crval = rec.crval))
    end
    label = "external:$id"
    model_set = ModelSet(label, :external, Inf, full, images, info,
                         (; kind = :external, archive = m.archive, epoch = m.epoch, dir = m.dir,
                            model_pattern = m.model_pattern, image_pattern = m.image_pattern,
                            note = m.note, all_components = true,
                            n_missing = length(missing_sources), missing_sources, files))
    ModelSets([label], [model_set])
end
