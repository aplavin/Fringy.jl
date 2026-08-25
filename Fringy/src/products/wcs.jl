
export wcs_manifest

"""
    wcs_manifest(results, cfg, facts, rows, sets; verbose = true) -> (manifest, files)

Write every registered image this run's model sets support, and return the manifest rows plus a
`(model_set, source) ⇒ relative path` map for the catalogue's `image_file` column.

`rows` are the catalogue rows: the registered position each image is to carry and the published
uncertainty its header states are columns of them, so this reads the table rather than
recomputing anything.

| column | what |
|---|---|
| `crval1_deg`, `crval2_deg` | the registered absolute position written into the file |
| `crval1_input_deg`, `crval2_input_deg` | what the source file carried — the phase-centre label of a published image; NaN for a map we wrote from nothing |
| `shift_mas` | the distance between them, where both exist |
| `input_sha256` | the file the copy was made from, by content |
"""
function wcs_manifest(results, cfg, facts, rows, sets::ModelSets; verbose = true)
    session = lowercase(facts.name)
    date_obs = string(facts.date)
    manifest = NamedTuple[]
    files = Dict{Tuple{String, Symbol}, String}()
    for r in rows
        label = r.model_set
        set = sets[label]
        s = Symbol(r.b1950)
        haskey(set.images, s) || continue
        slug = model_set_slug(label)
        dir = joinpath(results, "images", slug)
        mkpath(dir)
        rel = joinpath("images", slug, "$(r.b1950).fits")
        path = joinpath(results, rel)
        crval = (r.ra_deg, r.dec_deg)
        sigma = (r.sigma_ra_final_mas, r.sigma_dec_final_mas)
        img = set.images[s]
        if set.kind === :own
            write_registered_image(path, s, img, crval, sigma; model_set = label, session, date_obs)
            input = (NaN, NaN); sha = ""; npix = img.grid.npix; pixel = img.grid.pixel
        else
            input = write_registered_copy(path, img.path, crval, sigma;
                                          model_set = label, session, date_obs)
            sha = String(get(img.provenance, :sha256, ""))
            npix = img.npix; pixel = img.pixel_mas
        end
        shift = all(isfinite, input) ?
                hypot((crval[1] - input[1]) * cosd(crval[2]), crval[2] - input[2]) * MAS_PER_DEG : NaN
        files[(label, s)] = rel
        push!(manifest, (; b1950 = r.b1950, model_set = label, model_set_kind = r.model_set_kind,
                           file = rel, crval1_deg = crval[1], crval2_deg = crval[2],
                           crval1_input_deg = input[1], crval2_input_deg = input[2],
                           shift_mas = shift, npix, pixel_mas = pixel, input_sha256 = sha))
    end
    if verbose
        for l in unique(m.model_set for m in manifest)
            v = [m.shift_mas for m in manifest if m.model_set == l && isfinite(m.shift_mas)]
            @printf("  %-34s %3d images%s\n", l, count(m -> m.model_set == l, manifest),
                    isempty(v) ? "" :
                    @sprintf(", registered CRVAL vs the published label: median %.3f mas, max %.3f",
                             median(v), maximum(v)))
        end
    end
    (identity.(manifest), files)
end
