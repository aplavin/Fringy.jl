
export catalogue_rows, catalogue_metadata, write_catalogue, uncertainty_model, jet_geometry

"""
    jet_geometry(model) -> (jet_pa_deg, jet_r_mas)

The model's own jet direction and extent: the position angle (N through E) and radius of the
flux-weighted centroid of every component OTHER than the brightest, about the brightest one.

Undefined — `NaN`, not zero — for a one-component model and for a set with no model at all: there is
no second component to point at. Being taken about the brightest COMPONENT rather than about the map
origin, it is independent of where the model's frame puts its zero.
"""
function jet_geometry(model)
    isnothing(model) && return (NaN, NaN)
    cs = collect(components(model))
    length(cs) > 1 || return (NaN, NaN)
    f = [flux(c) for c in cs]
    x = [coords(c)[1] for c in cs]
    y = [coords(c)[2] for c in cs]
    k = argmax(f)
    keep = [i for i in eachindex(f) if i != k]
    w = f[keep]
    sw = sum(w)
    sw == 0 && return (NaN, NaN)
    cx = sum(w .* (x[keep] .- x[k])) / sw
    cy = sum(w .* (y[keep] .- y[k])) / sw
    (rad2deg(atan(cx, cy)), hypot(cx, cy))
end

"""
    uncertainty_model(reg, srcmap; reference_model_set = reg.reference_model_set, s = 1.5) -> NamedTuple

The published uncertainty model, fitted once on the reference model set's positions against the external
reference catalogue and then carried by every model set's row.

σ_final² = (s·σ_formal)² + F², with `s` at the geodetic-practice 1.5 and `F` fitted per coordinate.
Reusing the reference model set's `F` for a structure-corrected position is deliberately conservative: the floor
partly absorbed the structure term that the correction removes.

Returns the model, the rigid rotation of the reference model set frame against the reference catalogue (reported,
never removed), and the scatter before and after removing it.
"""
function uncertainty_model(reg::Registration, srcmap; reference_model_set = reg.reference_model_set, s = 1.5)
    P = reg.sets[reference_model_set].P
    srcs = sort(collect(P.source))
    ix = [findfirst(==(x), P.source) for x in srcs]
    dα = Float64[]; dδ = Float64[]; σa = Float64[]; σd = Float64[]
    σea = Float64[]; σed = Float64[]
    for (n, x) in enumerate(srcs)
        α0, δ0 = reg.radec[x]
        δ1 = δ0 + P.Δδ_mas[ix[n]] * MAS
        α1 = α0 + P.Δα★_mas[ix[n]] * MAS / cos(δ0)
        r = rfc_row(srcmap, x)
        Δu = separation(SphericalOffsetFlat, r.coords, ICRSCoords(α1, δ1))
        Δ = U.value(Δu)
        Cext = U.uncertainty(Δu).cov
        push!(dα, Δ[1] / MAS)
        push!(dδ, Δ[2] / MAS)
        push!(σa, P.σ_Δα★_mas[ix[n]]); push!(σd, P.σ_Δδ_mas[ix[n]])
        push!(σea, sqrt(ustrip(u"mas^2", Cext[1, 1])))
        push!(σed, sqrt(ustrip(u"mas^2", Cext[2, 2])))
    end
    rf = rotation_fit(srcs, reg.radec, dα, dδ)
    ema = error_model_fixed(rf.res_α, σa, σea; s)
    emd = error_model_fixed(rf.res_δ, σd, σed; s)
    (; s = ema.s, F_ra_mas = ema.F, F_dec_mas = emd.F, chi2_ra = ema.chi2, chi2_dec = emd.chi2,
       rotation_mas = rf.rot, rms_before_mas = rf.rms_before, rms_after_mas = rf.rms_after,
       n_sources = length(srcs), reference_model_set)
end

"""
    catalogue_rows(cfg, reg, sets, srcmap, um; closure, image_files) -> Vector{NamedTuple}

The catalogue, sorted by `(b1950, model_set)` so that the file is independent of the order
`cfg.structure_models` produced the sets in — the tuple fixes the RUN's iteration order and nothing in
any product.

A row exists for a (source, set) pair iff that set SAYS SOMETHING about that source: every source for
the point set, which is the set with no components and therefore says nothing about all of them
uniformly, and the sources it carries a model for otherwise. A source with no external counterpart is
an ABSENT ROW, never a row of NaNs.
"""
function catalogue_rows(cfg, reg::Registration, sets::ModelSets, srcmap, um;
                        closure = Dictionary{Tuple{String, Symbol}, NamedTuple}(),
                        image_files = Dict{Tuple{String, Symbol}, String}())
    rows = NamedTuple[]
    for (label, R) in pairs(reg.sets)
        set = sets[label]
        nsel = _selected_counts(R, reg.obs)
        srcs = R.kind === :point ? reg.sources : sort([s for s in reg.sources if haskey(set.models, s)])
        for s in srcs
            k = findfirst(==(s), R.P.source)
            isnothing(k) && continue
            α0, δ0 = reg.radec[s]
            fitted = registered_position(reg, label, s)
            info = get(set.info, s, nothing)
            model = get(set.models, s, nothing)
            jet_pa, jet_r = jet_geometry(model)
            σfa = R.P.σ_Δα★_mas[k]; σfd = R.P.σ_Δδ_mas[k]
            haspeak = !isnothing(info) && haskey(info, :peak_x_mas)
            pk = haspeak ? absolute_position(SVector(info.peak_x_mas, info.peak_y_mas), fitted,
                                             SVector(0.0, 0.0)) : SVector(NaN, NaN)
            fs = findfirst(==(String(s)), R.sources.source)
            ch = get(closure, (label, s), (; chi2_phase = NaN, n_phase = 0,
                                             chi2_amp = NaN, n_amp = 0))
            r = rfc_row(srcmap, s)
            push!(rows, (;
                b1950 = String(s), model_set = label, model_set_kind = String(R.kind),
                jname = r.jname,
                ra_reference_deg = rad2deg(α0), dec_reference_deg = rad2deg(δ0),
                ra_deg = rad2deg(fitted[1]), dec_deg = rad2deg(fitted[2]),
                ra_sex = ra_sexagesimal(fitted[1]), dec_sex = dec_sexagesimal(fitted[2]),
                Δα★_mas = R.P.Δα★_mas[k], Δδ_mas = R.P.Δδ_mas[k],
                sigma_ra_formal_mas = σfa, sigma_dec_formal_mas = σfd,
                sigma_ra_final_mas = sqrt((um.s * σfa)^2 + um.F_ra_mas^2),
                sigma_dec_final_mas = sqrt((um.s * σfd)^2 + um.F_dec_mas^2),
                peak_ra_deg = rad2deg(pk[1]), peak_dec_deg = rad2deg(pk[2]),
                peak_x_mas = haspeak ? info.peak_x_mas : NaN,
                peak_y_mas = haspeak ? info.peak_y_mas : NaN,
                peak_from = isnothing(info) ? "" : String(get(info, :peak_from, :none)),
                n_components = isnothing(info) ? 0 : info.n_components,
                n_components_total = isnothing(info) ? 0 : info.n_components_total,
                n_negative = isnothing(info) ? 0 : info.n_negative,
                flux_model_jy = isnothing(info) ? NaN : info.flux_model_jy,
                flux_total_jy = isnothing(info) ? NaN : info.flux_total_jy,
                compactness = isnothing(info) ? NaN : info.compactness,
                r_max_mas = isnothing(info) ? NaN : info.r_max_mas,
                jet_pa_deg = jet_pa, jet_r_mas = jet_r,
                model_radius_mas = set.radius_mas,
                peak_jy_beam = isnothing(info) ? NaN : Float64(get(info, :peak, NaN)),
                residual_rms_mjy = isnothing(info) ? NaN : 1e3 * Float64(get(info, :residual_rms, NaN)),
                beam_maj_mas = isnothing(info) ? NaN : Float64(get(info, :beam_maj_mas, NaN)),
                beam_min_mas = isnothing(info) ? NaN : Float64(get(info, :beam_min_mas, NaN)),
                beam_pa_deg = isnothing(info) ? NaN : Float64(get(info, :beam_pa_deg, NaN)),
                image_npix = isnothing(info) ? 0 : Int(get(info, :npix, 0)),
                image_pixel_mas = isnothing(info) ? NaN : Float64(get(info, :pixel_mas, NaN)),
                tau_structure_median_ps = R.tau[s].median_ps, tau_structure_max_ps = R.tau[s].max_ps,
                n_wrap_excluded = R.tau[s].n_wrap,
                wrms_ps = isnothing(fs) ? NaN : R.sources.wrms_ps[fs],
                n_accepted = isnothing(fs) ? 0 : R.sources.n_accepted[fs],
                n_selected = get(nsel, s, 0),
                chi2_closure_phase = ch.chi2_phase, n_closure_phase = ch.n_phase,
                chi2_closure_amp = ch.chi2_amp, n_closure_amp = ch.n_amp,
                image_file = get(image_files, (label, s), "")))
        end
    end
    sort!(rows; by = r -> (r.b1950, r.model_set))
    identity.(rows)
end

"Rows of the observable table this set's solve SELECTED, per source (accepted or not)."
function _selected_counts(R::SetResult, obs)
    d = Dictionary{Symbol, Int}()
    for i in R.F.selected
        s = obs.source[i]
        haskey(d, s) ? (d[s] += 1) : insert!(d, s, 1)
    end
    d
end


"""
    ensemble_floors(reg, sets) -> (; pooled, per_set)

THE ENSEMBLE-COUPLING FLOOR, measured rather than adopted: a source whose own structure delay is
identically zero still MOVES between two registrations, because they share antenna clocks,
tropospheres and a weighting procedure re-run from scratch with every other source. Whatever those
sources move by is the part of every OTHER source's displacement that is not its own structure, and it
is the floor under every per-source difference this pipeline publishes.

Per set, and pooled over every non-reference set — the pooled number is the schema's, the per-set numbers
are what a comparison against one particular set is read against.
"""
function ensemble_floors(reg::Registration, sets::ModelSets; tau_max_ps = 0.05)
    reference = reg.sets[reg.reference_model_set]
    per_set = Dictionary{String, Float64}()
    pooled = Float64[]
    for (label, R) in pairs(reg.sets)
        label == reg.reference_model_set && continue
        v = Float64[]
        for s in reg.sources
            haskey(sets[label].models, s) || continue
            R.tau[s].max_ps < tau_max_ps || continue
            i = get(sets[label].info, s, nothing)
            (isnothing(i) || !haskey(i, :peak_x_mas)) && continue
            ka = findfirst(==(s), reference.P.source); k = findfirst(==(s), R.P.source)
            (isnothing(ka) || isnothing(k)) && continue
            push!(v, hypot(reference.P.Δα★_mas[ka] - (R.P.Δα★_mas[k] + i.peak_x_mas),
                           reference.P.Δδ_mas[ka] - (R.P.Δδ_mas[k] + i.peak_y_mas)))
        end
        isempty(v) || (insert!(per_set, label, median(v)); append!(pooled, v))
    end
    (; pooled = isempty(pooled) ? NaN : median(pooled), per_set)
end

"""
    catalogue_metadata(cfg, facts, reg, sets, um, rows; revision, rfc_provenance) -> Vector{Pair}

The ECSV `meta:` block as ordered `key => value` pairs, ready to be spelled as YAML. Every number here
is the value of a key: the ensemble floor and χ² margin that a reader needs to recompute
the dropped verdict column, the uncertainty model, the frame rotation, and one entry per model set
saying what that set IS. Prose appears only under `derived` and `descriptions`, and carries no digit.
"""
function catalogue_metadata(cfg, facts, reg::Registration, sets::ModelSets, um, rows; revision = "",
                            rfc_provenance = Pair{String, String}[])
    fl = ensemble_floors(reg, sets)
    setmeta = Pair{String, Any}[]
    for (label, R) in pairs(reg.sets)
        set = sets[label]
        p = set.provenance
        d = Pair{String, Any}["kind" => String(set.kind),
                              "n_models" => length(set.models),
                              "n_images" => length(set.images),
                              "n_rows" => count(r -> r.model_set == label, rows)]
        isfinite(set.radius_mas) && push!(d, "radius_mas" => set.radius_mas)
        haskey(fl.per_set, label) && push!(d, "ensemble_floor_mas" => fl.per_set[label])
        if set.kind === :external
            push!(d, "all_components" => get(p, :all_components, true),
                  "archive" => p.archive, "epoch" => p.epoch, "dir" => p.dir,
                  "model_pattern" => p.model_pattern, "n_missing" => p.n_missing)
        elseif set.kind === :own
            g = get(p, :grid, nothing)
            cl = get(p, :clean, nothing)
            push!(d, "all_components" => false,
                  "component_nsigma" => Float64(get(p, :significance_nsigma, cfg.component_nsigma)),
                  "clean_threshold_nsigma" => isnothing(cl) ? NaN : Float64(cl.threshold_nsigma),
                  "clean_gain" => isnothing(cl) ? NaN : Float64(cl.gain),
                  "weighting" => String(get(p, :weighting, :unknown)),
                  "hybrid_iterations" => Int(get(p, :iterations, 0)),
                  "amp_iterations" => Int(get(get(p, :ampcal, (;)), :iterations, 0)),
                  "npix" => isnothing(g) ? 0 : Int(g.npix),
                  "pixel_mas" => isnothing(g) ? NaN : Float64(g.pixel),
                  "n_capped" => Int(get(p, :n_capped, 0)))
        end
        push!(setmeta, label => d)
    end
    [
     "session" => facts.name,
     "date_obs" => string(facts.date),
     "epoch_obs" => facts.epoch_obs,
     "doy" => facts.doy,
     "nu_eff_hz" => facts.ν_eff,
     "band_span_hz" => maximum(facts.ν) - minimum(facts.ν) + facts.Δν,
     "n_channels" => length(facts.ν),
     "n_sources" => length(reg.sources),
     "n_model_sets" => length(reg.sets),
     "n_rows" => length(rows),
     "reference_model_set" => reg.reference_model_set,
     "captured_phase_mode" => String(cfg.captured_phase_mode),
     "snr_gate" => cfg.min_snr,
     "structure_wrap_rad" => cfg.structure_wrap,
     "model_radius_mas" => cfg.model_radius_mas,
     "component_nsigma" => cfg.component_nsigma,
     "closure_amp_min_snr" => cfg.closure_amp_min_snr,
     "chi2_margin" => cfg.chi2_margin,
     "ensemble_floor_mas" => fl.pooled,
     "frame_rotation_mas" => collect(um.rotation_mas),
     "frame_rotation_removed" => false,
     "frame_rotation_reference" => "AstrodataRFC",
     "rfc" => ["version" => cfg.rfc_version; rfc_provenance],
     "rms_vs_reference_mas" => collect(um.rms_before_mas),
     "rms_vs_reference_derotated_mas" => collect(um.rms_after_mas),
     "sigma_final" => ["s" => um.s, "F_ra_mas" => um.F_ra_mas, "F_dec_mas" => um.F_dec_mas,
                       "chi2_ra" => um.chi2_ra, "chi2_dec" => um.chi2_dec],
     "model_sets" => setmeta,
     "pipeline" => ["package" => "Fringy", "revision" => revision, "julia" => string(VERSION)],
     "derived" => DERIVED,
     "descriptions" => DESCRIPTIONS,
    ]
end

"""
    DERIVED

The quantities that are NOT columns, as formulas over the ones that are. Each is a difference between
rows of this table, which is why it is not a column; stating the formula here is what
makes the drop documented rather than silent. No formula carries a number — the two constants they
need are the metadata keys `ensemble_floor_mas` and `chi2_margin`.
"""
const DERIVED = [
    "bias" => "the reference model set's position for a source minus a set's peak_position for it: the displacement a solve with no structure model makes on that source",
    "peak_position" => "the peak_ra_deg and peak_dec_deg of a row, the model set's own brightness peak looked up in its own registered image",
    "registration_systematic" => "spread of the position over the model sets whose kind is own: the imaging systematic of the registration",
    "delta" => "one model set's peak_position for a source minus another set's: the disagreement of two sets on the same physical feature",
    "resolvable" => "delta exceeds the larger of the registration systematic and the ensemble floor of the set being compared, which is stated beside that set under model_sets where it is measurable",
    "preferred_model_set" => "where resolvable and the ratio of chi2_closure_phase beats chi2_margin, the set with the smaller value; otherwise neither",
]

"""
    DESCRIPTIONS

One sentence per column, for a reader who opens the file and nothing else. Prose only: a description
that carried a number could go stale against the data beside it, and this file is generated from the
same values it describes.
"""
const DESCRIPTIONS = [
    "b1950" => "source name as the observing schedule spells it",
    "model_set" => "the set of brightness models this registration used; a data value, never a column name",
    "model_set_kind" => "point for the set with no components, own for this pipeline's own imaging, external for models read from files",
    "jname" => "the external reference catalogue's name for the same source",
    "ra_reference_deg" => "the a priori position the offsets are relative to",
    "ra_deg" => "the registered absolute position of THIS model set's own frame origin: an absolute WCS for the image behind it",
    "Δα★_mas" => "the offset the solve returned, East and North of the a priori position",
    "sigma_ra_formal_mas" => "the solve's own formal error",
    "sigma_ra_final_mas" => "the published uncertainty, from the model stated under sigma_final in this header",
    "peak_ra_deg" => "the brightness peak of this set's image, looked up in that WCS: an analysis quantity, not a second registration",
    "peak_from" => "whether the peak is the brightest pixel of an image or the brightest component of a model",
    "n_components" => "components that entered the structure delay",
    "n_components_total" => "components the model carries before any significance cut",
    "compactness" => "the brightest component's share of the model flux",
    "r_max_mas" => "the outermost component's distance from the peak",
    "jet_pa_deg" => "position angle, North through East, of the flux-weighted centroid of every component but the brightest, about the brightest",
    "model_radius_mas" => "the own-image CLEAN support radius of this set, infinite where none applies",
    "peak_jy_beam" => "the image's brightest pixel",
    "residual_rms_mjy" => "residual noise of the image the model was CLEANed from",
    "beam_maj_mas" => "restoring beam of that image: the Gaussian with the uv coverage's own weighted second moments",
    "image_npix" => "pixels per side of the image this set's model came with",
    "tau_structure_median_ps" => "the structure delay this model removed from the observables of this source",
    "n_wrap_excluded" => "observables the unwrap guard dropped from this set's solve",
    "wrms_ps" => "post-fit weighted rms of this source in this set's own solve",
    "n_accepted" => "observables of this source the solve accepted",
    "n_selected" => "observables of this source the solve considered before rejection",
    "chi2_closure_phase" => "closure-phase agreement of this set's model with the averaged calibrated visibilities: gain-invariant, so no self-calibration can flatter it",
    "chi2_closure_amp" => "the same for closure amplitude, over the quadrangles above the stated signal-to-noise floor",
    "image_file" => "the registered image this row is the WCS of, relative to this directory",
]


const CATALOGUE_UNITS = Dict(
    :ra_reference_deg => "deg", :dec_reference_deg => "deg", :ra_deg => "deg", :dec_deg => "deg",
    :peak_ra_deg => "deg", :peak_dec_deg => "deg", :peak_x_mas => "mas", :peak_y_mas => "mas",
    :Δα★_mas => "mas", :Δδ_mas => "mas",
    :sigma_ra_formal_mas => "mas", :sigma_dec_formal_mas => "mas",
    :sigma_ra_final_mas => "mas", :sigma_dec_final_mas => "mas",
    :flux_model_jy => "Jy", :flux_total_jy => "Jy", :r_max_mas => "mas", :jet_pa_deg => "deg",
    :jet_r_mas => "mas", :model_radius_mas => "mas", :peak_jy_beam => "Jy/beam",
    :residual_rms_mjy => "mJy/beam", :beam_maj_mas => "mas", :beam_min_mas => "mas",
    :beam_pa_deg => "deg", :image_pixel_mas => "mas",
    :tau_structure_median_ps => "ps", :tau_structure_max_ps => "ps", :wrms_ps => "ps")

_yaml_scalar(v::AbstractString) = occursin(r"^[A-Za-z_][A-Za-z0-9_:@.+-]*$", v) ? v : "'" * replace(v, "'" => "''") * "'"
_yaml_scalar(v::Bool) = v ? "true" : "false"
_yaml_scalar(v::Integer) = string(v)
_yaml_scalar(v::AbstractFloat) = isfinite(v) ? string(v) : (isnan(v) ? ".nan" : v > 0 ? ".inf" : "-.inf")
_yaml_scalar(v::AbstractVector) = "[" * join(_yaml_scalar.(v), ", ") * "]"
_yaml_scalar(v) = _yaml_scalar(string(v))

_yaml_key(k) = occursin(r"^[A-Za-z_][A-Za-z0-9_.-]*$", string(k)) ? string(k) :
               "'" * replace(string(k), "'" => "''") * "'"

"Write a `key => value` list as an ECSV comment-block YAML mapping, nested to any depth."
function _write_meta(io, pairs, indent)
    for (k, v) in pairs
        if v isa AbstractVector && !isempty(v) && first(v) isa Pair
            println(io, "# ", " "^indent, _yaml_key(k), ":")
            _write_meta(io, v, indent + 2)
        else
            println(io, "# ", " "^indent, _yaml_key(k), ": ", _yaml_scalar(v))
        end
    end
end

"""
    write_catalogue(dir, rows, meta) -> Vector{String}

`catalogue.ecsv`, `.csv` and `.md` — the same table three ways. The ECSV carries the column types,
their units and the metadata block; the CSV is the same rows for a reader that wants no header; the
markdown is the human summary.
"""
function write_catalogue(dir, rows, meta)
    isempty(rows) && error("write_catalogue: no rows")
    ks = keys(first(rows))
    ecsv = joinpath(dir, "catalogue.ecsv")
    open(ecsv, "w") do io
        println(io, "# %ECSV 1.0")
        println(io, "# ---")
        println(io, "# datatype:")
        for k in ks
            v = getproperty(first(rows), k)
            T = v isa AbstractString ? "string" : v isa Integer ? "int64" : "float64"
            println(io, "# - {name: $k, datatype: $T",
                    haskey(CATALOGUE_UNITS, k) ? ", unit: $(CATALOGUE_UNITS[k])}" : "}")
        end
        println(io, "# meta:")
        _write_meta(io, meta, 2)
        println(io, "# schema: astropy-2.0")
        println(io, join(string.(ks), " "))
        for r in rows
            println(io, join((x isa AbstractString ? (isempty(x) ? "\"\"" : replace(x, " " => "_")) :
                              _fmt(x) for x in (getproperty(r, k) for k in ks)), " "))
        end
    end
    csv = write_table(joinpath(dir, "catalogue.csv"), rows)
    md = joinpath(dir, "catalogue.md")
    open(md, "w") do io
        println(io, "| B1950 | name | model set | RA | Dec | σ α★ | σ δ | N cmp | τ max [ps] | wrms [ps] | χ² φ |")
        println(io, "|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows
            @printf(io, "| %s | %s | %s | %s | %s | %.3f | %.3f | %d | %.2f | %.2f | %s |\n",
                    r.b1950, r.jname, r.model_set, r.ra_sex, r.dec_sex,
                    r.sigma_ra_final_mas, r.sigma_dec_final_mas, r.n_components,
                    r.tau_structure_max_ps, r.wrms_ps,
                    isfinite(r.chi2_closure_phase) ? @sprintf("%.2f", r.chi2_closure_phase) : "—")
        end
    end
    [ecsv, csv, md]
end
