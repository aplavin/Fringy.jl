
export products

"""
    products(cfg, run, results; source_names_path, rfc_catalogue, rfc_provenance, verbose = true) -> Vector{String}

Write this session's deliverables into `results` and return the paths.

`run` carries the values the runner has: `facts`, the `registration`, the model `sets`, and — when the
session produced them — `calibration`, [`calibrate_and_average`](@ref)'s result. Nothing else is needed, and nothing here
reaches for anything else.

`rfc_catalogue` is the resolved RFC catalogue table for `cfg.rfc_version` — rows carrying
`names.J2000` and an uncertain `coords` value. The application constructs it (e.g.
`AstrodataRFC.table(AstrodataRFC.RFC(cfg.rfc_version))`); this package takes any table of that shape.
`rfc_provenance` is the key–value pairs recorded beside `cfg.rfc_version` in the catalogue metadata
and the manifest's `[inputs.rfc]` — the application states where its catalogue came from.
"""
function products(cfg, run, results; source_names_path, rfc_catalogue,
                  rfc_provenance = Pair{String, String}[], verbose = true, revision = "")
    reg = run.registration
    sets = run.sets
    facts = run.facts
    mkpath(results)
    paths = String[]

    srcmap = rfc_reference_map(source_names_path, rfc_catalogue; sources = reg.sources)
    um = uncertainty_model(reg, srcmap)
    verbose && @printf("uncertainty model on %d %s positions: s = %.1f, F = %.3f / %.3f mas (χ²/dof %.2f / %.2f)\n",
                       um.n_sources, um.reference_model_set, um.s, um.F_ra_mas, um.F_dec_mas,
                       um.chi2_ra, um.chi2_dec)

    closure = Dictionary{Tuple{String, Symbol}, NamedTuple}()
    if :closure in cfg.products
        if isnothing(run.calibration)
            @printf("products: :closure was requested but this run has no averaged visibilities — the closure columns are absent, not zero\n")
        else
            verbose && println("closure evaluation, on the averaged calibrated data of every model set:")
            closure = closure_evaluation(cfg, run.calibration, sets, reg.sources; verbose)
        end
    end

    rows = catalogue_rows(cfg, reg, sets, srcmap, um; closure)

    manifest = NamedTuple[]
    if :wcs in cfg.products
        verbose && println("registered images:")
        manifest, files = wcs_manifest(results, cfg, facts, rows, sets; verbose)
        rows = [merge(r, (; image_file = get(files, (r.model_set, Symbol(r.b1950)), ""))) for r in rows]
        isempty(manifest) ||
            push!(paths, write_table(joinpath(results, "wcs_manifest.csv"), manifest))
    end

    if :catalogue in cfg.products
        meta = catalogue_metadata(cfg, facts, reg, sets, um, rows; revision, rfc_provenance)
        append!(paths, write_catalogue(results, rows, meta))
    end
    if :fit_tables in cfg.products
        push!(paths, write_table(joinpath(results, "fit_summary.csv"), fit_summary_rows(reg)))
        push!(paths, write_table(joinpath(results, "fit_baselines.csv"), fit_baseline_rows(reg)))
    end
    if :report in cfg.products
        push!(paths, write_report(joinpath(results, "report.md"), cfg, facts, reg, sets, rows, um,
                                  ensemble_floors(reg, sets); manifest))
    end
    :figures in cfg.products && verbose &&
        println("figures are not drawn here: they need a plotting stack this package must not depend " *
                "on, and they live beside the session files.\n" *
                "  draw them with `sessions/figures/make_figures.jl`")

    written = [paths; [joinpath(results, m.file) for m in manifest]]

    if :manifest in cfg.products
        m = write_manifest(joinpath(results, "manifest.toml"), cfg, facts, sets, written;
                           source_names_path,
                           rfc_provenance,
                           revision,
                           extra = (; n_sources = length(reg.sources),
                                      n_model_sets = length(reg.sets),
                                      n_catalogue_rows = length(rows)))
        push!(paths, m)
        push!(written, m)
    end

    verbose && for p in paths
        @printf("  wrote %-24s %8.1f kB\n", relpath(p, results), filesize(p) / 1e3)
    end
    written
end
