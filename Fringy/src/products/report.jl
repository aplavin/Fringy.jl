
export write_report

"""
    write_report(path, cfg, facts, reg, sets, rows, um, floors; manifest) -> path

The headline numbers, per model set and over the session.
"""
function write_report(path, cfg, facts, reg::Registration, sets::ModelSets, rows, um, floors;
                      manifest = NamedTuple[])
    bylabel(l) = [r for r in rows if r.model_set == l]
    open(path, "w") do io
        println(io, "# ", facts.name, " — absolute registration\n")
        @printf(io, "%s, %d channels over %d IFs, ν_eff %.6f GHz. %d sources, %d model sets, %d catalogue rows.\n\n",
                facts.date, length(facts.ν), maximum(facts.if_of), facts.ν_eff / 1e9,
                length(reg.sources), length(reg.sets), length(rows))
        println(io, "Every registration below is a solve of its own, over the same observables with ",
                    "that model set's structure delay removed and the weighting procedure re-run from ",
                    "scratch. The reference model set — the set with no components — is `", reg.reference_model_set,
                    "`, and what each solve returns is the absolute position of that set's own frame ",
                    "origin: an absolute WCS for the image behind it.\n")

        println(io, "## The model sets\n")
        println(io, "| model set | kind | rows | wrms [ps] | Δwrms [ps] | χ²/dof | accepted | τ max [ps] | closure φ χ² (median) | images |")
        println(io, "|---|---|---|---|---|---|---|---|---|---|")
        for (l, R) in pairs(reg.sets)
            rs = bylabel(l)
            χ = filter(isfinite, [r.chi2_closure_phase for r in rs])
            @printf(io, "| `%s` | %s | %d | %.3f | %+.3f | %.4f | %d | %.1f | %s | %d |\n",
                    l, R.kind, length(rs), R.summary.wrms_ps,
                    R.summary.wrms_ps - R.reference_wrms_ps, R.summary.chi2, R.summary.n_accepted,
                    isempty(rs) ? NaN : maximum(r -> r.tau_structure_max_ps, rs),
                    isempty(χ) ? "—" : @sprintf("%.2f", median(χ)),
                    count(r -> !isempty(r.image_file), rs))
        end

        println(io, "\n## The uncertainty model\n")
        @printf(io, "`sigma_final^2 = (s * sigma_formal)^2 + F^2`, fitted on the `%s` positions against AstrodataRFC: s = %.1f, F = %.3f mas in RA* (χ²/dof %.2f) and %.3f mas in Dec (χ²/dof %.2f), over %d sources.\n",
                um.reference_model_set, um.s, um.F_ra_mas, um.chi2_ra, um.F_dec_mas, um.chi2_dec, um.n_sources)
        @printf(io, "\nScatter against that catalogue: %.3f / %.3f mas, and %.3f / %.3f after removing a rigid rotation of (%+.3f, %+.3f, %+.3f) mas — which is REPORTED, never removed: with EOP and the terrestrial frame held fixed for one session, such a rotation is the expected difference.\n",
                um.rms_before_mas..., um.rms_after_mas..., um.rotation_mas...)

        println(io, "\n## The ensemble-coupling floor\n")
        @printf(io, "A source whose own structure delay is identically zero still moves between two registrations, through the clocks, tropospheres and re-run weighting it shares with every other source. Measured over every non-reference model set: **%.4f mas**. Per set: %s.\n",
                floors.pooled,
                isempty(floors.per_set) ? "none measurable" :
                join([@sprintf("`%s` %.4f", l, v) for (l, v) in pairs(floors.per_set)], ", "))
        println(io, "\nIt is the floor under every difference between two rows of the catalogue: below it, a per-source difference is the SOLVE and not the models.")

        if !isempty(manifest)
            println(io, "\n## The registered images\n")
            for l in unique(m.model_set for m in manifest)
                v = [m.shift_mas for m in manifest if m.model_set == l && isfinite(m.shift_mas)]
                @printf(io, "- `%s`: %d files in `images/%s/`%s\n", l,
                        count(m -> m.model_set == l, manifest), model_set_slug(l),
                        isempty(v) ? " (written from this pipeline's own maps)" :
                        @sprintf(", registered CRVAL a median %.3f mas (max %.3f) from the published phase-centre label",
                                 median(v), maximum(v)))
            end
        end

        println(io, "\n## What is not here\n")
        println(io, "Differences between model sets are not columns: the bias vector, the ",
                    "registration systematic, and the disagreement between two sets are all ",
                    "arithmetic between rows of `catalogue.csv`, and the formulas are in ",
                    "the `derived` block of `catalogue.ecsv`'s header, with the two constants they ",
                    "need beside them as values.")
    end
    path
end
