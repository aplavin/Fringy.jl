
export fit_summary_rows, fit_baseline_rows

"""
    fit_summary_rows(reg) -> Vector{NamedTuple}

One row per model set. `reference_wrms_ps` is the solve this set's wrms is compared with: the reference model set's
own solve where the unwrap guard excluded nothing, and a matched reference solve on this set's own rows
where it did fire.
"""
fit_summary_rows(reg::Registration) =
    [(; model_set = label, model_set_kind = String(R.kind),
         wrms_ps = R.summary.wrms_ps, chi2 = R.summary.chi2, ndof = R.F.ndof,
         n_selected = R.summary.n_selected, n_accepted = R.summary.n_accepted,
         rejected_pct = R.summary.rejected_pct,
         sigma_floor_median_ps = R.summary.sigma_floor_median_ps,
         n_sources = R.summary.n_sources, n_wrap_excluded = R.n_wrap, n_no_model = R.n_no_model,
         reference_model_set = R.reference_model_set, reference_wrms_ps = R.reference_wrms_ps,
         wrms_change_ps = R.summary.wrms_ps - R.reference_wrms_ps)
     for (label, R) in pairs(reg.sets)]

"""
    fit_baseline_rows(reg) -> Vector{NamedTuple}

One row per (model set, baseline), sorted by set and then by baseline length.
"""
fit_baseline_rows(reg::Registration) =
    [(; model_set = label, b.a1, b.a2, baseline_km = b.km, sigma_floor_ps = b.sigma_floor_ps,
         wrms_ps = b.wrms_ps, n_accepted = b.n_accepted)
     for (label, R) in pairs(reg.sets) for b in R.baselines]
