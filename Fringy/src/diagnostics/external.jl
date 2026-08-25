
export external_comparison, structure_moment, beam_matched_correlation

"""
    structure_moment(model; rmax) -> (; pa, radius, flux, n)

The flux-weighted offset of every non-core positive component within `rmax` of the core, as a position
angle [deg] and a radius [mas]; the core is the brightest component.

The same window is used for both models being compared, deliberately: a deep archive model reaches out
to flux ours cannot see, and ours — cleaned unwindowed — carries the odd noise component far out, so
an unrestricted average would compare two different jets.
"""
function structure_moment(model; rmax)
    cs = collect(components(model))
    isempty(cs) && return (; pa = NaN, radius = NaN, flux = 0.0, n = 0)
    core = coords(cs[argmax(map(c -> abs(flux(c)), cs))])
    jet = filter(c -> 0 < hypot((coords(c) - core)...) ≤ rmax && flux(c) > 0, cs)
    isempty(jet) && return (; pa = NaN, radius = NaN, flux = 0.0, n = 0)
    F = sum(flux, jet)
    d = sum(c -> flux(c) .* (coords(c) - core), jet) ./ F
    (; pa = rad2deg(atan(d[1], d[2])), radius = hypot(d...), flux = F, n = length(jet))
end

"A model with every component reflected through the origin — the 180° rotation a wrong uv sign makes."
rotated_model(model) = MultiComponentModel([(@set c.coords = -coords(c)) for c in components(model)])

"""
    beam_matched_correlation(model, im) -> Float64

The reference `model` convolved with OUR restoring beam on OUR grid, correlated with our restored map,
with the CORE MASKED OUT — one beam around the peak. The core is symmetric, correlates just as well
with a rotated reference, and would otherwise drown the very asymmetry being tested.
"""
function beam_matched_correlation(model, im)
    m = restore(model, im.beam, zeros(im.grid.npix, im.grid.npix), im.grid)
    xs, ys = grid_axes(im.grid)
    r = ustrip(fwhm_max(im.beam))
    keep = [hypot(x - im.peak_xy[1], y - im.peak_xy[2]) > r for x in xs, y in ys]
    cor(m[keep], im.restored[keep])
end

"""
The reference brackets the comparison reads its numbers against. They are REPORTING thresholds — they
decide which lines a human is asked to look at, never whether anything succeeded.

`spread` is the one worth flagging and that is a considered choice rather than the obvious one: an
antenna gain is by definition per antenna, so the spread of the per-antenna |V| ratios measures it
directly and completely. The long/short TREND is the same information read through the accident that
an antenna's baselines have a characteristic length, plus whatever two independent deconvolutions
disagree about in compactness — it gets its own reference line and no flag at all.
"""
const EXTERNAL_TOLERANCES = (pa_deg = 25.0, flux_ratio = 1.5, spread = 1.4, trend = 1.5)

"Baseline-length bins of the |V| comparison [Mλ]."
const VIS_LENGTH_BINS = [0, 50, 100, 150, 200, 250, 300, 400, 1000]

"""
    external_comparison(cfg, images, reference_models, calibration; rmax = 3.0, weight_scale = calibration.weight_scale,
                        ampcal = nothing, tolerances = EXTERNAL_TOLERANCES,
                        bins = VIS_LENGTH_BINS) -> (; sources, handedness, notes, worst)

Our imaging against an archive's, per source, on values:

  * `images` — our own model set's images, keyed by source name (`sets[label].images`);
  * `reference_models` — the archive's model per source, already loaded (the caller reads the files;
    a diagnostics function takes values);
  * `calibration` — [`calibrate_and_average`](@ref)'s result, whose averaged calibrated visibilities are the uv points
    every amplitude reading is taken on.

`ampcal`, when given, are the amplitude self-calibration's component states: the |V| comparison is
then made on the visibilities the images were made from, including that solution. Without it the
comparison is on the Tsys-calibrated data, which is a different and equally honest reading — and
the returned rows say which one it was.

A source missing from either side is named in `notes` and skipped; if none survives, `handedness` is
empty and the caller must say the orientation is UNVERIFIED rather than assume it.
"""
function external_comparison(cfg, images, reference_models, calibration; rmax = 3.0,
                             weight_scale = calibration.weight_scale, ampcal = nothing,
                             tolerances = EXTERNAL_TOLERANCES, bins = VIS_LENGTH_BINS)
    notes = String[]
    present = Symbol[]
    for nm in keys(reference_models)
        haskey(images, nm) ? push!(present, nm) :
            push!(notes, "$nm has an external model but no image in this model set — skipped")
    end
    cal = isnothing(ampcal) ? calibration.avg :
          let s = Solution(calibration.avg, ampcal); calibrated_dataset(s, calibration.avg; terms = jones_terms(s)) end
    vis = stokes_i(cal)
    antnames = map(a -> a.name, calibration.avg.antennas)
    blen(k) = hypot(vis.u[k], vis.v[k]) / MAS_IN_RAD / 1e6

    rows = map(present) do nm
        ref = reference_models[nm]
        im = images[nm]
        ours = structure_moment(im.model; rmax)
        theirs = structure_moment(ref; rmax)
        ourf = sum(flux, components(im.model))
        reff = sum(flux, components(ref))
        dpa = rem(ours.pa - theirs.pa, 360, RoundNearest)
        resolved = min(ours.radius, theirs.radius) > 0.1 * ustrip(fwhm_max(im.beam))
        cd = beam_matched_correlation(ref, im)
        cr = beam_matched_correlation(rotated_model(ref), im)

        ks = findall(==(nm), vis.source)
        F = sum(abs ∘ flux, components(ref))
        M = [abs(visibility(ref, SVector(vis.u[k], vis.v[k]))) for k in ks]
        A = [abs(vis.V[k]) for k in ks]
        L = [blen(k) for k in ks]
        keep = findall(k -> M[k] > 0.02F &&
                            M[k] * sqrt(weight_scale * vis.w[ks[k]]) ≥ cfg.closure_amp_min_snr,
                       eachindex(M))
        vbins = NamedTuple[]
        for b in 1:length(bins) - 1
            sel = filter(k -> bins[b] ≤ L[k] < bins[b + 1], keep)
            length(sel) ≥ 50 || continue
            push!(vbins, (; lo = float(bins[b]), hi = float(bins[b + 1]), n = length(sel),
                            ratio = median(A[sel] ./ M[sel])))
        end
        trend = length(vbins) ≥ 2 ? last(vbins).ratio / first(vbins).ratio : NaN
        antenna_rows = [(; antenna = antnames[p],
                 ratio = (sel = filter(k -> p in vis.baseline_ix[ks[k]].antennas, keep);
                          isempty(sel) ? NaN : median(A[sel] ./ M[sel])))
              for p in eachindex(antnames)]
        fs = filter(isfinite, [s.ratio for s in antenna_rows])
        spread = isempty(fs) ? NaN : maximum(fs) / minimum(fs)

        v = view(vis, ks)
        ca = closure_amp_chi2(v, ref; weight_scale, min_snr = cfg.closure_amp_min_snr)
        cb = closure_amp_chi2(v, im.model; weight_scale, min_snr = cfg.closure_amp_min_snr)

        resolved && abs(dpa) ≥ tolerances.pa_deg &&
            push!(notes, @sprintf("%s: structure PA differs by %.1f° (reference bracket %.0f°)",
                                  nm, dpa, tolerances.pa_deg))
        cd > cr ||
            push!(notes, @sprintf("%s: the 180°-rotated reference correlates BETTER (%.3f vs %.3f) — check the uv sign",
                                  nm, cr, cd))
        (1 / tolerances.flux_ratio ≤ ourf / reff ≤ tolerances.flux_ratio) ||
            push!(notes, @sprintf("%s: CLEANed flux ratio %.2f outside ×%.1f of the external model",
                                  nm, ourf / reff, tolerances.flux_ratio))
        isfinite(spread) && spread > tolerances.spread &&
            push!(notes, @sprintf("%s: the per-antenna |V|/|V_ref| medians span ×%.2f (reference bracket ×%.2f) — an antenna-decomposable amplitude error survives",
                                  nm, spread, tolerances.spread))

        (; source = nm, ours, theirs, flux_ours = ourf, flux_reference = reff,
           flux_ratio = ourf / reff, dpa, resolved,
           correlation = (; direct = cd, rotated = cr, margin = cd - cr), handedness = cd > cr,
           n_points = length(ks), n_kept = length(keep), vis_bins = vbins,
           vis_median = isempty(keep) ? NaN : median(A[keep] ./ M[keep]),
           antennas = antenna_rows, spread, trend,
           closure_amp = (; reference = ca.chi2, ours = cb.chi2, rms_reference = ca.rms,
                            rms_ours = cb.rms, n = ca.n),
           amplitude_calibrated = !isnothing(ampcal))
    end
    (; sources = rows, notes,
       handedness = [(r.source, r.handedness) for r in rows],
       worst = (; pa_deg = isempty(rows) ? 0.0 :
                          maximum(r -> r.resolved ? abs(r.dpa) : 0.0, rows),
                  margin = isempty(rows) ? Inf : minimum(r -> r.correlation.margin, rows),
                  spread = isempty(rows) ? 1.0 :
                           maximum(r -> isfinite(r.spread) ? r.spread : 1.0, rows),
                  trend = isempty(rows) ? 1.0 :
                          maximum(r -> isfinite(r.trend) ? r.trend : 1.0, rows)),
       tolerances)
end
