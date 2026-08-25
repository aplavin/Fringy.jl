
export ImagingState, image_all!, image_run, own_imaging_set, imaging_model_set

"""
    ImagingState

The imaging chain's mutable working state: what every iteration reads, and the two things an iteration
can legitimately change.

| field | mutable | what |
|---|---|---|
| `avg` | no | the averaged calibrated `Dataset` every iteration re-calibrates from |
| `sources` | no | the sources being imaged, in order |
| `grid`, `settings`, `weighting`, `radius`, `nsigma_component` | no | the imaging configuration of THIS variant |
| `weight_scale` | no | `k` such that `k·w` is `1/σ²`, measured by [`calibrate_and_average`](@ref) |
| `optics` | **yes** | per source, the dirty beam and the map σ — both functions of the WEIGHTS, so an amplitude solution invalidates them and a phase solution does not |
"""
mutable struct ImagingState{D, S}
    const avg::D
    const sources::Vector{Symbol}
    const grid::ImageGrid
    const settings::S
    const weighting::Symbol
    const radius::Float64
    const nsigma_component::Float64
    const weight_scale::Float64
    optics::Dictionary{Symbol, NamedTuple{(:db, :σ)}}
end

imaging_weights(st::ImagingState, vis) =
    st.weighting === :natural ? vis.w : uniform_weights(vis, st.grid)

imaging_optics(st::ImagingState, vis) = Dictionary(st.sources, map(st.sources) do s
    v = view(vis, findall(==(s), vis.source))
    (; db = dirty_beam(v, st.grid; weights = imaging_weights(st, v)),
       σ = map_sigma(v; weight_scale = st.weight_scale, weights = imaging_weights(st, v)))
end)

disk_window(st::ImagingState) = (dirty, g) -> let (xs, ys) = grid_axes(g), c = peak_position(dirty, g)
    BitMatrix([hypot(x - c[1], y - c[2]) ≤ st.radius for x in xs, y in ys])
end

clean_source(st::ImagingState, v, s, provenance) =
    image_source(v, st.grid, st.optics[s].db; source = s, gain = st.settings.clean_gain,
                 niter = st.settings.clean_niter,
                 threshold = st.settings.clean_threshold_nsigma * st.optics[s].σ,
                 window = disk_window(st), weights = imaging_weights(st, v), provenance)

"""
    image_all!(st::ImagingState, sols::Tuple, iteration; refresh = false) -> (; images, vis)

Apply `sols` to the averaged dataset, form Stokes I, and CLEAN every source.

`refresh` re-derives the beams and the map σ from the weights this calibration produces: REQUIRED
after an amplitude solution (it scales every weight by |g|²), a waste after a phase one. Deconvolving
with a stale beam is not a small error — measured, it drove the residual rms from 27 to 39 σ_map,
pushed CLEAN onto its iteration cap and cost 26 % of the CLEANed flux, a divergence that looks exactly
like a failing self-calibration and is not one. That is why the beam cache is a mutable field of an
explicit state rather than a module binding assigned inside a loop.
"""
function image_all!(st::ImagingState, sols::Tuple, iteration; refresh = false, label = "own")
    cal = foldl((ds, s) -> calibrated_dataset(s, ds; terms = jones_terms(s)), sols; init = st.avg)
    vis = stokes_i(cal)
    refresh && (st.optics = imaging_optics(st, vis))
    imgs = map(st.sources) do s
        v = view(vis, findall(==(s), vis.source))
        clean_source(st, v, s, (; iteration, weighting = st.weighting, label))
    end
    (; images = Dictionary(st.sources, imgs), vis)
end

χphase(st::ImagingState, vis, s, model) =
    closure_chi2(view(vis, findall(==(s), vis.source)), model; weight_scale = st.weight_scale).chi2
χamp(st::ImagingState, vis, s, model, min_snr) =
    closure_amp_chi2(view(vis, findall(==(s), vis.source)), model;
                     weight_scale = st.weight_scale, min_snr).chi2


function diagonal_receptor_slot(ds, receptor::Int)
    w = map(j -> map(W -> SMatrix{2, 2, Float64}(receptor == 1 ? W[1, 1] : 0.0, 0.0, 0.0,
                                                 receptor == 2 ? W[2, 2] : 0.0), ds.rows.weight[j]),
            1:nrows(ds))
    @set ds.rows.weight = w
end

function receptor_image(st::ImagingState, ds, s, receptor::Int, iteration)
    vis = stokes_i(diagonal_receptor_slot(ds, receptor))
    v = view(vis, findall(==(s), vis.source))
    wts = imaging_weights(st, v)
    db = dirty_beam(v, st.grid; weights = wts)
    image_source(v, st.grid, db; source = s, gain = st.settings.clean_gain,
                 niter = st.settings.clean_niter,
                 threshold = st.settings.clean_threshold_nsigma *
                             map_sigma(v; weight_scale = st.weight_scale, weights = wts),
                 window = disk_window(st), weights = wts, provenance = (; iteration, receptor))
end

function receptor_comparison(st::ImagingState, ds, iteration)
    first_r = Dictionary(st.sources, map(s -> receptor_image(st, ds, s, 1, iteration), st.sources))
    pc = map(st.sources) do s
        r = first_r[s]; l = receptor_image(st, ds, s, 2, iteration)
        (; source = s, offset = hypot((r.peak_xy - l.peak_xy)...),
           beam = fwhm_max(st.optics[s].db.beam), peak_1 = r.peak, peak_2 = l.peak,
           flux_1 = cleaned_flux(r.model), flux_2 = cleaned_flux(l.model))
    end |> StructArray
    (; pc, seeds = first_r)
end

function restamp(seed, dst)
    p = dst.provenance
    @set seed.provenance = NamedTuple{keys(p)}(map(k -> hasproperty(seed.provenance, k) ?
                                                        getproperty(seed.provenance, k) :
                                                        getproperty(p, k), keys(p)))
end

"""
    image_run(cfg, own::OwnImaging, calibration; label = "own", sources = nothing, verbose = true)

Run the whole imaging chain once, on [`calibrate_and_average`](@ref)'s averaged visibilities, under ONE
`OwnImaging` configuration. Returns everything the chain measured:

| field | what |
|---|---|
| `images` | source ⇒ `SourceImage` — the restored maps and the CLEAN component models |
| `optics` | source ⇒ the final dirty beam and map σ |
| `solution`, `ampsolution` | the hybrid-phase and amplitude increments, as `Solution`s over `avg` |
| `trajectory`, `amptrajectory`, `fluxtrajectory`, `rmstrajectory` | per source, one entry per iteration |
| `receptor_comparison`, `receptor_comparison_amplitude` | the per-receptor-slot consistency census, before and after the amplitude loop |
| `amp_census` | sources per amplitude solution interval — the identifiability condition `:global` stands on |
| `amp_null` | closure-phase χ² after amplitude calibration against the preceding model; antenna gains cannot move it |
| `meta` | the settings this run used, plus `weight_scale`, `map_sigma` and `seeded_from_receptor1` |

The three quantities `amp_census`, `amp_null` and `receptor_comparison` are FIELDS rather than printed
lines because each has a stated pass/fail meaning and belongs to the product.
"""
function image_run(cfg, own::OwnImaging, calibration; label = "own", sources = nothing, verbose = true)
    avg = calibration isa Dataset ? calibration : calibration.avg
    wscale = calibration isa Dataset ? noise_scale(stokes_i(avg)) : calibration.weight_scale
    grid = ImageGrid(; npix = own.npix, pixel = own.pixel_mas)

    byscan = partition(avg, ByScan())
    byif = partition(avg, ByIF())
    wholeband = group(byif, (Tuple(1:maximum(avg.freq.if_of)),))

    vis_all = stokes_i(avg)
    srcnames = isnothing(sources) ? sort!(unique(vis_all.source)) : Symbol.(collect(sources))
    for s in srcnames
        any(==(s), vis_all.source) ||
            error("image_run: source $s has no usable Stokes-I visibility")
    end

    st = ImagingState(avg, srcnames, grid, own, own.weighting, cfg.model_radius_mas,
                      cfg.component_nsigma, wscale, Dictionary{Symbol, NamedTuple{(:db, :σ)}}())
    st.optics = imaging_optics(st, vis_all)

    t0 = time()
    it0 = image_all!(st, (), 0; label)
    images = it0.images
    timg0 = time() - t0

    (pc0, seeds) = receptor_comparison(st, avg, 0)
    seeded_from_receptor1 = Symbol[]
    for r in pc0
        r.offset > own.receptor_peak_separation_beam_fraction * r.beam || continue
        set!(images, r.source, restamp(seeds[r.source], images[r.source]))
        push!(seeded_from_receptor1, r.source)
        verbose && @printf("  seeding %s from one receptor slot alone (peak offset %.3f mas = %.2f beam)\n",
                           r.source, r.offset, r.offset / r.beam)
    end

    trajectory = Dictionary(srcnames, map(s -> [χphase(st, it0.vis, s, images[s].model)], srcnames))
    amptrajectory = Dictionary(srcnames,
        map(s -> [χamp(st, it0.vis, s, images[s].model, cfg.closure_amp_min_snr)], srcnames))
    fluxtrajectory = Dictionary(srcnames, map(s -> [cleaned_flux(images[s].model)], srcnames))
    rmstrajectory = Dictionary(srcnames, map(s -> [images[s].rms / st.optics[s].σ], srcnames))
    verbose && @printf("  iteration 0: %.1f s, %d sources; closure χ² phase median %.2f, |V| median %.1f\n",
                       timg0, length(srcnames), median(map(v -> v[end], trajectory)),
                       median(filter(isfinite, map(v -> v[end], amptrajectory))))

    sol = nothing
    receptors_phase_spread = Float64[]
    for it in 1:own.hybrid_iterations
        ms = Dictionary(srcnames, map(s -> images[s].model, srcnames))
        div = model_dataset(avg, ms; floor = own.model_floor)
        solq, _ = apply_step(refringe_step(cfg, own, byscan, byif, wholeband), zerosol(own, div, byscan, byif, wholeband))
        push!(receptors_phase_spread, _receptors_phase_spread(solq))
        own.refringe_receptor_coupling === :copy_1_to_2 && _copy_receptor1_to_2!(solq)
        sol = Solution(avg, solq.components)
        res = image_all!(st, (sol,), it; label)
        images = res.images
        for s in srcnames
            push!(trajectory[s], χphase(st, res.vis, s, images[s].model))
            push!(amptrajectory[s], χamp(st, res.vis, s, images[s].model, cfg.closure_amp_min_snr))
            push!(fluxtrajectory[s], cleaned_flux(images[s].model))
            push!(rmstrajectory[s], images[s].rms / st.optics[s].σ)
        end
        verbose && @printf("  hybrid %d: closure χ² median %.2f → %.2f, improved for %d of %d\n", it,
                           median(map(v -> v[end - 1], trajectory)),
                           median(map(v -> v[end], trajectory)),
                           count(v -> v[end] < v[end - 1], trajectory), length(srcnames))
    end

    ampsol = nothing
    receptor_comparison_amplitude = nothing
    amp_census = nothing
    amp_null = NaN
    amp_gains = nothing
    if own.amp_iterations > 0
        amptime = own.amp_interval === :scan ? byscan :
                  own.amp_interval === :session ? partition(avg, WholeObservation()) :
                  partition(avg, ByDuration(own.amp_interval);
                            within = partition(avg, WholeObservation()))
        ampfreq = own.amp_frequency === :band ? wholeband : byif
        amp_census = amp_identifiability(avg, amptime, srcnames)
        ampselection = Selection(@o _.source in srcnames)
        ampsol = let a = Solution(avg; components = (LogAmplitudeBandpass(; time = amptime,
                                                                          frequency = ampfreq),))
            isnothing(sol) ? a : Solution(avg, merge(sol.components, a.components))
        end
        for it in 1:own.amp_iterations
            ms = Dictionary(srcnames, map(srcnames) do s
                own.amp_model === :full && return images[s].model
                m = significant_components(images[s].model, images[s].rms;
                                           nsigma = cfg.component_nsigma)
                isempty(components(m)) ?
                    MultiComponentModel([argmax(flux, components(images[s].model))]) : m
            end)
            step = AmpSelfCal(LogAmplitudeBandpass; models = ms, weight_scale = wscale,
                              min_snr = own.amp_min_snr, floor = own.model_floor,
                              regularization = own.amp_regularization,
                              normalize = own.amp_normalize, selection = ampselection)
            ampsol, (_, amp_gains) = apply_step(Capture(Gains() => :gains, step), ampsol)
            res = image_all!(st, (ampsol,), it; refresh = true, label)
            amp_null = median(map(s -> χphase(st, res.vis, s, images[s].model), srcnames))
            images = res.images
            for s in srcnames
                push!(trajectory[s], χphase(st, res.vis, s, images[s].model))
                push!(amptrajectory[s], χamp(st, res.vis, s, images[s].model, cfg.closure_amp_min_snr))
                push!(fluxtrajectory[s], cleaned_flux(images[s].model))
                push!(rmstrajectory[s], images[s].rms / st.optics[s].σ)
            end
            verbose && @printf("  amp %d: closure |V| χ² median %.1f → %.1f (null test %.2f vs %.2f)\n",
                               it, median(filter(isfinite, map(v -> v[end - 1], amptrajectory))),
                               median(filter(isfinite, map(v -> v[end], amptrajectory))),
                               amp_null, median(map(v -> v[end - 1], trajectory)))
        end
        receptor_comparison_amplitude = receptor_comparison(
            st, calibrated_dataset(ampsol, avg; terms = jones_terms(ampsol)), own.amp_iterations).pc
    end

    meta = (; label, grid, weighting = st.weighting, iterations = own.hybrid_iterations,
              seeded_from_receptor1, weight_scale = wscale, map_sigma = map(o -> o.σ, st.optics),
              clean = (; gain = own.clean_gain, niter = own.clean_niter,
                       threshold_nsigma = own.clean_threshold_nsigma, radius = cfg.model_radius_mas),
              component_nsigma = cfg.component_nsigma,
              receptors_phase_spread, refringe_receptor_coupling = own.refringe_receptor_coupling,
              ampcal = (; iterations = own.amp_iterations, interval = own.amp_interval,
                        frequency = own.amp_frequency, normalize = own.amp_normalize,
                        regularization = own.amp_regularization, min_snr = own.amp_min_snr,
                        components = isnothing(ampsol) ? nothing : ampsol.components),
              n_capped = count(s -> images[s].provenance.cleaned == own.clean_niter, srcnames))
    (; images, optics = st.optics, solution = sol, ampsolution = ampsol,
       trajectory, amptrajectory, fluxtrajectory, rmstrajectory,
       receptor_comparison = pc0, receptor_comparison_amplitude, amp_census, amp_null, amp_gains, meta)
end

zerosol(own::OwnImaging, ds, byscan, byif, wholeband) = Solution(ds; components =
    own.refringe_targets === :full ?
        (PhaseOffset(; time = byscan, frequency = byif),
         Delay(; time = byscan, frequency = wholeband),
         Rate(; time = byscan, frequency = wholeband)) :
        (PhaseOffset(; time = byscan, frequency = byif),))

refringe_step(cfg, own::OwnImaging, byscan, byif, wholeband) = FringeSelf(
    FringeFit(; tiles = (; time = byscan, frequency = wholeband),
                window = own.hybrid_window, oversample = own.hybrid_oversample,
                refine = own.refringe_refine === :ml ? cfg.refine : NoRefine()),
    (own.refringe_targets === :full ? (PhaseOffset, Delay, Rate) : (PhaseOffset,))...;
    min_snr = own.hybrid_min_snr, globalization = SNRWeighted(iterations = 50),
    reference = cfg.reference_antenna)

_receptors_phase_spread(sol) = let v = sol[PhaseOffset].values
    sqrt(mean(abs2, rem.(filter(isfinite, vec(v[:, 2, :, :] .- v[:, 1, :, :])), 2π, RoundNearest)))
end
function _copy_receptor1_to_2!(sol)
    for st in values(sol.components)
        st.values[:, 2, :, :] .= st.values[:, 1, :, :]
    end
    sol
end

"""
    amp_identifiability(avg, amptime, sources) -> (; per_cell, min, max, median, n_cells)

THE identifiability census: how many distinct sources fall in each amplitude solution interval. It is
the condition `amp_normalize = :global` stands on and the one thing an interval choice can silently
destroy — a cell's common-mode gain is degenerate with the sources in it being brighter than their
models, and only an ensemble of sources per cell breaks that. A number in
the product, not a printed line, because it has a stated pass/fail meaning: a single-source cell under
`:global` is over-fitting waiting to happen.
"""
function amp_identifiability(avg, amptime, sources)
    prows = parentrows(avg)
    sets = [Set{Int}() for _ in 1:ncells(amptime)]
    for j in 1:nrows(avg)
        avg.rows.source[j] in sources || continue
        push!(sets[cell_of_row(amptime, prows[j])], avg.rows.source_ix[j])
    end
    n = filter(>(0), length.(sets))
    (; per_cell = n, min = minimum(n; init = 0), max = maximum(n; init = 0),
       median = isempty(n) ? NaN : median(n), n_cells = length(n))
end

"""
    imaging_model_set(cfg, run; label = "own") -> ModelSet

The imaging run, seen as a model set: the components retained by the significance policy, and nothing
else — no recentering and no separate origin argument. Each model stays in the frame CLEAN produced it
in, whose zero is the map reference point, and that is the model origin whose absolute position the
registration returns.

`run` is [`image_run`](@ref)'s result, or any value carrying `images` and `meta` — which is what an
imaging product deserialized from a cache is.
"""
function imaging_model_set(cfg, run; label = "own", nsigma = cfg.component_nsigma)
    ms = Dictionary{Symbol, Any}()
    info = Dictionary{Symbol, NamedTuple}()
    for (s, si) in pairs(run.images)
        sig = significant_components(si.model, si.rms; nsigma)
        cs = collect(components(sig))
        set!(ms, s, sig)
        rmax = isempty(cs) ? 0.0 : maximum(c -> hypot((coords(c) .- si.peak_xy)...), cs)
        set!(info, s, (; n_components = length(cs),
                         n_components_total = length(components(si.model)),
                         n_negative = count(c -> flux(c) < 0, cs),
                         flux_model_jy = sum(flux, cs; init = 0.0),
                         flux_total_jy = cleaned_flux(si.model),
                         compactness = isempty(cs) ? 1.0 :
                                       maximum(flux, cs) / sum(flux, cs; init = 0.0),
                         r_max_mas = rmax, peak_x_mas = si.peak_xy[1], peak_y_mas = si.peak_xy[2],
                         peak = si.peak, peak_from = :pixel,
                         residual_rms = si.rms, npix = si.grid.npix, pixel_mas = si.grid.pixel,
                         beam_maj_mas = fwhm_max(si.beam), beam_min_mas = fwhm_min(si.beam),
                         beam_pa_deg = rad2deg(position_angle(si.beam)),
                         n_visibilities = si.nvis, n_cleaned = si.provenance.cleaned))
    end
    ModelSet(label, :own, cfg.model_radius_mas, ms, Dictionary{Symbol, Any}(run.images), info,
             (; kind = :own, run.meta..., significance_nsigma = nsigma, all_components = false))
end

"""
    own_imaging_set(cfg, own::OwnImaging, calibration; label = "own", sources = nothing) -> ModelSet

Image this session under `own` and return the resulting model set — [`image_run`](@ref) followed by
[`imaging_model_set`](@ref). This is what one member of `cfg.structure_models` expands into, once per
variant.
"""
own_imaging_set(cfg, own::OwnImaging, calibration; label = "own", sources = nothing, verbose = true) =
    imaging_model_set(cfg, image_run(cfg, own, calibration; label, sources, verbose); label)
