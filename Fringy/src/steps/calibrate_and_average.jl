
export calibrate_and_average

"""
    calibrate_and_average(cfg, ld, instr, astrometry_inputs, if_alignment = nothing; tsys = true)
        -> (; avg::Dataset, logamp, weight_scale, summary)

Self-calibrate and average this session's visibilities.

* `instr` is [`instrumental`](@ref)'s result and `if_alignment` [`derive_if_alignment`](@ref)'s `IFAlignment`, both
  installed through the ordinary gain chain — the same instrument the astrometric pass measures
  against, so the two halves cannot drift apart.
* `astrometry_inputs` is [`astrometry_models`](@ref)'s result. Gain curves use its modelled source elevation at each
  datum's canonical epoch.

Returns

* `avg` — the averaged, calibrated root `Dataset`;
* `logamp` — the installed `LogAmplitudeBandpass` states, i.e. the Tsys-derived amplitude scale;
* `weight_scale` — [`noise_scale`](@ref)'s `k`, for which `k·w` is `1/σ²` in flux units. Measured
  here so that nothing downstream needs an imaging product to get it;
* `summary` — the SEFD census (per antenna and overall) and the calibrated-amplitude quantiles,
  as data rather than as printed lines.

`tsys = false` requests calibration without the a priori amplitude scale; it is a diagnostic control,
not a production mode.
"""
function calibrate_and_average(cfg, ld, instr, astrometry_inputs, if_alignment = nothing;
                               tsys::Bool = true)
    (; ds, byscan, byif, wholeband) = ld
    sel = ld.selection
    tsys_calibration = cfg.tsys_calibration
    visibility_averaging = cfg.visibility_averaging

    base = Solution(ds; components = (
        PhaseOffset(; time = byscan, frequency = byif),
        Delay(; time = byscan, frequency = byif),
        Rate(; time = byscan, frequency = wholeband),
        LogAmplitudeBandpass(; time = byscan, frequency = byif)))
    sol = Solution(base.dataset, (;
        PhaseOffset = ComponentState(base[PhaseOffset].definition, copy(instr.applied.phase)),
        Delay = ComponentState(base[Delay].definition, copy(instr.applied.delay)),
        Rate = base[Rate], LogAmplitudeBandpass = base[LogAmplitudeBandpass]))

    elevation(antenna, source, tt) =
        antenna_delay_terms(astrometry_inputs.astrometry_model, antenna, astrometry_inputs.astrometry_model.sources[source].ra, astrometry_inputs.astrometry_model.sources[source].dec, tt).el
    if tsys
        sol, _ = apply_step(TsysInit(ld.tsys, ld.gain; antennas = ld.antennas, elevation,
                                     tsys_ceiling = tsys_calibration.tsys_ceiling,
                                     quantization = tsys_calibration.quantization,
                                     reference_sefd = tsys_calibration.reference_sefd,
                                     fallback_cap = tsys_calibration.fallback_cap), sol)
    end
    isnothing(if_alignment) || ((sol, _) = apply_step(InstallIFAlignment(if_alignment), sol))

    sol, _ = apply_step(
        FringeSelf(FringeFit(; tiles = (; time = byscan, frequency = wholeband),
                                    window = cfg.window, oversample = cfg.oversample,
                                    refine = cfg.refine, selection = sel),
                        PhaseOffset, Delay, Rate;
                        min_snr = cfg.min_snr, globalization = SNRWeighted(iterations = 50),
                        reference = cfg.reference_antenna), sol)

    avtime = partition(ds, ByDuration(visibility_averaging.time); within = byscan)
    avfreq = group(partition(ds, ByChannel()),
                   Tuple(Iterators.partition(1:nchannels(ds), visibility_averaging.channels)))
    cal = calibrated_dataset(sol, select(ds, sel); terms = jones_terms(sol))
    avg = average(cal, Averaging(; time = avtime, frequency = avfreq))

    logamp = sol[LogAmplitudeBandpass].values
    weight_scale = noise_scale(stokes_i(avg))
    (; avg, logamp, weight_scale,
       summary = calibration_and_averaging_summary(ds, avg, logamp, tsys_calibration, weight_scale))
end

"""
    calibration_and_averaging_summary(ds, avg, logamp, tsys_calibration, weight_scale) -> NamedTuple

The census this step measures: the SEFD implied by the installed amplitude scale, overall and per
antenna and receptor slot, and the calibrated visibility amplitudes it produced.
"""
function calibration_and_averaging_summary(ds, avg, logamp, tsys_calibration, weight_scale)
    scale = tsys_calibration.quantization * tsys_calibration.reference_sefd
    sefd(v) = exp.(-2 .* filter(!iszero, v)) .* scale
    all_sefd = sefd(logamp)
    antennas = NamedTuple[]
    for p in 1:nantennas(ds)
        v = @view logamp[p, :, :, :]
        any(!iszero, v) || continue
        s = sefd(v)
        sefd_by_receptor_slot = ntuple(2) do i
            receptor_slot_sefd = sefd(@view logamp[p, i, :, :])
            isempty(receptor_slot_sefd) ? NaN : median(receptor_slot_sefd)
        end
        push!(antennas, (; antenna = ds.antennas[p].name, n = length(s), median = median(s),
                           sefd_by_receptor_slot))
    end
    amp = [abs(avg.rows.visibility[j][c][i, i]) for j in 1:nrows(avg), c in 1:nchannels(avg), i in 1:2
           if avg.rows.weight[j][c][i, i] > 0]
    (; n_cells = length(logamp), n_filled = length(all_sefd),
       sefd = isempty(all_sefd) ? (; median = NaN, p10 = NaN, p90 = NaN, max = NaN) :
              (; median = median(all_sefd), p10 = quantile(all_sefd, 0.1),
                 p90 = quantile(all_sefd, 0.9), max = maximum(all_sefd)),
       per_antenna = StructArray(identity.(antennas)),
       n_rows = nrows(avg), n_channels = nchannels(avg),
       amplitude = (; p05 = quantile(amp, 0.05), median = median(amp),
                      p95 = quantile(amp, 0.95), max = maximum(amp)),
       weight_scale)
end
