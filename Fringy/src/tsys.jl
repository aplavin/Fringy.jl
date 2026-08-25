
export TsysInit

"""
    TsysInit(records, gains; antennas, elevation, tsys_ceiling, quantization, reference_sefd, fallback_cap)

Step that fills a solution's `LogAmplitudeBandpass` state with the Tsys-derived amplitude scale
`a = ½·(log η + log SEFD_ref − log SEFD)` per (antenna, receptor slot, time cell, frequency window), from the recorded
system temperatures and gain curves (see the file header for the sign, which is load-bearing).

`records` is a SYSTEM_TEMPERATURE table as read by `VLBIFiles.system_temperature`: rows with
`time::DateTime`, `interval`, `antenna_no::Int` and `tsys_1/2` (one value per band and receptor slot).
`gains` is a GAIN_CURVE table (`VLBIFiles.gain_curve`): one row per antenna with `antenna_no`,
`dpfu_1/2` [K/Jy], the polynomial coefficients `gain_1/2` (`[coefficient, band]`), `nterm_1/2` and
`x_typ_1/2` (0 = the polynomial is a function of elevation in degrees, 1 = of zenith angle;
`y_typ` must be 2, a relative gain multiplying the DPFU). `antennas` maps file antenna numbers to
dataset antenna names, exactly as for [`PcalInit`](@ref).

`elevation(antenna::Symbol, source::Symbol, t::Float64) -> Float64` [rad] supplies the elevation the
gain curve is evaluated at—the a priori model's (`antenna_delay_terms(astrometry_model, …).el`), evaluated at
each time cell's own reference epoch. It is a callable rather than a table because the elevation is a
*model* quantity, and Fringy never carries a second, silently-divergent copy of the geometry.

Data conditions, all explicit:

- `tsys_ceiling` [K] rejects non-finite, non-positive, and implausibly large readings. A cell's value
  is the median of its accepted records.
- A cell with no accepted record falls back to that (antenna, receptor slot, frequency window)'s
  session median. More than `fallback_cap` of an antenna's cells falling back is an error; a group
  with no accepted record anywhere also fails loudly.
- `quantization` is the sampler efficiency η (0.88 for VLBA 2-bit, `NO_LEVELS = 4`), entering as
  `√η` per antenna. `reference_sefd` [Jy] is the global scale: 1.0 makes calibrated amplitudes
  janskys. Both are global constants and both are stated rather than assumed.

The target partitions are the design's `time = per-scan`, `frequency = ByIF` — the frequency partition
must resolve the IFs (each band of the table maps to exactly one frequency cell, fail loud otherwise),
and each time cell must carry a single source (the elevation is a per-source quantity). All keyword
arguments are required.
"""
struct TsysInit{R, G, A, E} <: Step
    records::R
    gains::G
    antennas::A
    elevation::E
    tsys_ceiling::Float64
    quantization::Float64
    reference_sefd::Float64
    fallback_cap::Float64
end

function TsysInit(records, gains; antennas, elevation, tsys_ceiling, quantization, reference_sefd,
                  fallback_cap)
    tsys_ceiling > 0 || error("TsysInit: tsys_ceiling must be positive, got $tsys_ceiling K")
    0 < quantization ≤ 1 || error("TsysInit: quantization efficiency must be in (0, 1], got $quantization")
    reference_sefd > 0 || error("TsysInit: reference_sefd must be positive, got $reference_sefd Jy")
    0 ≤ fallback_cap ≤ 1 || error("TsysInit: fallback_cap is a fraction, got $fallback_cap")
    TsysInit(records, gains, antennas, elevation, Float64(tsys_ceiling), Float64(quantization),
             Float64(reference_sefd), Float64(fallback_cap))
end

function _tsys_band_cells(part::FreqPartition, if_of::AbstractVector{<:Integer}, nband::Integer)
    cells = map(1:nband) do b
        chans = findall(==(b), if_of)
        isempty(chans) && return 0
        cs = unique(cell_of_channel(part, c) for c in chans)
        length(cs) == 1 ||
            error("TsysInit: IF $b spans frequency cells $(cs) — the LogAmplitudeBandpass frequency partition must resolve the IFs (ByIF)")
        only(cs)
    end
    present = filter(!=(0), cells)
    allunique(present) ||
        error("TsysInit: two IFs map to the same frequency cell $(cells) — the LogAmplitudeBandpass frequency partition must resolve the IFs (ByIF)")
    cells
end

function _gaincurve(coeffs, nterm::Integer, x_typ::Integer, el::Real)
    x = x_typ == 0 ? rad2deg(el) :
        x_typ == 1 ? 90 - rad2deg(el) :
        error("TsysInit: GAIN_CURVE X_TYP $(x_typ) is not supported (0 = elevation in degrees, 1 = zenith angle)")
    g = 0.0
    for k in nterm:-1:1
        g = g * x + coeffs[k]
    end
    g > 0 || error("TsysInit: gain curve evaluates to $g at elevation $(rad2deg(el))° — a non-positive gain is not a calibration")
    g
end

"""
    apply_step(step::TsysInit, sol, ::Nothing) -> (sol′, nothing)

Fill the `LogAmplitudeBandpass` state of `sol` from the step's SYSTEM_TEMPERATURE and GAIN_CURVE
tables (see [`TsysInit`](@ref) for the recipe, the sign and the data conditions). Cells of a
(antenna, time cell) the antenna has no data in keep their previous values (zero = identity gain for a
fresh solution). Produces no product.
"""
function apply_step(step::TsysInit, sol::Solution, ::Nothing)
    ds = dataset(sol)
    haskind(sol, LogAmplitudeBandpass) ||
        error("TsysInit needs a LogAmplitudeBandpass component in the solution")
    def = sol[LogAmplitudeBandpass].definition
    Tt = def.time; Tf = def.frequency

    rec = step.records
    isempty(rec) && error("TsysInit: the SYSTEM_TEMPERATURE table is empty")
    nband = length(first(rec.tsys_1))
    length(first(rec.tsys_2)) == nband ||
        error("TsysInit: the two receptor slots of the SYSTEM_TEMPERATURE table carry different band counts")

    name2idx = Dictionary(map(a -> a.name, ds.antennas), eachindex(ds.antennas))
    antidx = map(rec.antenna_no) do no
        haskey(step.antennas, no) ||
            error("TsysInit: SYSTEM_TEMPERATURE antenna number $no is not in the `antennas` map $(collect(keys(step.antennas)))")
        nm = step.antennas[no]
        haskey(name2idx, nm) || error("TsysInit: antenna $nm has SYSTEM_TEMPERATURE records but is not in the dataset")
        name2idx[nm]
    end

    nant = nantennas(ds)
    dpfu = fill(NaN, nant, 2, nband)
    gcoef = [zeros(0) for _ in 1:nant, _ in 1:2, _ in 1:nband]
    gxtyp = zeros(Int, nant, 2, nband)
    for g in step.gains
        haskey(step.antennas, g.antenna_no) ||
            error("TsysInit: GAIN_CURVE antenna number $(g.antenna_no) is not in the `antennas` map")
        nm = step.antennas[g.antenna_no]
        haskey(name2idx, nm) || continue
        p = name2idx[nm]
        for (i, (dp, gain, nterm, xt, yt)) in enumerate(((g.dpfu_1, g.gain_1, g.nterm_1, g.x_typ_1, g.y_typ_1),
                                                         (g.dpfu_2, g.gain_2, g.nterm_2, g.x_typ_2, g.y_typ_2)))
            for b in 1:nband
                yt[b] == 2 ||
                    error("TsysInit: GAIN_CURVE Y_TYP $(yt[b]) at antenna $nm is not supported (2 = relative gain multiplying the DPFU)")
                dpfu[p, i, b] = ustrip(u"K/Jy", dp[b])
                gcoef[p, i, b] = Float64[gain[t, b] for t in 1:nterm[b]]
                gxtyp[p, i, b] = xt[b]
            end
        end
    end

    ntc = ncells(Tt)
    src_of_tc = fill(Symbol(""), ntc)
    cells_of_ant = [Set{Int}() for _ in 1:nant]
    for j in 1:nrows(ds)
        tc = cell_of_row(Tt, j)
        s = ds.rows.source[j]
        if src_of_tc[tc] == Symbol("")
            src_of_tc[tc] = s
        else
            src_of_tc[tc] == s ||
                error("TsysInit: time cell $tc spans sources $(src_of_tc[tc]) and $s — the elevation " *
                      "the gain curve needs is a per-source quantity, so the time partition must not " *
                      "merge sources (ByScan on source-aware scans)")
        end
        p, q = ds.rows.baseline_ix[j].antennas
        push!(cells_of_ant[p], tc); push!(cells_of_ant[q], tc)
    end

    tcell = _record_cells(Tt, rec.time, rec.interval, ds.time0)
    cellts = fill(NaN, nant, 2, ntc, nband)
    medts = fill(NaN, nant, 2, nband)
    accepted(x) = (v = ustrip(u"K", x); isfinite(v) && 0 < v ≤ step.tsys_ceiling ? v : NaN)
    for p in 1:nant, i in 1:2
        col = i == 1 ? rec.tsys_1 : rec.tsys_2
        rows_p = findall(k -> antidx[k] == p, eachindex(antidx))
        isempty(rows_p) && continue
        for b in 1:nband
            good = filter(isfinite, [accepted(col[k][b]) for k in rows_p])
            isempty(good) || (medts[p, i, b] = median(good))
        end
        for (tc, poss) in pairs(groupfind(k -> tcell[k], rows_p))
            tc == 0 && continue
            ks = view(rows_p, poss)
            for b in 1:nband
                good = filter(isfinite, [accepted(col[k][b]) for k in ks])
                isempty(good) || (cellts[p, i, tc, b] = median(good))
            end
        end
    end

    fcell = _tsys_band_cells(Tf, ds.freq.if_of, nband)
    trefs = ustrip.(u"s", references(Tt))
    vals = copy(sol[LogAmplitudeBandpass].values)
    nfallback = zeros(Int, nant); ncell = zeros(Int, nant)
    scale = 0.5 * (log(step.quantization) + log(step.reference_sefd))
    for p in 1:nant
        isempty(cells_of_ant[p]) && continue
        nm = ds.antennas[p].name
        for tc in sort!(collect(cells_of_ant[p]))
            el = Float64(step.elevation(nm, src_of_tc[tc], trefs[tc]))
            for i in 1:2, b in 1:nband
                fc = fcell[b]
                fc == 0 && continue
                isnan(dpfu[p, i, b]) &&
                    error("TsysInit: antenna $nm receptor slot $i has no GAIN_CURVE entry for IF $b — no DPFU, no SEFD")
                ncell[p] += 1
                ts = cellts[p, i, tc, b]
                if isnan(ts)
                    ts = medts[p, i, b]
                    isnan(ts) &&
                        error("TsysInit: antenna $nm receptor slot $i IF $b has no usable " *
                              "SYSTEM_TEMPERATURE record in the whole session (ceiling " *
                              "$(step.tsys_ceiling) K) — cannot even fall back to its own median")
                    nfallback[p] += 1
                end
                sefd = ts / (dpfu[p, i, b] * _gaincurve(gcoef[p, i, b], length(gcoef[p, i, b]), gxtyp[p, i, b], el))
                vals[p, i, tc, fc] = scale - 0.5 * log(sefd)
            end
        end
    end

    for p in 1:nant
        ncell[p] == 0 && continue
        frac = nfallback[p] / ncell[p]
        frac > step.fallback_cap && error(
            "TsysInit: $(round(100frac, digits = 1)) % of antenna $(ds.antennas[p].name)'s $(ncell[p]) " *
            "(receptor slot, scan, IF) cells have no usable SYSTEM_TEMPERATURE record and fell back to its " *
            "session median — above the $(round(100 * step.fallback_cap, digits = 1)) % cap")
    end
    total = sum(nfallback)
    total > 0 && @warn("TsysInit: $total of $(sum(ncell)) (antenna, receptor slot, scan, IF) cells fell " *
                       "back to the antenna's session-median Tsys",
                       per_antenna = Dictionary(map(a -> a.name, ds.antennas), nfallback))

    comps = merge(sol.components, (; LogAmplitudeBandpass = ComponentState(def, vals)))
    (Solution(sol.dataset, comps), nothing)
end
