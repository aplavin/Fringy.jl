
export Verdict, verdict, dataset_verdicts, input_verdicts, scan_verdicts, tone_health_verdicts,
       cable_verdicts, clock_break_verdicts, amplitude_verdicts, identifiability_verdicts,
       fringe_census_verdicts, orientation_verdicts,
       closure_triangles, closure_floor, receptor_slots_column

"""
    Verdict

One check's outcome as DATA: `level` (`:pass` / `:warn` / `:fail`), the `gate` it belongs to, the
`measured` numbers as a NamedTuple, the `message` a human reads and an actionable `hint`.
"""
const Verdict = @NamedTuple begin
    level::Symbol
    gate::String
    measured::NamedTuple
    message::String
    hint::String
end

verdict(level::Symbol, gate, message; measured = (;), hint = "") =
    Verdict((level, String(gate), measured, String(message), String(hint)))

_wrapπ(x) = rem(x, 2π, RoundNearest)
_mjd(dt::DateTime) = datetime2julian(dt) - 2400000.5


"""
    input_verdicts(cfg, facts; mjd0, mjd1, eop, ionex, antenna_geometry, antennas,
                   external_model_coverage = nothing, orientation_external_available = nothing)
        -> Vector{Verdict}

Whether the external reference data covers this session: the EOP series, each IONEX product, the
antenna table, and optionally external-model coverage or the own-image orientation comparison.

`external_model_coverage` is `nothing` without an external structure-model set, otherwise a named
tuple with `available` and `n_sources`. `orientation_external_available` is `nothing` unless own
imaging and explicit orientation-reference sources are configured; then it states whether their
external models are present.

`eop` is [`read_finals2000A`](@ref)'s result, `ionex` a `NamedTuple` of [`IonexSeries`](@ref) keyed
the way `cfg.ionex_files` is, `antenna_geometry` [`load_antenna_geometry`](@ref)'s result, and
`antennas` the antenna
names the data actually carries. All values — the reading is the caller's, the judgement is here.
"""
function input_verdicts(cfg, facts; mjd0, mjd1, eop, ionex, antenna_geometry, antennas,
                        external_model_coverage = nothing, orientation_external_available = nothing)
    out = Verdict[]
    lo, hi = extrema(eop)
    push!(out, lo ≤ mjd0 && mjd1 ≤ hi ?
        verdict(:pass, "eop-coverage",
                @sprintf("finals2000A covers the session (%d rows, MJD %.0f..%.0f)", length(eop), lo, hi);
                measured = (; n = length(eop), lo, hi, mjd0, mjd1)) :
        verdict(:fail, "eop-coverage",
                @sprintf("finals2000A covers MJD %.0f..%.0f but the session is %.3f..%.3f — every position would carry an extrapolated Earth orientation",
                         lo, hi, mjd0, mjd1);
                measured = (; n = length(eop), lo, hi, mjd0, mjd1)))
    for (k, s) in pairs(ionex)
        ilo, ihi = extrema(s.mjd)
        covered = ilo ≤ mjd0 && mjd1 ≤ ihi
        m = (; product = k, n_maps = length(s.mjd), lo = ilo, hi = ihi, nlat = length(s.lat),
               nlon = length(s.lon), height_km = s.height / 1e3, covered,
               primary = k === cfg.iono_primary)
        msg = @sprintf("IONEX %s: %d maps, MJD %.3f..%.3f, %d×%d grid, shell %.0f km",
                       k, length(s.mjd), ilo, ihi, length(s.lat), length(s.lon), s.height / 1e3)
        push!(out, covered ? verdict(:pass, "ionex-$k", msg; measured = m) :
              k === cfg.iono_primary ?
              verdict(:fail, "ionex-$k",
                      msg * @sprintf(" — the PRIMARY product does NOT cover %.3f..%.3f; every ionospheric correction would be extrapolated (add the neighbouring day's map)", mjd0, mjd1);
                      measured = m) :
              verdict(:warn, "ionex-$k", msg * @sprintf(" — does not cover %.3f..%.3f", mjd0, mjd1);
                      measured = m,
                      hint = "this is not the correction ($(cfg.iono_primary) is): the three products' spread is what `iono_inflation` turns into the ionospheric uncertainty, so an incomplete one distorts the σ column of every observable. Add the neighbouring day's map (ionex/PROVENANCE.md has the URLs)"))
    end
    have = Set(keys(antenna_geometry))
    missing_antennas = [a for a in antennas if !(a in have)]
    push!(out, isempty(missing_antennas) ?
        verdict(:pass, "antenna_geometry.toml",
                @sprintf("%d antennas at epoch %.3f; all %d antennas of the file present",
                         length(antenna_geometry), facts.epoch_obs, length(antennas));
                measured = (; n_geometry_antennas = length(antenna_geometry), facts.epoch_obs,
                              n_data_antennas = length(antennas))) :
        verdict(:fail, "antenna_geometry.toml",
                "no a priori position for " * join(missing_antennas, ", ") *
                " — regenerate the antenna table with `sessions/acquire/make_antenna_geometry_toml.py` for this array";
                measured = (; missing = missing_antennas)))
    if !isnothing(external_model_coverage)
        (; available, n_sources) = external_model_coverage
        push!(out, available ?
            verdict(:pass, "external-model-coverage",
                    "external model files present for all $n_sources dataset sources";
                    measured = external_model_coverage) :
            verdict(:warn, "external-model-coverage",
                    "external model files are missing for one or more of $n_sources dataset sources";
                    measured = external_model_coverage,
                    hint = "those sources use zero structure correction in this external model set"))
    end
    if !isnothing(orientation_external_available)
        push!(out, orientation_external_available ?
            verdict(:pass, "external-reference",
                    "external models present — the orientation check will run";
                    measured = (; available = true)) :
            verdict(:warn, "external-reference",
                    "no external image models configured or present";
                    measured = (; available = false),
                    hint = "the orientation check is skipped: without an independent image of the same sky, a mirrored registration cannot be detected"))
    end
    out
end


"How many runs there are, and how many of them carry more than one source."
function scan_census(scanid::AbstractVector, sid::AbstractVector)
    d = Dict{Int, Set{Int}}()
    for j in eachindex(scanid)
        push!(get!(Set{Int}, d, scanid[j]), sid[j])
    end
    (length(d), count(v -> length(v) > 1, values(d)))
end

"""
How many scans a given rule finds, source-aware and gap-only — the stability sweep, recomputed from the
row index. A schedule whose gaps are comparable to the rule's `min_gap` gives a scan count that moves
with it, and then "per scan" does not mean what the reader thinks.
"""
function run_count(tt::AbstractVector, sid::AbstractVector, ord::Vector{Int}, mingap::Float64,
                   source_aware::Bool)
    n = 1; tprev = tt[ord[1]]; sprev = sid[ord[1]]
    for j in ord
        t = tt[j]; s = sid[j]
        (t - tprev > mingap || (source_aware && s != sprev)) && (n += 1)
        tprev = max(tprev, t); sprev = s
    end
    n
end

"""
    scan_verdicts(cfg; scan, source_ix, t, n_rows, n_flagged, gaps = (5.0, 30.0, 60.0, 120.0))
        -> Vector{Verdict}

The scan definition and what the FLAG table removes.

Fringe tiles are scans. A gap-only definition merges adjacent scans on different sources, and
a tile spanning two sources is not a fringe measurement at all (`FringeFit` fails loud on it, and the
Tsys initialization needs one elevation per cell) — so that one is a `:fail`, not a `:warn`.
"""
function scan_verdicts(cfg; scan, source_ix, t, n_rows, n_flagged,
                       gaps = (5.0, 30.0, 60.0, 120.0))
    out = Verdict[]
    nscan, multi = scan_census(scan, source_ix)
    push!(out, multi == 0 ?
        verdict(:pass, "scan-single-source", "every run carries exactly one source";
                measured = (; n_runs = nscan, n_multi = 0)) :
        verdict(:fail, "scan-single-source",
                "$multi of $nscan runs span more than one source — a fringe tile would mix sources";
                measured = (; n_runs = nscan, n_multi = multi)))
    tt = collect(t); sid = collect(source_ix); ord = sortperm(tt)
    counts = [run_count(tt, sid, ord, g, true) for g in gaps]
    gaponly = [run_count(tt, sid, ord, g, false) for g in gaps]
    push!(out, allequal(counts) ?
        verdict(:pass, "scan-stability",
                @sprintf("the run count is %d at every min_gap from %.0f s to %.0f s",
                         first(counts), first(gaps), last(gaps));
                measured = (; gaps = collect(gaps), counts, gap_only = gaponly)) :
        verdict(:warn, "scan-stability",
                @sprintf("the run count moves with min_gap: %s", join(counts, " → "));
                measured = (; gaps = collect(gaps), counts, gap_only = gaponly),
                hint = "the schedule's gaps are comparable to `cfg.scans.min_gap`; verify the scan definition before interpreting per-scan quantities"))
    frac = 100n_flagged / n_rows
    push!(out, frac < 25 ?
        verdict(:pass, "flags", @sprintf("%.1f%% of rows flagged", frac);
                measured = (; n_rows, n_flagged, flagged_pct = frac, n_runs = nscan)) :
        verdict(:warn, "flags", @sprintf("%.1f%% of rows flagged", frac);
                measured = (; n_rows, n_flagged, flagged_pct = frac, n_runs = nscan),
                hint = "a quarter of the session was removed by the FLAG table; verify whether flags are antenna- or frequency-specific before trusting the census"))
    out
end


"""
    tone_health_verdicts(cfg; tones, antennas, antenna_names, weak_pct = 5.0)
        -> Vector{Verdict}

The tone-pair decoding `PcalInit` needs, at the RAW record cadence: exactly two tones per band after
the session's tone pick, and how much of each (antenna, receptor slot)'s tone block sits below the floor
that `PcalInit` itself applies.

`tones` is the PHASE-CAL table as [`pick_tones`](@ref) left it — the comb the calibration will
actually use, so the census is of what happens and not of what the file happens to carry.
"""
function tone_health_verdicts(cfg; tones, antennas, antenna_names, weak_pct = 5.0)
    out = Verdict[]
    ntone, nband = size(first(tones.freq_1))
    push!(out, ntone == 2 ?
        verdict(:pass, "pcal-tones",
                @sprintf("%d tones × %d bands per record, %d records", ntone, nband, length(tones));
                measured = (; ntone, nband, n_records = length(tones))) :
        verdict(:fail, "pcal-tones",
                "$ntone tones per band — the tone-pair recipe of `PcalInit` needs exactly 2; every instrumental delay would be decoded from the wrong comb";
                measured = (; ntone, nband, n_records = length(tones))))
    rows_of = _pcal_rows_by_antenna(tones, antennas)
    census = NamedTuple[]
    weakest = NamedTuple[]
    for st in antenna_names
        ks = get(rows_of, st, Int[])
        if isempty(ks)
            push!(out, verdict(:warn, "pcal-$st", "no PHASE-CAL records at all";
                               measured = (; antenna = st, n_records = 0),
                               hint = "`PcalInit` leaves this antenna uncalibrated (its cells keep their previous values); expected only for an antenna whose tone generator was off"))
            continue
        end
        for (i, col) in ((1, tones.pcal_1), (2, tones.pcal_2))
            amp = [abs(ComplexF64(col[k][tn, b])) for k in ks, tn in 1:ntone, b in 1:nband]
            weak = 0; dead = 0
            for tn in 1:ntone, b in 1:nband
                a = @view amp[:, tn, b]
                m = median(a)
                weak += count(<(cfg.tone_floor * m), a)
                dead += count(iszero, a)
            end
            pw = 100weak / length(amp)
            push!(census, (; antenna = st, receptor_slot = i, n_records = length(ks),
                             weak_pct = pw, dead_pct = 100dead / length(amp),
                             median_amplitude = median(amp)))
            pw > weak_pct && push!(weakest, (; antenna = st, receptor_slot = i, pct = pw))
        end
    end
    push!(out, isempty(weakest) ?
        verdict(:pass, "pcal-tone-health",
                @sprintf("every (antenna, receptor slot) keeps > %.0f%% of its tones above the %.2f floor",
                         100 - weak_pct, cfg.tone_floor);
                measured = (; rows = census)) :
        verdict(:warn, "pcal-tone-health",
                "tones below the floor for " *
                join((@sprintf("%s-receptor-slot-%d %.1f%%", w.antenna, w.receptor_slot, w.pct)
                      for w in weakest), ", ");
                measured = (; rows = census, weakest),
                hint = "`tone_floor` drops those records; a few per cent is a synthesiser dropout, a large fraction means the floor or the tone generator needs looking at"))
    out
end

"FITS antenna number ⇒ the PHASE-CAL row indices of that antenna."
function _pcal_rows_by_antenna(tones, antennas)
    rows_of = Dictionary(antennas, [Int[] for _ in antennas])
    for k in eachindex(tones.antenna_no)
        no = tones.antenna_no[k]
        (1 ≤ no ≤ length(antennas)) || continue
        push!(rows_of[antennas[no]], k)
    end
    rows_of
end

"The tone spacing and the delay ambiguity it implies, per band [Hz] and [s]."
function tone_comb(tones)
    _, nband = size(first(tones.freq_1))
    Δf = [ustrip(u"Hz", first(tones.freq_1)[2, b] - first(tones.freq_1)[1, b]) for b in 1:nband]
    νmid = [ustrip(u"Hz", (first(tones.freq_1)[1, b] + first(tones.freq_1)[2, b]) / 2) for b in 1:nband]
    (; spacing = Δf, ambiguity = 1 ./ Δf, νmid)
end


"""
    cable_verdicts(cfg; tones, antennas, antenna_names, time0) -> Vector{Verdict}

What `cfg.cable_antennas` is: the antennas whose CABLE_CAL readout is a real measurement. The three
failure modes seen in practice — no reading at all, a dead constant, and a readout that wraps — are
exactly what this census measures, and it is what produces the exclusion list.

Note this is only about WHICH antennas are read; whether the cable correction is applied at all is
`cable_apply`, settled separately by the cable-application experiment.
"""
function cable_verdicts(cfg; tones, antennas, antenna_names, time0)
    rows_of = _pcal_rows_by_antenna(tones, antennas)
    census = NamedTuple[]
    suspect = String[]
    for st in antenna_names
        ks = get(rows_of, st, Int[])
        cs = [ustrip(u"s", c) for c in @view(tones.cable_cal[ks]) if !isnan(c)]
        inlist = Symbol(st) in cfg.cable_antennas
        if isempty(cs)
            push!(census, (; antenna = st, n = 0, median_ns = NaN, ptp_ns = NaN, sigma_ps = NaN,
                             jumps = 0, verdict = "no readings", in_list = inlist))
            inlist && push!(suspect, "$st is in cable_antennas but has no CABLE_CAL readings")
            continue
        end
        ts = [ustrip(u"s", tones.time[k] - time0) for k in ks if !isnan(tones.cable_cal[k])]
        o = sortperm(ts); cs = cs[o]
        d = diff(cs)
        σ = robust_sigma(d)
        njump = count(x -> abs(x) > max(10σ, 100e-12), d)
        ptp = maximum(cs) - minimum(cs)
        v = if ptp == 0 || (σ == 0 && ptp < 1e-15)
            "DEAD CONSTANT"
        elseif njump > 5 && median(abs.(filter(x -> abs(x) > max(10σ, 100e-12), d))) > 0.5e-9
            @sprintf("%d WRAP-LIKE STEPS", njump)
        else
            njump == 0 ? "ok" : @sprintf("%d step(s)", njump)
        end
        push!(census, (; antenna = st, n = length(cs), median_ns = 1e9median(cs), ptp_ns = 1e9ptp,
                         sigma_ps = 1e12σ, jumps = njump, verdict = v, in_list = inlist))
        inlist && v != "ok" && !startswith(v, "0") &&
            push!(suspect, "$st is in cable_antennas but its readout is $v")
        !inlist && v == "ok" &&
            push!(suspect, "$st is EXCLUDED from cable_antennas but its readout looks healthy")
    end
    [isempty(suspect) ?
     verdict(:pass, "cable-census",
             @sprintf("cable_antennas (%s) matches what the readouts look like",
                      join(cfg.cable_antennas, ","));
             measured = (; rows = census)) :
     verdict(:warn, "cable-census", join(suspect, "; ");
             measured = (; rows = census, suspect),
             hint = "edit `cable_antennas` in the session descriptor. This controls which antenna readouts are trusted; `cable_apply` separately controls whether the correction is applied")]
end


"The tunable constants of the gap-crossing scan, each with the measurement that set it."
const CLOCK_BREAK_DEFAULTS = (
    gap_floor = 90.0,
    gap_cadence = 2.5,
    nsigma = 4.0,
    single_receptor_nsigma = 6.0,
    single_receptor_floor = 150e-12,
    npre = 20, npost = 10)

"""
    pcal_if_series(ks, cols, times, time0, ntone, nband; tone_floor) -> (t, Φ, wrapmargin)

The per-record, per-IF instrumental phase at the TONE MIDPOINT, for every receptor slot at once, over the
records where EVERY tone of EVERY receptor slot stands above `tone_floor` × its own session median. One
mask shared by both receptor slots is what makes a gap index mean the same thing in both, which the
coincidence test relies on; it also keeps a collapsing tone from entering as a phase excursion.

`wrapmargin[p, b]` is how far that (receptor slot, band)'s session tone-pair difference sits from the ±180°
wrap, in radians — property (a). It is small exactly where a naive half-angle would have flipped by π.
"""
function pcal_if_series(ks, cols, times, time0, ntone::Int, nband::Int; tone_floor)
    npol = length(cols); n = length(ks)
    amp = zeros(n, npol, ntone, nband); ph = zeros(n, npol, ntone, nband)
    for (j, k) in enumerate(ks), p in 1:npol, tn in 1:ntone, b in 1:nband
        z = ComplexF64(cols[p][k][tn, b])
        amp[j, p, tn, b] = abs(z); ph[j, p, tn, b] = angle(z)
    end
    med = [median(@view amp[:, p, tn, b]) for p in 1:npol, tn in 1:ntone, b in 1:nband]
    js = [j for j in 1:n if all(amp[j, p, tn, b] > tone_floor * med[p, tn, b]
                                for p in 1:npol, tn in 1:ntone, b in 1:nband)]
    t = [ustrip(u"s", times[ks[j]] - time0) for j in js]
    o = sortperm(t); js = js[o]; t = t[o]
    Φ = [Matrix{Float64}(undef, length(js), nband) for _ in 1:npol]
    wrapmargin = fill(NaN, npol, nband)
    for p in 1:npol, b in 1:nband
        dφ = [_wrapπ(ph[j, p, 2, b] - ph[j, p, 1, b]) for j in js]
        d0 = isempty(dφ) ? 0.0 : angle(sum(cis, dφ))
        wrapmargin[p, b] = π - abs(d0)
        for (i, j) in enumerate(js)
            Φ[p][i, b] = _wrapπ(ph[j, p, 1, b] + (d0 + _wrapπ(dφ[i] - d0)) / 2)
        end
    end
    (t, Φ, wrapmargin)
end

"Make a phase series continuous WITHIN a scan (never across the gap — the step there is the unknown)."
unwrap_local(v) = cumsum(vcat(v[1], _wrapπ.(diff(v))))

"""
    gap_step(t, Φ, g, νmid, gapthr; npre, npost) -> NamedTuple or `nothing`

The drift fitted over up to `npre` CONTIGUOUS records before the gap at index `g`, PER IF,
extrapolated across the gap and compared with the mean of up to `npost` contiguous records after it;
the per-IF residuals are decomposed into a COMMON PHASE step `dφ` [rad] and a GROUP-DELAY step `dτ`
[s] over the IF-centre lever arm. `nothing` when neither side has enough contiguous records to fit.

`σφ`/`στ` are the OLS PREDICTION standard error of that extrapolation at the post-gap epoch, which
grows with the lever arm and is what bounds the authority of a step measured across three hours
(property (c)); `scatter` is the rms of the per-IF residuals ABOUT the two-parameter model — a clock
step is coherent across every IF, a tone excursion is not; `ifmax`/`dφx`/`dτx`/`στx`/`scatterx` are
the same step with the single worst IF DROPPED (property (e)), which the caller uses only on a gap
that already flagged.
"""
function gap_step(t, Φ, g, νmid, gapthr; npre = 20, npost = 10)
    n = length(t)
    lo = g
    while lo > 1 && g - lo < npre && t[lo] - t[lo - 1] ≤ gapthr
        lo -= 1
    end
    hi = g + 1
    while hi < n && hi - g < npost && t[hi + 1] - t[hi] ≤ gapthr
        hi += 1
    end
    npre_eff = g - lo + 1; npost_eff = hi - g
    (npre_eff ≥ 4 && npost_eff ≥ 3) || return nothing
    pre = lo:g; post = (g + 1):hi
    tbar = mean(@view t[pre]); Sxx = sum(x -> (x - tbar)^2, @view t[pre])
    lever = mean(@view t[post]) - tbar
    X = [ones(npre_eff) (t[pre] .- tbar)]
    nb = length(νmid)
    res = zeros(nb); σr = zeros(nb)
    for b in 1:nb
        y = unwrap_local(Φ[pre, b])
        c = X \ y
        σr[b] = sqrt(sum(abs2, y .- X * c) / (npre_eff - 2))
        res[b] = _wrapπ(mean(unwrap_local(Φ[post, b])) - (c[1] + c[2] * lever))
    end
    σband = median(σr) * sqrt(1 / npre_eff + lever^2 / Sxx + 1 / npost_eff)
    ord = sortperm(νmid)
    ν = νmid[ord]
    resu = cumsum(vcat(res[ord[1]], _wrapπ.(diff(res[ord]))))
    A = [ones(nb) 2π .* (ν .- mean(ν))]
    rms(v, k) = length(v) > k ? sqrt(sum(abs2, v) / (length(v) - k)) : 0.0
    solve_(idx) = (S = A[idx, :]; p = S \ resu[idx]; (p, resu[idx] .- S * p, inv(S' * S)))
    p, r, C = solve_(1:nb)
    imax = nb ≥ 4 ? argmax(abs.(r)) : 0
    px, rx, Cx = imax == 0 ? (p, r, C) : solve_(filter(!=(imax), 1:nb))
    (dφ = _wrapπ(p[1]), dτ = p[2], σφ = σband * sqrt(C[1, 1]), στ = σband * sqrt(C[2, 2]),
     scatter = rms(r, 2), σband, gaplen = t[g + 1] - t[g], lever,
     ifmax = imax == 0 ? 0 : ord[imax],
     dφx = _wrapπ(px[1]), dτx = px[2], στx = σband * sqrt(Cx[2, 2]), scatterx = rms(rx, 2))
end

"σ = 0 or NaN is `cannot judge`, not `everything is an outlier`."
_exceeds(x, nσ, σs...) = (σ = maximum(σs); isfinite(σ) && σ > 0 && isfinite(x) && abs(x) > nσ * σ)

"""
    clock_break_verdicts(cfg; tones, antennas, antenna_names, time0, νmid, ntone, nband,
                         params = CLOCK_BREAK_DEFAULTS) -> Vector{Verdict}

The gap-crossing scan, as two gates and three reported censuses.

Each census row for one affected receptor slot carries `receptor_slot::Int`. The aggregate row used
when both receptor slots are jointly unscannable carries `receptor_slots = (1, 2)` instead.

WHICH CHANNEL MAKES A CANDIDATE: the DELAY. Every session adjudicated so far has spent its
adjudication throwing phase-only candidates away — a clock break IS a delay step and the phase step
follows from it — so a phase-only outlier is counted, reported as an LO offset, and kept out of the
candidate list. Nothing is hidden: the count and the epochs are in the returned data.
"""
function clock_break_verdicts(cfg; tones, antennas, antenna_names, time0, νmid, ntone, nband,
                              params = CLOCK_BREAK_DEFAULTS)
    rows_of = _pcal_rows_by_antenna(tones, antennas)
    candidates = NamedTuple[]
    single = NamedTuple[]
    toneevents = NamedTuple[]
    loffsets = NamedTuple[]
    census = NamedTuple[]
    for st in antenna_names
        ks = get(rows_of, st, Int[])
        isempty(ks) && continue
        t, Φ, wrapmargin = pcal_if_series(ks, (tones.pcal_1, tones.pcal_2), tones.time, time0,
                                          ntone, nband; cfg.tone_floor)
        if length(t) ≤ params.npre + params.npost + 2
            push!(census, (; antenna = st, receptor_slots = (1, 2), n_gaps = 0, sigma_phase_deg = NaN,
                             sigma_delay_ps = NaN, wrap_margin_deg = NaN, n_lo_offsets = 0,
                             outliers = NamedTuple[],
                             note = "too few usable records ($(length(t))) — not scanned"))
            continue
        end
        dt = diff(t)
        gapthr = max(params.gap_cadence * median(dt), params.gap_floor)
        gaps = findall(>(gapthr), dt)
        isempty(gaps) && continue
        steps = [[gap_step(t, Φp, g, νmid, gapthr; params.npre, params.npost) for g in gaps]
                 for Φp in Φ]
        dflag = Vector{Vector{Bool}}(); pflag = Vector{Vector{Bool}}(); στrob = Float64[]
        for (i, s) in enumerate(steps)
            σφ = robust_sigma([x.dφ for x in s if !isnothing(x)])
            στ = robust_sigma([x.dτ for x in s if !isnothing(x)])
            push!(στrob, στ)
            df = [isnothing(s[j]) ? false : _exceeds(s[j].dτ, params.nsigma, στ, s[j].στ)
                  for j in eachindex(gaps)]
            pf = [isnothing(s[j]) ? false : _exceeds(s[j].dφ, params.nsigma, σφ, s[j].σφ)
                  for j in eachindex(gaps)]
            push!(census, (; antenna = st, receptor_slot = i, n_gaps = length(gaps),
                             sigma_phase_deg = rad2deg(σφ), sigma_delay_ps = 1e12στ,
                             wrap_margin_deg = rad2deg(minimum(@view wrapmargin[i, :])),
                             n_lo_offsets = count(pf .& .!df),
                             outliers = [(; t = 0.5(t[gaps[j]] + t[gaps[j] + 1]), s[j].gaplen,
                                            dφ_deg = rad2deg(s[j].dφ), dτ_ps = 1e12s[j].dτ)
                                         for j in findall(df)],
                             note = ""))
            push!(dflag, df); push!(pflag, pf)
        end
        for j in eachindex(gaps)
            ss = [s[j] for s in steps]
            when = 0.5(t[gaps[j]] + t[gaps[j] + 1])
            gaplen = t[gaps[j] + 1] - t[gaps[j]]
            flagged = [p for p in eachindex(ss) if dflag[p][j]]
            if !isempty(flagged) &&
               all(p -> (x = ss[p]; x.ifmax != 0 && abs(x.dτx) < 0.5abs(x.dτ) &&
                                    !_exceeds(x.dτx, params.nsigma, στrob[p], x.στx)), flagged)
                push!(toneevents, (; antenna = st, receptor_slots = Tuple(flagged), t = when,
                                     datetime = time0 + Second(round(Int, when)),
                                     band = ss[flagged[1]].ifmax,
                                     dτ_ps = 1e12 * mean(ss[p].dτ for p in flagged),
                                     dτx_ps = 1e12 * mean(ss[p].dτx for p in flagged)))
                continue
            end
            if length(flagged) == length(ss)
                push!(candidates, (; antenna = st, t = when,
                                     datetime = time0 + Second(round(Int, when)), gaplen,
                                     dτ_ps = 1e12 * mean(x.dτ for x in ss),
                                     dφ_deg = rad2deg(mean(x.dφ for x in ss))))
            elseif isempty(flagged)
                all(pflag[p][j] for p in eachindex(pflag)) &&
                    push!(loffsets, (; antenna = st, t = when,
                                       datetime = time0 + Second(round(Int, when)),
                                       dφ_deg = rad2deg(mean(x.dφ for x in ss))))
            else
                for p in flagged
                    x = ss[p]; other = ss[3 - p]
                    _exceeds(x.dτ, params.single_receptor_nsigma, στrob[p], x.στ) || continue
                    abs(x.dτ) ≥ params.single_receptor_floor || continue
                    x.scatter ≤ 3x.σband || continue
                    (isnothing(other) || abs(other.dτ) < 0.5abs(x.dτ)) || continue
                    push!(single, (; antenna = st, receptor_slot = p, t = when,
                                     datetime = time0 + Second(round(Int, when)), gaplen,
                                     dτ_ps = 1e12x.dτ, dφ_deg = rad2deg(x.dφ),
                                     other_ps = isnothing(other) ? NaN : 1e12other.dτ))
                end
            end
        end
    end
    m = (; census, candidates, single_receptor = single, tone_events = toneevents,
           lo_offsets = loffsets)
    [isempty(candidates) ?
     verdict(:pass, "clock-breaks",
             @sprintf("no gap-crossing DELAY step is a %.0fσ outlier in both receptor slots",
                      params.nsigma); measured = m) :
     verdict(:warn, "clock-breaks",
             @sprintf("%d gap(s) where the DELAY step is a %.0fσ outlier in BOTH receptor slots — candidates, not verdicts",
                      length(candidates), params.nsigma); measured = m,
             hint = "inspect those epochs before solving: the PWL clock model of `GlobalDelayFit` has no break machinery, so a real break is absorbed as troposphere and biases the positions. This is the delay-series half of the test; the ambiguity-free tone-pair delay series around the epoch must also agree across every IF and persist"),
     isempty(single) ?
     verdict(:pass, "clock-breaks-1receptor",
             @sprintf("no delay step in a single receptor slot reaches %.0fσ and %.0f ps",
                      params.single_receptor_nsigma, 1e12params.single_receptor_floor);
             measured = m) :
     verdict(:warn, "clock-breaks-1receptor",
             @sprintf("%d delay step(s) in a single receptor slot beyond %.0fσ and %.0f ps — the other receptor slot does not show them",
                      length(single), params.single_receptor_nsigma,
                      1e12params.single_receptor_floor); measured = m,
             hint = "an antenna clock break moves both receptor slots, so most candidates in a single receptor slot are LO phase offsets or tone excursions; inspect them because the rule requiring both receptor slots cannot detect a genuine event in only one receptor slot")]
end


"""
    amplitude_verdicts(tsys_calibration; tsys, gain, antennas, antenna_names) -> Vector{Verdict}

The Tsys-derived amplitude scale: the Tsys sentinels and coverage against
`tsys_calibration.tsys_ceiling` and `tsys_calibration.fallback_cap`, and whether there are gain
curves to turn a Tsys into an SEFD.

These are shared calibrated-visibility settings. The astrometric delay table does not read them, so a
file without them still produces positions; it cannot produce calibrated visibilities, which is why
neither condition is a `:fail`.
"""
function amplitude_verdicts(tsys_calibration; tsys, gain, antennas, antenna_names)
    out = Verdict[]
    if isempty(tsys)
        push!(out, verdict(:warn, "tsys", "no SYSTEM_TEMPERATURE table"; measured = (; n = 0),
                           hint = "calibrated visibilities have no Tsys-derived scale because `TsysInit` cannot run"))
    else
        nb = length(first(tsys.tsys_1))
        census = NamedTuple[]; bad = String[]
        for st in antenna_names
            ks = [k for k in eachindex(tsys.antenna_no)
                  if 1 ≤ tsys.antenna_no[k] ≤ length(antennas) && antennas[tsys.antenna_no[k]] == st]
            if isempty(ks)
                push!(bad, "$st has no Tsys records")
                push!(census, (; antenna = st, n = 0, median_k = NaN, over_pct = NaN, nan_pct = NaN))
                continue
            end
            v = Float64[]
            for k in ks, b in 1:nb, col in (tsys.tsys_1, tsys.tsys_2)
                push!(v, ustrip(u"K", col[k][b]))
            end
            good = filter(x -> isfinite(x) && 0 < x ≤ tsys_calibration.tsys_ceiling, v)
            nover = count(x -> isfinite(x) && x > tsys_calibration.tsys_ceiling, v)
            nnan = count(!isfinite, v)
            push!(census, (; antenna = st, n = length(ks),
                             median_k = isempty(good) ? NaN : median(good),
                             over_pct = 100nover / length(v), nan_pct = 100nnan / length(v)))
            frac = (nover + nnan) / length(v)
            frac > tsys_calibration.fallback_cap &&
                push!(bad, @sprintf("%s: %.0f%% of its Tsys values are sentinels or non-finite",
                                    st, 100frac))
        end
        push!(out, isempty(bad) ?
            verdict(:pass, "tsys-census",
                    @sprintf("every antenna stays under the %.0f%% fallback cap against the %.0f K ceiling",
                             100tsys_calibration.fallback_cap, tsys_calibration.tsys_ceiling);
                    measured = (; rows = census)) :
            verdict(:warn, "tsys-census", join(bad, "; "); measured = (; rows = census, bad),
                    hint = "the Tsys initialization falls back to the antenna's own median for those cells; above `fallback_cap` it errors, so calibrated-visibility production stops instead of using an inadequately constrained amplitude scale. The census counts invalid Tsys values, while the cap applies to receptor-slot/scan/IF cells that actually fell back"))
    end
    push!(out, length(gain) > 0 ?
        verdict(:pass, "gain-curves", "$(length(gain)) GAIN_CURVE entries (DPFU + relative gain per IF)";
                measured = (; n = length(gain))) :
        verdict(:warn, "gain-curves", "no GAIN_CURVE table"; measured = (; n = 0),
                hint = "the Tsys initialization has no DPFU and cannot turn a Tsys into an SEFD: calibrated visibilities have no absolute amplitude scale — the astrometric delay stages are unaffected. Treat any image as uncalibrated in flux"))
    out
end


"""
Distinct sources in each fixed-duration cell of the schedule, and each cell's own occupied span. A tail
cell the session ends in the middle of is not a thin schedule — it is a stub, and is reported
separately rather than setting the minimum.
"""
function amp_cells(tt::AbstractVector, sid::AbstractVector, Δ::Float64)
    t0, t1 = extrema(tt)
    nc = max(ceil(Int, (t1 - t0) / Δ), 1)
    sets = [Set{Int}() for _ in 1:nc]
    lo = fill(Inf, nc); hi = fill(-Inf, nc)
    for j in eachindex(tt)
        c = clamp(floor(Int, (tt[j] - t0) / Δ) + 1, 1, nc)
        push!(sets[c], sid[j])
        lo[c] = min(lo[c], tt[j]); hi[c] = max(hi[c], tt[j])
    end
    occ = findall(!isempty, sets)
    (n = [length(sets[c]) for c in occ], span = [hi[c] - lo[c] for c in occ])
end

"""
    identifiability_verdicts(img; t, source_ix) -> Vector{Verdict}

The condition `amp_normalize = :global` stands on: every amplitude solution interval must contain an
ENSEMBLE of sources, or its common-mode gain is degenerate with the one source in it being brighter
than its model. That is a property of the OBSERVING SCHEDULE, so it is measured here, before any
imaging.
"""
function identifiability_verdicts(img; t, source_ix)
    if img.amp_interval isa Symbol
        return [verdict(:warn, "amp-identifiability",
                        "amp_interval = $(img.amp_interval) is not a duration";
                        measured = (; amp_interval = img.amp_interval),
                        hint = "the census assumes a fixed-duration interval; with :scan every cell has exactly one source, which is the configuration measured to over-fit")]
    end
    Δ = float(ustrip(u"s", img.amp_interval))
    cells = amp_cells(collect(t), collect(source_ix), Δ)
    full = findall(≥(0.5Δ), cells.span)
    n = cells.n[full]
    m = (; interval = img.amp_interval, n_full = length(full), n_occupied = length(cells.n),
           min = minimum(n), max = maximum(n), median = median(n),
           stubs = cells.n[setdiff(eachindex(cells.n), full)])
    if img.amp_normalize === :global && minimum(n) < 2
        [verdict(:warn, "amp-identifiability",
                 @sprintf("%d of %d intervals contain a single source, with amp_normalize = :global",
                          count(<(2), n), length(n)); measured = m,
                 hint = "in such a cell the array-common gain is exactly degenerate with that source being brighter than its model, and the solve takes the freedom — measured on BK255AQ per-scan: closure-|V| χ² 0.7, below what the external models themselves reach, and 0851+202's structure PA 48.6° off. Lengthen `amp_interval` or use :percell")]
    elseif minimum(n) < 4
        [verdict(:warn, "amp-identifiability",
                 @sprintf("the thinnest interval has only %d sources", minimum(n)); measured = m,
                 hint = "the common mode is constrained by that ensemble alone; watch the closure-amplitude χ² and the external flux comparison of `external_comparison`")]
    else
        [verdict(:pass, "amp-identifiability",
                 @sprintf("every %s interval carries ≥ %d sources", img.amp_interval, minimum(n));
                 measured = m)]
    end
end


"""
    closure_triangles(obs) -> (; closure, min_snr, thermal)

Triangle sums of the raw measured delays per (time cell, receptor-slot pair). Antenna-based terms cancel exactly
in a triangle, so what remains is measurement noise plus whatever does not close — and that is the
number that says what the delays are worth before any model or estimator exists.
"""
function closure_triangles(obs)
    byscan = Dict{Tuple{Int,NTuple{2,Int}}, Dict{Tuple{Int,Int}, Int}}()
    receptor_slots = receptor_slots_column(obs)
    for i in eachindex(obs)
        d = get!(() -> Dict{Tuple{Int,Int},Int}(), byscan, (obs.tcell[i], receptor_slots[i]))
        d[obs.baseline_ix[i].antennas] = i
    end
    cl = Float64[]; ms = Float64[]; th = Float64[]
    for (_, d) in byscan, (pq, i1) in d
        p, q = pq
        for (rs, i2) in d
            (rs[1] == q && haskey(d, (p, rs[2]))) || continue
            i3 = d[(p, rs[2])]
            push!(cl, obs.delay[i1] + obs.delay[i2] - obs.delay[i3])
            push!(ms, min(obs.snr[i1], obs.snr[i2], obs.snr[i3]))
            push!(th, sqrt(obs.sigma_delay[i1]^2 + obs.sigma_delay[i2]^2 + obs.sigma_delay[i3]^2))
        end
    end
    (; closure = cl, min_snr = ms, thermal = th)
end

"""
The column containing the ordered pair of structural receptor-slot identifiers in a tidy peak table.
"""
receptor_slots_column(tbl) = tbl.receptor_slots

"""
The SNR bins the closure floor is read in, weakest leg first in the tuple's own order. The floor is
quoted on the STRONG triangles, where the thermal term is negligible and what is left is the
systematic.
"""
const CLOSURE_SNR_BINS = ((1000.0, Inf, "> 1000"), (300.0, 1000.0, "300–1000"),
                          (100.0, 300.0, "100–300"), (0.0, 100.0, "< 100"))

"""
    closure_floor(obs; bins = CLOSURE_SNR_BINS, strong = 1000.0) -> (; rows, floor_ps, n_strong)

The non-closing systematic per baseline: over the triangles whose WEAKEST leg is above `strong`, the
excess of the triangle-sum scatter over thermal, divided by √3 because three baselines contribute
independently to one triangle sum.

`rows` carries every SNR bin — the floor is a systematic and must be seen to be flat against SNR
before it is believed — and `floor_ps` is the strong-triangle value the error budget stands on. A
session with no strong triangle gets `NaN` and `n_strong == 0`, which is an absence and not a zero.
"""
function closure_floor(obs; bins = CLOSURE_SNR_BINS, strong = 1000.0)
    (; closure, min_snr, thermal) = closure_triangles(obs)
    rows = NamedTuple[]
    for (lo, hi, label) in bins
        m = findall(j -> lo ≤ min_snr[j] < hi, eachindex(min_snr))
        isempty(m) && continue
        σ = robust_sigma(closure[m]); tm = median(thermal[m])
        push!(rows, (; label, lo, hi, n = length(m), sigma_ps = 1e12σ, thermal_ps = 1e12tm,
                       excess_ps = 1e12sqrt(max(σ^2 - tm^2, 0)) / sqrt(3)))
    end
    k = findall(j -> min_snr[j] ≥ strong, eachindex(min_snr))
    (; rows, n_triangles = length(closure), n_strong = length(k),
       floor_ps = isempty(k) ? NaN :
                  (σ = robust_sigma(closure[k]); tm = median(thermal[k]);
                   1e12sqrt(max(σ^2 - tm^2, 0)) / sqrt(3)))
end

"""
    fringe_census_verdicts(cfg; obs) -> Vector{Verdict}

What the fringe product says about itself: the detection fraction against the session's `min_snr`,
the window occupancy measured on the DETECTIONS (the captured product is ungated, and a
noise-dominated peak lands anywhere in the search window by construction), and the closure floor.
"""
function fringe_census_verdicts(cfg; obs)
    out = Verdict[]
    det = count(≥(cfg.min_snr), obs.snr) / length(obs)
    push!(out, det ≥ 0.9 ?
        verdict(:pass, "detection-fraction",
                @sprintf("%.4f of peaks above SNR %.0f", det, cfg.min_snr);
                measured = (; n = length(obs), detection_fraction = det, gate = cfg.min_snr,
                              snr_median = median(obs.snr))) :
        verdict(:warn, "detection-fraction",
                @sprintf("%.4f of peaks reach SNR %.0f", det, cfg.min_snr);
                measured = (; n = length(obs), detection_fraction = det, gate = cfg.min_snr,
                              snr_median = median(obs.snr)),
                hint = "weak sources, short scans, and calibration errors can all lower this fraction; the estimator applies the same SNR floor again"))
    det_ix = findall(≥(cfg.min_snr), obs.snr)
    dmax = isempty(det_ix) ? NaN : maximum(abs, view(obs.delay, det_ix))
    rmax = isempty(det_ix) ? NaN : maximum(abs, view(obs.rate, det_ix))
    dlim = ustrip(u"s", rightendpoint(cfg.window.delay))
    rlim = ustrip(u"Hz", rightendpoint(cfg.window.rate))
    m = (; n_detections = length(det_ix), delay_max_ns = 1e9dmax, rate_max_mHz = 1e3rmax,
           delay_limit_ns = 1e9dlim, rate_limit_mHz = 1e3rlim,
           delay_pct = 100dmax / dlim, rate_pct = 100rmax / rlim)
    push!(out, isempty(det_ix) ?
        verdict(:warn, "search-window", "no peak reaches the session's SNR gate — nothing to measure";
                measured = m,
                hint = "the window occupancy is a statement about the DETECTED residuals; with no detection there is none to make, and the detection fraction above is the thing to read") :
        (dmax < 0.9dlim && rmax < 0.9rlim) ?
        verdict(:pass, "search-window",
                @sprintf("both maxima over the %d detections sit inside 90%% of the window",
                         length(det_ix)); measured = m) :
        verdict(:warn, "search-window",
                @sprintf("|delay| reaches %.0f%% and |rate| %.0f%% of the window over the %d detections",
                         100dmax / dlim, 100rmax / rlim, length(det_ix)); measured = m,
                hint = "peaks are piling up at the edge: `cfg.window` was sized on this project's reference session's residuals and this dataset's clocks are larger — widen it and re-run the fringe pass"))
    cf = closure_floor(obs)
    push!(out, !isfinite(cf.floor_ps) ?
        verdict(:warn, "closure-floor", "no triangle with all three baselines above SNR 1000";
                measured = cf,
                hint = "the floor is only measurable on the strong triangles; on a weaker session read the next bin down instead") :
        verdict(:pass, "closure-floor",
                @sprintf("%.1f ps per baseline at min-SNR > 1000, over %d triangles",
                         cf.floor_ps, cf.n_strong); measured = cf))
    out
end


"""
    orientation_verdicts(; external_available, models_available) -> Vector{Verdict}

Whether the INPUTS of the external comparison are here — the part this check can settle from the file
system, and the part whose absence has a cost worth stating. The comparison itself is
[`external_comparison`](@ref), which reports and never fails: handedness is read by comparing our
reduction with somebody else's, which a human weighs.
"""
orientation_verdicts(; external_available, models_available) =
    [!external_available ?
     verdict(:warn, "orientation", "no external models — the comparison cannot run";
             measured = (; external_available, models_available),
             hint = "without an independent image of the same sky, nothing pins map handedness; internal agreement alone cannot detect a global reflection through the map origin") :
     !models_available ?
     verdict(:warn, "orientation", "this run has no model set to compare — the `models` step has not run";
             measured = (; external_available, models_available),
             hint = "the comparison runs after the `models` step; `sessions/diagnostics/external_comparison.jl` is what performs it") :
     verdict(:pass, "orientation",
             "external models and our own model set both present — the comparison can run";
             measured = (; external_available, models_available))]


"""
    dataset_verdicts(sections...) -> (; verdicts, n_pass, n_warn, n_fail)

The verdicts of every section, counted. `n_fail > 0` is the one outcome a caller should act on: a
condition under which everything downstream would be silently wrong, established from the input data
alone.
"""
function dataset_verdicts(sections...)
    v = reduce(vcat, sections; init = Verdict[])
    (; verdicts = v, n_pass = count(x -> x.level === :pass, v),
       n_warn = count(x -> x.level === :warn, v), n_fail = count(x -> x.level === :fail, v))
end
