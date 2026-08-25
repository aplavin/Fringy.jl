
export FringeFit

"""
    FringeFit(; tiles, window, oversample, refine, selection = Selection())

Coherent FFT fringe measurement. For every occupied time cell, frequency cell, and baseline, it
divides out the solution's current gain chain, gathers the residual, takes the windowed FFT peak for
each structural receptor-slot pair, optionally refines it, and records it. It solves nothing and applies nothing: its
only output is a [`FringeTable`](@ref).

`tiles = (; time::TimePartition, frequency::FreqPartition)` are the measurement cells (root-bound);
`window = (; rate, delay)` the retained search windows (Unitful closed intervals); `oversample =
(; rate, delay)` the padding factors (≥1); `refine` a `LocalML`/`NoRefine` (the sub-bin peak refinement,
the `LocalML` docstring); `selection` restricts the data. All keyword arguments except `selection` are required.

All peaks are recorded UNGATED: an SNR floor is a property of a solve, not of a measurement, and belongs
to the step that consumes the table ([`FringeSelf`](@ref), [`DeriveIFAlignment`](@ref)).

Run it directly on a solution — `table = fit(sol)` — or as a schedule step wrapped in
`Capture(Fringes() => :name, fit)`. Execution uses
`min(Threads.nthreads(), number_of_occupied_time_cells)` tasks; results do not depend on the thread
count.
"""
struct FringeFit{TG, W, O, R, S} <: Step
    tiles::TG
    window::W
    oversample::O
    refine::R
    selection::S
end

function FringeFit(; tiles, window, oversample, refine, selection = Selection())
    all(k -> k ≥ 1, (oversample.rate, oversample.delay)) || error("FringeFit oversample factors must be ≥ 1")
    refine isa Union{LocalML, NoRefine} ||
        error("FringeFit refine must be a LocalML or NoRefine, got $(typeof(refine))")
    FringeFit(tiles, window, oversample, refine, selection)
end

function _foreach_tile(f, n::Integer)
    ntasks = min(Threads.nthreads(), n)
    ntasks ≤ 1 && return foreach(f, 1:n)
    next = Threads.Atomic{Int}(1)
    @sync for _ in 1:ntasks
        Threads.@spawn while true
            k = Threads.atomic_add!(next, 1)
            k > n && break
            f(k)
        end
    end
    nothing
end

function _validate_axes(seldata::Dataset, rows_ts, chans_fs, ts, fs)
    length(unique(@view seldata.freq.ν[chans_fs])) ≥ 2 ||
        error("FringeFit: tile (time cell $ts, freq cell $fs) has <2 distinct frequency channels — delay is unconstrained")
    length(unique(@view seldata.rows.t[rows_ts])) ≥ 2 ||
        error("FringeFit: tile (time cell $ts, freq cell $fs) has <2 distinct time samples — rate is unconstrained")
end

"""
    apply_step(step::FringeFit, sol, ::Fringes) -> (sol, table)

Measure the fringe table of `sol`'s residual. The solution is returned UNCHANGED (a `FringeFit` installs
nothing); the product is the [`FringeTable`](@ref). Reached through `Capture(Fringes() => :name, fit)`,
or directly as `fit(sol)`.
"""
function apply_step(step::FringeFit, sol::Solution, ::Fringes)
    ds = dataset(sol)
    Tt = step.tiles.time
    Tf = step.tiles.frequency
    states = _resolve_terms(sol, jones_terms(sol))

    seldata = select(ds, step.selection)
    nrows(seldata) ≥ 1 || error("FringeFit: selection matched no rows")
    prows = parentrows(seldata)
    pchans = parentchannels(seldata)
    row_tcell = map(r -> cell_of_row(Tt, r), prows)
    chan_fcell = map(c -> cell_of_channel(Tf, pchans[c]), 1:nchannels(seldata))
    occupied_fs = sort!(unique(chan_fcell))
    chans_of_fs = Dictionary(occupied_fs, map(fs -> findall(==(fs), chan_fcell), occupied_fs))
    ifs_of_fs = Dictionary(occupied_fs, map(fs -> sort!(unique(@view seldata.freq.if_of[chans_of_fs[fs]])), occupied_fs))
    ν_fs = ustrip.(u"Hz", references(Tf))

    rows_of_ts = groupfind(identity, row_tcell)
    tile_cells = sort!(collect(keys(rows_of_ts)))
    tile_rows = Vector{Vector{NamedTuple}}(undef, length(tile_cells))

    _foreach_tile(length(tile_cells)) do k
        ts = tile_cells[k]
        rows_ts = rows_of_ts[ts]
        srcs = unique(@view seldata.rows.source[rows_ts])
        length(srcs) == 1 ||
            error("FringeFit: tile (time cell $ts) spans multiple sources $(srcs) — selected data within one tile must be single-source")
        t0 = ustrip(u"s", references(Tt)[ts])

        bl_of_ts = seldata.rows.baseline_ix[rows_ts]
        baselines_ts = sort(unique(bl_of_ts); by = b -> b.antennas)
        cells_bl = Dictionary(baselines_ts, map(_ -> [fill(_nanmeas(), 2, 2) for _ in occupied_fs], baselines_ts))
        ifw_bl = Dictionary(baselines_ts, map(_ -> [fill(zero(_IFW), length(ifs_of_fs[fs])) for fs in occupied_fs], baselines_ts))
        bin_bl = Dictionary(baselines_ts, map(_ -> (; delay = fill(NaN, length(occupied_fs)),
                                                     rate = fill(NaN, length(occupied_fs))), baselines_ts))

        for (kfs, fs) in enumerate(occupied_fs)
            chans_fs = chans_of_fs[fs]
            _validate_axes(seldata, rows_ts, chans_fs, ts, fs)
            ν0 = ν_fs[fs]
            for bl in baselines_ts
                bkeep = rows_ts[findall(==(bl), bl_of_ts)]
                bdata = _subset(seldata, bkeep, chans_fs)
                g = fringe_grid(bdata, states)
                ifw_bl[bl][kfs] = g.Dif
                delay_bin = NaN
                rate_bin = NaN
                for i in 1:2, j in 1:2
                    Dij = g.D[i, j]
                    Dij > 0 || continue
                    S = getindex.(g.grids, i, j)
                    tr = fringe_transform(S, g.tlat, g.νlat; window = step.window, oversample = step.oversample, freq_snap = g.νsnap)
                    pk = fringe_measure(S, tr.G, tr.rate, tr.delay, g.tlat, g.νlat, t0, ν0, Dij;
                                        σν = g.σν[i, j], refine = step.refine)
                    cells_bl[bl][kfs][i, j] =
                        (; pk.coeff, pk.delay, pk.rate, pk.snr, pk.sigma_delay, weight = Dij, pk.Q, pk.q_ab)
                    if isnan(delay_bin)
                        delay_bin = length(tr.delay) > 1 ? tr.delay[2] - tr.delay[1] : 1.0
                        rate_bin = length(tr.rate) > 1 ? tr.rate[2] - tr.rate[1] : 1.0
                    end
                end
                bin_bl[bl].delay[kfs] = delay_bin
                bin_bl[bl].rate[kfs] = rate_bin
            end
        end

        src = seldata.rows.source[rows_ts[1]]
        sid = seldata.rows.source_ix[rows_ts[1]]
        dt = datetime_at(ds, t0)
        rows_out = NamedTuple[]
        for bl in baselines_ts
            p, q = bl.antennas
            blsym = Baseline((ds.antennas[p].name, ds.antennas[q].name))
            push!(rows_out, (;
                source = src, source_ix = sid, baseline = blsym, baseline_ix = bl,
                tcell = ts, datetime = dt, t = t0,
                fcells = collect(occupied_fs), ν = ν_fs[occupied_fs],
                cells = [SMatrix{2,2,_MEAS,4}(cells_bl[bl][kk]) for kk in eachindex(occupied_fs)],
                ifs = [copy(ifs_of_fs[fs]) for fs in occupied_fs],
                ifweights = ifw_bl[bl],
                delay_binwidth = bin_bl[bl].delay, rate_binwidth = bin_bl[bl].rate))
        end
        tile_rows[k] = rows_out
    end

    table = FringeTable(StructArray(identity.(reduce(vcat, tile_rows))), step.tiles,
                        nantennas(ds), map(a -> a.name, ds.antennas))
    (sol, table)
end

apply_step(::FringeFit, ::Solution, ::Nothing) =
    error("FringeFit produces only a measurement; wrap it in `Capture(Fringes() => :name, …)` or call it directly")

"""
    (step::FringeFit)(sol::Solution) -> FringeTable

Measure `sol`'s fringe table directly — the pure-function form, for scripts and for
[`FringeSelf`](@ref)/[`DeriveIFAlignment`](@ref) inputs.
"""
(step::FringeFit)(sol::Solution) = last(apply_step(step, sol, Fringes()))
