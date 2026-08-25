
_soln_spec(::Type{PhaseOffset})          = (; label = "φ₀",      unit = "rad", scale = 1.0,  phase = true)
_soln_spec(::Type{Delay})                = (; label = "delay",   unit = "ns",  scale = 1e9,  phase = false)
_soln_spec(::Type{Rate})                 = (; label = "rate",    unit = "mHz", scale = 1e3,  phase = false)
_soln_spec(::Type{PhaseBandpass})        = (; label = "φ(bp)",   unit = "rad", scale = 1.0,  phase = true)
_soln_spec(::Type{LogAmplitudeBandpass}) = (; label = "log|bp|", unit = "",    scale = 1.0,  phase = false)
_kind_label(K::Type) = _soln_spec(K).label

receptor_label(ds, a::Integer, r::Integer) =
    (P = ds.antennas[a].poltypes[r]; P === :U ? string(r) : string(P))
function receptor_label(ds, r::Integer)
    ps = unique(a.poltypes[r] for a in ds.antennas)
    length(ps) == 1 && only(ps) !== :U ? string(only(ps)) : string(r)
end

correlation_product_label(ds, p, q, i, j) = receptor_label(ds, p, i) * receptor_label(ds, q, j)

"""
    enabled_apply(apply_sa) -> Tuple

Enabled component kinds of `view.grid.apply`, in `ctx.components` order — the `terms=` tuple for
`calibrated_dataset`. `()` ⇒ identity (raw data).
"""
enabled_apply(apply_sa) = Tuple(apply_sa.kind[k] for k in eachindex(apply_sa) if apply_sa.enabled[k])

resolve_pair(baselines, A, B) =
    any(bl -> bl.antennas == (A, B), baselines) ? (A, B) :
    any(bl -> bl.antennas == (B, A), baselines) ? (B, A) : nothing

"""
    window_ds(ds, sel; baseline=nothing, channels=Colon()) -> Dataset

Parent-identity-preserving subset of `ds` to the DateTime interval `sel` (optionally one directed
`baseline` and a channel subset). Built via `select(ds, Selection(...))`: the subset keeps the parent
antenna/source tables, dense encodings and `time0`, so root-bound partitions stay resolvable.
"""
window_ds(ds, sel; baseline = nothing, channels = Colon()) =
    select(ds, Selection(r -> r.datetime ∈ sel &&
                              (baseline === nothing || r.baseline_ix.antennas == baseline);
                         channels))


"""
    chan_info(ds)

Per-channel frequency-axis summary for the grid/channel x-axis. `if_bounds` = channels `c` where
`c`/`c+1` fall in different IFs (x-tick/divider); single-IF ⇒ no breaks. For the "sort by frequency"
toggle it also carries the frequency-sort permutation `perm` (displayed index → true channel), the
sorted `ν_sorted = ν[perm]`, and `if_bounds_sorted` (IF breaks in sorted order). IF membership is read
directly from `ds.freq.if_of`.
"""
function chan_info(ds)
    ifmem = ds.freq.if_of
    bound(mem) = findall(c -> mem[c] != mem[c + 1], 1:length(mem) - 1)
    perm = sortperm(ds.freq.ν)
    (; n = length(ds.freq.ν), ν = ds.freq.ν, if_bounds = bound(ifmem),
       perm, ν_sorted = ds.freq.ν[perm], if_bounds_sorted = bound(ifmem[perm]))
end

"""
    uv_points(ds)

`(u, v)` per time-baseline row (uvw rides per row, not per channel). uvw stored in METRES (Unitful) →
ustrip to Float64. The panel plots (u,v) AND (−u,−v) (Hermitian); the selected-interval highlight gates
on `ds.rows.datetime` (1:1 with these points), so no time coordinate is stored here.
"""
uv_points(ds) = StructArray(map(uvw -> (; u = ustrip(u"m", uvw[1]), v = ustrip(u"m", uvw[2])), ds.rows.uvw))

"""
    antenna_lonlat(ds)

`(name, lon, lat)` per antenna in dense `ds.antennas` order (index keys `_antcolor`). Longitude and
latitude are bare WGS84 radians derived from ECEF coordinates; degrees belong at the plot layer.
"""
antenna_lonlat(ds) = StructArray(map(ds.antennas) do a
    g = LLA(ECEF(a.xyz), wgs84)
    (; name = String(a.name), lon = deg2rad(g.lon), lat = deg2rad(g.lat))
end)


_stage_time_partition(st::FringeFit)      = st.tiles.time
_stage_time_partition(st::FringeSelf) = st.fit.tiles.time
_stage_time_partition(st::StEFCal)         = st.coherence.time
_stage_time_partition(st::Capture)         = _stage_time_partition(st.step)
_stage_time_partition(::Step)              = nothing

_stage_name(::FringeFit)       = :fringefit
_stage_name(::FringeSelf) = :fringeself
_stage_name(::StEFCal)         = :stefcal
_stage_name(st::Capture)       = _stage_name(st.step)
_stage_name(::Step)            = :stage

_same_cells(a, b) = length(a) == length(b) &&
    all(abs(leftendpoint(a[k]) - leftendpoint(b[k])) ≤ 1e-6 &&
        abs(rightendpoint(a[k]) - rightendpoint(b[k])) ≤ 1e-6 for k in eachindex(a))

function _wrap_terms(terms; width = 12)
    lines = String[]; cur = ""
    for t in terms
        cur = isempty(cur) ? t : length(cur) + 1 + length(t) ≤ width ? cur * " " * t :
              (push!(lines, cur); t)
    end
    isempty(cur) || push!(lines, cur)
    join(lines, "\n")
end

"""
    timeline_lines(ctx)

Selectable horizontal interval lines: scan + whole + each schedule stage's tile time partition + each
seed-solution component's time partition. Partitions with identical cells MERGE into one line, each
contributing its single-word term; scan/whole go first so they lead merged rows and sort to the top.
Merged terms are line-wrapped into a multi-line label. `StructArray((; label, cells))` with
`cells::Vector{ClosedInterval{Float64}}` in absolute Unix seconds.
"""
function timeline_lines(ctx)
    ds = ctx.ds
    t0u = datetime2unix(ds.time0)
    cellsof(part) = ClosedInterval{Float64}[
        (ustrip(u"s", leftendpoint(iv)) + t0u)..(ustrip(u"s", rightendpoint(iv)) + t0u) for iv in supports(part)]

    entries = Tuple{String, Vector{ClosedInterval{Float64}}}[]
    push!(entries, ("scan",  cellsof(partition(ds, ByScan()))))
    push!(entries, ("whole", cellsof(partition(ds, WholeObservation()))))
    for st in ctx.schedule
        tp = _stage_time_partition(st)
        tp === nothing && continue
        push!(entries, (string(_stage_name(st)), cellsof(tp)))
    end
    if ctx.sol0 !== nothing
        for K in ctx.components
            push!(entries, (_kind_label(K), cellsof(ctx.sol0[K].definition.time)))
        end
    end

    merged = Tuple{Vector{String}, Vector{ClosedInterval{Float64}}}[]
    for (lab, cells) in entries
        k = findfirst(m -> _same_cells(m[2], cells), merged)
        k === nothing ? push!(merged, ([lab], cells)) : push!(merged[k][1], lab)
    end
    StructArray((; label = [_wrap_terms(terms) for (terms, _) in merged], cells = last.(merged)))
end

"""
    presence(ds)

Present (time, source/antenna) scatter points for the presence lanes, one pass over `ds.rows`: each row
contributes its time to its source and to both its antennas. `(; sources, antennas)`, each
`(; names, pts)` with `pts` a `StructArray((; t, row))` — `t` absolute Unix seconds, `row` the 1-based
category index.
"""
function presence(ds)
    ts = datetime2unix.(ds.rows.datetime)
    usrc = sort(unique(ds.rows.source))
    srcidx = Dict(n => i for (i, n) in enumerate(usrc))
    srcnames = string.(usrc)
    antnames = [String(a.name) for a in ds.antennas]
    sources = StructArray((t = ts, row = [srcidx[s] for s in ds.rows.source]))
    antennas = StructArray(flatmap(eachindex(ds.rows)) do k
        [(; t = ts[k], row = a) for a in ds.rows.baseline_ix[k].antennas]
    end)
    (; sources = (; names = srcnames, pts = sources), antennas = (; names = antnames, pts = antennas))
end


"""
    occupancy(sol) -> Dict{Type, BitArray{4}}

Per component kind, a `[antenna, receptor slot, tcell, fcell]` bitmask flagging cells that carry any present
datum — built in ONE traversal of the root dataset. For each present correlation `(i,j)` of a
row/channel, both endpoint nodes `(p,i)` / `(q,j)` are OR-ed into every component's own cell (its time
cell of the row, its frequency cell of the channel). Masks never-observed identity-zero cells in the
Solution panel.
"""
function occupancy(sol::Solution)
    ds = dataset(sol)
    nant = nantennas(ds)
    kinds = jones_terms(sol)
    defs = map(K -> sol[K].definition, kinds)
    occs = map(d -> falses(nant, 2, ncells(d.time), ncells(d.frequency)), defs)
    blix = ds.rows.baseline_ix; wcol = ds.rows.weight
    for j in 1:nrows(ds)
        p, q = blix[j].antennas
        tcs = map(d -> cell_of_row(d.time, j), defs)
        Wrow = wcol[j]
        for c in 1:nchannels(ds)
            fcs = map(d -> cell_of_channel(d.frequency, c), defs)
            for (i, jj) in present(Wrow[c])
                for k in eachindex(defs)
                    o = occs[k]
                    @inbounds o[p, i, tcs[k], fcs[k]] = true
                    @inbounds o[q, jj, tcs[k], fcs[k]] = true
                end
            end
        end
    end
    Dict{Type, BitArray{4}}(K => o for (K, o) in zip(kinds, occs))
end


"""
    varying_axis(state::ComponentState) -> Symbol

Which partition axis a component's curve is drawn against: `:time` for an alignment term whose time
partition has >1 cell, else `:freq`; bandpass kinds always `:freq`.
"""
varying_axis(s::ComponentState{<:PhaseBandpass})        = :freq
varying_axis(s::ComponentState{<:LogAmplitudeBandpass}) = :freq
varying_axis(s::ComponentState) = ncells(s.definition.time) > 1 ? :time : :freq

const _SolnTrace = @NamedTuple{antenna::Int, receptor_slot::Int, xs::Vector{Float64}, ys::Vector{Float64}}
const _TimeSolnTrace = @NamedTuple{antenna::Int, receptor_slot::Int, frequency_cell::Int,
                                  xs::Vector{Float64}, ys::Vector{Float64}}

"""
    soln_traces(state, occ; xaxis) -> Vector

For `:time`, one trace per (antenna, receptor slot, frequency cell). For `:freq`, one trace per
(antenna, receptor slot). `xs` = the varying-axis partition references (SI: seconds-from-time0 for
`:time`, Hz for `:freq`), `ys` = `state.values`, over every occupied cell (masked by `occ`, `occupancy`'s
bitmask for this kind). Points are sorted by `xs`. Never-observed (identity-zero) cells are dropped.
"""
function soln_traces(state::ComponentState, occ::BitArray{4}; xaxis::Symbol)
    soln_traces(state, occ, Val(xaxis))
end

function soln_traces(state::ComponentState, occ::BitArray{4}, ::Val{:time})
    vals = state.values
    nant, nrec, ntc, nfc = size(vals)
    size(occ) == size(vals) || error("soln_traces: occupancy mask size $(size(occ)) ≠ values size $(size(vals))")
    trefs = ustrip.(u"s", references(state.definition.time))
    torder = sortperm(trefs)
    filtermap(Iterators.product(1:nant, 1:nrec, 1:nfc)) do (a, r, fc)
        tcs = filter(tc -> @inbounds(occ[a, r, tc, fc]), torder)
        isempty(tcs) && return nothing
        (; antenna = a, receptor_slot = r, frequency_cell = fc,
           xs = trefs[tcs], ys = [@inbounds(vals[a, r, tc, fc]) for tc in tcs])
    end
end

function soln_traces(state::ComponentState, occ::BitArray{4}, ::Val{:freq})
    vals = state.values
    nant, nrec, ntc, nfc = size(vals)
    size(occ) == size(vals) || error("soln_traces: occupancy mask size $(size(occ)) ≠ values size $(size(vals))")
    frefs = ustrip.(u"Hz", references(state.definition.frequency))
    filtermap(Iterators.product(1:nant, 1:nrec)) do (a, r)
        pts = filtermap(Iterators.product(1:ntc, 1:nfc)) do (tc, fc)
            @inbounds occ[a, r, tc, fc] || return nothing
            (; x = frefs[fc], y = @inbounds(vals[a, r, tc, fc]))
        end
        isempty(pts) && return nothing
        sa = StructArray(pts); ord = sortperm(sa.x)
        (; antenna = a, receptor_slot = r, xs = sa.x[ord], ys = sa.y[ord])
    end
end


"""
    DeriveSlot{T}

Per-frame, one-task-at-a-time, coalesce-to-latest compute engine: spawn heavy work on a bg thread,
install IFF the result still matches the latest desired sig (else DROP). At most one task in flight.
"""
mutable struct DeriveSlot{T}
    desired::Any
    shown::Union{Nothing,@NamedTuple{result::T, sig::Any}}
    task::Union{Nothing,Task}
    failed_sig::Any
end
DeriveSlot{T}() where {T} = DeriveSlot{T}(nothing, nothing, nothing, nothing)

isbusy(slot::DeriveSlot) = slot.task !== nothing && (slot.shown === nothing || slot.shown.sig != slot.desired)

"""
    derive!(slot, sig, compute) -> (result_or_nothing, isbusy)

Run once per frame. `compute` is a 0-arg thunk closing over a frozen snapshot. Harvest installs IFF still
desired else DROP; spawn IFF stale & idle (coalesce to newest). A worker exception is caught and surfaced
as one `@error` (NOT rethrown, so a failing compute never crashes the render loop); that sig is not
respawned until `desired` changes, and the panel keeps its last good result.
"""
function derive!(slot::DeriveSlot, sig, compute)
    slot.desired = sig
    if slot.task !== nothing && istaskdone(slot.task)
        res, rsig, err = fetch(slot.task)
        slot.task = nothing
        if err === nothing
            rsig == slot.desired && (slot.shown = (; result = res, sig = rsig))
        else
            slot.failed_sig = rsig
            @error "derive! compute failed" exception = err maxlog = 5
        end
    end
    if slot.task === nothing && (slot.shown === nothing || slot.shown.sig != slot.desired) && slot.failed_sig != slot.desired
        slot.task = Threads.@spawn try
            (compute(), sig, nothing)
        catch e
            (nothing, sig, (e, catch_backtrace()))
        end
    end
    (slot.shown === nothing ? nothing : slot.shown.result, isbusy(slot))
end

"""
    cached!(f, ref, sig)

Synchronous sig-keyed memoize: `ref` holds `(; sig, value)` (or `nothing`); recompute `f()` only when
`sig` differs. The non-threaded sibling of `derive!`.
"""
function cached!(f, ref::Base.RefValue, sig)
    cur = ref[]
    (cur !== nothing && cur.sig == sig) && return cur.value
    v = f()
    ref[] = (; sig, value = v)
    v
end


function _combine_by_baseline(av)
    nfc = length(av.freq.ν)
    g = groupview(r -> r.baseline_ix, av.rows)
    StructArray(map(collect(keys(g)), collect(g)) do bl, rows
        phase = map(1:nfc) do c
            num = sum(r -> ifelse.(r.weight[c] .> 0, r.weight[c] .* r.visibility[c], zero(r.visibility[c])), rows)
            den = sum(r -> r.weight[c], rows)
            ifelse.(den .> 0, angle.(num ./ den), NaN)
        end
        ΣW = map(c -> sum(r -> r.weight[c], rows), 1:nfc)
        (; baseline = bl, phase, ΣW)
    end)
end

_grid_averaging(root) = Averaging(time = partition(root, WholeObservation()),
                                  frequency = partition(root, ByChannel()))

"""
    compute_grid_data(sol, terms_enabled, sel) -> StructArray (; baseline, phase, ΣW)

Windowed, optionally-calibrated coherent-average phase per baseline/channel over `sel`. `terms_enabled ==
()` ⇒ RAW windowed datums (identity); else the datums calibrated by those component kinds. `phase[b][c]`
and `ΣW[b][c]` are 2×2 SMatrices over the present baselines/channels.
"""
function compute_grid_data(sol::Solution, terms_enabled::Tuple, sel::ClosedInterval{DateTime})
    root = dataset(sol)
    dsw = window_ds(root, sel)
    dsg = terms_enabled == () ? dsw : calibrated_dataset(sol, dsw; terms = terms_enabled)
    _combine_by_baseline(average(dsg, _grid_averaging(root)))
end

_unit_visibility(ds) = (vcol = ds.rows.visibility;
    mapview(j -> mapview(c -> map(Returns(complex(1.0)), vcol[j][c]), eachindex(vcol[j])), eachindex(vcol)))

"""
    compute_grid_model(sol, terms_enabled, sel) -> StructArray (; baseline, model, ΣW)

The solution's predicted baseline corruption phase `φ_{p,i}−φ_{q,j}` for the enabled terms, window-
averaged — computed by calibrating a UNIT-visibility dataset (`V ≡ 1`) so `calibrated_dataset` yields
`1/(g_{p,i}·conj(g_{q,j}))`; `model = -angle(V̄)` (negate: inverse→forward).
"""
function compute_grid_model(sol::Solution, terms_enabled::Tuple, sel::ClosedInterval{DateTime})
    root = dataset(sol)
    dsw = window_ds(root, sel)
    dsu = @set dsw.rows.visibility = _unit_visibility(dsw)
    dsc = calibrated_dataset(sol, dsu; terms = terms_enabled)
    comb = _combine_by_baseline(average(dsc, _grid_averaging(root)))
    StructArray((; comb.baseline, model = map(-, comb.phase), comb.ΣW))
end

grid_sig(view) = (view.version[], Tuple(view.grid.apply.enabled), view.sel[], view.refresh[])


_plane_chans(ds, if_sel) = if_sel == 0 ? Colon() : findall(==(if_sel), ds.freq.if_of)

_slot_present(dsw, i, j) = any(W -> any(w -> w[i, j] > 0, W), dsw.rows.weight)

"""
    compute_fft_planes(ctx, sol, terms_enabled, pq, slots, sel, chans, rate_os, delay_os, rate_frac, delay_frac)

`|FFT|` delay-rate planes for present baseline `pq=(p,q)`, over `sel` × `chans`, one per slot. The RAW
windowed data + `solution`/`terms` go to `Fringy.fringe_plane` (it calibrates internally). `rate_frac`/
`delay_frac ∈ (0,1]` set the search window as a fraction of Nyquist (Δt = median Δt, Δν = `ds.freq.Δν`);
each result carries `nyq_rate`/`nyq_delay` so the UI labels the window sliders in absolute units. `snr` is
the native `[rate, delay]` plane in SNR units (ImPlotExtra.image! uses Makie indexing) — `|G|/σ`, obtained
from `|FFT|` as `magnitude · peak.snr/max(magnitude)` (σ cancels), so the peak pixel reads its labelled
`peak.snr`; `scale_max = peak.snr` is the colorrange top. An absent slot ⇒ `empty=true`.
"""
function compute_fft_planes(ctx, sol::Solution, terms_enabled::Tuple, pq::Tuple{Int,Int}, slots,
                            sel::ClosedInterval{DateTime}, chans, rate_os, delay_os, rate_frac, delay_frac)
    root = dataset(sol)
    dsw = window_ds(root, sel; baseline = pq, channels = chans)
    ts = sort(unique(dsw.rows.t))
    length(ts) > 1 || error("compute_fft_planes: selected window has <2 time samples for baseline $pq — the rate axis is unconstrained (widen the time selection)")
    Δt = median(diff(ts))
    Δν = root.freq.Δν
    nyq_r = 1 / (2Δt); nyq_d = 1 / (2Δν)
    sol_arg = terms_enabled == () ? nothing : sol
    window = (; rate = (-(rate_frac * nyq_r) * u"Hz")..((rate_frac * nyq_r) * u"Hz"),
                delay = (-(delay_frac * nyq_d) * u"s")..((delay_frac * nyq_d) * u"s"))
    os = (; rate = rate_os, delay = delay_os)
    map(slots) do (i, j)
        lbl = correlation_product_label(ctx.ds, pq[1], pq[2], i, j)
        _slot_present(dsw, i, j) || return (; correlation_product = (i, j), label = lbl, empty = true)
        fp = fringe_plane(dsw, pq, (i, j); refine = NoRefine(), window, oversample = os, solution = sol_arg, terms = terms_enabled)
        mag = collect(fp.magnitude); amax = maximum(mag)
        snr = amax > 0 ? mag .* (fp.peak.snr / amax) : mag
        (; correlation_product = (i, j), label = lbl, empty = false, fp.rate, fp.delay,
           snr, scale_max = fp.peak.snr, fp.peak,
           nyq_rate = nyq_r, nyq_delay = nyq_d)
    end
end

plane_sig(view) = (grid_sig(view), Tuple(view.grid.slots.enabled), view.plane.pair[], view.plane.if_sel[],
                   view.plane.rate_os[], view.plane.delay_os[], view.plane.rate_win[], view.plane.delay_win[])

_cell_or_nothing(part, x) = findfirst(s -> x in s, supports(part))

function _node_value(sol::Solution, ::Type{K}, p, i, t_s, ν_hz) where {K}
    haskind(sol, K) || return nothing
    st = sol[K]
    tc = _cell_or_nothing(st.definition.time, t_s * u"s")
    fc = _cell_or_nothing(st.definition.frequency, ν_hz * u"Hz")
    (tc === nothing || fc === nothing) ? nothing : st.values[p, i, tc, fc]
end

_node_dr(sol::Solution, p, i, t_s, ν_hz) =
    (; delay = _node_value(sol, Delay, p, i, t_s, ν_hz), rate = _node_value(sol, Rate, p, i, t_s, ν_hz))

function pair_dr(sol::Solution, p, i, q, j, t_s, ν_hz)
    np = _node_dr(sol, p, i, t_s, ν_hz); nq = _node_dr(sol, q, j, t_s, ν_hz)
    (; delay = (np.delay === nothing || nq.delay === nothing) ? nothing : np.delay - nq.delay,
       rate  = (np.rate  === nothing || nq.rate  === nothing) ? nothing : np.rate  - nq.rate)
end


uvdist_wavelengths(uvw, ν_hz) = NoUnits(hypot(uvw[1], uvw[2]) * (ν_hz * u"Hz") / Unitful.c0)

_uv_empty(status) =
    (; x = Float64[], y = Float64[], correlation_product = Tuple{Int,Int}[], status)

_cadence(ds) = (t = sort(unique(ds.rows.t));
    length(t) > 1 ? median(diff(t)) : error("_cadence: dataset has <2 distinct time samples — cannot infer an integration cadence"))

abstract type UVTimeMode end
struct UVScan <: UVTimeMode end
struct UVWhole <: UVTimeMode end
struct UVIntegration <: UVTimeMode end
struct UVCustom <: UVTimeMode end
abstract type UVFreqMode end
struct UVChannel <: UVFreqMode end
struct UVIF <: UVFreqMode end
struct UVBand <: UVFreqMode end

_uv_time_part(ds, ::UVScan, _)          = partition(ds, ByScan())
_uv_time_part(ds, ::UVWhole, _)         = partition(ds, WholeObservation())
_uv_time_part(ds, ::UVIntegration, _)   = partition(ds, ByDuration(_cadence(ds) * u"s"); within = partition(ds, WholeObservation()))
_uv_time_part(ds, ::UVCustom, custom_s) = partition(ds, ByDuration(custom_s * u"s"); within = partition(ds, WholeObservation()))

_uv_freq_part(ds, ::UVChannel) = partition(ds, ByChannel())
_uv_freq_part(ds, ::UVIF)      = partition(ds, ByIF())
_uv_freq_part(ds, ::UVBand)    = (byif = partition(ds, ByIF()); group(byif, (Tuple(1:ncells(byif)),)))

"""
    compute_uv_amplitude(ctx, sol, terms_enabled, source, time_mode, freq_mode, custom_s, slots)

Calibrated-data UV profile: source-subset (optionally calibrated) averaged over the time/freq partitions,
converting each averaged frequency cell to UV distance in wavelengths. One point per averaged
row/channel/enabled slot with weight>0.
"""
function compute_uv_amplitude(ctx, sol::Solution, terms_enabled::Tuple, source::Symbol,
                              time_mode::UVTimeMode, freq_mode::UVFreqMode, custom_s, slots)
    root = dataset(sol)
    dss = select(root, Selection(r -> r.source == source))
    nrows(dss) == 0 && return _uv_empty("no data for source $source")
    dsc = terms_enabled == () ? dss : calibrated_dataset(sol, dss; terms = terms_enabled)
    av = average(dsc, Averaging(time = _uv_time_part(root, time_mode, custom_s),
                                frequency = _uv_freq_part(root, freq_mode)))
    pts = filtermap(Iterators.product(av.rows, eachindex(av.freq.ν), slots)) do (row, fc, ij)
        row.weight[fc][ij...] > 0 || return nothing
        (; x = uvdist_wavelengths(row.uvw, av.freq.ν[fc]), y = abs(row.visibility[fc][ij...]), correlation_product = ij)
    end
    isempty(pts) && return _uv_empty(nothing)
    sa = StructArray(pts)
    (; sa.x, sa.y, sa.correlation_product, status = nothing)
end

"""
    compute_uv_snr(ctx, fringes, source, slots) -> (; x, y, correlation_product, status)

Fringe-SNR UV profile from the captured `Fringes` product (no FFT). Each product row matching `source`
places one point per freq-tile × enabled slot at the uv-distance of that (baseline, tile) — the tile's
averaged uvw is joined by `(baseline, tile datetime)` over the measurement tile partitions. NaN-sentinel
cells (no computable peak) are skipped. `fringes === nothing` ⇒ a status message.
"""
function compute_uv_snr(ctx, fringes, source::Symbol, slots)
    fringes === nothing &&
        return _uv_empty("no captured fringes (add a Capture(Fringes()=>…, FringeFit) to the schedule)")
    tiles = ctx.plane_defaults.tiles
    tiles === nothing && return _uv_empty("schedule has no FringeFit tiles for the SNR join")
    rows = filter(r -> r.source == source, fringes)
    isempty(rows) && return _uv_empty("no captured fringes for source $source")
    dss = select(ctx.ds, Selection(r -> r.source == source))
    av = average(dss, Averaging(time = tiles.time, frequency = tiles.frequency))
    uvmap = Dict((r.baseline_ix, r.datetime) => r.uvw for r in av.rows)
    matched = filter(fr -> haskey(uvmap, (fr.baseline_ix, fr.datetime)), rows)
    dropped = length(rows) - length(matched)
    dropped > 0 && @warn "compute_uv_snr: $dropped fringe row(s) had no matching averaged uvw (dropped from the SNR join)" maxlog=1
    pts = flatmap(matched) do fr
        uvw = uvmap[(fr.baseline_ix, fr.datetime)]
        filtermap(Iterators.product(enumerate(fr.ν), slots)) do (kν, ij)
            k, ν = kν
            snr = fr.cells[k][ij...].snr
            isnan(snr) && return nothing
            (; x = uvdist_wavelengths(uvw, ν), y = snr, correlation_product = ij)
        end
    end
    isempty(pts) && return _uv_empty("no captured fringes for the enabled slots")
    sa = StructArray(pts)
    (; sa.x, sa.y, sa.correlation_product, status = nothing)
end

uvprof_sig(view) = view.uvprof.ymode[] === :snr ?
    (:snr, view.uvprof.source[], Tuple(view.grid.slots.enabled), view.refresh[]) :
    (:amplitude, view.version[], Tuple(view.grid.apply.enabled), Tuple(view.grid.slots.enabled),
     view.uvprof.source[], view.uvprof.time_mode[], view.uvprof.freq_mode[], view.uvprof.custom_time[],
     view.refresh[])
