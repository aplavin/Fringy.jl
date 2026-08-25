
"""
    Dataset

Observed visibilities, weights, and metadata. A `Dataset` retains the column storage supplied by its
producer: file-backed datasets use lazy views, while simulations, averaging, and tests may use eager
in-memory vectors.
`rows.visibility[j]::AbstractVector{SMatrix{2,2,ComplexF64,4}}` over channels (weight likewise,
`Float64`) — mmap-backed views (file) or eager per-row vectors (in-memory producers). In the 2×2
receptor-slot layout, `V[i,j]` is the correlation product between receptor slot `i` of antenna `p` and receptor slot `j` of
antenna `q`. Receptor labels are opaque metadata and do not determine matrix indexing or numerical
behavior. An absent correlation has weight zero and is excluded by [`present`](@ref).
`parentrows`/`parentchannels` index into the root so root-bound partitions stay resolvable through a
subset; a root has identity ranges.
"""
struct Dataset{R,A,S,F,PR,PC}
    rows::R
    antennas::A
    sources::S
    freq::F
    time0::DateTime
    parentrows::PR
    parentchannels::PC
end

function Dataset(rows::StructArray, antennas, sources, freq;
                 time0::DateTime=minimum(rows.datetime),
                 parentrows=Base.OneTo(length(rows)),
                 parentchannels=Base.OneTo(length(freq.ν)))
    rows_t = @insert rows.t = float.(ustrip.(u"s", rows.datetime .- time0))
    Dataset{typeof(rows_t),typeof(antennas),typeof(sources),typeof(freq),typeof(parentrows),typeof(parentchannels)}(
        rows_t, antennas, sources, freq, time0, parentrows, parentchannels)
end

ConstructionBase.constructorof(::Type{<:Dataset}) =
    (rows, antennas, sources, freq, time0, parentrows, parentchannels) ->
        Dataset{typeof(rows),typeof(antennas),typeof(sources),typeof(freq),typeof(parentrows),typeof(parentchannels)}(
            rows, antennas, sources, freq, time0, parentrows, parentchannels)

nrows(ds::Dataset)     = length(ds.rows)
nchannels(ds::Dataset) = length(ds.freq.ν)
nantennas(ds::Dataset) = length(ds.antennas)
nsources(ds::Dataset)  = length(ds.sources)
parentrows(ds::Dataset)     = ds.parentrows
parentchannels(ds::Dataset) = ds.parentchannels

function _resolve_reference(ds::Dataset, name::Symbol)
    idx = findfirst(a -> a.name == name, ds.antennas)
    isnothing(idx) &&
        error("reference antenna $name not in dataset antennas $(map(a -> a.name, ds.antennas))")
    idx
end

datetime_at(time0::DateTime, t::Real)     = time0 + round(Nanosecond, float(t) * u"s")
datetime_at(time0::DateTime, t::Quantity) = time0 + round(Nanosecond, float(t))
datetime_at(ds::Dataset, t) = datetime_at(ds.time0, t)

"""
    present(W::StaticMatrix)

The `(i, j)` mask over `W .> 0` — the SOLE usability predicate. Flags (≤0), absent products, and zeros
all fold in (a non-finite V with positive weight propagates to a loud NaN, never a silent wrong answer).
Iteration order is column-major over the 2×2 — `(1,1),(2,1),(1,2),(2,2)`, `i` fastest.
"""
present(W::StaticMatrix) = ((i, j) for i in 1:2, j in 1:2 if W[i, j] > 0)

_c64(x) = ComplexF64(x)

_nonneg(w) = (x = Float64(w); x > 0 ? x : 0.0)
_weights_nonneg(col) = coherencymatrices(col; value=_nonneg, absent=0.0)
_vis_column(col) = coherencymatrices(col; value=_c64, absent=ComplexF64(NaN, NaN))

function _if_of(windows, ν)
    map(ν) do νk
        hits = findall(fw -> νk * u"Hz" ∈ frequency(fw, Interval), windows)
        length(hits) == 1 || error("channel $(νk) Hz lies in $(length(hits)) frequency windows (expected 1)")
        only(hits)
    end
end

function _common_step(windows)
    steps = unique(abs.(ustrip.(u"Hz", step.(frequencies.(windows)))))
    length(steps) == 1 || error("frequency windows have unequal channel steps: $(steps) Hz")
    only(steps)
end

function _freq_meta(uv)
    windows = uv.freq_windows
    ν = ustrip.(u"Hz", frequencies(windows))
    (; windows, ν, if_of = _if_of(windows, ν), Δν = _common_step(windows))
end

"""
    load_dataset(path::AbstractString; scans) -> Dataset

Load one correlator file into a root `Dataset`: relabel + wrap + view, NO visibility copy. Rows carry
IDENTITY + ENCODING column pairs — `baseline`/`source` (names) are stable identities selection
predicates target; `baseline_ix`/`source_ix` are dense `1:n` encodings. Autocorrelations (p == q) are
dropped. `scans` (a `GapBasedScans`-style strategy) is REQUIRED — files carry no scan table. Reading
goes through `VLBIFiles.VLBI.load`, which auto-detects the container by content (uvfits and fits-idi
alike), so no per-format loader hook is needed.
"""
function load_dataset(path::AbstractString; scans)
    uv = VLBIFiles.VLBI.load(path)
    wt = uvtable_wide(uv)
    hasproperty(wt, :int_time) ||
        error("uvtable_wide has no int_time column (file lacks INTTIM) — required by Fringy.Dataset")
    srctable = VLBIFiles.sources(uv)
    wt_scan = add_scan_ids(scans, wt)

    antennas = collect(only(uv.ant_arrays).antennas)
    name2idx = Dictionary(map(a -> a.name, antennas), eachindex(antennas))
    baseline_ix = map(bl -> Baseline(map(n -> name2idx[n], bl.antennas)), wt.baseline)
    keep = findall(bl -> bl.antennas[1] != bl.antennas[2], baseline_ix)

    rawids = sort!(unique(wt.source_id[keep]))
    srcremap = Dictionary(rawids, eachindex(rawids))
    sources = StructArray(map(rid -> (; name = srctable[rid].name, srctable[rid].coords, rawid = rid), rawids))

    rows = StructArray((
        baseline    = wt.baseline[keep],
        baseline_ix = baseline_ix[keep],
        source      = wt.source[keep],
        source_ix   = map(sid -> srcremap[sid], wt.source_id[keep]),
        scan        = wt_scan.scan_id[keep],
        datetime    = wt.datetime[keep],
        uvw         = wt.uvw[keep],
        int_time    = wt.int_time[keep],
        visibility  = view(_vis_column(wt.visibility), keep),
        weight      = view(_weights_nonneg(wt.weight), keep),
    ))
    Dataset(rows, antennas, sources, _freq_meta(uv))
end

struct ConcatVector{T,V} <: AbstractVector{T}
    parts::V
    offsets::Vector{Int}
    n::Int
end
function ConcatVector(parts)
    offsets = [0; cumsum([length(p) for p in parts])]
    ConcatVector{eltype(first(parts)), typeof(parts)}(parts, offsets, last(offsets))
end
function ConcatVector{T}(parts) where {T}
    offsets = [0; cumsum([length(p) for p in parts])]
    ConcatVector{T, typeof(parts)}(parts, offsets, last(offsets))
end
Base.size(c::ConcatVector) = (c.n,)
Base.IndexStyle(::Type{<:ConcatVector}) = IndexLinear()
Base.@propagate_inbounds function Base.getindex(c::ConcatVector, i::Int)
    @boundscheck checkbounds(c, i)
    k = searchsortedlast(c.offsets, i - 1)
    @inbounds c.parts[k][i - c.offsets[k]]
end

struct ScatterVector{T,S} <: AbstractVector{T}
    src::S
    pos::Vector{Int}
    zero::T
    n::Int
end
ScatterVector(src::AbstractVector{T}, pos::Vector{Int}, n::Integer) where {T} =
    ScatterVector{T,typeof(src)}(src, pos, zero(T), n)
Base.size(s::ScatterVector) = (s.n,)
Base.IndexStyle(::Type{<:ScatterVector}) = IndexLinear()
Base.@propagate_inbounds function Base.getindex(s::ScatterVector, c::Int)
    @boundscheck checkbounds(s, c)
    j = searchsortedfirst(s.pos, c)
    (j ≤ length(s.pos) && @inbounds(s.pos[j]) == c) ? (@inbounds s.src[j]) : s.zero
end

function _ordered_scans(scans, wide::StructArray)
    ord = view(wide, sortperm(wide.datetime))
    add_scan_ids(scans, ord)
end

"""
    load_dataset(paths::AbstractVector{<:AbstractString}; scans) -> Dataset

Load several correlator files and [`combine`](@ref) them into ONE root Dataset. Files sharing a
frequency setup are row-unioned; files at differing setups (e.g. per-band jobs) are joined along the
channel axis into one wideband Dataset (see [`combine`](@ref)). Each file is loaded independently.
"""
load_dataset(paths::AbstractVector{<:AbstractString}; scans) =
    combine(map(p -> load_dataset(p; scans), paths); scans)

const _VIS_ET = SMatrix{2,2,ComplexF64,4}
const _WGT_ET = SMatrix{2,2,Float64,4}

_concat_block(dsets) = (;
    baseline   = ConcatVector(map(d -> d.rows.baseline, dsets)),
    source     = reduce(vcat, map(d -> d.rows.source, dsets)),
    datetime   = reduce(vcat, map(d -> d.rows.datetime, dsets)),
    uvw        = reduce(vcat, map(d -> d.rows.uvw, dsets)),
    int_time   = reduce(vcat, map(d -> d.rows.int_time, dsets)),
    visibility = ConcatVector(map(d -> d.rows.visibility, dsets)),
    weight     = ConcatVector(map(d -> d.rows.weight, dsets)),
)

function _group_bands(datasets)
    setups = unique(d.freq.ν for d in datasets)
    sets = map(Set, setups)
    n = length(setups)
    g = SimpleGraph(n)
    for i in 1:n, j in (i + 1):n
        isempty(intersect(sets[i], sets[j])) || add_edge!(g, i, j)
    end
    bands = map(connected_components(g)) do members
        mi = members[argmax(map(m -> length(sets[m]), members))]
        all(m -> issubset(sets[m], sets[mi]), members) ||
            error("combine: frequency setups partially overlap (neither channel set contains the other) — only disjoint or nested setups are allowed")
        setups[mi]
    end
    sort!(bands; by = minimum)
    for k in 2:length(bands)
        maximum(bands[k - 1]) < minimum(bands[k]) ||
            error("combine: frequency bands interleave — each band must occupy a disjoint frequency range")
    end
    map(bands) do bandν
        (; band_dsets = filter(d -> issubset(Set(d.freq.ν), Set(bandν)), datasets),
           bfreq = first(filter(d -> d.freq.ν == bandν, datasets)).freq)
    end
end

function _band_block(band_dsets, bfreq)
    base = _concat_block(band_dsets)
    allequal(d.freq.ν for d in band_dsets) && return base
    nB = length(bfreq.ν)
    gpos = Dictionary(bfreq.ν, eachindex(bfreq.ν))
    padded(getcol, d) = let col = getcol(d), pos = map(ν -> gpos[ν], d.freq.ν)
        length(pos) == nB ? col : mapview(l -> ScatterVector(col[l], pos, nB), eachindex(col))
    end
    setproperties(base, (;
        visibility = ConcatVector{AbstractVector{_VIS_ET}}(map(d -> padded(x -> x.rows.visibility, d), band_dsets)),
        weight     = ConcatVector{AbstractVector{_WGT_ET}}(map(d -> padded(x -> x.rows.weight, d), band_dsets)),
    ))
end

function _union_freq(freqs)
    Δνs = unique(f.Δν for f in freqs)
    length(Δνs) == 1 ||
        error("combine: frequency setups have differing channel steps Δν=$(collect(Δνs)) Hz — a single wideband comb needs one common Δν")
    ν = reduce(vcat, (collect(f.ν) for f in freqs))
    issorted(ν) && allunique(ν) ||
        error("combine: frequency setups overlap or interleave (channel frequencies are not strictly increasing across blocks) — combine expects disjoint frequency bands")
    if_of = Int[]; off = 0
    for f in freqs
        append!(if_of, f.if_of .+ off)
        off += maximum(f.if_of)
    end
    wins = flatmap(f -> @something(f.windows, ()), freqs)
    (; windows = isempty(wins) ? nothing : wins, ν, if_of, Δν = only(Δνs))
end

function _assert_no_near_duplicates(baseline, source, datetime, int_time)
    for idxs in groupfind(g -> (baseline[g], source[g]), eachindex(baseline))
        length(idxs) < 2 && continue
        ord = sort(idxs; by = g -> datetime[g])
        for m in 2:length(ord)
            g1, g2 = ord[m - 1], ord[m]
            Δt = datetime[g2] - datetime[g1]
            zero(Δt) < Δt < int_time[g2] / 2 && error(
                "combine: rows on baseline $(baseline[g1]) source $(source[g1]) at $(datetime[g1]) and " *
                "$(datetime[g2]) differ by $(Δt) (< half the $(int_time[g2]) integration) — nearby but " *
                "unequal timestamps across bands split one integration into separate rows; check " *
                "correlator time alignment")
        end
    end
end

function _wideband_wide(blocks, freqs)
    nB = length(blocks)
    nchans = map(f -> length(f.ν), freqs)
    KT = Tuple{eltype(first(blocks).baseline), eltype(first(blocks).datetime), eltype(first(blocks).source)}
    keyof(blk) = KT[(blk.baseline[i], blk.datetime[i], blk.source[i]) for i in eachindex(blk.datetime)]
    blkkeys = map(keyof, blocks)
    for (b, ks) in enumerate(blkkeys)
        allunique(ks) ||
            error("combine: block $b (one frequency setup) has duplicate (baseline, datetime, source) rows — the same measurement appears twice")
    end
    dicts = map(ks -> Dictionary(ks, eachindex(ks)), blkkeys)

    gkeys = unique(Iterators.flatten(blkkeys))
    ng = length(gkeys)
    localof = map(d -> map(k -> get(d, k, 0), gkeys), dicts)

    baseline = map(k -> k[1], gkeys)
    datetime = map(k -> k[2], gkeys)
    source   = map(k -> k[3], gkeys)
    firstblk = map(g -> findfirst(b -> localof[b][g] != 0, 1:nB), 1:ng)
    uvw      = map(g -> blocks[firstblk[g]].uvw[localof[firstblk[g]][g]], 1:ng)
    int_time = map(g -> blocks[firstblk[g]].int_time[localof[firstblk[g]][g]], 1:ng)

    _assert_no_near_duplicates(baseline, source, datetime, int_time)

    offs = [0; cumsum(collect(nchans))]; ntot = last(offs)
    padcol(blkcol, b, zcell) = mapview(g -> (l = localof[b][g]; l == 0 ? zcell : blkcol[l]), 1:ng)
    padvis = ntuple(b -> padcol(blocks[b].visibility, b, fill(zero(_VIS_ET), nchans[b])), nB)
    padwgt = ntuple(b -> padcol(blocks[b].weight,     b, fill(zero(_WGT_ET), nchans[b])), nB)
    visibility = mapview(g -> (p = map(c -> c[g], padvis); ConcatVector{_VIS_ET, typeof(p)}(p, offs, ntot)), 1:ng)
    weight     = mapview(g -> (p = map(c -> c[g], padwgt); ConcatVector{_WGT_ET, typeof(p)}(p, offs, ntot)), 1:ng)

    StructArray((; baseline, source, datetime, uvw, int_time, visibility, weight))
end

"""
    combine(datasets::AbstractVector{<:Dataset}; scans) -> Dataset

Combine several root datasets into one root dataset. Inputs are grouped
into frequency BANDS (setups sharing channels must be nested, one ⊆ the other; partial overlap fails loud)
whose axes concatenate into one wideband comb, and rows are outer-joined on the `(baseline, datetime,
source)` key: same-band rows are row-unioned (a channel-subset member scatter-padded onto the band axis),
differing bands laid side by side (a row absent from a band ⇒ that band's channels weight-0). Requires a
common `Δν`. Antennas/sources are unioned (conflicting metadata/coords fail loud) and name columns
re-encoded. Rows stay lazy, are stably datetime-sorted, and scans are re-derived on the merged timeline
under a single `time0`. Fails loud on a subset input, differing `Δν`, interleaving or partially-overlapping
bands, or nearby-but-unequal cross-band timestamps.
"""
function combine(datasets::AbstractVector{<:Dataset}; scans)
    all(d -> parentrows(d) isa Base.OneTo && parentchannels(d) isa Base.OneTo, datasets) ||
        error("combine: input is a subset (has non-trivial parent indices) — combine root datasets only")

    antennas = sort!(unique(flatmap(d -> d.antennas, datasets)); by = a -> a.name)
    allunique(map(a -> a.name, antennas)) || error("combine: antenna name with conflicting metadata")
    name2idx = Dictionary(map(a -> a.name, antennas), eachindex(antennas))

    srcnames = sort!(unique(flatmap(d -> d.sources.name, datasets)))
    srcidx = Dictionary(srcnames, eachindex(srcnames))
    namecoord = Dictionary{Symbol,Any}()
    for d in datasets, s in d.sources
        haskey(namecoord, s.name) ? (namecoord[s.name] == s.coords || error("combine: source $(s.name) has conflicting coords across inputs")) :
                                    insert!(namecoord, s.name, s.coords)
    end
    sources = StructArray(map((k, nm) -> (; name = nm, coords = namecoord[nm], rawid = k), eachindex(srcnames), srcnames))

    bands = _group_bands(datasets)
    blocks = map(b -> _band_block(b.band_dsets, b.bfreq), bands)
    bfreqs = map(b -> b.bfreq, bands)
    if length(bands) == 1
        wide = StructArray(only(blocks))
        allunique(zip(wide.baseline, wide.datetime, wide.source)) ||
            error("combine: duplicate (baseline, datetime, source) rows within one frequency band — the same measurement appears twice")
        freq = only(bfreqs)
    else
        wide = _wideband_wide(blocks, bfreqs)
        freq = _union_freq(bfreqs)
    end
    wide_scan = _ordered_scans(scans, wide)

    rows = StructArray((
        baseline    = wide_scan.baseline,
        baseline_ix = map(bl -> Baseline(map(n -> name2idx[n], bl.antennas)), wide_scan.baseline),
        source      = wide_scan.source,
        source_ix   = map(nm -> srcidx[nm], wide_scan.source),
        scan        = wide_scan.scan_id,
        datetime    = wide_scan.datetime,
        uvw         = wide_scan.uvw,
        int_time    = wide_scan.int_time,
        visibility  = wide_scan.visibility,
        weight      = wide_scan.weight,
    ))
    Dataset(rows, antennas, sources, freq)
end


"""
    Selection(pred=Returns(true); channels=Colon())

A row predicate (`@o` optic, evaluated per traversal) + a channel restriction (`Colon()` or a
strictly-increasing unique `Vector{Int}`). Unbound to any dataset — channel in-bounds validation
happens at use time in [`select`](@ref).
"""
struct Selection{P,C}
    predicate::P
    channels::C
end
Selection(pred=Returns(true); channels=Colon()) = Selection(pred, _check_channels(channels))
_check_channels(::Colon) = Colon()
function _check_channels(chs::AbstractVector{<:Integer})
    allunique(chs) && issorted(chs) || error("Selection channels must be strictly increasing and unique")
    chs
end

_resolve_channels(::Dataset, ::Colon) = Colon()
function _resolve_channels(ds::Dataset, chs::AbstractVector)
    all(c -> 1 ≤ c ≤ nchannels(ds), chs) || error("Selection channels out of bounds 1:$(nchannels(ds))")
    chs
end

_channel_project(ds::Dataset, sub, ::Colon) = (sub.visibility, sub.weight, ds.freq, ds.parentchannels)
function _channel_project(ds::Dataset, sub, chs::AbstractVector)
    vis = mapview(cv -> view(cv, chs), sub.visibility)
    wgt = mapview(cv -> view(cv, chs), sub.weight)
    freq = setproperties(ds.freq, (; ν = ds.freq.ν[chs], if_of = ds.freq.if_of[chs]))
    (vis, wgt, freq, ds.parentchannels[chs])
end

function _subset(ds::Dataset, rowkeep, chankeep)
    sub = view(ds.rows, rowkeep)
    vis, wgt, freq, pchan = _channel_project(ds, sub, chankeep)
    rows = StructArray((; sub.baseline, sub.baseline_ix, sub.source, sub.source_ix,
                          sub.scan, sub.datetime, sub.uvw, sub.int_time,
                          visibility = vis, weight = wgt))
    Dataset(rows, ds.antennas, ds.sources, freq;
            time0 = ds.time0, parentrows = parentrows(ds)[rowkeep], parentchannels = pchan)
end

"""
    select(ds::Dataset, sel::Selection) -> Dataset

Parent-identity-preserving subset: keep the parent antenna/source tables, dense encodings, and
`time0`; record parent row and channel indices so root-bound partitions stay resolvable. Rows are
filtered by the column-projected optic (visibility/weight wrappers never built for rejected rows);
columns stay lazy views. Re-evaluated per traversal — callers do not cache it across stages.
"""
function select(ds::Dataset, sel::Selection)
    chs = _resolve_channels(ds, sel.channels)
    keep = findall(sel.predicate, ds.rows)
    _subset(ds, keep, chs)
end
