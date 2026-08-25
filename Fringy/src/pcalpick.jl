
export slice_tones, tone_health, pick_tones

_tonedim(::Type{<:StaticMatrix{N, M}}) where {N, M} = N
_tonedim(::Type) = 0

"""
    slice_tones(records, tones) -> records′

Every `[tone, band]`-shaped field of the PHASE-CAL record table, restricted to the tone indices
`tones`; all other fields unchanged. The result is an ordinary PHASE-CAL table with `length(tones)`
tones per band.
"""
function slice_tones(records, tones)
    ntone = size(first(records.freq_1), 1)
    ix = SVector(tones)
    cols = map(StructArrays.components(records)) do col
        _tonedim(eltype(col)) == ntone ? map(m -> m[ix, :], col) : col
    end
    StructArray(cols)
end

"""
    tone_health(records; amplitude_floor) -> Matrix{Bool} [tone, band]

Which tones of the comb are alive everywhere. A tone is HEALTHY in a band when its session-median
amplitude clears `amplitude_floor` × the median over the band's tones at every (antenna, receptor slot) that
has records — the same "fraction of the local typical amplitude" scale `cfg.tone_floor` uses to drop
collapsed tones inside `PcalInit`, applied here to the tone rather than to the single record.
"""
function tone_health(records; amplitude_floor)
    ntone, nband = size(first(records.freq_1))
    ok = trues(ntone, nband)
    for no in unique(records.antenna_no)
        ks = findall(==(no), records.antenna_no)
        for col in (records.pcal_1, records.pcal_2)
            amed = [median(abs(ComplexF64(col[k][tn, b])) for k in ks) for tn in 1:ntone, b in 1:nband]
            for b in 1:nband
                m = median(@view amed[:, b])
                m > 0 || continue
                ok[:, b] .&= (@view amed[:, b]) .> amplitude_floor * m
            end
        end
    end
    ok
end

"""
    pick_tones(records, mode; amplitude_floor) -> records′

The PHASE-CAL table cut down to the two tones per band that `PcalInit` decodes.

- `mode = nothing` — the table is returned UNCHANGED (`===` the input). This is the no-op every
  2-tone session runs, and it is the configuration default.
- `mode = :outermost` — the outermost tone pair `(k, NO_TONES+1−k)` that is healthy in every band at
  every antenna and receptor slot, falling inward from `k = 1` until one is. One comb for the whole array.

A table that already carries exactly 2 tones is returned unchanged whatever the mode: there is nothing
to pick, and slicing it would only be an opportunity to get the order wrong.

`mode` is typed. Without the annotation this method and the `(cfg, records)` one below would share the
positional signature `(Any, Any)`, and in a precompiled module that is an error rather than the
accidental "the keyword picks the body" merge Julia performs at top level — which is how these two
coexisted while both lived in an `include`d script.
"""
function pick_tones(records, mode::Union{Nothing,Symbol}; amplitude_floor)
    isnothing(mode) && return records
    mode === :outermost ||
        error("pick_tones: mode must be `nothing` or `:outermost`, got $(repr(mode))")
    ntone, nband = size(first(records.freq_1))
    if ntone == 2
        println("pick_tones: the table already carries 2 tones per band — nothing to pick")
        return records
    end
    ok = tone_health(records; amplitude_floor)
    pair = nothing
    for k in 1:ntone ÷ 2
        all(@view ok[k, :]) && all(@view ok[ntone + 1 - k, :]) && (pair = (k, ntone + 1 - k); break)
    end
    isnothing(pair) &&
        error("pick_tones: no symmetric tone pair of the $ntone-tone comb is above the $(amplitude_floor) health floor in all $nband bands — tone health: $(ok)")
    f = first(records.freq_1)
    @printf("pick_tones(:outermost): tones %d and %d of %d%s; lever %s MHz\n", pair[1], pair[2], ntone,
            pair == (1, ntone) ? "" : " — fell inward, an outer tone is dead somewhere",
            join((@sprintf("%.3f", ustrip(u"MHz", f[pair[2], b] - f[pair[1], b])) for b in 1:nband), "/"))
    slice_tones(records, pair)
end

"""
    pick_tones(cfg, records) -> records′

`pick_tones` with the session's configured choice (`cfg.tone_pick`, `cfg.tone_floor`) — the form every
load site calls, so that `tone_pick = nothing` is a guaranteed no-op.
"""
pick_tones(cfg, records) = pick_tones(records, cfg.tone_pick; amplitude_floor = cfg.tone_floor)
