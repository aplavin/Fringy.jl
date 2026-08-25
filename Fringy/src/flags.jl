
export flag_selection

struct _AntFlags
    starts::Vector{DateTime}
    maxends::Vector{DateTime}
end

function _AntFlags(ivs)
    ord = sortperm(ivs; by = leftendpoint)
    starts = [leftendpoint(ivs[k]) for k in ord]
    maxends = accumulate(max, [rightendpoint(ivs[k]) for k in ord])
    _AntFlags(starts, maxends)
end

function _overlaps(f::_AntFlags, a::DateTime, b::DateTime)
    k = searchsortedlast(f.starts, b)
    k ≥ 1 && f.maxends[k] ≥ a
end

struct _FlagPredicate{D, P} <: Function
    byantenna::D
    within::P
end

function (p::_FlagPredicate)(row)
    p.within(row) || return false
    t0 = row.datetime
    t1 = datetime_at(t0, row.int_time)
    for name in row.baseline.antennas
        f = get(p.byantenna, name, nothing)
        isnothing(f) && continue
        _overlaps(f, t0, t1) && return false
    end
    true
end

"""
    flag_selection(flags; antennas, within=Selection()) -> Selection

Turn a FLAG table into a [`Selection`](@ref) that drops every row whose integration overlaps a flag of
either of its baseline's antennas. `flags` is a table of rows carrying `antenna_no::Int`,
`timerange::ClosedInterval{DateTime}` and `source_rawid::Int` (as read by `VLBIFiles.flags`);
`antennas` maps the file's antenna numbers to the dataset's antenna names (e.g.
`map(a -> a.name, only(uv.ant_arrays).antennas)`) — a number the mapping does not cover fails loud,
since silently ignoring an antenna's flags would quietly admit bad data. `within` is a base `Selection`
the flag predicate is ANDed onto (its channel restriction is carried through unchanged).

Convention: a row occupies `[datetime, datetime + int_time]` and is dropped when that window touches a
flag interval AT ALL (closed-interval overlap). This is deliberately conservative: the extra data lost
at the boundaries is a fraction of one integration per flag, which is not worth the risk of keeping a
partially-slewing one.

Flags restricted to one source (`source_rawid != 0`) are NOT representable as a time/antenna predicate
here and fail loud; all rows of the tables this reader accepts carry 0 (all sources).
"""
function flag_selection(flags; antennas, within::Selection = Selection())
    all(==(0), flags.source_rawid) ||
        error("flag_selection: $(count(!=(0), flags.source_rawid)) flag row(s) are restricted to one " *
              "source (source_rawid ≠ 0) — only whole-session (antenna, time-range) flags are supported")
    names = map(flags.antenna_no) do no
        haskey(antennas, no) ||
            error("flag_selection: FLAG antenna number $no is not in the `antennas` map $(collect(keys(antennas)))")
        antennas[no]
    end
    byantenna = map(idxs -> _AntFlags(flags.timerange[idxs]), groupfind(identity, names))
    Selection(_FlagPredicate(byantenna, within.predicate), within.channels)
end
