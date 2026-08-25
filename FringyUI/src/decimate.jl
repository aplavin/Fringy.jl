
"""
    grid_decimate(points::StructArray, region::NTuple{N,ClosedInterval}, nbins::NTuple{N,Int}) -> Vector{Int}

Indices of one representative point per occupied cell of an `nbins`-per-axis grid over `region`. Each
point's N fields are the axes (matching `region`/`nbins`); points outside `region` are dropped; the
first point seen in each cell is kept, in input order. Dimension-agnostic — callers `@view` the result
(or index any array sharing the point indexing).
"""
function grid_decimate(points::StructArray, region::NTuple{N,ClosedInterval}, nbins::NTuple{N,Int}) where {N}
    (any(≤(0), width.(region)) || any(<(1), nbins)) && return Int[]
    seen = falses(nbins .+ 1)
    keep = Int[]
    for (k, p) in pairs(points)
        all(values(p) .∈ region) || continue
        b = floor.(Int, (values(p) .- leftendpoint.(region)) ./ width.(region) .* nbins) .+ 1
        seen[b...] || (seen[b...] = true; push!(keep, k))
    end
    keep
end
