
export FringeTable

const _MEAS = NamedTuple{(:coeff, :delay, :rate, :snr, :sigma_delay, :weight, :Q, :q_ab),
                         Tuple{ComplexF64, Float64, Float64, Float64, Float64, Float64,
                               SMatrix{2,2,Float64,4}, Float64}}
_nanmeas() = (; coeff = ComplexF64(NaN, NaN), delay = NaN, rate = NaN, snr = NaN,
                sigma_delay = NaN, weight = 0.0, Q = zero(SMatrix{2,2,Float64,4}), q_ab = NaN)

const _IFW = SMatrix{2,2,Float64,4}

const _EDGE = @NamedTuple{p::Int, i::Int, q::Int, j::Int, delay::Float64, rate::Float64,
                          coeff::ComplexF64, snr::Float64, Q::SMatrix{2,2,Float64,4}, q_ab::Float64}

"""
    FringeTable

The measured fringe product: per (solution interval × frequency cell), the per-baseline, per-correlation-product
peaks, UNGATED. `rows` is a `StructArray`, one row per occupied (source, baseline, time cell); `tiles`
are the `(; time, frequency)` partitions the measurement was made on, so a table carries its own solution
granularity; `nantennas` and `antenna_names` are the antenna census `rows.baseline_ix` indexes into.
The matrix indices are structural endpoint receptor slots; receptor labels are not propagated into the derived
table.

Row columns: `source`, `source_ix`, `baseline`, `baseline_ix`, `tcell`, `datetime`, `t`, plus, per entry
`k` of the row's frequency cells `fcells` (cell references `ν`):

  * `cells[k]::SMatrix{2,2}` of per-correlation-product `(; coeff, delay, rate, snr, sigma_delay, weight,
    Q, q_ab)` — a pair with no computable peak carries NaN sentinels and `weight = 0`, the exact reason
    it is absent;
  * `ifs[k]` / `ifweights[k]` — the per-IF Σw′ breakdown inside that frequency cell;
  * `delay_binwidth[k]` / `rate_binwidth[k]` — that row's own transform lattice bin widths, NaN where
    the baseline had no computable pair in the cell.
"""
struct FringeTable{R, TG}
    rows::R
    tiles::TG
    nantennas::Int
    antenna_names::Vector{Symbol}
end

Base.show(io::IO, t::FringeTable) = print(io, "FringeTable(", length(t.rows), " rows, ",
    length(unique(t.rows.tcell)), " of ", ncells(t.tiles.time), " time cells × ",
    ncells(t.tiles.frequency), " frequency cells, ", t.nantennas, " antennas)")

function _resolve_reference(table::FringeTable, name::Symbol)
    idx = findfirst(==(name), table.antenna_names)
    isnothing(idx) &&
        error("reference antenna $name not in the table's antennas $(table.antenna_names)")
    idx
end

_rows_of_tile(table::FringeTable, ts::Integer) = view(table.rows, findall(==(ts), table.rows.tcell))

"""
    _edges_of(rows, fs; min_snr, pairs) -> StructArray of committed peaks

One (time cell, frequency cell)'s [`stationize`](@ref) edge table, reconstructed from that time cell's
table `rows`. ORDER IS PART OF THE CONTRACT: row order (the measurement emits a time cell's rows sorted
by `baseline_ix.antennas`) × `(i in 1:2, j in 1:2)`, `j` fastest — `_scalar_wls` and `stefcal!` accumulate
their normal equations by float summation over the edges, so edge order is load-bearing at the ULP level.

`min_snr` is the hard gate; `pairs` is a predicate on the edge selecting which correlation-product
peaks enter. The current edge fields `p`, `i`, `q`, and `j` identify the two antenna endpoints and
their structural receptor slots; labels are not part of the derived table.
"""
function _edges_of(rows, fs::Integer; min_snr::Real, pairs)
    edges = StructArray{_EDGE}(undef, 0)
    for r in rows
        k = findfirst(==(fs), r.fcells)
        isnothing(k) && continue
        p, q = r.baseline_ix.antennas
        for i in 1:2, j in 1:2
            c = r.cells[k][i, j]
            c.weight > 0 || continue
            c.snr ≥ min_snr || continue
            e = (; p, i, q, j, c.delay, c.rate, c.coeff, c.snr, c.Q, c.q_ab)
            pairs(e) || continue
            push!(edges, e)
        end
    end
    edges
end

_edges_of(table::FringeTable, ts::Integer, fs::Integer; min_snr::Real, pairs) =
    _edges_of(_rows_of_tile(table, ts), fs; min_snr, pairs)

function _binwidths_of(rows, fs::Integer)
    for r in rows
        k = findfirst(==(fs), r.fcells)
        isnothing(k) && continue
        isnan(r.delay_binwidth[k]) && continue
        return (; delay = r.delay_binwidth[k], rate = r.rate_binwidth[k])
    end
    (; delay = 1.0, rate = 1.0)
end
