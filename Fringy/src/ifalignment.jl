
export IFAlignment, InstallIFAlignment, DeriveIFAlignment

"""
    IFAlignment

Static per-(antenna, receptor slot, frequency cell) instrumental `delay` [s] and `phase` [rad] — the
table [`InstallIFAlignment`](@ref) installs. `delay`, `phase`, and `nruns` have shape
`(nantennas, 2, nfrequency_cells)`. The second axis is structural and always has the two supported
receptor-slot positions; an unavailable receptor slot simply contributes no measurements. `νref` has one reference
frequency [Hz] per frequency cell. `phase` is referred to that cell's `νref`, the same convention
used by `Delay` and `PhaseOffset`, so the states can be added directly.

**The gauge**, stated exactly — the table satisfies all three conditions, and [`DeriveIFAlignment`](@ref)
enforces them rather than merely aiming at them:

- **(G1)** `delay[reference, i, k] = phase[reference, i, k] = 0` for every cell — reference gauge
  (the delay exactly; the phase to the floating-point dust left by the gauge node's `angle`, ~1e-18 rad);
- **(G2)** `mean_k delay[p, i, k] = 0` for every (antenna, receptor slot) — the static/band-common delay split;
- **(G3)** `circmean_k phase[p, i, k] = 0` for every (antenna, receptor slot) — the same split in phase.

The alignment is thereby defined up to a per-(antenna, receptor slot) (delay, phase) which is fixed by
(G1)–(G3); the reference antenna's own pattern is absorbed into every other antenna's, which is why the
gauge cancels on every baseline. The (G2) shift is degenerate with antenna clock; the (G3) shift is a
common antenna/receptor-slot phase that fringe measurement profiles out.

`nruns[p, i, k]` counts the solution intervals that contributed — the intervals in which this (antenna,
receptor slot) was solved, in this gauge, in every frequency cell. A cell with none is left at zero
(identity), reported rather than guessed, and is excluded from (G2)/(G3). `reference` is the reference
antenna index of the underlying graph solve; [`zero_if_alignment`](@ref) uses `0` because it has no anchor.
"""
struct IFAlignment
    delay::Array{Float64, 3}
    phase::Array{Float64, 3}
    nruns::Array{Int, 3}
    νref::Vector{Float64}
    reference::Int
end

Base.show(io::IO, al::IFAlignment) = print(io, "IFAlignment(",
    size(al.delay, 1), " antennas × ", size(al.delay, 3), " frequency cells, ",
    count(>(0), al.nruns), " of ", length(al.nruns), " cells solved, |delay| max ",
    round(1e9 * maximum(abs, al.delay); digits = 2), " ns)")

"""
    zero_if_alignment(nant, nfcell, νref) -> IFAlignment

The identity alignment: all delays and phases zero. Applying it is a no-op by construction.
"""
zero_if_alignment(nant::Integer, nfcell::Integer, νref) =
    IFAlignment(zeros(nant, 2, nfcell), zeros(nant, 2, nfcell), zeros(Int, nant, 2, nfcell),
                collect(Float64, νref), 0)

_circmean(xs) = (s = sum(cis, xs); iszero(s) ? NaN : angle(s))

"""
    DeriveIFAlignment(; min_snr, reference, pairs = e -> e.i == e.j,
                 globalization = SNRWeighted(iterations = 50))

The session-static per-(antenna, receptor slot, frequency cell) instrumental solution — the classical manual
phase-cal, complementing the tone-derived [`PcalInit`](@ref). It consumes a PER-IF [`FringeTable`](@ref)
(a table whose frequency cells resolve the IFs, measured over scans strong enough to reach every
IF) and returns the [`IFAlignment`](@ref) that [`InstallIFAlignment`](@ref) installs.

Recipe, per solution interval, per frequency cell:

1. **Stationize.** The interval's peaks, gated at `min_snr` and filtered by `pairs`, are solved into
   per-(antenna, receptor slot) delays `d` and phases `φ` by the ordinary [`stationize`](@ref), gauged at
   `reference`. **Only solution components gauged at the reference antenna are read.** A component
   in which the reference antenna is undetected is gauged elsewhere, so its values carry an arbitrary
   offset that depends on both the interval and the cell — an interaction term the two-way separation of
   steps 2–3 has no place for, and which the per-interval constants removed there cannot touch. Its nodes
   are therefore not measured in that cell; the completeness rule below drops the interval for them.
2. **Delay.** `d[p, k] − mean_k d[p, k]` is the antenna's per-cell delay deviation, free of its
   band-common delay. Its MEDIAN over the intervals is the alignment's delay.
3. **Phase.** `φ[p, k]` still carries the interval's own band-common delay as `2π·ν_k·τ̄[p]` (many turns
   across the comb). It is removed using `τ̄[p] = mean_k d[p, k]`, the within-cell slope average, which is
   unambiguous (see the file header); a further per-(antenna, interval) constant is removed as the
   circular mean over the cells. The alignment's phase is the circular mean of what is left, over the
   intervals.
4. **Gauge.** The two combinations above preserve the per-interval conditions `mean_k d = 0` and
   `circmean_k φ = 0` only approximately (a median is not a mean; circular means over intervals and over
   cells do not commute), so the table is re-centred into the gauge [`IFAlignment`](@ref) documents: the
   row's own `mean_k` delay and `circmean_k` phase are subtracted per (antenna, receptor slot), giving (G2)
   and (G3) exactly. Each is a per-(antenna, receptor slot) constant, and both are zero at the reference
   antenna, so (G1) is preserved. Where a row's phasors cancel exactly the phase gauge is undetermined
   and the row is left alone rather than made `NaN`.

An (antenna, receptor slot) with no interval in which every frequency cell was solved gets a zero (identity)
alignment and `nruns == 0` — reported, never guessed.

**Reference-antenna choice, a rule and not a preference.** Under the default `pairs` predicate the two
receptor slots form disjoint solution components, so receptor slot `j` can use the reference gauge only
if the reference antenna has receptor slot `j`. A reference antenna with only one receptor slot therefore
leaves the other receptor slot with no session-consistent gauge at all — and gets `nruns == 0` for that
entire receptor slot, which
the caller's report names, instead of a table silently built in a per-interval arbitrary gauge. Choose a
reference antenna carrying every receptor slot the session solves for.

**No rate.** Rate is a scan-dependent quantity (atmosphere, LO), not a static filter property. It is
absent from [`IFAlignment`](@ref) and [`InstallIFAlignment`](@ref) touches no `Rate` component, so an installed
instrumental solution is structurally unable to move the per-scan rate signal.

Everything else is read off the table: the antenna count, the frequency-cell references `νref`, and the
reference index that the `reference` antenna name resolves to. `min_snr` is the peak SNR floor and
`globalization.iterations` the phase-StEFCal count.

`pairs` selects which receptor-slot pairs enter the instrumental solve. For each edge, `e.i` and
`e.j` are structural endpoint receptor-slot indices; the default `e -> e.i == e.j` therefore keeps
the diagonal pairs `(1, 1)` and `(2, 2)`. It does not compare or interpret receptor labels. An antenna
with one available receptor slot remains a supported graph node because unavailable products have zero weight.
"""
struct DeriveIFAlignment{G, P}
    min_snr::Float64
    reference::Symbol
    pairs::P
    globalization::G
end

DeriveIFAlignment(; min_snr, reference, pairs = e -> e.i == e.j,
             globalization = SNRWeighted(iterations = 50)) =
    DeriveIFAlignment(Float64(min_snr), Symbol(reference), pairs, globalization)

"""
    (step::DeriveIFAlignment)(table::FringeTable) -> IFAlignment

Derive the instrumental alignment from a per-IF fringe table. A pure derivation — it never touches a
`Solution`; the application is [`InstallIFAlignment`](@ref).
"""
function (step::DeriveIFAlignment)(table::FringeTable)
    nant = table.nantennas
    νr = ustrip.(u"Hz", references(table.tiles.frequency))
    nfc = length(νr)
    isempty(table.rows) && error("DeriveIFAlignment: the fringe table is empty")
    refidx = _resolve_reference(table, step.reference)
    ν̄ = mean(νr)

    tcells = sort!(unique(table.rows.tcell))
    tpos = Dictionary(tcells, eachindex(tcells))
    nt = length(tcells)
    D = fill(NaN, nant, 2, nt, nfc)
    P = fill(NaN, nant, 2, nt, nfc)

    rows_of_t = groupfind(r -> r.tcell, table.rows)
    for (tc, poss) in pairs(rows_of_t)
        ti = tpos[tc]
        trows = view(table.rows, poss)
        for k in 1:nfc
            edges = _edges_of(trows, k; min_snr = step.min_snr, pairs = step.pairs)
            length(edges) ≥ 1 || continue
            bw = _binwidths_of(trows, k)
            st = stationize(edges, nant; globalization = step.globalization,
                            reference_antenna = refidx,
                            rate_binwidth = bw.rate, delay_binwidth = bw.delay)
            for c in st.components
                c.anchored_at_reference || continue
                for n in c.nodes
                    D[node_antenna(n), node_receptor(n), ti, k] = st.delay[n]
                    P[node_antenna(n), node_receptor(n), ti, k] = st.phase[n]
                end
            end
        end
    end

    delay = zeros(nant, 2, nfc)
    phase = zeros(nant, 2, nfc)
    nruns = zeros(Int, nant, 2, nfc)
    for p in 1:nant, i in 1:2
        good = [ti for ti in 1:nt if all(k -> isfinite(D[p, i, ti, k]) && isfinite(P[p, i, ti, k]), 1:nfc)]
        isempty(good) && continue
        τ̄ = [mean(@view D[p, i, ti, :]) for ti in good]
        χ = [P[p, i, good[m], k] - 2π * (νr[k] - ν̄) * τ̄[m] for m in eachindex(good), k in 1:nfc]
        for m in eachindex(good)
            θ = _circmean(@view χ[m, :])
            isnan(θ) || (@views χ[m, :] .-= θ)
        end
        for k in 1:nfc
            delay[p, i, k] = median(D[p, i, good[m], k] - τ̄[m] for m in eachindex(good))
            ψ = _circmean(@view χ[:, k])
            phase[p, i, k] = isnan(ψ) ? 0.0 : ψ
            nruns[p, i, k] = length(good)
        end
        μ = mean(@view delay[p, i, :])
        @views delay[p, i, :] .-= μ
        ψ̄ = _circmean(@view phase[p, i, :])
        isnan(ψ̄) || (@views phase[p, i, :] .= rem.(phase[p, i, :] .- ψ̄, 2π, RoundNearest))
    end
    IFAlignment(delay, phase, nruns, νr, refidx)
end

"""
    InstallIFAlignment(alignment::IFAlignment)

Step that adds a static per-(antenna, receptor slot, frequency window) instrumental [`IFAlignment`](@ref) into a solution's
`Delay` and `PhaseOffset` states — the same two components `PcalInit` fills, on the same per-IF
partition, so the alignment reaches the data through the ordinary gain chain with no side channel and is
seen identically by every later stage (the astrometric fringe pass and the imaging pass alike).

The same value is added to every TIME cell: the alignment is a session-static instrumental property, and
what varies within the session is the phase-cal's business, not this layer's.

Sign: the stored components are the CORRUPTION, divided out as `V′ = V/(g_p·conj(g_q))`, and the values
here are antenna-based per-window peaks in exactly the convention [`FringeSelf`](@ref) installs
its graph solution with (`values .+= solved`), so adding them removes the measured structure.

Requirements (all fail loud): both components present, sharing their partitions; the frequency partition
must be the one the alignment was solved on (checked against its stored cell references); the antenna
count must match. A zero alignment is a no-op by construction — which is what makes a
`no alignment configured` pipeline provably identical to one without this stage.
"""
struct InstallIFAlignment{A} <: Step
    alignment::A
end

"""
    apply_step(step::InstallIFAlignment, sol, ::Nothing) -> (sol′, nothing)

Add the step's [`IFAlignment`](@ref) into the `Delay`/`PhaseOffset` states of `sol`, in every time cell.
Produces no product.
"""
function apply_step(step::InstallIFAlignment, sol::Solution, ::Nothing)
    al = step.alignment
    (haskind(sol, Delay) && haskind(sol, PhaseOffset)) ||
        error("InstallIFAlignment needs both Delay and PhaseOffset components in the solution")
    dd = sol[Delay].definition; dp = sol[PhaseOffset].definition
    (dd.time.lookup == dp.time.lookup && dd.frequency.lookup == dp.frequency.lookup) ||
        error("InstallIFAlignment: the Delay and PhaseOffset components must share their partitions (one instrumental (delay, phase) pair per (antenna, receptor slot, IF))")
    vals_d = copy(sol[Delay].values)
    vals_p = copy(sol[PhaseOffset].values)
    nant, _, ntc, nfc = size(vals_d)
    size(al.delay, 1) == nant ||
        error("InstallIFAlignment: the alignment covers $(size(al.delay, 1)) antennas, the solution has $nant")
    size(al.delay, 3) == nfc ||
        error("InstallIFAlignment: the alignment covers $(size(al.delay, 3)) frequency cells, the components have $nfc")
    νref = ustrip.(u"Hz", references(dd.frequency))
    maximum(abs, νref .- al.νref) ≤ 1.0 ||
        error("InstallIFAlignment: the alignment was solved on frequency cells centred at $(al.νref) Hz but the " *
              "components' cells are centred at $(νref) Hz — it must be installed on the partition it was solved on")
    for p in 1:nant, i in 1:2, fc in 1:nfc, tc in 1:ntc
        vals_d[p, i, tc, fc] += al.delay[p, i, fc]
        vals_p[p, i, tc, fc] += al.phase[p, i, fc]
    end
    comps = merge(sol.components, (;
        Delay = ComponentState(dd, vals_d),
        PhaseOffset = ComponentState(dp, vals_p)))
    (Solution(sol.dataset, comps), nothing)
end
