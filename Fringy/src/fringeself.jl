
export FringeSelf

"""
    FringeSelf(fit::FringeFit, targets...; min_snr, globalization, reference,
                    pairs = Returns(true))

Phase/delay/rate self-calibration: measure `fit`'s [`FringeTable`](@ref) on the solution being
calibrated, solve its peaks into per-(antenna, receptor slot) corrections with [`stationize`](@ref), and ADD
them into the solution's components. It does not apply visibilities or average them. `targets` ⊆
`(PhaseOffset, Delay, Rate)` (≥1, unique) are the
component kinds to seed. Each target component's time and frequency partitions must refine the
corresponding measurement partition: every target cell must be wholly contained in one `fit.tiles`
cell.

One correction is solved per measurement tile and broadcast to every target cell contained in that
tile. Per-IF self-calibration, for data whose IFs cannot be assumed alignable, is this same step fed a
per-IF `fit`. Under `Capture(Fringes() => :name, …)` the measured table comes back as the step's
product.

`min_snr` is the hard SNR gate on the peaks that enter the solve; `globalization` an
`SNRWeighted`/`CurvatureWeighted`; `reference` the reference antenna name (resolved to a dense antenna
index at execution; absent ⇒ fail loud); `pairs` a predicate on an edge row
`(; p, i, q, j, delay, rate, coeff, snr, Q, q_ab)`, where `i` and `j` are structural receptor-slot
indices. The default admits every pair: a self-calibration solve wants every available edge, and
off-diagonal pairs of receptor slots connect the two sets of receptor-slot nodes. Receptor labels are not inspected.
"""
struct FringeSelf{F <: FringeFit, G, P} <: Step
    fit::F
    targets::Tuple
    min_snr::Float64
    globalization::G
    reference::Symbol
    pairs::P
end

function FringeSelf(fit::FringeFit, targets...; min_snr, globalization, reference,
                         pairs = Returns(true))
    isempty(targets) && error("FringeSelf needs at least one target kind")
    allunique(targets) || error("FringeSelf targets must be unique; got $(map(nameof, targets))")
    all(K -> K in (PhaseOffset, Delay, Rate), targets) ||
        error("FringeSelf targets must be a subset of (PhaseOffset, Delay, Rate); got $(map(nameof, targets))")
    FringeSelf(fit, targets, Float64(min_snr), globalization, Symbol(reference), pairs)
end

_cells_in(contain, tile) = findall(==(tile), contain)

function _validate_targets(step::FringeSelf, sol::Solution, tiles)
    for K in step.targets
        haskind(sol, K) || error("FringeSelf target $(nameof(K)) is not present in the solution")
        assert_refines(sol[K].definition.time, tiles.time)
        assert_refines(sol[K].definition.frequency, tiles.frequency)
    end
    PhaseOffset in step.targets && Delay in step.targets &&
        assert_refines(sol[PhaseOffset].definition.frequency, sol[Delay].definition.frequency)
    PhaseOffset in step.targets && Rate in step.targets &&
        assert_refines(sol[PhaseOffset].definition.time, sol[Rate].definition.time)
end

"""
    apply_step(step::FringeSelf, sol, capture::Union{Nothing,Fringes}) -> (sol′, product_or_nothing)

Measure, solve, install. `capture === nothing` (bare step) ⇒ no product; `capture isa Fringes` (wrapped
in `Capture`) ⇒ also return the [`FringeTable`](@ref) the solve consumed — the measurement is made either
way, so capturing it is free.
"""
function apply_step(step::FringeSelf, sol::Solution, capture::Union{Nothing, Fringes})
    _resolve_reference(dataset(sol), step.reference)
    _validate_targets(step, sol, step.fit.tiles)
    table = step.fit(sol)
    (_solve_and_install(step, sol, table), capture isa Fringes ? table : nothing)
end

function _solve_and_install(step::FringeSelf, sol::Solution, table::FringeTable)
    ds = dataset(sol)
    nant = nantennas(ds)
    nant == table.nantennas ||
        error("FringeSelf: the table covers $(table.nantennas) antennas, the solution's dataset has $nant")
    refidx = _resolve_reference(ds, step.reference)
    _validate_targets(step, sol, table.tiles)

    Tt = table.tiles.time
    Tf = table.tiles.frequency
    ν_fs = ustrip.(u"Hz", references(Tf))

    tcontain = Dictionary(step.targets, map(K -> _containing(sol[K].definition.time, Tt), step.targets))
    fcontain = Dictionary(step.targets, map(K -> _containing(sol[K].definition.frequency, Tf), step.targets))

    hasoff = PhaseOffset in step.targets
    hasdel = Delay in step.targets
    hasrate = Rate in step.targets
    off2del_f = (hasoff && hasdel) ? _containing(sol[PhaseOffset].definition.frequency, sol[Delay].definition.frequency) : Int32[]
    νref_del = hasdel ? ustrip.(u"Hz", references(sol[Delay].definition.frequency)) : Float64[]
    off2rat_t = (hasoff && hasrate) ? _containing(sol[PhaseOffset].definition.time, sol[Rate].definition.time) : Int32[]
    tref_rat = hasrate ? ustrip.(u"s", references(sol[Rate].definition.time)) : Float64[]

    tv = Dictionary(step.targets, map(K -> copy(sol[K].values), step.targets))

    rows_of_ts = groupfind(r -> r.tcell, table.rows)
    for ts in sort!(collect(keys(rows_of_ts)))
        trows = view(table.rows, rows_of_ts[ts])
        t0 = ustrip(u"s", references(Tt)[ts])
        for fs in sort!(unique(flatmap(r -> r.fcells, trows)))
            ν0 = ν_fs[fs]
            edges = _edges_of(trows, fs; min_snr = step.min_snr, pairs = step.pairs)
            bw = _binwidths_of(trows, fs)
            st = stationize(edges, nant; globalization = step.globalization, reference_antenna = refidx,
                            rate_binwidth = bw.rate, delay_binwidth = bw.delay)

            for n in 1:2nant
                st.node_present[n] || continue
                p = node_antenna(n); i = node_receptor(n)
                d_n = st.delay[n]; r_n = st.rate[n]; φ_n = st.phase[n]
                if hasdel
                    for tc in _cells_in(tcontain[Delay], ts), fc in _cells_in(fcontain[Delay], fs)
                        tv[Delay][p, i, tc, fc] += d_n
                    end
                end
                if hasrate
                    for tc in _cells_in(tcontain[Rate], ts), fc in _cells_in(fcontain[Rate], fs)
                        tv[Rate][p, i, tc, fc] += r_n
                    end
                end
                if hasoff
                    written = NTuple{2,Int}[]
                    for tc in _cells_in(tcontain[PhaseOffset], ts), fc in _cells_in(fcontain[PhaseOffset], fs)
                        Δφ = φ_n
                        hasdel && (Δφ += 2π * d_n * (νref_del[off2del_f[fc]] - ν0))
                        hasrate && (Δφ += 2π * r_n * (tref_rat[off2rat_t[tc]] - t0))
                        tv[PhaseOffset][p, i, tc, fc] += Δφ
                        push!(written, (tc, fc))
                    end
                    if !isempty(written)
                        tc1, fc1 = first(written)
                        Δwrap = 2π * round(tv[PhaseOffset][p, i, tc1, fc1] / 2π)
                        for (tc, fc) in written
                            tv[PhaseOffset][p, i, tc, fc] -= Δwrap
                        end
                    end
                end
            end
        end
    end

    comps = sol.components
    for K in step.targets
        comps = merge(comps, NamedTuple{(nameof(K),)}((ComponentState(sol[K].definition, tv[K]),)))
    end
    Solution(sol.dataset, comps)
end
