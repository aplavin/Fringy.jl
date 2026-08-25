
export fringefit, peak_observables

"""
    fringefit(cfg, ld, instr, if_alignment = nothing; frequency = :band) -> (; table, obs, comb)

Measure this session's fringes.

* `instr` is [`instrumental`](@ref)'s result; its `applied` states are installed into a fresh
  solution — a `ComponentState` construction, never a re-solve, so the states the measurement is made
  against are bit-for-bit the ones the instrumental step derived.
* `if_alignment`, when given, is [`derive_if_alignment`](@ref)'s `IFAlignment`, applied through the same gain chain
  (`InstallIFAlignment`) so that everything downstream sees one instrument.
* `frequency` is `:band` (all IFs in one cell) or `:if` (one cell per IF).

Returns

* `table::FringeTable` — the captured cells produced by the full fringe step;
* `obs` — the tidy observable table, one row per (scan, baseline, selected receptor-slot pair), built by
  [`peak_observables`](@ref) from diagonal matrix cells. Only defined for `frequency = :band`,
  where there is one frequency cell to be a whole-band observable of; `nothing` otherwise;
* `comb` — the channel comb and the cell references needed to interpret the table without reopening
  the visibility file.
"""
function fringefit(cfg, ld, instr, if_alignment = nothing; frequency = :band)
    freqpart = frequency === :band ? ld.wholeband :
               frequency === :if   ? ld.byif :
               error("fringefit: `frequency` must be :band or :if, got $(repr(frequency))")
    sol = fringe_states(ld, instr, freqpart)
    isnothing(if_alignment) || ((sol, _) = apply_step(InstallIFAlignment(if_alignment), sol))
    table = FringeFit(; tiles = (; time = ld.byscan, frequency = freqpart),
                        window = cfg.window, oversample = cfg.oversample, refine = cfg.refine,
                        selection = ld.selection)(sol)
    comb = (; ld.ds.freq.ν, ld.ds.freq.if_of, ld.ds.freq.Δν,
              if_reference_ν = ustrip.(u"Hz", references(ld.byif)),
              band_reference_ν = ustrip.(u"Hz", references(ld.wholeband)),
              cell_reference_ν = ustrip.(u"Hz", references(freqpart)))
    (; table, obs = frequency === :band ? peak_observables(table.rows) : nothing, comb)
end

"""
    fringe_states(ld, instr, freqpart) -> Solution

The solution the search runs on: `PhaseOffset` and `Delay` carrying [`instrumental`](@ref)'s states
per (scan, IF), and a zero `Rate` on the searched frequency partition.

`Rate` follows the tiling because a rate is estimated per tile: over the whole band for the
astrometric pass, per IF for the alignment pass. `PhaseOffset`/`Delay` do NOT — they carry the
instrument, which is per-IF whatever is being searched.
"""
function fringe_states(ld, instr, freqpart)
    base = Solution(ld.ds; components = (
        PhaseOffset(; time = ld.byscan, frequency = ld.byif),
        Delay(; time = ld.byscan, frequency = ld.byif),
        Rate(; time = ld.byscan, frequency = freqpart)))
    Solution(base.dataset, (;
        PhaseOffset = ComponentState(base[PhaseOffset].definition, copy(instr.applied.phase)),
        Delay = ComponentState(base[Delay].definition, copy(instr.applied.delay)),
        Rate = base[Rate]))
end

"""
    peak_observables(rows) -> StructArray

The tidy astrometric observable table: one row per (scan, baseline, selected receptor-slot pair) from
the diagonal `(1, 1)` and `(2, 2)` cells of a whole-band fringe table. This is the astrometry
pipeline's structural product selection; receptor labels are not inspected. Off-diagonal products
remain available in the raw `FringeTable`.

Peaks are kept **ungated**: a cell that
produced a finite SNR is a row, whatever that SNR is, because acceptance is the estimator's decision
and a table that pre-filters it cannot be re-cut afterwards.

The `receptor_slots` column carries the ordered receptor-slot pair at the two baseline endpoints.
"""
function peak_observables(rows)
    out = NamedTuple[]
    for r in rows
        length(r.cells) == 1 ||
            error("peak_observables: the tidy table is a whole-band product — this row carries " *
                  "$(length(r.cells)) frequency cells, so run the fit with `frequency = :band`")
        for ip in 1:2
            c = r.cells[1][ip, ip]
            isnan(c.snr) && continue
            push!(out, (; r.source, r.source_ix, r.baseline, r.baseline_ix, r.tcell, r.datetime, r.t,
                          receptor_slots = (ip, ip), c.delay, c.rate, c.snr, c.sigma_delay, c.coeff,
                          weight = c.weight, ifweights = [w[ip, ip] for w in r.ifweights[1]]))
        end
    end
    StructArray(identity.(out))
end
