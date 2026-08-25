
export observables

"""
    observables(cfg, fringes, instr, astrometry_inputs; captured_phase_mode = cfg.captured_phase_mode, structure = nothing,
                source_offsets = ()) -> StructArray

The astrometric observable table: correlator polynomial and clock bookkeeping, independent geometry,
troposphere and ionosphere, and measured delay, composed into `τ_residual` per row.

* `fringes` is [`fringefit`](@ref)'s result (anything carrying the tidy `obs` table);
* `instr` is [`instrumental`](@ref)'s result, or the `CapturedAppliedPhase` itself;
* `astrometry_inputs` is [`astrometry_models`](@ref)'s result.

`captured_phase_mode` (`:retain` | `:add_back`, the configured value by default) decides whether this table
includes the `CapturedAppliedPhase` slope: `:retain` leaves it out and `:add_back` algebraically adds it.
In the current application the capture is tone-derived and precedes IF alignment. Add-back does not undo that
alignment or other uncaptured terms, and it does not rerun or refit the fringe measurement. The
`τ_applied_phase` column is reported either way, so the table-column choice is auditable. It is a keyword because
the validation suite forms both tables from one fringe product.

`structure` turns the point-source observable into a registration observable (a
`StructureModels`, whose absent source gives `τ_structure = 0.0` and a bit-identical table), and
`source_offsets` shifts the a priori model's source positions — the Gauss–Newton relinearization hook that
[`solve`](@ref) builds internally.
"""
observables(cfg, fringes, instr, astrometry_inputs; captured_phase_mode = cfg.captured_phase_mode, structure = nothing,
            source_offsets = ()) =
    total_delays(fringes.obs, captured_applied_phase(instr), astrometry_inputs.correlator_model, astrometry_inputs.astrometry_model;
                 captured_phase_mode, structure, source_offsets)

"""
    captured_applied_phase(x) -> CapturedAppliedPhase

The instrumental states, whether `x` is [`instrumental`](@ref)'s result or the states themselves —
so a caller holding a cached product's `CapturedAppliedPhase` need not wrap it to look like a step
result.
"""
captured_applied_phase(x::CapturedAppliedPhase) = x
captured_applied_phase(x) = x.applied
