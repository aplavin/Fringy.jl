
export derive_if_alignment

"""
    derive_if_alignment(cfg, table::FringeTable) -> (; if_alignment::IFAlignment, summary)

Derive the session's static instrumental alignment from a PER-IF fringe table (the one
[`fringefit`](@ref) produces with `frequency = :if`).

A pure derivation: it never touches a `Solution` and never reopens the data — the table alone is a
complete input. `cfg.if_alignment_min_snr` is the per-IF peak SNR floor of the solve and
`cfg.if_alignment_pairs` the predicate selecting pairs of receptor slots that it stationizes over.

`summary` carries, per (antenna, receptor slot): `nruns` (the solution-interval count behind the solve),
`delay_rms` and `delay_ptp` [s] of the per-IF pattern it removes, and `curvature_rms` [rad], the rms
second difference of the inter-IF phase — which is the quantity the whole exercise is about. Plus
`n_identity`, the number of (antenna, receptor slot, IF) cells that got the identity alignment because no
scan solved every IF: a silent identity at one antenna is exactly how a session keeps one
lobe-locked corner, so the count is data, not a log line.

Whether this runs at all is `cfg.if_alignment`, decided by the caller. That is a configuration value
deciding whether a STEP RUNS, not a code-path mode: this function has no `:none` branch.
"""
function derive_if_alignment(cfg, table::FringeTable)
    cfg.reference_antenna in table.antenna_names ||
        error("derive_if_alignment: reference antenna $(cfg.reference_antenna) is not in the table's antenna " *
              "census $(table.antenna_names)")
    if_alignment = DeriveIFAlignment(; min_snr = cfg.if_alignment_min_snr, reference = cfg.reference_antenna,
                                      pairs = cfg.if_alignment_pairs)(table)
    (; if_alignment, summary = if_alignment_summary(if_alignment, table.antenna_names))
end

"""
    if_alignment_summary(al::IFAlignment, antenna_names)
        -> (; rows, n_identity, delay_max, phase_max, reference)

The measured content of an alignment, per (antenna, receptor slot), as data. `rows` is a
`StructArray` of `(; antenna, receptor_slot, nruns, delay_rms, delay_ptp, curvature_rms)`.
"""
function if_alignment_summary(al::IFAlignment, antenna_names)
    rows = NamedTuple[]
    for p in axes(al.delay, 1), i in 1:2
        d = @view al.delay[p, i, :]
        φ = @view al.phase[p, i, :]
        d2 = [rem(φ[k+1] - 2φ[k] + φ[k-1], 2π, RoundNearest) for k in 2:(length(φ)-1)]
        push!(rows, (; antenna = antenna_names[p], receptor_slot = i,
                       nruns = maximum(@view al.nruns[p, i, :]),
                       delay_rms = sqrt(mean(abs2, d)), delay_ptp = maximum(d) - minimum(d),
                       curvature_rms = isempty(d2) ? 0.0 : sqrt(mean(abs2, d2))))
    end
    (; rows = StructArray(identity.(rows)), n_identity = count(iszero, al.nruns),
       n_cells = length(al.nruns), delay_max = maximum(abs, al.delay),
       phase_max = maximum(abs, al.phase), reference = al.reference)
end
