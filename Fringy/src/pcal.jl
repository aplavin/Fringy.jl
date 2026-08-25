
export PcalInit

"""
    PcalInit(records; antennas, cable, cable_apply, amplitude_floor)

Step that fills a solution's `Delay` and `PhaseOffset` states from a PHASE-CAL table. `records` is a
table of rows with `time::DateTime`, `interval`, `antenna_no::Int`, `freq_1/2` and `pcal_1/2`
(`[tone, band]` matrices, the FITS-IDI `_1`/`_2` pair = receptor slots 1/2) and `cable_cal` (NaN when
unavailable), as read by `VLBIFiles.phase_cal`. The phasors are taken in the SKY-frequency convention,
which is what that reader delivers: an LSB band's tones are detected in the BASEBAND and arrive
conjugated and in reverse tone order, and `VLBIFiles._tones_to_sky` undoes both, so that
`wrap(φ₂ − φ₁)` and the band-centre
phase mean the same thing on every IF of a mixed-sideband file.
`antennas` maps the file's antenna numbers to antenna names (e.g.
`map(a -> a.name, only(uv.ant_arrays).antennas)`). `cable` lists the antennas whose `cable_cal` is
read (`k = +1`); every other antenna's cable monitor is ignored. A listed antenna's cell with no
cable reading at all (difx2fits NaNs the last record of a scan) is left unfilled rather than silently
calibrated without it — that holds in every `cable_apply` mode, so the filled-cell set is identical
across them and the modes differ only in the correction applied.

`cable_apply` selects WHICH halves of the tone measurement the cable monitor corrects:

| value | delay `τ ← τ + cable` | phase `φ ← φ + 2πν_mid·cable` | meaning |
|---|---|---|---|
| `:both` | yes | yes | remove the monitor term from both tone delay and midpoint phase |
| `:delay` | yes | no | corrects only the within-IF tone-pair slope; the whole-band group delay, which rides on the inter-IF phase pattern, then keeps ~15/16 of the cable term |
| `:none` | no | no | the tones are applied exactly as measured, cable variation included |

`amplitude_floor` (fraction of a tone's own session-median amplitude) drops collapsed tones — the
dropouts that would otherwise inject a random phase into a scan average. All keyword
arguments are required.

Both target components must be present and must share their partitions: the tone pair measures ONE
(delay, band-centre phase) pair per (scan, IF), so the two halves of one measurement cannot live on
different supports. The design partitions are `time = per-scan`, `frequency = ByIF`. Records are
assigned to the time cell their `time ± interval/2` window overlaps most (typically 4–5 of them per
scan; a zero-interval record is an instantaneous sample and goes to the cell that CONTAINS its time),
and their tone values are averaged COHERENTLY inside the cell. A cell with no usable record keeps
its previous value (zero = identity gain for a fresh solution).

**Only the variation about each (antenna, receptor slot, frequency window) session median is installed.**
The tone pair determines the delay only modulo `1/Δf_tone`; those unknown integers are not recoverable
from the tones. The static part is degenerate with clock and receptor-dependent offsets. The per-scan
series is unwrapped to the nearest turn about its own median, and the
band-centre phase is likewise referred to its circular session mean. A turn taken on the delay is
taken on the PHASE too: the pair determines `(τ, φ_mid)` jointly up to `(τ + k/Δf_tone, φ_mid + kπ)`,
so moving the delay by k ambiguities without moving the midpoint phase by kπ would install a model
180° wrong at the tones for odd k. The phase reference is defined only modulo `2π`; any static
per-(antenna, receptor slot, frequency window) phase remains outside this measurement.

**The frequency windows of one (antenna, receptor slot, scan) are then put on one 2π branch** — each
cell's per-window phase
variations are re-wrapped about their own circular mean rather than independently about ±π. Each
value moves by a multiple of 2π, so the applied gain `cis(φ)` is untouched; what changes is the
whole-band group delay the calibration applies, which is carried by the inter-window phase pattern.
Independent wrapping can place one window a full turn from its neighbours and introduce a spurious
whole-band slope.

The branch reference phase is the same-cell circular mean, not a temporal unwrap, because only the inter-window pattern
must be consistent. A `2π` shift common to a cell is invisible to both the gain and band slope, whereas
a temporal unwrap must survive gaps and missing scans. Validity needs the per-window variations of one
cell to span less than half a turn: a set inside an arc shorter than `π` contains
its own circular mean, so every value is then within π of the branch reference and lands on the right branch;
this is the validity condition for the branch choice.

**Sign:** the values are stored as
MEASURED, i.e. `Delay = +τ_instrumental`, `PhaseOffset = +φ_instrumental` evaluated at the
frequency cell's own
reference. Fringy's stored components are the CORRUPTION, which `calibrated_dataset` divides out as
`V′ = V/(g_p·conj(g_q))`, so applying this state SUBTRACTS `τ_p − τ_q` from the measured baseline delay
(covered by the sign and composition tests in `test/test_pcal.jl`).
"""
struct PcalInit{R, A, C} <: Step
    records::R
    antennas::A
    cable::C
    cable_apply::Symbol
    amplitude_floor::Float64
end

function PcalInit(records; antennas, cable, cable_apply, amplitude_floor)
    Symbol(cable_apply) in (:both, :delay, :none) ||
        error("PcalInit: cable_apply must be :both, :delay or :none, got $(cable_apply)")
    PcalInit(records, antennas, collect(Symbol, cable), Symbol(cable_apply), Float64(amplitude_floor))
end

_wrapπ(x::Real) = rem(x, 2π, RoundNearest)

function _record_cells(part::TimePartition, times, intervals, time0::DateTime)
    sups = supports(part)
    los = map(iv -> ustrip(u"s", leftendpoint(iv)), sups)
    his = map(iv -> ustrip(u"s", rightendpoint(iv)), sups)
    map(eachindex(times)) do k
        half = ustrip(u"s", intervals[k]) / 2
        c = ustrip(u"s", times[k] - time0)
        if half == 0
            i = findfirst(i -> los[i] ≤ c ≤ his[i], eachindex(los))
            return isnothing(i) ? 0 : i
        end
        a = c - half; b = c + half
        best = 0; bestov = 0.0
        for i in eachindex(los)
            ov = min(b, his[i]) - max(a, los[i])
            ov > bestov && (best = i; bestov = ov)
        end
        best
    end
end

function _band_cells(part::FreqPartition, freqs)
    cells = map(b -> cellof(part, (freqs[1, b] + freqs[2, b]) / 2), axes(freqs, 2))
    allunique(cells) ||
        error("PcalInit: two PHASE-CAL bands map to the same frequency cell $(cells) — the Delay/PhaseOffset frequency partition must resolve the IFs (ByIF)")
    cells
end

"""
    apply_step(step::PcalInit, sol, ::Nothing) -> (sol′, nothing)

Fill the `Delay`/`PhaseOffset` states of `sol` from the step's PHASE-CAL records (see [`PcalInit`](@ref)
for the recipe, the ambiguity handling and the sign convention). Cells with no usable record keep their
previous values. Produces no product.
"""
function apply_step(step::PcalInit, sol::Solution, ::Nothing)
    ds = dataset(sol)
    (haskind(sol, Delay) && haskind(sol, PhaseOffset)) ||
        error("PcalInit needs both Delay and PhaseOffset components in the solution")
    dd = sol[Delay].definition; dp = sol[PhaseOffset].definition
    (dd.time.lookup == dp.time.lookup && dd.frequency.lookup == dp.frequency.lookup) ||
        error("PcalInit: the Delay and PhaseOffset components must share their partitions (one tone pair measures one delay AND one band-centre phase per (scan, IF))")
    Tt = dd.time; Tf = dd.frequency

    rec = step.records
    isempty(rec) && error("PcalInit: the PHASE-CAL table is empty")
    ntone, nband = size(first(rec.freq_1))
    ntone == 2 ||
        error("PcalInit: the tone-pair recipe needs exactly 2 tones per band, the table has $ntone")

    name2idx = Dictionary(map(a -> a.name, ds.antennas), eachindex(ds.antennas))
    antidx = map(rec.antenna_no) do no
        haskey(step.antennas, no) ||
            error("PcalInit: PHASE-CAL antenna number $no is not in the `antennas` map $(collect(keys(step.antennas)))")
        nm = step.antennas[no]
        haskey(name2idx, nm) || error("PcalInit: antenna $nm has PHASE-CAL records but is not in the dataset")
        name2idx[nm]
    end
    for nm in step.cable
        haskey(name2idx, nm) || error("PcalInit: `cable` antenna $nm is not in the dataset")
    end
    addcable = [ds.antennas[p].name in step.cable for p in eachindex(ds.antennas)]

    tf = first(rec.freq_1)
    (all(==(tf), rec.freq_1) && all(==(tf), rec.freq_2)) ||
        error("PcalInit: the PHASE-CAL tone frequencies are not the same for every record and " *
              "receptor — this recipe reads one comb (`freq_1` of the first record) for all of them")

    fcell = _band_cells(Tf, first(rec.freq_1))
    Δftone = [ustrip(u"Hz", first(rec.freq_1)[2, b] - first(rec.freq_1)[1, b]) for b in 1:nband]
    all(>(0), Δftone) || error("PcalInit: non-positive tone spacing $(Δftone) Hz")
    νmid = [ustrip(u"Hz", (first(rec.freq_1)[1, b] + first(rec.freq_1)[2, b]) / 2) for b in 1:nband]
    νref = ustrip.(u"Hz", references(Tf))
    tcell = _record_cells(Tt, rec.time, rec.interval, ds.time0)

    nant = nantennas(ds)
    ntc = ncells(Tt)
    amed = zeros(nant, 2, ntone, nband)
    for p in 1:nant, i in 1:2
        idxs = findall(==(p), antidx)
        isempty(idxs) && continue
        col = i == 1 ? rec.pcal_1 : rec.pcal_2
        for tn in 1:ntone, b in 1:nband
            amed[p, i, tn, b] = median(abs(col[k][tn, b]) for k in idxs)
        end
    end

    τ = fill(NaN, nant, 2, ntc, nband)
    φ = fill(NaN, nant, 2, ntc, nband)
    nused = 0
    for (p, i) in Iterators.product(1:nant, 1:2)
        col = i == 1 ? rec.pcal_1 : rec.pcal_2
        rows_p = findall(k -> antidx[k] == p && tcell[k] > 0, eachindex(antidx))
        isempty(rows_p) && continue
        for (tc, poss) in pairs(groupfind(k -> tcell[k], rows_p))
            ks = view(rows_p, poss)
            Σ = zeros(ComplexF64, ntone, nband)
            n = zeros(Int, ntone, nband)
            for k in ks, tn in 1:ntone, b in 1:nband
                z = ComplexF64(col[k][tn, b])
                abs(z) ≥ step.amplitude_floor * amed[p, i, tn, b] || continue
                Σ[tn, b] += z; n[tn, b] += 1
            end
            cab = 0.0
            if addcable[p]
                cs = [ustrip(u"s", c) for c in @view(rec.cable_cal[ks]) if !isnan(c)]
                isempty(cs) && continue
                cab = median(cs)
            end
            cabτ = step.cable_apply === :none ? 0.0 : cab
            cabφ = step.cable_apply === :both ? cab : 0.0
            for b in 1:nband
                (n[1, b] > 0 && n[2, b] > 0) || continue
                dφ = _wrapπ(angle(Σ[2, b]) - angle(Σ[1, b]))
                τ[p, i, tc, b] = dφ / (2π * Δftone[b]) + cabτ
                φ[p, i, tc, b] = _wrapπ(angle(Σ[1, b]) + dφ / 2 + 2π * νmid[b] * cabφ)
                nused += 1
            end
        end
    end
    nused > 0 || error("PcalInit: no usable PHASE-CAL record fell inside any time cell — do the record times match the dataset's scans?")

    vals_d = copy(sol[Delay].values)
    vals_p = copy(sol[PhaseOffset].values)
    Δτ = fill(NaN, nband); Δφ = fill(NaN, nband)
    for p in 1:nant, i in 1:2
        med = fill(NaN, nband); φmed = fill(NaN, nband)
        for b in 1:nband
            cells = findall(tc -> isfinite(τ[p, i, tc, b]), 1:ntc)
            isempty(cells) && continue
            amb = 1 / Δftone[b]
            med[b] = amb / 2π * angle(sum(t -> cis(2π * t / amb), @view τ[p, i, cells, b]))
            φmed[b] = angle(sum(cis, @view φ[p, i, cells, b]))
        end
        for tc in 1:ntc
            bands = findall(b -> isfinite(τ[p, i, tc, b]), 1:nband)
            isempty(bands) && continue
            for b in bands
                amb = 1 / Δftone[b]
                d = τ[p, i, tc, b] - med[b]
                nturn = round(d / amb)
                Δτ[b] = d - nturn * amb
                Δφ[b] = _wrapπ(φ[p, i, tc, b] - φmed[b] - nturn * π)
            end
            phase_reference = angle(sum(b -> cis(Δφ[b]), bands))
            for b in bands
                fc = fcell[b]
                vals_d[p, i, tc, fc] = Δτ[b]
                vals_p[p, i, tc, fc] = phase_reference + _wrapπ(Δφ[b] - phase_reference) + 2π * Δτ[b] * (νref[fc] - νmid[b])
            end
        end
    end

    comps = merge(sol.components, (;
        Delay = ComponentState(dd, vals_d),
        PhaseOffset = ComponentState(dp, vals_p)))
    (Solution(sol.dataset, comps), nothing)
end
