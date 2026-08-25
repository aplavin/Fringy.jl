
using Fringy, Unitful, IntervalSets, Accessors
include(joinpath(@__DIR__, "common.jl"))

const PATH = "e21f19-1-b2-b7mar20-3C279.fits/E21F19.0.bin0000.source0000.FITS"
const REF  = :AA
const SCAN = 5
const WIN  = (; rate = (-0.5u"Hz")..(0.5u"Hz"), delay = (-50e-9u"s")..(50e-9u"s"))
const OVS  = (; rate = 4.0, delay = 4.0)

ds = load_dataset(PATH; scans = GapBasedScans(min_gap = 10u"s"))
println("loaded $(nrows(ds)) rows, $(nchannels(ds)) ch, $(nantennas(ds)) ant, $(length(unique(ds.rows.scan))) scans")

byscan = partition(ds, ByScan()); byif = partition(ds, ByIF())
fband  = group(byif, (Tuple(1:maximum(ds.freq.if_of)),))
sel = Selection(@o _.scan == SCAN)

whole_band_ok = try
    probesol  = Solution(ds; components = (PhaseOffset(; time = byscan, frequency = fband),
                                           Delay(; time = byscan, frequency = fband),
                                           Rate(;  time = byscan, frequency = fband)))
    probestep = FringeFit(; tiles = (; time = byscan, frequency = fband),
                            window = WIN, oversample = OVS, refine = NoRefine(), selection = sel)
    probestep(probesol)
    true
catch e
    @warn "whole-band 32-IF tile refused (non-commensurate lattice) — falling back to per-IF tiles" exception = e
    false
end
freqtiles = whole_band_ok ? fband : byif
println("frequency tiling: ", whole_band_ok ? "whole-band (32-IF, snapped)" : "per-IF (whole-band refused)")

sol0 = Solution(ds; components = (
    PhaseOffset(; time = byscan, frequency = freqtiles),
    Delay(;       time = byscan, frequency = freqtiles),
    Rate(;        time = byscan, frequency = freqtiles)))

sched = [Capture(Fringes() => :fft,
                 FringeSelf(FringeFit(; tiles = (; time = byscan, frequency = freqtiles),
                                             window = WIN, oversample = OVS, refine = NoRefine(), selection = sel),
                                 PhaseOffset, Delay, Rate; min_snr = 6.0,
                                 globalization = SNRWeighted(iterations = 50), reference = REF))]

run_example(; ds, sol0, sched, scan = SCAN, pair = (:AA, :PV), name = "eht_fitsidi")
