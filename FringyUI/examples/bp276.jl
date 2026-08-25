
using Fringy, Unitful, IntervalSets, Accessors
include(joinpath(@__DIR__, "common.jl"))

const PATH  = "BP276/BP276A_L.idifits"
const REF   = :LA
const SCANS = (12, 18, 23)
const WIN   = (; rate = (-0.2u"Hz")..(0.2u"Hz"), delay = (-0.5e-6u"s")..(0.5e-6u"s"))
const OVS   = (; rate = 4.0, delay = 4.0)

ds = load_dataset(PATH; scans = GapBasedScans(min_gap = 30u"s"))
println("loaded $(nrows(ds)) rows, $(nchannels(ds)) ch, $(nantennas(ds)) ant")

byscan = partition(ds, ByScan()); byif = partition(ds, ByIF())
whole  = partition(ds, WholeObservation()); bychan = partition(ds, ByChannel())
sel = Selection(@o _.scan ∈ SCANS)

sol0 = Solution(ds; components = (
    PhaseOffset(; time = byscan, frequency = byif),
    Delay(;       time = byscan, frequency = byif),
    Rate(;        time = byscan, frequency = byif),
    PhaseBandpass(;        time = whole, frequency = bychan),
    LogAmplitudeBandpass(; time = whole, frequency = bychan)))

sched = [
    Capture(Fringes() => :fft,
            FringeSelf(FringeFit(; tiles = (; time = byscan, frequency = byif),
                                        window = WIN, oversample = OVS, refine = NoRefine(), selection = sel),
                            PhaseOffset, Delay, Rate; min_snr = 7.0,
                            globalization = SNRWeighted(iterations = 100), reference = REF)),
    StEFCal(PhaseBandpass, LogAmplitudeBandpass; coherence = (; time = byscan, frequency = byif),
            iterations = 20, regularization = 1e-3, reference = (REF, 1), selection = sel),
]

run_example(; ds, sol0, sched, scan = 12, pair = (:FD, :LA), name = "bp276")
