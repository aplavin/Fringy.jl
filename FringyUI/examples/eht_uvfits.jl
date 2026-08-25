
using Fringy, Unitful, IntervalSets, Accessors
include(joinpath(@__DIR__, "common.jl"))

const PATH = "data/ehthops-345-mixedpol-Rev2-r7/ehthops/hops-b1/6.uvfits/3769/hops_3769_3C279.uvfits"
const REF  = :AA
const WIN  = (; rate = (-0.5u"Hz")..(0.5u"Hz"), delay = (-4e-9u"s")..(4e-9u"s"))
const OVS  = (; rate = 4.0, delay = 4.0)

ds = load_dataset(PATH; scans = GapBasedScans(min_gap = 60u"s"))
println("loaded $(nrows(ds)) rows, $(nchannels(ds)) ch, $(nantennas(ds)) ant, $(length(unique(ds.rows.scan))) scans")

byscan = partition(ds, ByScan()); byif = partition(ds, ByIF())
whole  = partition(ds, WholeObservation()); bychan = partition(ds, ByChannel())
fband  = group(byif, (Tuple(1:nchannels(ds)),))

sol0 = Solution(ds; components = (
    PhaseOffset(; time = byscan, frequency = fband),
    Delay(;       time = byscan, frequency = fband),
    Rate(;        time = byscan, frequency = fband),
    PhaseBandpass(;        time = whole, frequency = bychan),
    LogAmplitudeBandpass(; time = whole, frequency = bychan)))

sched = [
    Capture(Fringes() => :fft,
            FringeSelf(FringeFit(; tiles = (; time = byscan, frequency = fband),
                                        window = WIN, oversample = OVS, refine = NoRefine()),
                            PhaseOffset, Delay, Rate; min_snr = 6.0,
                            globalization = SNRWeighted(iterations = 50), reference = REF)),
    StEFCal(PhaseBandpass, LogAmplitudeBandpass; coherence = (; time = byscan, frequency = fband),
            iterations = 20, regularization = 1e-3, reference = (REF, 1)),
]

run_example(; ds, sol0, sched, scan = 4, pair = (:AA, :PV), name = "eht_uvfits")
