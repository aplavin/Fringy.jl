
using Fringy, Unitful, IntervalSets, Accessors
include(joinpath(@__DIR__, "common.jl"))

const BASE = "data/ehthops-345-mixedpol-Rev2-r7/ehthops"
const BANDS = ["hops-b1", "hops-b2", "hops-b3", "hops-b4"]
const SRCS  = ["3C273", "3C279", "J1058+0133", "J1146+3958", "J1337-1257", "J1512-0905", "M87"]
const REF  = :AA
const SRC_SHOWN = Symbol("J1337-1257")
const PAIR = (:AA, :MM)
const WIN  = (; rate = (-0.5u"Hz")..(0.5u"Hz"), delay = (-4e-9u"s")..(4e-9u"s"))
const OVS  = (; rate = 4.0, delay = 4.0)

paths = filter(isfile, [joinpath(BASE, b, "6.uvfits/3769/hops_3769_$(s).uvfits") for b in BANDS for s in SRCS])
ds = load_dataset(paths; scans = GapBasedScans(min_gap = 60u"s"))
println("combined $(length(paths)) files → $(nrows(ds)) rows, $(nchannels(ds)) ch, $(nantennas(ds)) ant, ",
        "$(nsources(ds)) sources, $(length(unique(ds.rows.scan))) scans, ",
        "$(round(minimum(ds.freq.ν)/1e9; digits=2))–$(round(maximum(ds.freq.ν)/1e9; digits=2)) GHz")

nband = 4
bandcells = Tuple(Tuple((b - 1) * 32 + 1 : b * 32) for b in 1:nband)
byscan = partition(ds, ByScan())
whole  = partition(ds, WholeObservation()); bychan = partition(ds, ByChannel())
perband = group(partition(ds, ByIF()), bandcells)

sol0 = Solution(ds; components = (
    PhaseOffset(; time = byscan, frequency = perband),
    Delay(;       time = byscan, frequency = perband),
    Rate(;        time = byscan, frequency = perband),
    PhaseBandpass(;        time = whole, frequency = bychan),
    LogAmplitudeBandpass(; time = whole, frequency = bychan)))

sched = [
    Capture(Fringes() => :fft,
            FringeSelf(FringeFit(; tiles = (; time = byscan, frequency = perband),
                                        window = WIN, oversample = OVS, refine = NoRefine()),
                            PhaseOffset, Delay, Rate; min_snr = 6.0,
                            globalization = SNRWeighted(iterations = 50), reference = REF)),
    StEFCal(PhaseBandpass, LogAmplitudeBandpass; coherence = (; time = byscan, frequency = perband),
            iterations = 20, regularization = 1e-3, reference = (REF, 1)),
]

calscan = first(sort(unique(ds.rows.scan[ds.rows.source .== SRC_SHOWN])))
run_example(; ds, sol0, sched, scan = calscan, pair = PAIR, name = "eht_wideband", solve_budget = 400000)
