"""
    Fringy

Generic VLBI visibility processing, calibration, delay modelling, and imaging primitives.

The package core defines [`Dataset`](@ref), selections and partitions, calibration components and
[`Solution`](@ref), fringe measurement and self-calibration, phase-calibration and amplitude
calibration, averaging, visibility-domain models, and imaging primitives. These APIs operate on
values. The generic delay layer consists of [`CorrelatorDelayModel`](@ref), for the model recorded by the
correlator, and [`AstrometryModel`](@ref), which combines the independently evaluated geometric,
tropospheric, and ionospheric terms. These APIs can be composed into alignment, imaging, astrometry,
or other workflows.

The repository currently also co-locates an astrometry application layer inside the package:
[`SessionConfig`](@ref), the functions under `src/steps/`, astrometric product writers under
`src/products/`, and the routines under `src/diagnostics/`. Its documentation lives under
`applications/astrometry/`; the currently committed orchestration, dataset descriptors, acquisition
helpers, and reference data remain under `sessions/`. Those interfaces describe one consumer of the
generic package rather than the definition of Fringy as a whole.

See `docs/README.md` for the package documentation map and `applications/astrometry/README.md` for
the astrometry workflow.
"""
module Fringy

using LinearAlgebra, StaticArrays, StructArrays, Statistics, Unitful, Dates
using Printf: @printf, @sprintf
using UnitfulAstro
using IntervalSets
using Graphs: SimpleGraph, add_edge!, connected_components
using SplitApplyCombine: mapview
using DataManipulation: flatmap, groupfind, groupview
using AxisKeys: KeyedArray
using Dictionaries: Dictionary, set!
using VLBIData: Antenna, Baseline, GapBasedScans, add_scan_ids
import VLBIData
import VLBIFiles
using VLBIFiles: FrequencyWindow, uvtable_wide, coherencymatrices, frequencies, frequency
using InterferometricModels: Point, MultiComponentModel, Beam, EllipticGaussian, components,
                             flux, coords, fwhm_max, fwhm_min, position_angle, convolve, intensity,
                             visibility
import AccessorsExtra
import ERFA
using UncertainSkyCoords: ICRSCoords, SphericalOffsetFlat, separation, U
using TOML: TOML
import SHA
using Skipper: filterview
using Accessors: @o, @insert, @set, setproperties
import ConstructionBase
using Random: AbstractRNG, MersenneTwister

include("dataset.jl")
include("partition.jl")
include("component.jl")
include("solution.jl")
include("apply.jl")
include("flags.jl")
include("fft.jl")
include("stationize.jl")
include("schedule.jl")
include("fringetable.jl")
include("fringefit.jl")
include("fringeself.jl")
include("ifalignment.jl")
include("pcal.jl")
include("pcalpick.jl")
include("tsys.jl")
include("stefcal.jl")
include("simulate.jl")
include("corrmodel.jl")
include("geometry.jl")
include("troposphere.jl")
include("ionosphere.jl")
include("structure.jl")
include("totals.jl")
include("estimate.jl")
include("imaging.jl")
include("ampcal.jl")
include("sessionconfig.jl")
include("steps/load.jl")
include("steps/astrometry_models.jl")
include("steps/instrumental.jl")
include("steps/fringe.jl")
include("steps/derive_if_alignment.jl")
include("steps/calibrate_and_average.jl")
include("steps/observables.jl")
include("steps/solve.jl")
include("steps/models.jl")
include("steps/external.jl")
include("steps/imaging.jl")
include("steps/register.jl")

include("products/util.jl")
include("products/fits.jl")
include("products/closure.jl")
include("products/catalogue.jl")
include("products/fittables.jl")
include("products/wcs.jl")
include("products/report.jl")
include("products/manifest.jl")
include("steps/products.jl")

include("diagnostics.jl")

export Dataset, load_dataset, combine, Selection, select, present,
       nrows, nchannels, nantennas, nsources, parentrows, parentchannels,
       Antenna, Baseline, GapBasedScans, @o
export Partition, TimePartition, FreqPartition, TimeAxis, FreqAxis,
       partition, WholeObservation, ByScan, ByDuration, ByIF, ByChannel, TimeBoundaries,
       coarsen, group, refines, assert_refines,
       ncells, references, supports, cellof, cell_of_row, cell_of_channel
export PhaseOffset, Delay, Rate, PhaseBandpass, LogAmplitudeBandpass,
       ComponentState, kindof, response, gain,
       Solution, dataset, jones_terms, haskind
export calibrated_dataset, Averaging, average, simulate, corrupt
export CorrelatorDelayModel, τ_correlator, τ_correlator_clock, τ_correlator_troposphere,
       UTCEpoch, utc_mjd, AntennaGeometry, load_antenna_geometry, EOPSeries, read_finals2000A, eop_at,
       GeometryTerms, FULL_GEOMETRY, geometric_delay, azel,
       zenith_hydrostatic_path, standard_pressure, surface_pressure, pressure_series,
       hydrostatic_mapping, wet_mapping, gradient_mapping,
       IonexSeries, read_ionex, vtec, pierce_point, slant_iono_delay, iono_delays,
       iono_baseline_correction, tec_to_delay, group_delay_ν_eff

end
