"""
    FringyUI

FringyUI — Dear ImGui / ImPlot frontend for the alignment-only `Fringy`: `Solution`/`ComponentState`
inspector with Statistics, Timeline, Solution, baseline-grid, fringe-plane and UV-profile panels.
"""
module FringyUI

import CImGui as ig
import CImGui.lib as lib
import ImPlot
import ImPlotExtra
import ImGuiThemes
import GLFW
import ModernGL as GL
using Fringy
using Fringy: Dataset, Solution, ComponentState, Step, Capture, Fringes, FringeFit, FringeSelf, StEFCal,
              PhaseOffset, Delay, Rate, PhaseBandpass, LogAmplitudeBandpass,
              jones_terms, dataset, haskind, present, fringe_plane,
              ncells, references, supports, cell_of_row, cell_of_channel,
              nantennas, nrows, nchannels, partition,
              WholeObservation, ByScan, ByIF, ByChannel, ByDuration, group,
              Averaging, average, calibrated_dataset,
              select, Selection
using Geodesy: ECEF, LLA, wgs84
using StructArrays, IntervalSets
using Accessors
using Logging
using Printf
using Statistics: median
import Unitful
using Unitful: @u_str, ustrip, NoUnits
using Dates: DateTime, datetime2unix, unix2datetime
using DataManipulation: flatmap, filtermap, groupview, mapview

ig.set_backend(:GlfwOpenGL3)

include("decimate.jl")
include("derive.jl")
include("runtime.jl")

function __init__()
    _SCREEN_CAPTURE_C[] = @cfunction(_screen_capture, Bool,
        (Cuint, Cint, Cint, Cint, Cint, Ptr{Cuint}, Ptr{Cvoid}))
    _AXIS_FMT_C[] = @cfunction(_axis_fmt, Cint, (Cdouble, Ptr{Cchar}, Cint, Ptr{Cvoid}))
end

export inspect, stop!

end
