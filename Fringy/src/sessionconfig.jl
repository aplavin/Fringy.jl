
export SessionConfig, PointSource, ExternalModels, TsysCalibration, VisibilityAveraging, OwnImaging,
       model_path, image_path, models_available, external_models, own_imaging

"""
    PointSource()

The zero-structure-delay model: a point source at the visibility phase centre.
"""
struct PointSource end

"""
    ExternalModels(; dir, epoch = "", model_pattern = "{src}.mod", image_pattern = nothing,
                     archive = "", note = "")

A per-source external brightness-model set: one model file per source in `dir`, optionally with a
restored image beside it.

`model_pattern` and `image_pattern` may contain `{src}` and `{epoch}` placeholders. `epoch` is the
substitution value; `archive` and `note` are provenance metadata. `image_pattern = nothing` declares
that the set has no restored images. Every component in each usable model file enters the model set.
"""
Base.@kwdef struct ExternalModels
    dir::String
    epoch::String = ""
    model_pattern::String = "{src}.mod"
    image_pattern::Union{Nothing, String} = nothing
    archive::String = ""
    note::String = ""
end

_subst(pattern, m::ExternalModels, src) =
    replace(pattern, "{src}" => String(Symbol(src)), "{epoch}" => m.epoch)

"""
    model_path(m::ExternalModels, src) -> String

Return `joinpath(m.dir, m.model_pattern)` after substituting `{src}` and `{epoch}`. File existence is
not checked.
"""
model_path(m::ExternalModels, src) = joinpath(m.dir, _subst(m.model_pattern, m, src))

"""
    image_path(m::ExternalModels, src) -> String

Return the restored-image path after substituting `{src}` and `{epoch}`. Errors when
`m.image_pattern === nothing`; file existence is not checked.
"""
function image_path(m::ExternalModels, src)
    isnothing(m.image_pattern) &&
        error("image_path: this external model set declares no image_pattern (dir $(m.dir))")
    joinpath(m.dir, _subst(m.image_pattern, m, src))
end

"""
    models_available(m, sources) -> Bool

Return whether every source has the nonempty files declared by the external-model configuration: its
model and, when configured, its restored image. This is a lightweight filesystem check; it does not
load or validate file contents. `models_available(nothing, sources)` is `false`.
"""
_nonempty_file(path) = isfile(path) && filesize(path) > 0

_source_files_available(m::ExternalModels, src) =
    _nonempty_file(model_path(m, src)) &&
    (isnothing(m.image_pattern) || _nonempty_file(image_path(m, src)))

models_available(m::ExternalModels, sources) = all(s -> _source_files_available(m, s), sources)
models_available(::Nothing, sources) = false

"""
    TsysCalibration(; tsys_ceiling = 400.0, quantization = 0.88,
                      reference_sefd = 1.0, fallback_cap = 0.2)

Settings for the a priori Tsys-derived amplitude scale.
"""
Base.@kwdef struct TsysCalibration
    tsys_ceiling::Float64 = 400.0
    quantization::Float64 = 0.88
    reference_sefd::Float64 = 1.0
    fallback_cap::Float64 = 0.2
end

"""
    VisibilityAveraging(; time = 20.0u"s", channels = 128)

Time and channel averaging settings for calibrated visibilities.
"""
Base.@kwdef struct VisibilityAveraging{T}
    time::T = 20.0u"s"
    channels::Int = 128
end

"""
    OwnImaging(; ...)

Configuration for the astrometry application's CLEAN, hybrid-calibration, amplitude self-calibration,
and model-variant stages. Every keyword has a default. Symbol-valued options are restricted by their
consumers and fail there when unsupported.

| keyword | unit | default | what reads it |
|---|---|---|---|
| `npix` | pixels | `256` | the imaging grid (`steps/imaging.jl`) |
| `pixel_mas` | mas | `0.1` | the imaging grid (`steps/imaging.jl`) |
| `clean_gain` | — | `0.1` | `image_source` (`steps/imaging.jl`) |
| `clean_threshold_nsigma` | σ_map | `5.0` | `image_source`'s stopping rule |
| `clean_niter` | iterations | `100_000` | `image_source`'s runaway guard |
| `hybrid_iterations` | iterations | `0` | the hybrid loop (`steps/imaging.jl`) |
| `model_floor` | fraction | `0.05` | `model_dataset`'s near-null floor |
| `hybrid_window` | Hz / s | ±0.005 Hz, ±4 ns | the hybrid loop's re-fringe |
| `hybrid_oversample` | — | `4.0` / `4.0` | the hybrid loop's re-fringe |
| `hybrid_min_snr` | — | `5.0` | the hybrid loop's self-calibration |
| `amp_interval` | duration or `:scan`/`:session` | `30u"minute"` | the amplitude self-cal cell |
| `amp_normalize` | `:percell`/`:global`/`:none` | `:global` | the amplitude self-cal gauge |
| `amp_frequency` | `:band`/`:if` | `:band` | the amplitude self-cal granularity |
| `amp_model` | `:significant`/`:full` | `:significant` | which components calibrate |
| `amp_iterations` | iterations | `1` | the amplitude self-cal loop |
| `amp_regularization` | — | `1e-3` | the amplitude self-cal solve |
| `amp_min_snr` | — | `5.0` | the amplitude self-cal per-datum floor |
| `receptor_peak_separation_beam_fraction` | beams | `0.1` | the per-receptor-slot seeding contingency |
| `weighting` | — | `:natural` | the uv weighting of the maps and their beams (`image_run`) |
| `refringe_targets`, `refringe_refine`, `refringe_receptor_coupling` | — | `:full`, `:ml`, `:independent` | the hybrid loop's re-fringe |
| `variants` | suffix ⇒ keyword overrides | `n10`, `uniform` | the registration-stability model sets |
"""
Base.@kwdef struct OwnImaging
    npix::Int = 256
    pixel_mas::Float64 = 0.1
    clean_gain::Float64 = 0.1
    weighting::Symbol = :natural
    clean_threshold_nsigma::Float64 = 5.0
    clean_niter::Int = 100_000

    hybrid_iterations::Int = 0
    model_floor::Float64 = 0.05
    hybrid_window = (; rate = (-0.005u"Hz")..(0.005u"Hz"), delay = (-4e-9u"s")..(4e-9u"s"))
    hybrid_oversample = (; rate = 4.0, delay = 4.0)
    hybrid_min_snr::Float64 = 5.0
    refringe_targets::Symbol = :full
    refringe_refine::Symbol = :ml
    refringe_receptor_coupling::Symbol = :independent

    amp_interval = 30u"minute"
    amp_normalize::Symbol = :global
    amp_frequency::Symbol = :band
    amp_model::Symbol = :significant
    amp_iterations::Int = 1
    amp_regularization::Float64 = 1e-3
    amp_min_snr::Float64 = 5.0

    receptor_peak_separation_beam_fraction::Float64 = 0.1

    variants::Vector{Pair{String, NamedTuple}} = [
        "n10" => (; clean_threshold_nsigma = 10.0),
        "uniform" => (; weighting = :uniform),
    ]
end


"""
    SessionConfig(; <keywords>)

Immutable configuration for one astrometry application run. Required filesystem, catalogue, and
gauge inputs have no defaults; optional fields use the defaults shown below. Configuration contains
choices only. [`SessionFacts`](@ref) contains identity, epoch, and channel-comb values derived from the
visibility data.

`(req)` below marks a required keyword — omitting it is an `UndefKeywordError`, never a silent zero.

## Derived identity and epoch

There is no `name`, `date`, `epoch_obs`, `doy` or `ν_eff` field: those five are data-derived facts.
[`load_session`](@ref) derives them from the file into [`SessionFacts`](@ref), which travels in every
product's metadata; storing them in configuration would allow the descriptor to contradict its data.

## Data and physical-model inputs

| field | type | default | meaning · consumers |
|---|---|---|---|
| `data` | `String` | (req) | the FITS-IDI file. Everything that reads visibilities, tones, weather or the correlator model |
| `antenna_geometry` | `String` | (req) | antenna TOML: ITRF positions + velocities, axis offsets, ocean loading. Array-specific, not session-specific |
| `eop` | `String` | (req) | IERS `finals2000A.all`; `read_finals2000A` |
| `ionex_dir` | `String` | (req) | directory holding the IONEX maps named by `ionex_files` |
| `ionex_files` | `NamedTuple` | (req) | product ⇒ file list covering the session (a run crossing UT midnight needs two days). One key per GIM product |
| `iono_primary` | `Symbol` | `:COD` | which product IS the correction; the others give its uncertainty |
| `iono_inflation` | `Float64` | `1.5` | the ionospheric σ is this × the products' spread on the baseline differential |
| `geometry` | `GeometryTerms` | `FULL_GEOMETRY` | enabled physical geometry terms; recorded in the observables product's `meta.terms` |

## Independent calibration and estimator gauges

| field | type | default | meaning · consumers |
|---|---|---|---|
| `reference_antenna` | `Symbol` | (req) | reference antenna for calibration graph gauges and the estimator clock gauge |
| `reference_receptor_slot` | `Int` | (req) | estimator reference receptor slot, `1` or `2` |

Both fields are explicit application choices. `reference_antenna` must be present in the dataset and
must provide every structural receptor slot that [`DeriveIFAlignment`](@ref) is asked to solve.

## Fringe measurement

| field | type | default | meaning · consumers |
|---|---|---|---|
| `scans` | `GapBasedScans` | `GapBasedScans(min_gap = 1u"minute")` | how visibility rows become scans; `load_dataset` |
| `window` | `NamedTuple` | ±0.06 Hz, ±80 ns | the fringe search rectangle (`rate`, `delay` intervals); `FringeFit` |
| `oversample` | `NamedTuple` | `4.0` / `4.0` | FFT zero-padding factors per axis; `FringeFit` |
| `refine` | `LocalML` | `LocalML(iterations = 2)` | sub-bin refinement of the peak; `FringeFit` |
| `min_snr` | `Float64` | `7.0` | SNR floor for installed fringe solutions and default `GlobalDelayFit` acceptance; captured peaks stay ungated |

## A priori amplitude calibration and visibility averaging

| field | type | default | meaning · consumers |
|---|---|---|---|
| `tsys_calibration` | `TsysCalibration` | `TsysCalibration()` | Tsys acceptance, amplitude scale, and fallback policy; `calibrate_and_average` and the dataset diagnostics |
| `visibility_averaging` | `VisibilityAveraging` | `VisibilityAveraging()` | calibrated-visibility time and channel averaging; `calibrate_and_average` |

## Instrumental calibration — the tone-derived solution (`PcalInit`)

| field | type | default | meaning · consumers |
|---|---|---|---|
| `tone_floor` | `Float64` | `0.25` | drop a tone whose amplitude fell below this fraction of its own session median |
| `tone_pick` | `Union{Nothing,Symbol}` | `nothing` | how a >2-tone comb is cut to the pair `PcalInit` decodes; `nothing` is the no-op, `:outermost` the widest healthy pair |
| `cable_antennas` | `Vector{Symbol}` | (req) | the antennas whose CABLE_CAL readout is trusted |
| `cable_apply` | `Symbol` | `:none` | `:both` / `:delay` / `:none` — which halves of the tone measurement the cable monitor corrects |

## IF alignment — the fringe-derived per-IF solution (`DeriveIFAlignment`)

| field | type | default | meaning · consumers |
|---|---|---|---|
| `if_alignment` | `Bool` | `false` | solve and install the session-static per-(antenna, receptor slot, IF) alignment |
| `if_alignment_min_snr` | `Float64` | `10.0` | per-IF peak SNR floor of that solve |
| `if_alignment_pairs` | function | diagonal pairs of receptor slots | which peaks for pairs of receptor slots enter it |

## Observable form and the global solve

| field | type | default | meaning · consumers |
|---|---|---|---|
| `captured_phase_mode` | `Symbol` | `:retain` | `:retain` keeps the captured tone-derived phase slope in the observable; `:add_back` adds it algebraically back |
| `elevation_cutoff` | `Float64` | `deg2rad(7.0)` | rad — observations below it are not fitted |
| `clock_node` | `Float64` | `3600.0` | s — PWL clock knot spacing about the correlator's linear clock model |
| `zwd_node` | `Float64` | `1800.0` | s — PWL zenith wet delay knot spacing |
| `clock_rate_constraint` | `Float64` | `72e-12/3600` | s/s — constraint σ on the clock rate between knots |
| `zwd_rate_constraint` | `Float64` | `40e-12/3600` | s/s — constraint σ on the ZWD rate |
| `gradient_constraint` | `Float64` | `0.5e-3/c` | s — constraint σ on the tropospheric gradients (0.5 mm of path) |
| `outlier_nsigma` | `Float64` | `3.5` | rejection threshold, WITH restoration |
| `robust_iterations` | `Int` | `8` | maximum reweight/reject passes |
| `sigma_floor_init` | `Float64` | `10e-12` | s — starting per-baseline additive noise |
| `min_baseline_obs` | `Int` | `5` | fewer accepted rows on a baseline ⇒ keep the initial floor |
| `gn_iterations` | `Int` | `2` | Gauss–Newton steps (the second checks stationarity) |

The estimator fields are read by `delay_fit_spec`, which also supplies the reference gauge and
`min_snr`. `GlobalDelayFit` requires all sixteen of its keywords.

## Structure models and registration

| field | type | default | meaning · consumers |
|---|---|---|---|
| `structure_wrap` | `Float64` | `π/2` | rad — the per-observable unwrap guard: an inter-channel structure-phase step this large means the unwrap is undetermined, and the observable is dropped |
| `model_radius_mas` | `Float64` | `5.0` | mas — the CLEAN disk of our own imaging |
| `component_nsigma` | `Float64` | `5.0` | components entering a MODEL must clear this × the residual rms |
| `closure_amp_min_snr` | `Float64` | `5.0` | per-baseline SNR floor of the closure-amplitude statistic |
| `structure_models` | `Tuple` | `(PointSource(),)` | ordered [`PointSource`](@ref), [`OwnImaging`](@ref), or [`ExternalModels`](@ref) specifications; at most one imaging and one external specification |
| `chi2_margin` | `Float64` | `1.2` | the factor one closure-phase χ² must beat another by before the comparison says anything; PUBLISHED metadata, read by nothing |

`model_radius_mas` is measured about each own image's dirty-map peak and must fit inside the CLEAN
field. `closure_amp_min_snr` is top-level because closure evaluation also
applies to external model sets. `chi2_margin` is published as metadata; no stage branches on it.

## Where the files go

| field | type | default | meaning · consumers |
|---|---|---|---|
| `work` | `String` | (req) | this session's INTERMEDIATES directory — the runner's `.jls` products |
| `rfc_version` | `String` | (req) | the named RFC catalogue release; the application resolves it to the catalogue it passes to `products` |
| `products` | `Tuple` | all seven | which deliverables the products step emits; a step runs iff a requested product needs it |

These fields are required and have no default. `work` holds rerunnable intermediates; the application
runner supplies the separate results directory for deliverables.

The application validates `products`; a step runs when at least one requested product requires it.

## Diagnostics fixture

| field | type | default | meaning · consumers |
|---|---|---|---|
| `reference_sources` | `Vector{Symbol}` | `Symbol[]` | sources for the optional own-image versus external-model orientation comparison and its gallery panels |

This comparison applies only when [`OwnImaging`](@ref) and an external model set are both configured;
an empty vector disables it.
"""
Base.@kwdef struct SessionConfig
    data::String
    antenna_geometry::String
    eop::String
    ionex_dir::String
    ionex_files::NamedTuple
    iono_primary::Symbol = :COD
    iono_inflation::Float64 = 1.5
    geometry::GeometryTerms = FULL_GEOMETRY

    reference_antenna::Symbol
    reference_receptor_slot::Int

    scans::GapBasedScans = GapBasedScans(min_gap = 1u"minute")
    window::NamedTuple = (; rate = (-0.06u"Hz")..(0.06u"Hz"), delay = (-8e-8u"s")..(8e-8u"s"))
    oversample::NamedTuple = (; rate = 4.0, delay = 4.0)
    refine::LocalML = LocalML(iterations = 2)
    min_snr::Float64 = 7.0

    tsys_calibration::TsysCalibration = TsysCalibration()
    visibility_averaging::VisibilityAveraging = VisibilityAveraging()

    tone_floor::Float64 = 0.25
    tone_pick::Union{Nothing, Symbol} = nothing
    cable_antennas::Vector{Symbol}
    cable_apply::Symbol = :none

    if_alignment::Bool = false
    if_alignment_min_snr::Float64 = 10.0
    if_alignment_pairs::Function = e -> e.i == e.j

    captured_phase_mode::Symbol = :retain
    elevation_cutoff::Float64 = deg2rad(7.0)
    clock_node::Float64 = 60 * 60.0
    zwd_node::Float64 = 30 * 60.0
    clock_rate_constraint::Float64 = 72e-12 / 3600
    zwd_rate_constraint::Float64 = 40e-12 / 3600
    gradient_constraint::Float64 = 0.5e-3 / 299792458.0
    outlier_nsigma::Float64 = 3.5
    robust_iterations::Int = 8
    sigma_floor_init::Float64 = 10e-12
    min_baseline_obs::Int = 5
    gn_iterations::Int = 2

    structure_wrap::Float64 = π / 2
    model_radius_mas::Float64 = 5.0
    component_nsigma::Float64 = 5.0
    closure_amp_min_snr::Float64 = 5.0
    structure_models::Tuple = (PointSource(),)
    chi2_margin::Float64 = 1.2

    work::String
    rfc_version::String
    products::Tuple = (:catalogue, :fit_tables, :wcs, :closure, :figures, :report, :manifest)

    reference_sources::Vector{Symbol} = Symbol[]

    function SessionConfig(args...)
        length(args) == 46 || throw(MethodError(SessionConfig, args))
        _validate_receptor_slot(args[10], "SessionConfig")
        new(args...)
    end
end

"""
    external_models(cfg) -> ExternalModels | nothing

Return the configured external model set, or `nothing`. Errors when more than one
[`ExternalModels`](@ref) value is present.
"""
function external_models(cfg::SessionConfig)
    ms = filter(m -> m isa ExternalModels, collect(cfg.structure_models))
    length(ms) > 1 && error("external_models: $(length(ms)) external model sets configured; one at most")
    isempty(ms) ? nothing : ms[1]
end

"""
    own_imaging(cfg) -> OwnImaging | nothing

Return the configured [`OwnImaging`](@ref) value, or `nothing`. Errors when more than one is present.
"""
function own_imaging(cfg::SessionConfig)
    ms = filter(m -> m isa OwnImaging, collect(cfg.structure_models))
    length(ms) > 1 && error("own_imaging: $(length(ms)) imaging configurations; one at most")
    isempty(ms) ? nothing : ms[1]
end
