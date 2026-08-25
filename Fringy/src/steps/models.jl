
export ModelSet, ModelSets, models, point_model_set, model_set_slug

"""
    ModelSet

One set of per-source brightness models, and everything known about where it came from.

| field | meaning |
|---|---|
| `label` | the set's name and its key in every product — `point`, `own`, `own:n10`, or `external:<id>`. A DATA VALUE, never an identifier in the code |
| `kind` | `:point` / `:own` / `:external` — what KIND of set it is, for the writers that treat images differently |
| `radius_mas` | the own-image CLEAN support radius, `Inf` where none applies |
| `models` | source ⇒ `MultiComponentModel`, each in ITS OWN frame. Empty for `point` |
| `images` | source ⇒ a `SourceImage` (ours) or an external image record (path + WCS + provenance). May be empty: a set without images is a complete model set |
| `info` | source ⇒ the per-source model description the catalogue publishes |
| `provenance` | kind-specific: the archive/epoch/dir and per-file SHA-256s, or the imaging run's own metadata |

Sources absent from `models` are absent, not zero-filled: `StructureModels` reports them and gives
them `τ_structure = 0`, which is exactly "this set says nothing about this source".
"""
struct ModelSet
    label::String
    kind::Symbol
    radius_mas::Float64
    models::Dictionary{Symbol, Any}
    images::Dictionary{Symbol, Any}
    info::Dictionary{Symbol, NamedTuple}
    provenance::NamedTuple
end

"""
    ModelSets

Label ⇒ [`ModelSet`](@ref), in the order `cfg.structure_models` produced them. The order is the RUN's
iteration order and the report's; it carries no meaning in any product, whose rows are sorted by
`(source, model_set)` so that no set occupies a privileged position.
"""
const ModelSets = Dictionary{String, ModelSet}

Base.show(io::IO, s::ModelSet) = print(io,
    "ModelSet($(repr(s.label)), $(s.kind), $(length(s.models)) models, $(length(s.images)) images",
    isfinite(s.radius_mas) ? ", r ≤ $(s.radius_mas) mas)" : ")")

"""
    model_set_slug(label) -> String

A model-set label as a path component: `own:n10` ⇒ `own-n10`. The label is the value; the slug replaces
directory-awkward punctuation.
"""
model_set_slug(label::AbstractString) = replace(label, ':' => '-', '@' => '-')

"""
    point_model_set() -> ModelSet

The model set with no components — the default reference model set for registration comparisons, and
the reason the register step has no special case (see the file header).
"""
point_model_set() = ModelSet("point", :point, Inf, Dictionary{Symbol, Any}(),
                             Dictionary{Symbol, Any}(), Dictionary{Symbol, NamedTuple}(),
                             (; kind = :point))

"""
    models(cfg, ld, calibration = nothing; verbose = true) -> ModelSets
    models(cfg, sources; facts, calibration = nothing, verbose = true) -> ModelSets

Materialize every structure-model source this session configures.

* `ld` is [`load_session`](@ref)'s result — used for the session's source list and for `facts.date`,
  which the external reader checks against each image header and warns about when they disagree;
* `calibration` is [`calibrate_and_average`](@ref)'s result, or `nothing`. An `OwnImaging` member needs it (it is what the
  imaging chain runs on); a session with none never looks at it, which is why a configuration without
  imaging needs no averaged visibilities and no mode.

The second form takes the source list directly, for a caller holding a cached fringe product rather
than an open dataset.

The expansion rule, which lives here and nowhere else:

| member | sets | labels |
|---|---|---|
| `PointSource()` | 1 | `point` |
| `OwnImaging(; variants)` | `1 + length(variants)` | `own`, `own:<suffix>` |
| `ExternalModels()` | 1 | `external:<id>` |
"""
models(cfg, ld, calibration = nothing; kwargs...) =
    models(cfg, sort!(unique(ld.ds.rows.source)); facts = ld.facts, calibration, kwargs...)

function models(cfg, sources::AbstractVector; facts, calibration = nothing, verbose = true)
    sets = ModelSets()
    for m in cfg.structure_models
        for (label, set) in pairs(model_sets_of(cfg, m, sources; facts, calibration, verbose))
            haskey(sets, label) &&
                error("models: two configured members both produce the model set $(repr(label)) — " *
                      "give one of them its own archive/epoch label")
            insert!(sets, label, set)
        end
    end
    verbose && for (label, s) in pairs(sets)
        @printf("  %-34s %-9s %3d models, %3d images%s\n", label, s.kind, length(s.models),
                length(s.images), isfinite(s.radius_mas) ? @sprintf(", r ≤ %.1f mas", s.radius_mas) : "")
    end
    sets
end

"""
    model_sets_of(cfg, member, sources; facts, calibration, verbose) -> ModelSets

The sets ONE configuration member produces. Three methods, one per member type, and no branch
anywhere else in the pipeline.
"""
model_sets_of(cfg, ::PointSource, sources; kwargs...) =
    ModelSets(["point"], [point_model_set()])

model_sets_of(cfg, m::ExternalModels, sources; facts, calibration = nothing, verbose = true) =
    external_model_sets(m, sources; date = facts.date, verbose)

function model_sets_of(cfg, own::OwnImaging, sources; facts, calibration = nothing, verbose = true)
    isnothing(calibration) &&
        error("models: this session configures OwnImaging, so it needs the averaged calibrated " *
              "visibilities — run `calibrate_and_average` and pass its result")
    sets = ModelSets()
    for (label, settings) in own_variants(own)
        insert!(sets, label, own_imaging_set(cfg, settings, calibration; label, sources, verbose))
    end
    sets
end

"""
    own_variants(own::OwnImaging) -> Vector{Pair{String, OwnImaging}}

The imaging run and its registration-stability variants, as `label ⇒ the OwnImaging that produces
it`. A variant is a set of KEYWORD OVERRIDES of `own`'s own fields, so an unknown key is a
construction error here rather than an unused command-line string.
"""
own_variants(own::OwnImaging) =
    ["own" => own,
     ("own:$suffix" => setproperties(own, over) for (suffix, over) in own.variants)...]
