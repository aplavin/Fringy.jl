
export StructureModels, StructureComb, structure_delay, absolute_position, uv_from_partials


mutable struct _SlopeAccum
    Sw::Float64
    Swν::Float64
    Swνν::Float64
    Swφ::Float64
    Swνφ::Float64
end
_SlopeAccum() = _SlopeAccum(0.0, 0.0, 0.0, 0.0, 0.0)

@inline function _accumulate!(a::_SlopeAccum, ν::Float64, φ::Float64, w::Float64)
    a.Sw += w; a.Swν += w * ν; a.Swνν += w * ν * ν; a.Swφ += w * φ; a.Swνφ += w * ν * φ
    a
end

@inline function _slope(a::_SlopeAccum)
    den = a.Sw * a.Swνν - a.Swν^2
    den > 0 ? (a.Sw * a.Swνφ - a.Swν * a.Swφ) / den : NaN
end


"""
    PARTIAL_UV_SIGN

The sign relating an observable's source partials `(∂τ_∂α★, ∂τ_∂δ)` to the uv coordinate of the imaging/model
convention ([`uv_of`](@ref), [`UV_SIGN`](@ref)). `−1`, and see the file header: it follows from
`total_delays`' delay identity plus `InterferometricModels`' `cis(2π·u·x)`, is asserted by the
synthetic mirrored-model gate, and is measured against the correlator's own `uvw` on the real session.
Flipping this one alone DOUBLES each source's structure bias instead of removing it; flipping it
together with `UV_SIGN` leaves every registered position alone and mirrors every map instead.
"""
const PARTIAL_UV_SIGN = -1.0

"""
    uv_from_partials(∂τ_∂α★, ∂τ_∂δ, ν) -> SVector{2,Float64}

The uv point [inverse milliarcseconds] of an observable with baseline source partials `(∂τ_∂α★, ∂τ_∂δ)`
[s/rad, w.r.t. local `(Δα★ = cos(δ) Δα, Δδ)`, as `total_delays` reports them] at frequency `ν` [Hz]:
`PARTIAL_UV_SIGN·(∂τ_∂α★, ∂τ_∂δ)·ν` converted from radians⁻¹ to mas⁻¹.

The analogue of [`uv_of`](@ref) for the astrometric plane, and the ONE place a source partial becomes
a uv coordinate. The two must agree on the same datum — that identity is the registration's sign
gate.
"""
@inline function uv_from_partials(∂τ_∂α★::Real, ∂τ_∂δ::Real, ν::Real)
    s = PARTIAL_UV_SIGN * ν * MAS
    SVector(Float64(∂τ_∂α★) * s, Float64(∂τ_∂δ) * s)
end


"""
    StructureComb(ν, if_of, ifs)

The frequency support a whole-band fringe measurement (and hence its structure correction) lives on:
the channel frequencies `ν` [Hz] in ASCENDING order, and, per channel, the index into the per-IF
weight vector that the fringe sum captured (0 for a channel whose IF is outside the tile).

Built once per pass — `ifs` is the tile's IF identities in the order the captured weights are stored
(ascending, the capture's own order) — and then reused for every observable, which supplies only its
own weights, `length(ifs)` of them ([`structure_delay`](@ref) fails loud on any other count, as
`applied_phase_delay` does). The ascending order is required, not sorted into: the phase unwrap of
[`structure_delay`](@ref) walks the comb, and a comb presented out of order would unwrap along a path
that is not the frequency axis. A file whose channel axis is not ascending — any lower-sideband IF,
whose channel frequencies DESCEND — is sorted at the construction site (`Fringy._structure_comb`,
`totals.jl`), where `ν` and `if_of` can be permuted together; the refusal here is what makes a comb
assembled by any other route fail loud instead of unwrapping sideways.
"""
struct StructureComb
    ν::Vector{Float64}
    wix::Vector{Int}
    nifs::Int

    function StructureComb(ν, if_of, ifs)
        length(ν) == length(if_of) ||
            error("StructureComb: $(length(ν)) frequencies but $(length(if_of)) IF identities")
        issorted(ν) ||
            error("StructureComb: the channel frequencies are not ascending — the phase unwrap walks the comb in the order given")
        wix = map(a -> something(findfirst(==(a), ifs), 0), if_of)
        new(collect(Float64, ν), collect(Int, wix), length(ifs))
    end
end

Base.show(io::IO, c::StructureComb) = print(io, "StructureComb(", length(c.ν), " channels, ",
    count(>(0), c.wix), " in the tile, ", round(1e-6 * (last(c.ν) - first(c.ν)); digits = 1), " MHz span)")


"""
    structure_delay(model, ∂τ_∂α★, ∂τ_∂δ, comb::StructureComb, weights; wrap_threshold) -> Float64

The structure delay [s] of one observable: the contribution the source's own brightness distribution
makes to its measured total delay, relative to the model's coordinate origin.

`model` is an `InterferometricModels` model in its own coordinate frame — its zero is the model
origin and nothing recentres it — which has already passed the significance policy
([`significant_components`](@ref)) — a model, never a bare component vector: the caller
carries a MODEL from the imaging product all the way here, so there is one representation of a
brightness distribution in this pipeline and no place where a list of components can lose the type
that says what it is. `(∂τ_∂α★, ∂τ_∂δ)` are the observable's baseline source partials
[s/rad] and `weights`
the per-IF weights the fringe sum accumulated, in `comb`'s own order.

It is the intercept-marginalized weighted LSQ slope of `arg V_model(u(ν))` over exactly the channels
and weights the measured delay used, negated as `total_delays` negates `delay` — see the file header,
and the one-component identity it implies:

    structure_delay(one Point at x) == ∂τ_∂α★·x_α★ + ∂τ_∂δ·x_δ        (exactly)

**Wrap guard, per observable.** The phase is unwrapped along the comb; a step whose
minimal representation reaches `wrap_threshold` [rad] between adjacent channels means the unwrap is
not determined — an in-band null crossing of a strong two-component source, real structure and exactly
where the fringe measurement itself is suspect. That observable returns **NaN**, which makes its
`τ_residual`
non-finite and drops it from the registration solve; it is counted and reported per source, never a
global failure. Where the unwrap STARTS is immaterial: a different starting channel changes the
unwrapped phase by a constant, and the slope marginalizes the intercept.

The step is between adjacent RETAINED channels: an IF the fringe sum gave no weight leaves a
larger gap in the walk. The unwrap across such a gap is unambiguous only while its phase step remains
under `π` — a bound on `τ_structure·Δν_gap`, not directly on source size.

The step tested is the step's MINIMAL representation, hence bounded by π by construction: a
`wrap_threshold` of exactly π is degenerate (nothing can trip it), and π/2 is
the operative one. This is a policy boundary rather than a fitted parameter.

An empty model (a source CLEAN found nothing significant in) gives exactly 0.0 — the "unresolved ⇒
registration = point-source position" property, which the significance policy is what
makes true.
"""
function structure_delay(model, ∂τ_∂α★::Real, ∂τ_∂δ::Real, comb::StructureComb, weights;
                         wrap_threshold::Real)
    wrap_threshold > 0 ||
        error("structure_delay: wrap_threshold must be positive [rad], got $wrap_threshold")
    length(weights) == comb.nifs ||
        error("structure_delay: $(length(weights)) captured weights for a comb of $(comb.nifs) IFs — the fringe tile and the comb disagree")
    isempty(components(model)) && return 0.0
    a = _SlopeAccum()
    φprev = 0.0; turns = 0.0; started = false
    @inbounds for c in eachindex(comb.ν)
        k = comb.wix[c]
        k == 0 && continue
        w = Float64(weights[k])
        w > 0 || continue
        ν = comb.ν[c]
        φ = angle(_structure_visibility(model, uv_from_partials(∂τ_∂α★, ∂τ_∂δ, ν)))
        if started
            d = rem(φ - φprev, 2π, RoundNearest)
            abs(d) ≥ wrap_threshold && return NaN
            turns += d - (φ - φprev)
        end
        φprev = φ; started = true
        _accumulate!(a, ν, φ + turns, w)
    end
    s = _slope(a)
    isnan(s) ? 0.0 : -s / 2π
end

@inline function _structure_visibility(model, uv)
    z = zero(ComplexF64)
    for c in components(model)
        z += visibility(c, uv)
    end
    z
end

"""
    StructureModels(models; wrap_threshold)

The structure correction as `total_delays` consumes it: per-source component models — coordinates
ALREADY relative to the chosen origin, significance policy ALREADY applied — plus the per-observable
wrap threshold of [`structure_delay`](@ref).

`models` supports `haskey`/`getindex` by source name. A source with no entry gets
`τ_structure = 0.0`; callers must report that absence because the fallback is scientifically valid only
for an unresolved source.
"""
struct StructureModels{M}
    models::M
    wrap_threshold::Float64
end
StructureModels(models; wrap_threshold::Real) = StructureModels(models, Float64(wrap_threshold))

Base.show(io::IO, s::StructureModels) = print(io, "StructureModels(", length(s.models),
    " sources, wrap guard ", round(s.wrap_threshold; digits = 3), " rad)")


"""
    absolute_position(x, fitted, model_origin) -> SVector{2,Float64}

The absolute `(ra, dec)` [rad] of the map coordinate `x` [mas, (East, North)], given the registered
absolute position `fitted = (ra, dec)` [rad] of the model origin and the
map coordinate `model_origin` [mas] corresponding to `fitted`:

    absolute(x) = fitted + (x − model_origin)

with the East offset converted at the target declination — the same convention `total_delays` uses for
its `source_offsets` hook, so a position round-tripped through both is exact.

In production `model_origin` is the model's own coordinate origin, `SVector(0.0, 0.0)`, so this reads
"the sky position of map coordinate `x`, in the WCS the solve delivered". The argument stays explicit
because the gauge statement needs it: displacing a
model's coordinates by Δ moves `fitted` by exactly Δ and leaves every `absolute_position` unchanged,
which is what makes the model origin a gauge rather than a result, and it is the property the origin-gauge
test asserts on synthetic data.

This origin function maps any pixel of any panel to the sky, so a brightness
peak's position is a LOOKUP here and never a second registration.
"""
function absolute_position(x, fitted, model_origin)
    dec = Float64(fitted[2]) + (Float64(x[2]) - Float64(model_origin[2])) * MAS
    ra = Float64(fitted[1]) + (Float64(x[1]) - Float64(model_origin[1])) * MAS / cos(dec)
    SVector(ra, dec)
end
