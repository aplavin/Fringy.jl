
export Step, Capture, Fringes, apply_step

"""
    Step

A stage in a calibration schedule (`FringeFit`, `FringeSelf`, …, and the composition wrapper
`Capture`). Run one via
[`apply_step`](@ref); a whole schedule is a `Vector{Step}` folded with `apply_step` (see the module header).
"""
abstract type Step end

"""
    Fringes()

Product-KIND singleton: wrapping a `FringeFit` (or a `FringeSelf`, which measures one on its way to
the solve) in `Capture(Fringes() => :name, step)` materializes the measured [`FringeTable`](@ref) — all
computable peaks, ungated.
"""
struct Fringes end

"""
    Capture(product::Pair, step::Step)

Wrap `step` so its product is materialized under a name: `Capture(Fringes() => :fr, fit)`. The pair's
key is the product KIND (e.g. `Fringes()`), its value the `Symbol` name; [`apply_step`](@ref) on the
`Capture` returns that product as a `name => product` pair. A `(kind, stage)` pair the stage does not
support fails loud at first use (no `apply_step` method).
"""
struct Capture{K, ST<:Step} <: Step
    product::Pair{K, Symbol}
    step::ST
end

"""
    apply_step(step::Step, solution) -> (solution′, product_or_nothing)

Run one step against `solution`, returning the updated solution and, for a `Capture`, its product as a
`name => product` pair; for any other step (e.g. a bare `FringeSelf`, which does not materialize its
measured table)
`nothing`. A full schedule is this folded over a `Vector{Step}` (see the module header).
"""
apply_step(step::Step, sol) = apply_step(step, sol, nothing)

function apply_step(cap::Capture, sol, ::Nothing)
    kind = first(cap.product)
    name = last(cap.product)
    sol′, product = apply_step(cap.step, sol, kind)
    (sol′, name => product)
end
