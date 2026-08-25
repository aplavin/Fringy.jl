# Fringy.jl

*Fast & composable radio interferometry (VLBI) calibration in Julia.*

`Fringy` is a Julia toolkit for radio inteferometry calibration. What shapes it:

- **Data stays on disk.** Files are read in place (FITS-IDI and UVFITS supported for now) — no import or conversion step, no whole-dataset load into memory.
- **Extensible composable interface.** Calibration terms and solvers are designed as building blocks that are easy to assemble in the needed structure. Flexible representation of frequency- and time-dependent gains, delays/rates, and more.
- **Batch execution, interactive exploration – both first-class.** Runs from scripts, and ships a native UI (`FringyUI.jl`) designed alongside the library.

> **Status:** alpha / early access. The overall framework is in place and stable in shape; the catalog of terms and solvers in flux, being developed to suit specific usecases.

## What's in the repo

The repo contains two Julia packages designed to work together. The project as a whole is **[Fringy.jl](https://github.com/aplavin/Fringy.jl)**; the core individual package also shares that name.

- **[`Fringy/`](Fringy/)** — the core library. Defines the problem, the math, and the solvers. Currently, `Fringy` also contains FITS-IDI/UVFITS support and some application-specific tools. These parts are intended to be extracted into separate packages.
- **[`FringyUI/`](FringyUI/)** — the interactive calibration inspection UI. Live views of visibilities and calibration terms.

## Quickstart

See scripts in the [`FringyUI/examples/`](FringyUI/examples/) folder. More documentation and examples coming soon.

## Development timeline

Efficient visibility data access:

- since 2020: [`VLBIData.jl`](https://github.com/JuliaAPlavin/VLBIData.jl)+ packages to manipulate uvfits datasets in Julia
- 2025 fall: support for FITSIDI as lazily-queried tables in [`VLBIFiles.jl`](https://github.com/JuliaAPlavin/VLBIFiles.jl)
- 2026 April-May: FITSIDI and UVFITS memory mapping in [`VLBIFiles.jl`](https://github.com/JuliaAPlavin/VLBIFiles.jl)

Earlier analyzis tools:

- 2025: collection of ad-hoc tools for inspecting/analyzing already-calibrated VLBI data: e.g., [VLBInspect.jl](https://github.com/aplavin/VLBInspect.jl), [InterFit.jl](https://github.com/aplavin/InterFit.jl)
- 2025 December: [FringeHunt.jl](https://github.com/aplavin/FringeHunt.jl), baseline-based fringe fitting for VLBI data with individual fringe inspection
- 2026 March: cleaned up version of that fringe fitting code used in a published paper ([full code](https://github.com/aplavin/txs2005-refractive-substructure))

[`Fringy.jl`](https://github.com/aplavin/Fringy.jl), this collection of packages:

- 2026: converging on a modular consistent design, proof-of-concept implementations
- 2026 May: first alpha release
  - contains overall setup + fitting + UIs infrastructure,
  - simple source model support,
  - full support fot fits-idi and uvfits files
- **2026 August: second alpha release**
  - refinements, multi-band solutions, and more
  - astrometry-specific functionality
