
using FringyUI
using Fringy
using IntervalSets
using Printf
import ImGuiTestEngine as te
using ImGuiTestEngine: @register_test, @imcheck

antix(ds, name::Symbol) = @something(findfirst(a -> a.name === name, ds.antennas),
                                     error("antenna $name not found in $(getproperty.(ds.antennas, :name))"))
pair_ix(ds, (a, b)::Tuple{Symbol,Symbol}) = minmax(antix(ds, a), antix(ds, b))

function scan_interval(ds, scan)
    dts = ds.rows.datetime[ds.rows.scan .== scan]
    isempty(dts) && error("scan $scan has no rows (present scans: $(sort(unique(ds.rows.scan))))")
    ClosedInterval(extrema(dts)...)
end

"""
    run_example(; ds, sol0, sched, scan, pair, name, window_size=(1400,900), solve_budget=200000)

Drive the inspector over `(ds, sol0, sched)` and save the 10 example screenshots under
`examples/output/<name>/`. `scan` is the scan number whose interval is selected; `pair` is a
`(:ANT1, :ANT2)` name tuple for the fringe plane. Returns the vector of written PNG paths.
"""
function run_example(; ds, sol0, sched, scan, pair, name,
                     window_size = (1400, 900), solve_budget = 200000)
    exdir = @__DIR__
    cap_auto = joinpath(exdir, "output", "captures")
    cap_dst  = joinpath(exdir, "output", name)
    rm(cap_auto; force = true, recursive = true); mkpath(cap_auto)
    rm(cap_dst;  force = true, recursive = true); mkpath(cap_dst)

    println("nthreads default=", Threads.nthreads(:default), " interactive=", Threads.nthreads(:interactive))
    @assert Threads.nthreads(:interactive) >= 1 "need an interactive thread (launch julia -t auto,1)"

    pairix = pair_ix(ds, pair)
    tsel = scan_interval(ds, scan)
    println("dataset '$name': datums=", length(ds.rows) * length(ds.freq.ν), " nant=", length(ds.antennas),
            " nbl=", length(unique(ds.rows.baseline_ix)), " nscan=", length(unique(ds.rows.scan)),
            "\n  scan $scan interval=", tsel, "  fringe pair=", pair, " → indices ", pairix)

    state = FringyUI._build_state(ds, sol0, sched)
    view = state.view

    engine = te.CreateContext(; exit_on_completion = true, show_test_window = false)
    engine.IO.ConfigWatchdogWarning = 3600f0
    engine.IO.ConfigWatchdogKillTest = 3600f0

    shot_labels = String[]
    nshots = Ref(0)
    results = Dict{Symbol,Bool}(:solve_done => false, :solve_ok => false, :grid => false,
                                :plane => false, :uvamp => false, :uvsnr => false)

    function yield_until(pred; budget = solve_budget, step = 5)
        te.Yield(3)
        spent = 3
        while spent < budget
            pred() && return true
            te.Yield(step); spent += step
        end
        pred()
    end
    shot!(label) = (te.Yield(8); te.CaptureReset(); te.Yield(5); te.CaptureScreenshot(); te.Yield(20);
                    push!(shot_labels, label); nshots[] += 1)

    griddone()  = !FringyUI.isbusy(state.cache.grid)  && state.cache.grid.shown  !== nothing
    planedone() = !FringyUI.isbusy(state.cache.plane) && state.cache.plane.shown !== nothing
    uvdone()    = !FringyUI.isbusy(state.cache.uvprof) && state.cache.uvprof.shown !== nothing

    @register_test(engine, "FringyUI", "example") do
        te.Yield(30); te.WindowFocus("Stage progress"); te.Yield(3)

        srcname = only(unique(ds.rows.source[ds.rows.scan .== scan]))
        view.uvprof.source[] = something(findfirst(==(srcname), ds.sources.name))

        FringyUI._start_solve!(state)
        solved = yield_until(() -> (t = state.solve.task[]; t !== nothing && istaskdone(t)) &&
                                   lock(() -> state.solve.latest[] !== nothing, state.solve.lock))
        results[:solve_done] = solved
        results[:solve_ok] = solved && lock(state.solve.lock) do
            !state.solve.failed[] && length(state.solve.snapshots) == state.solve.nstages[]
        end
        @imcheck solved
        @imcheck results[:solve_ok]
        te.Yield(10); shot!("00_overview")

        view.soln.split[] = false; te.WindowFocus("Solution"); te.Yield(12); shot!("01_solution_overlay")
        view.soln.split[] = true; te.Yield(12); shot!("02_solution_split")
        view.soln.split[] = false

        view.sel[] = tsel

        view.grid.apply.enabled .= false
        gok = yield_until(griddone); results[:grid] = gok; @imcheck gok
        te.WindowFocus("Baseline grid"); te.Yield(12); shot!("03_grid_raw")

        view.grid.apply.enabled .= true
        yield_until(griddone); te.Yield(12); shot!("04_grid_calibrated")

        view.plane.pair[] = pairix
        view.grid.apply.enabled .= false
        pok = yield_until(() -> planedone() && griddone()); results[:plane] = pok; @imcheck pok
        te.WindowFocus("Fringe plane"); te.Yield(12); shot!("05_fringe_plane_raw")

        view.grid.apply.enabled .= true
        yield_until(() -> planedone() && griddone()); te.Yield(12); shot!("06_fringe_plane_calibrated")

        view.uvprof.ymode[] = :amplitude
        te.WindowFocus("UV profile")
        uok = yield_until(uvdone); results[:uvamp] = uok; @imcheck uok
        te.Yield(12); shot!("07_uv_amplitude")

        view.uvprof.ymode[] = :snr
        usok = yield_until(uvdone); results[:uvsnr] = usok; @imcheck usok
        te.Yield(12); shot!("08_uv_snr")

        view.sort_chan[] = true
        te.WindowFocus("Statistics"); te.Yield(14); shot!("09_stats_sorted")

        te.Yield(80)
    end

    threw = false
    cd(exdir) do
        try
            FringyUI._renderloop!(state; window_size, frames = 300000, engine = engine)
        catch e
            threw = true
            @error "render loop threw" exception = (e, catch_backtrace())
        end
    end
    let t = state.solve.task[]; t !== nothing && istaskdone(t) && wait(t) end
    foreach((state.cache.grid, state.cache.model, state.cache.plane, state.cache.uvprof)) do slot
        slot.task === nothing || wait(slot.task)
    end

    produced = isdir(cap_auto) ? sort(filter(f -> endswith(f, ".png"), readdir(cap_auto; join = true))) : String[]
    pngs = map(eachindex(produced)) do i
        lbl = i ≤ length(shot_labels) ? shot_labels[i] : @sprintf("shot_%02d", i - 1)
        dst = joinpath(cap_dst, "$lbl.png"); mv(produced[i], dst; force = true); dst
    end

    summary = te.GetResultSummary(engine)
    println("\n=== ENGINE RESULT SUMMARY ($name) ===")
    println("tests: ", summary.CountTested, " tested, ", summary.CountSuccess, " succeeded")
    println("scenario results: ", sort(collect(results); by = first))
    println("render loop threw: ", threw, "   shots requested: ", nshots[], "   PNGs produced: ", length(pngs))
    println("=== ARTIFACTS (output/$name/) ===")
    for f in pngs
        println(@sprintf("  %8d bytes  %s", filesize(f), basename(f)))
    end

    @assert !threw "render loop must not throw"
    @assert results[:solve_done] "solve must finish"
    @assert results[:solve_ok] "solve must complete successfully with all stage snapshots"
    @assert results[:grid] "baseline grid must show a result for the selected scan"
    @assert results[:plane] "fringe plane must show a result for the selected pair"
    @assert results[:uvamp] "UV amplitude profile must show a result"
    @assert results[:uvsnr] "UV SNR profile must show a result from the captured Fringes"
    @assert all(f -> filesize(f) > 0, pngs) "all PNG screenshots must be non-empty"
    @assert length(pngs) >= 9 "expected >=9 non-empty PNGs (got $(length(pngs)))"

    te.DestroyContext(engine)
    println("\nEXAMPLE '$name' DONE — $(length(pngs)) PNGs in output/$name/")
    pngs
end
