

"""
    AxisLink

Shared-axis handle (one per `:time`/`:chan`/`:freq`): every plot links its ImPlot axis to it ⇒ shared
cross-panel pan/zoom + one crosshair. The crosshair is a 1-frame-lag latch — detect and draw are
decoupled across frames so it shows on every linked plot regardless of draw order.
"""
struct AxisLink
    min::Base.RefValue{Cdouble}
    max::Base.RefValue{Cdouble}
    cursor::Base.RefValue{Cdouble}
    active::Base.RefValue{Bool}
    _hovered::Base.RefValue{Bool}
    extent::ClosedInterval{Float64}
end
AxisLink(extent::ClosedInterval) =
    AxisLink(Ref(Cdouble(leftendpoint(extent))), Ref(Cdouble(rightendpoint(extent))),
             Ref(Cdouble(NaN)), Ref(false), Ref(false), extent)

function setup_linked_axis!(ax, link::AxisLink)
    ImPlot.SetupAxisLinks(ax, link.min, link.max)
    ImPlot.SetupAxisLimitsConstraints(ax, leftendpoint(link.extent), rightendpoint(link.extent))
end

struct AxisRange
    min::Base.RefValue{Cdouble}
    max::Base.RefValue{Cdouble}
end
AxisRange() = AxisRange(Ref(Cdouble(0)), Ref(Cdouble(1)))

setup_axis_range!(ax, range::AxisRange) = ImPlot.SetupAxisLinks(ax, range.min, range.max)

const _CROSSHAIR_COL = ig.ImVec4(0.345f0, 0.431f0, 0.459f0, 0.7f0)

function axis_crosshair!(link::AxisLink)
    (link.active[] && !isnan(link.cursor[])) &&
        ImPlot.PlotInfLines("##cur", link.cursor, 1, ImPlot.ImPlotSpec(LineColor = _CROSSHAIR_COL, LineWeight = 1f0))
    if ImPlot.IsPlotHovered()
        link.cursor[] = ImPlot.GetPlotMousePos().x
        link._hovered[] = true
    end
end

function _advance_crosshair!(link::AxisLink)
    link.active[] = link._hovered[]
    link._hovered[] = false
end


_unwrap_fringefit(st::FringeFit)      = st
_unwrap_fringefit(st::FringeSelf) = st.fit
_unwrap_fringefit(st::Capture)         = _unwrap_fringefit(st.step)
_unwrap_fringefit(::Step)              = nothing

function _plane_defaults(schedule)
    for st in schedule
        fi = _unwrap_fringefit(st)
        fi === nothing || return (; fi.tiles, fi.window, fi.oversample)
    end
    (; tiles = nothing,
       window = (; rate = (-5.0u"mHz")..(5.0u"mHz"), delay = (-64.0u"ns")..(64.0u"ns")),
       oversample = (; rate = 2.0, delay = 8.0))
end

function _fringes_name(schedule)
    for st in schedule
        st isa Capture && first(st.product) isa Fringes && return last(st.product)
    end
    nothing
end


"""
    _build_state(ds, sol, schedule) -> (; ctx, solve, view, cache)

Build the full state. Mutability only in Ref/Atomic/Lock/StructArray leaves. `ctx` is immutable, built
once (pure functions of `ds`, `sol`, `schedule`); `solve` is the only cross-thread state (accessed under
`lock`); `view` holds render-only selections (ImGui binds the Refs directly); `cache` holds
render-managed derived results. `sol === nothing` ⇒ data-viewer only. Partitions are built on the ROOT
`ds`; subsets are passed only as data.
"""
function _build_state(ds::Dataset, sol::Union{Nothing,Solution}, schedule::AbstractVector)
    components = sol === nothing ? () : jones_terms(sol)
    solvable = sol !== nothing && !isempty(components) && !isempty(schedule)
    chan = chan_info(ds)
    extents = (; time = ClosedInterval(extrema(datetime2unix.(ds.rows.datetime))...),
                 chan = ClosedInterval(0.5, chan.n + 0.5),
                 freq = ClosedInterval(extrema(ds.freq.ν)...))
    uv = uv_points(ds)
    ctx0 = (; ds, sol0 = sol, components, solvable,
              schedule,
              antennas = sort(ds.antennas; by = a -> a.name),
              antenna_ix = Dict(a => i for (i, a) in enumerate(ds.antennas)),
              baselines = unique(ds.rows.baseline_ix),
              nif = ncells(partition(ds, ByIF())),
              chan, extents,
              uv, uv_R = isempty(uv) ? 1.0 : max(maximum(abs, uv.u), maximum(abs, uv.v)),
              lonlat = antenna_lonlat(ds),
              occupancy = sol === nothing ? Dict{Type,BitArray{4}}() : occupancy(sol),
              plane_defaults = _plane_defaults(schedule),
              fringes_name = _fringes_name(schedule))
    ctx = (; ctx0..., timeline = (; lines = timeline_lines(ctx0), presence = presence(ds)))

    solve = (; lock = ReentrantLock(),
               latest = Ref{Any}(nothing),
               snapshots = StructArray((stage_ix = Int[], stage_name = Symbol[], sol = Any[], wall = Float64[])),
               products = Ref{Any}(Dict{Symbol,Any}()),
               nstages = Ref(0),
               failed = Ref(false),
               log = Tuple{Logging.LogLevel,String}[],
               stop = Threads.Atomic{Bool}(false),
               task = Ref{Any}(nothing))

    rep = first(ds.rows.baseline_ix).antennas
    slots = StructArray([(; ij = (i, j), label = correlation_product_label(ds, rep[1], rep[2], i, j), enabled = true)
                         for j in 1:2 for i in 1:2])
    applysa = StructArray((kind = collect(Type, components),
                           label = String[_kind_label(K) for K in components],
                           enabled = fill(false, length(components))))
    pd = ctx.plane_defaults
    view = (; version = Ref{Union{Symbol,Int}}(:current), refresh = Ref(0), sort_chan = Ref(false),
              sel = Ref{Union{Nothing,ClosedInterval{DateTime}}}(nothing),
              axes = (; time = AxisLink(extents.time), chan = AxisLink(extents.chan), freq = AxisLink(extents.freq)),
              grid = (; full_matrix = Ref(false), only_active = Ref(true), show_model = Ref(false),
                        marker = Ref(1.5f0), slots, apply = applysa),
              soln = (; split = Ref(false), fit_sig = Ref{Any}(nothing),
                        yranges = Dict(K => AxisRange() for K in components)),
              plane = (; pair = Ref{Union{Nothing,Tuple{Int,Int}}}(nothing), if_sel = Ref(0),
                         rate_os = Ref(Float32(pd.oversample.rate)), delay_os = Ref(Float32(pd.oversample.delay)),
                         rate_win = Ref(1f0), delay_win = Ref(1f0), fit_planes = Ref{Any}(nothing)),
              uvprof = (; source = Ref(1), ymode = Ref(:amplitude), time_mode = Ref{UVTimeMode}(UVScan()),
                          freq_mode = Ref{UVFreqMode}(UVChannel()), custom_time = Ref(60f0), fit_sig = Ref{Any}(nothing)))

    cache = (; grid = DeriveSlot{Any}(), model = DeriveSlot{Any}(),
               plane = DeriveSlot{Any}(), plane_marks = Ref{Any}(nothing),
               uvprof = DeriveSlot{Any}(),
               decim = (; tl_src = Ref{Any}(nothing), tl_ant = Ref{Any}(nothing),
                          uv = Ref{Any}(nothing), uvc = Ref{Any}(nothing),
                          uv_sel = Ref{Any}(nothing), uv_selc = Ref{Any}(nothing)))

    (; ctx, solve, view, cache)
end


function _publish_running!(solve, k, name)
    lock(solve.lock) do
        prev = solve.latest[]
        solve.latest[] = (; stage_ix = k, stage_name = name, sol = prev === nothing ? nothing : prev.sol,
                            wall = 0.0, running = true)
    end
end

function _publish_done!(solve, k, st, snext, product, wall)
    lock(solve.lock) do
        push!(solve.snapshots, (; stage_ix = k, stage_name = _stage_name(st), sol = snext, wall))
        if product !== nothing
            name, val = product
            d = solve.products[]
            haskey(d, name) ? push!(d[name], val) : (d[name] = Any[val])
        end
        solve.latest[] = (; stage_ix = k, stage_name = _stage_name(st), sol = snext, wall, running = false)
    end
end

"""
    _start_solve!(state)

Spawn the background solve on the `:default` pool: fold `apply_step` over `ctx.schedule` from
`ctx.sol0`, publishing per-stage snapshots + products and checking the cooperative `stop` between stages.
Runs under `_UILogger` so `@warn`/`@error` reach the log panel (and the global logger). Any exception is
caught, flagged (`solve.failed[] = true`, under lock) and re-emitted as one `@error` (no rethrow, no
teardown; the task itself never fails). Never double-spawns.
"""
function _start_solve!(state)
    ctx, solve = state.ctx, state.solve
    (!ctx.solvable || solve.task[] !== nothing) && return state
    lock(solve.lock) do; solve.nstages[] = length(ctx.schedule) end
    solve.task[] = Threads.@spawn with_logger(_UILogger(solve)) do
        try
            foldl(enumerate(ctx.schedule); init = ctx.sol0) do s, (k, st)
                solve.stop[] && return s
                _publish_running!(solve, k, _stage_name(st))
                t0 = time()
                snext, product = Fringy.apply_step(st, s)
                _publish_done!(solve, k, st, snext, product, time() - t0)
                snext
            end
        catch e
            lock(solve.lock) do; solve.failed[] = true end
            @error "FringyUI solve failed" exception = (e, catch_backtrace())
        end
    end
    state
end

"""
    _UILogger

Lightweight capture logger the solve task runs under: routes `@warn`/`@error` (Warn-level and up) into
`solve.log` under lock and fans everything to the global logger too. No state stream — the fold publishes
snapshots directly.
"""
struct _UILogger{S} <: Logging.AbstractLogger
    solve::S
end
Logging.min_enabled_level(::_UILogger) = Logging.Warn
Logging.shouldlog(::_UILogger, _...) = true
Logging.catch_exceptions(::_UILogger) = false
function Logging.handle_message(lg::_UILogger, level, message, _mod, grp, id, file, line; kwargs...)
    msg = haskey(kwargs, :exception) ? sprint(showerror, first(kwargs[:exception])) : string(message)
    lock(lg.solve.lock) do; push!(lg.solve.log, (level, msg)) end
    Logging.handle_message(Base.global_logger(), level, message, _mod, grp, id, file, line; kwargs...)
end

"""
    _active_solution(state) -> Solution

The Solution for `view.version`: `:current` ⇒ the latest published stage's solution (or `ctx.sol0` before
any), an integer `k` ⇒ the k-th completed snapshot. Read under the solve lock.
"""
function _active_solution(state)
    ctx, solve, view = state.ctx, state.solve, state.view
    lock(solve.lock) do
        v = view.version[]
        if v isa Integer
            1 ≤ v ≤ length(solve.snapshots) ? solve.snapshots.sol[v] : ctx.sol0
        else
            lat = solve.latest[]
            (lat === nothing || lat.sol === nothing) ? ctx.sol0 : lat.sol
        end
    end
end

function _captured_fringes(state)
    name = state.ctx.fringes_name
    name === nothing && return nothing
    lock(state.solve.lock) do
        vec = get(state.solve.products[], name, nothing)
        (vec === nothing || isempty(vec)) ? nothing : reduce(vcat, map(t -> t.rows, vec))
    end
end

stop!(state) = Threads.atomic_or!(state.solve.stop, true)

const _IMWCHAR = Cushort
const _GLYPH_RANGES = _IMWCHAR[
    0x0020, 0x00FF,
    0x0370, 0x03FF,
    0x2080, 0x209F,
    0x2190, 0x21FF,
    0x27F0, 0x27FF,
    0x0000,
]
const _FONT_PATH = normpath(joinpath(@__DIR__, "..", "assets", "JuliaMono-Regular.ttf"))

function _load_glyph_font!(io)
    isfile(_FONT_PATH) || error("FringyUI: bundled font missing at $_FONT_PATH")
    atlasptr = Ptr{lib.ImFontAtlas}(unsafe_load(Ptr{Ptr{Cvoid}}(io.Fonts)))
    ranges = Base.unsafe_convert(Ptr{_IMWCHAR}, _GLYPH_RANGES)
    font = lib.ImFontAtlas_AddFontFromFileTTF(atlasptr, _FONT_PATH, 16f0, C_NULL, ranges)
    font == C_NULL && error("FringyUI: failed to load bundled font $_FONT_PATH")
    return font
end

function _screen_capture(viewport_id::Cuint, x::Cint, y::Cint, w::Cint, h::Cint,
                         pixels::Ptr{Cuint}, user_data::Ptr{Cvoid})::Bool
    io = ig.GetIO()
    disp_h = Cint(round(unsafe_load(io.DisplaySize).y))
    y2 = disp_h - (y + h)
    GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, Cint(1))
    GL.glReadPixels(x, y2, w, h, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, Ptr{Cvoid}(pixels))
    stride = Int(w)
    a = 0; b = (Int(h) - 1) * stride
    tmp = Vector{Cuint}(undef, stride)
    while a < b
        unsafe_copyto!(pointer(tmp),     pixels + a * sizeof(Cuint), stride)
        unsafe_copyto!(pixels + a * sizeof(Cuint), pixels + b * sizeof(Cuint), stride)
        unsafe_copyto!(pixels + b * sizeof(Cuint), pointer(tmp),     stride)
        a += stride; b -= stride
    end
    return true
end
const _SCREEN_CAPTURE_C = Ref{Ptr{Cvoid}}(C_NULL)

"""
    _renderloop!(state; window_size, frames, on_frame, engine)

Manual bounded render loop. `frames`/`on_frame`/`engine` are internal verification + test-engine hooks
(not on the public `inspect` signature). Draw exceptions propagate.
"""
function _renderloop!(state; window_size = (1600, 1000),
                      frames = nothing, on_frame = nothing, engine = nothing)
    w, h = window_size
    GLFW.Init()
    ig.set_backend(:GlfwOpenGL3)
    ctx  = ig.CreateContext()
    pctx = ImPlot.CreateContext()
    ImPlot.SetImGuiContext(ctx)
    io = ig.GetIO()
    io.ConfigFlags = unsafe_load(io.ConfigFlags) | ig.ImGuiConfigFlags_DockingEnable
    if engine !== nothing
        io.ConfigErrorRecoveryEnableTooltip = false
        io.ConfigErrorRecoveryEnableAssert = false
        io.ConfigDebugHighlightIdConflicts = false
    end

    GLFW.WindowHint(GLFW.CONTEXT_VERSION_MAJOR, 3)
    GLFW.WindowHint(GLFW.CONTEXT_VERSION_MINOR, 2)
    if Sys.isapple()
        GLFW.WindowHint(GLFW.OPENGL_PROFILE, GLFW.OPENGL_CORE_PROFILE)
        GLFW.WindowHint(GLFW.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
    end
    engine === nothing || GLFW.WindowHint(GLFW.SCALE_FRAMEBUFFER, false)
    window = GLFW.CreateWindow(w, h, "FringyUI")
    GLFW.MakeContextCurrent(window)
    GLFW.SwapInterval(1)
    lib.ImGui_ImplGlfw_InitForOpenGL(Ptr{lib.GLFWwindow}(window.handle), true)
    _load_glyph_font!(io)
    lib.ImGui_ImplOpenGL3_Init("#version 150")
    if engine !== nothing
        engine.IO.ScreenCaptureFunc = _SCREEN_CAPTURE_C[]
        ig._start_test_engine(engine, ctx)
    end

    ImGuiThemes.apply_theme!("Solarized Light")

    try
        frame = 0
        while !GLFW.WindowShouldClose(window) && (frames === nothing || frame < frames)
            frame += 1
            GLFW.PollEvents()
            lib.ImGui_ImplOpenGL3_NewFrame()
            lib.ImGui_ImplGlfw_NewFrame()
            ig.NewFrame()
            ig.DockSpaceOverViewport()
            draw(state)
            (engine !== nothing && engine.show_test_window) && ig._show_test_window(engine)
            on_frame === nothing || on_frame(frame)
            ig.Render()
            dw, dh = GLFW.GetFramebufferSize(window)
            GL.glViewport(0, 0, dw, dh)
            GL.glClearColor(0.2f0, 0.2f0, 0.2f0, 1f0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            lib.ImGui_ImplOpenGL3_RenderDrawData(Ptr{Cint}(ig.GetDrawData()))
            GLFW.SwapBuffers(window)
            engine === nothing || ig._post_swap(engine)
            if engine !== nothing && engine.exit_on_completion && !ig._test_engine_is_running(engine)
                GLFW.SetWindowShouldClose(window, true)
            end
            GC.safepoint()
        end
    finally
        lib.ImGui_ImplOpenGL3_Shutdown()
        lib.ImGui_ImplGlfw_Shutdown()
        ImPlot.DestroyContext(pctx)
        ig.DestroyContext(ctx)
        GLFW.DestroyWindow(window)
        GLFW.Terminate()
    end
end

"""
    _smoke_window(; frames=60, window_size=(800,600))

Open an empty docked window for `frames` frames and exit cleanly — the windowed-environment gate.
Returns `true` on success, `false` (with a warning) if GLFW/GL init fails (e.g. headless).
"""
function _smoke_window(; frames = 60, window_size = (800, 600))
    try
        _renderloop!(nothing; window_size, frames)
        return true
    catch e
        @warn "FringyUI._smoke_window failed (headless environment?)" exception = (e, catch_backtrace())
        return false
    end
end

function _axis_fmt(value::Cdouble, buf::Ptr{Cchar}, size::Cint, data::Ptr{Cvoid})::Cint
    scale = unsafe_load(Ptr{Cdouble}(data))
    n = @ccall snprintf(buf::Ptr{Cchar}, size::Csize_t, "%.3g"::Cstring; (value * scale)::Cdouble)::Cint
    return n < 0 ? Cint(0) : min(n, size - one(Cint))
end
const _AXIS_FMT_C = Ref{Ptr{Cvoid}}(C_NULL)
const _FMT_SCALES = Dict{Float64,Base.RefValue{Cdouble}}()

function _setup_unit_axis(axis, scale::Float64)
    ref = get!(() -> Ref(Cdouble(scale)), _FMT_SCALES, scale)
    ImPlot.SetupAxisFormat(axis, _AXIS_FMT_C[], Base.unsafe_convert(Ptr{Cvoid}, ref))
end

const _SNR_SYMLOG = ImPlotExtra.SymLog(20.0)

const _PALETTE = NTuple{3,Float32}[
    (0.122f0, 0.467f0, 0.706f0), (1.000f0, 0.498f0, 0.055f0), (0.173f0, 0.627f0, 0.173f0),
    (0.839f0, 0.153f0, 0.157f0), (0.580f0, 0.404f0, 0.741f0), (0.549f0, 0.337f0, 0.294f0),
    (0.890f0, 0.467f0, 0.761f0), (0.498f0, 0.498f0, 0.498f0), (0.737f0, 0.741f0, 0.133f0),
    (0.090f0, 0.745f0, 0.812f0)]
_palcolor(k) = (rgb = _PALETTE[mod1(k, length(_PALETTE))]; ig.ImVec4(rgb[1], rgb[2], rgb[3], 1f0))
_antcolor(a) = _palcolor(a)
_reccolor(r) = _palcolor(r == 1 ? 1 : 4)
_recmarker(r) = r == 1 ? ImPlot.ImPlotMarker_Circle : ImPlot.ImPlotMarker_Cross
_slotcolor(k) = _palcolor(k)

const _DOT_SPEC = ImPlot.ImPlotSpec(Marker = ImPlot.ImPlotMarker_Square,
                                    MarkerLineColor = ig.ImVec4(0f0, 0f0, 0f0, 0f0))

"""
    timeline_layout(ctx)

Row table `Vector{(; y::Int, kind::Symbol, label::String, idx::Int)}`. `kind` ∈ (:line, :src, :ant);
`idx` indexes back into `timeline.lines` / `presence.sources.names` / `presence.antennas.names`.
"""
function timeline_layout(ctx)
    tl = ctx.timeline
    entries = [map(enumerate(tl.lines.label))             do (i, lab); (; kind = :line, label = lab, idx = i) end;
               map(enumerate(tl.presence.sources.names))  do (i, nm);  (; kind = :src,  label = nm,  idx = i) end;
               map(enumerate(tl.presence.antennas.names)) do (i, nm);  (; kind = :ant,  label = nm,  idx = i) end]
    [(; y, e...) for (y, e) in enumerate(entries)]
end

"""
    timeline_hit(ctx, layout, mx, my)

`(; lineidx, cellidx, lo, hi)` or `nothing`. Find the SELECTION line whose row-y is within 0.4 of `my`,
then the cell whose `[lo,hi]` contains `mx`. PURE.
"""
function timeline_hit(ctx, layout, mx, my)
    for r in layout
        r.kind === :line || continue
        abs(my - r.y) ≤ 0.4 || continue
        for (c, iv) in enumerate(ctx.timeline.lines.cells[r.idx])
            lo = leftendpoint(iv); hi = rightendpoint(iv)
            lo ≤ mx ≤ hi && return (; lineidx = r.idx, cellidx = c, lo, hi)
        end
        return nothing
    end
    nothing
end

function _cell_label(txt; voff = 0f0)
    ig.SetCursorPosX(ig.GetCursorPosX() + max(0f0, (ig.GetContentRegionAvail().x - ig.CalcTextSize(txt).x) / 2))
    voff > 0 && ig.SetCursorPosY(ig.GetCursorPosY() + voff)
    ig.TextUnformatted(txt)
end

function draw(state)
    _advance_crosshair!(state.view.axes.time)
    _advance_crosshair!(state.view.axes.chan)
    _stage_progress(state)
    _solution(state)
    _grid(state)
    _timeline(state)
    _statistics(state)
    _uv_profile(state)
    _plane(state)
    nothing
end

function draw(::Nothing)
    if ig.Begin("FringyUI (smoke)")
        ig.TextUnformatted("empty smoke window — render infrastructure OK")
    end
    ig.End()
    nothing
end

const _SOLN_CELLFLAGS = ImPlot.ImPlotFlags_NoLegend | ImPlot.ImPlotFlags_NoTitle | ImPlot.ImPlotFlags_NoMenus

function _setup_soln_axes(spec, xaxis, view; yfit = false, ylims = nothing, yrange = nothing)
    xflags = xaxis === :time ? ImPlot.ImPlotAxisFlags_NoTickLabels : ImPlot.ImPlotAxisFlags_None
    ImPlot.SetupAxis(ImPlot.ImAxis_X1, C_NULL, xflags)
    if xaxis === :time
        ImPlot.SetupAxis(ImPlot.ImAxis_Y1, C_NULL)
        setup_axis_range!(ImPlot.ImAxis_Y1, yrange)
    elseif spec.phase
        ImPlot.SetupAxis(ImPlot.ImAxis_Y1, C_NULL)
        ImPlot.SetupAxisLimits(ImPlot.ImAxis_Y1, -π, π, ImPlot.ImPlotCond_Always)
    elseif ylims !== nothing
        ImPlot.SetupAxis(ImPlot.ImAxis_Y1, C_NULL)
        ImPlot.SetupAxisLimits(ImPlot.ImAxis_Y1, ylims[1], ylims[2], ImPlot.ImPlotCond_Always)
    else
        ImPlot.SetupAxis(ImPlot.ImAxis_Y1, C_NULL, yfit ? ImPlot.ImPlotAxisFlags_AutoFit : ImPlot.ImPlotAxisFlags_None)
    end
    if xaxis === :time
        setup_linked_axis!(ImPlot.ImAxis_X1, view.axes.time)
    else
        setup_linked_axis!(ImPlot.ImAxis_X1, view.axes.freq)
        _setup_unit_axis(ImPlot.ImAxis_X1, 1e-9)
    end
    _setup_unit_axis(ImPlot.ImAxis_Y1, spec.scale)
end

function _plot_trace!(lab, xs, ys, col, marker)
    ImPlot.PlotLine(lab, xs, ys; spec = ImPlot.ImPlotSpec(LineColor = col, LineWeight = 1f0))
    ImPlot.PlotScatter(lab, xs, ys; spec = ImPlot.ImPlotSpec(Marker = marker, MarkerSize = 3f0, LineColor = col, FillColor = col))
end

_trace_suffix(::_SolnTrace) = ""
_trace_suffix(tr::_TimeSolnTrace) = "/f$(tr.frequency_cell)"

function _solution_ylims(spec, traces)
    spec.phase && return (-π, π)
    ys = filter(isfinite, reduce(vcat, (tr.ys for tr in traces); init = Float64[]))
    isempty(ys) && return nothing
    lo, hi = extrema(ys)
    pad = hi > lo ? 0.05 * (hi - lo) : max(abs(hi), 1.0) * 0.05
    (lo - pad, hi + pad)
end

function _fit_solution_axis!(range, spec, traces)
    limits = _solution_ylims(spec, traces)
    limits === nothing && return nothing
    range.min[], range.max[] = limits
    nothing
end

function _solution_overlay(state, kinds, do_fit)
    ds = state.ctx.ds; view = state.view; occ = state.ctx.occupancy; sol = _active_solution(state)
    nt = length(kinds); specs = _soln_spec.(kinds)
    avail = ig.GetContentRegionAvail()
    capw = maximum(s -> ig.CalcTextSize("$(s.label) ($(s.unit))").x, specs) + 6f0
    ph = avail.y / nt
    vmid = max(0f0, (ph - ig.GetTextLineHeight()) / 2)
    ig.PushStyleVar(ig.ImGuiStyleVar_CellPadding, ig.ImVec2(0, 0))
    if ig.BeginTable("##soln_overlay", 2, ig.ImGuiTableFlags_NoSavedSettings, avail)
        ig.TableSetupColumn("##cap", ig.ImGuiTableColumnFlags_WidthFixed, capw)
        ig.TableSetupColumn("##plot", ig.ImGuiTableColumnFlags_WidthStretch)
        for (ti, K) in enumerate(kinds)
            spec = specs[ti]; st = sol[K]; xaxis = varying_axis(st)
            traces = soln_traces(st, occ[K]; xaxis)
            xaxis === :time && do_fit && _fit_solution_axis!(view.soln.yranges[K], spec, traces)
            xoff = xaxis === :time ? datetime2unix(ds.time0) : 0.0
            ig.TableNextRow()
            ig.TableSetColumnIndex(0); _cell_label("$(spec.label) ($(spec.unit))"; voff = vmid)
            ig.TableSetColumnIndex(1)
            ig.PushID(ti)
            if ImPlot.BeginPlot("##o$ti", ig.ImVec2(-1, ph), _SOLN_CELLFLAGS)
                _setup_soln_axes(spec, xaxis, view; yfit = do_fit, yrange = view.soln.yranges[K])
                for tr in traces
                    label = "$(ds.antennas[tr.antenna].name)" *
                            receptor_label(ds, tr.antenna, tr.receptor_slot) * _trace_suffix(tr)
                    _plot_trace!(label,
                                 tr.xs .+ xoff, tr.ys, _antcolor(tr.antenna), _recmarker(tr.receptor_slot))
                end
                xaxis === :time && axis_crosshair!(view.axes.time)
                ImPlot.EndPlot()
            end
            ig.PopID()
        end
        ig.EndTable()
    end
    ig.PopStyleVar()
end

function _solution_split(state, kinds, do_fit)
    ds = state.ctx.ds; view = state.view; occ = state.ctx.occupancy; sol = _active_solution(state)
    ants = 1:nantennas(ds)
    na = length(ants); nt = length(kinds); specs = _soln_spec.(kinds)
    xaxes = map(K -> varying_axis(sol[K]), kinds)
    traces = map(K -> soln_traces(sol[K], occ[K]; xaxis = varying_axis(sol[K])), kinds)
    ylims = _solution_ylims.(specs, traces)
    foreach(eachindex(kinds)) do ti
        xaxes[ti] === :time && do_fit && _fit_solution_axis!(view.soln.yranges[kinds[ti]], specs[ti], traces[ti])
    end
    avail = ig.GetContentRegionAvail()
    lh = ig.GetTextLineHeightWithSpacing()
    capw = maximum(a -> ig.CalcTextSize(String(ds.antennas[a].name)).x, ants) + 6f0
    ph = (avail.y - lh) / na
    vmid = max(0f0, (ph - ig.GetTextLineHeight()) / 2)
    ig.PushStyleVar(ig.ImGuiStyleVar_CellPadding, ig.ImVec2(0, 0))
    ImPlot.PushStyleVar(ImPlot.ImPlotStyleVar_PlotPadding, ig.ImVec2(10, 2))
    if ig.BeginTable("##soln_split", nt + 1, ig.ImGuiTableFlags_NoSavedSettings, avail)
        ig.TableSetupColumn("##cap", ig.ImGuiTableColumnFlags_WidthFixed, capw)
        foreach(ti -> ig.TableSetupColumn("##c$ti", ig.ImGuiTableColumnFlags_WidthStretch), 1:nt)
        ig.TableNextRow()
        foreach(ti -> (ig.TableSetColumnIndex(ti); _cell_label("$(specs[ti].label) ($(specs[ti].unit))")), 1:nt)
        for (ri, a) in enumerate(ants)
            ig.TableNextRow()
            ig.TableSetColumnIndex(0); _cell_label(String(ds.antennas[a].name); voff = vmid)
            for ti in 1:nt
                spec = specs[ti]; xaxis = xaxes[ti]
                xoff = xaxis === :time ? datetime2unix(ds.time0) : 0.0
                ig.TableSetColumnIndex(ti)
                ig.PushID((ri - 1) * nt + ti)
                if ImPlot.BeginPlot("##s$(ri)_$(ti)", ig.ImVec2(-1, ph), _SOLN_CELLFLAGS)
                    _setup_soln_axes(spec, xaxis, view; ylims = ylims[ti], yrange = view.soln.yranges[kinds[ti]])
                    for tr in traces[ti]
                        tr.antenna == a || continue
                        label = receptor_label(ds, tr.antenna, tr.receptor_slot) * _trace_suffix(tr)
                        _plot_trace!(label, tr.xs .+ xoff, tr.ys,
                                     _reccolor(tr.receptor_slot), _recmarker(tr.receptor_slot))
                    end
                    xaxis === :time && axis_crosshair!(view.axes.time)
                    ImPlot.EndPlot()
                end
                ig.PopID()
            end
        end
        ig.EndTable()
    end
    ImPlot.PopStyleVar()
    ig.PopStyleVar()
end

function _solution(state)
    ctx, view = state.ctx, state.view
    if ig.Begin("Solution")
        if ctx.sol0 === nothing || isempty(ctx.components)
            ig.TextUnformatted(ctx.sol0 === nothing ? "no solution loaded" : "solution has no components")
        else
            ds = ctx.ds
            ig.Checkbox("split", view.soln.split)
            split = view.soln.split[]
            for a in 1:nantennas(ds)
                ig.SameLine()
                split ? ig.TextUnformatted(String(ds.antennas[a].name)) :
                        ig.TextColored(_antcolor(a), String(ds.antennas[a].name))
            end
            for r in 1:2
                ig.SameLine()
                split ? ig.TextColored(_reccolor(r), receptor_label(ds, r)) :
                        ig.TextUnformatted(receptor_label(ds, r))
            end
            cur = (view.version[], view.refresh[])
            do_fit = cur != view.soln.fit_sig[]
            view.soln.fit_sig[] = cur
            kinds = collect(ctx.components)
            split ? _solution_split(state, kinds, do_fit) : _solution_overlay(state, kinds, do_fit)
        end
    end
    ig.End()
end

const _TL_HIGHLIGHT = ig.ImVec4(0.7961f0, 0.2941f0, 0.0863f0, 1f0)
const _TL_LINECOL   = ig.ImVec4(0.149f0, 0.5451f0, 0.8235f0, 1f0)
const _TL_SRCCOL    = ig.ImVec4(0.396f0, 0.482f0, 0.514f0, 1f0)
const _TL_ANTCOL    = ig.ImVec4(0.5216f0, 0.6f0, 0.0f0, 1f0)

function _timeline(state)
    ctx, view = state.ctx, state.view
    if ig.Begin("Timeline")
        layout = timeline_layout(ctx)
        nrow = length(layout)
        if nrow == 0
            ig.Text("(no timeline data)")
        else
            labels = [r.label for r in layout]
            if ImPlot.BeginPlot("##timeline", ig.ImVec2(-1, -1),
                                ImPlot.ImPlotFlags_NoLegend | ImPlot.ImPlotFlags_NoMenus | ImPlot.ImPlotFlags_NoMouseText)
                ImPlot.SetupAxes(C_NULL, C_NULL, ImPlot.ImPlotAxisFlags_None, ImPlot.ImPlotAxisFlags_NoLabel)
                ImPlot.SetupAxisScale(ImPlot.ImAxis_X1, ImPlot.ImPlotScale_Time)
                setup_linked_axis!(ImPlot.ImAxis_X1, view.axes.time)
                ImPlot.SetupAxisLimits(ImPlot.ImAxis_Y1, 0.5, nrow + 0.5, ImPlot.ImPlotCond_Always)
                ImPlot.SetupAxisTicks(ImPlot.ImAxis_Y1, Cdouble.(1:nrow), nrow, labels, false)

                hovered = nothing
                if ImPlot.IsPlotHovered()
                    mp = ImPlot.GetPlotMousePos()
                    hovered = timeline_hit(ctx, layout, mp.x, mp.y)
                    if hovered !== nothing && ig.IsMouseClicked(0)
                        view.sel[] = ClosedInterval(unix2datetime(hovered.lo), unix2datetime(hovered.hi))
                    end
                end
                sel = view.sel[]
                selu = sel === nothing ? nothing : (datetime2unix(leftendpoint(sel)), datetime2unix(rightendpoint(sel)))

                if selu !== nothing
                    ImPlot.PlotInfLines("##selband", Cdouble[selu[1], selu[2]];
                        spec = ImPlot.ImPlotSpec(LineColor = _TL_HIGHLIGHT, LineWeight = 1.5f0))
                end

                for r in layout
                    r.kind === :line || continue
                    y = Cdouble(r.y)
                    for (c, iv) in enumerate(ctx.timeline.lines.cells[r.idx])
                        lo = leftendpoint(iv); hi = rightendpoint(iv)
                        is_hov = hovered !== nothing && hovered.lineidx == r.idx && hovered.cellidx == c
                        is_sel = selu !== nothing && lo ≥ selu[1] - 1e-6 && hi ≤ selu[2] + 1e-6
                        col = is_sel ? _TL_HIGHLIGHT : _TL_LINECOL
                        lw  = is_hov ? 5f0 : (is_sel ? 3.5f0 : 1.5f0)
                        spec = is_hov ?
                            ImPlot.ImPlotSpec(LineColor = col, LineWeight = lw,
                                              Marker = ImPlot.ImPlotMarker_Circle, MarkerSize = 5f0, FillColor = col) :
                            ImPlot.ImPlotSpec(LineColor = col, LineWeight = lw)
                        ImPlot.PlotLine("##l$(r.idx)c$c", Cdouble[lo, hi], Cdouble[y, y]; spec)
                    end
                end

                pr = ctx.timeline.presence
                srcrows = filter(r -> r.kind === :src, layout)
                antrows = filter(r -> r.kind === :ant, layout)
                lim = ImPlot.GetPlotLimits()
                iv = ClosedInterval(lim.X.Min, lim.X.Max)
                nbinx = round(Int, ImPlot.GetPlotSize().x)
                dc = state.cache.decim
                if !isempty(srcrows)
                    region = (iv, ClosedInterval(0.5, length(srcrows) + 0.5)); nbins = (nbinx, length(srcrows))
                    ks = cached!(dc.tl_src, (region, nbins)) do
                        pr.sources.pts[grid_decimate(pr.sources.pts, region, nbins)]
                    end
                    ImPlot.PlotScatter("##src", ks.t, [Cdouble(srcrows[r].y) for r in ks.row];
                        spec = setproperties(_DOT_SPEC; MarkerSize = 2.5f0, MarkerFillColor = _TL_SRCCOL))
                end
                if !isempty(antrows)
                    region = (iv, ClosedInterval(0.5, length(antrows) + 0.5)); nbins = (nbinx, length(antrows))
                    ka = cached!(dc.tl_ant, (region, nbins)) do
                        pr.antennas.pts[grid_decimate(pr.antennas.pts, region, nbins)]
                    end
                    ImPlot.PlotScatter("##ant", ka.t, [Cdouble(antrows[r].y) for r in ka.row];
                        spec = setproperties(_DOT_SPEC; MarkerSize = 2.5f0, MarkerFillColor = _TL_ANTCOL))
                end

                axis_crosshair!(view.axes.time)
                ImPlot.EndPlot()
            end
        end
    end
    ig.End()
end

const _ST_UVGREY = ig.ImVec4(0.396f0, 0.482f0, 0.514f0, 0.55f0)
const _ST_UVSEL  = ig.ImVec4(0.7961f0, 0.2941f0, 0.0863f0, 1f0)

_chan_display(ctx, view) = view.sort_chan[] ?
    (; ν = ctx.chan.ν_sorted, perm = ctx.chan.perm, if_bounds = ctx.chan.if_bounds_sorted) :
    (; ν = ctx.chan.ν,        perm = 1:ctx.chan.n,  if_bounds = ctx.chan.if_bounds)

function _statistics(state)
    ctx, view = state.ctx, state.view
    if ig.Begin("Statistics")
        ig.Checkbox("sort by frequency", view.sort_chan)
        if ImPlot.BeginSubplots("##stats", 2, 2, ig.ImVec2(-1, -1))
            _stat_uv(ctx, view, state.cache.decim)
            _stat_map(ctx)
            _stat_chanfreq(state)
            if ImPlot.BeginPlot("##stat_blank")
                ImPlot.EndPlot()
            end
            ImPlot.EndSubplots()
        end
    end
    ig.End()
end

function _stat_uv(ctx, view, dc)
    uv = ctx.uv
    if ImPlot.BeginPlot("uv coverage", ig.ImVec2(-1, -1), ImPlot.ImPlotFlags_Equal | ImPlot.ImPlotFlags_NoLegend)
        ImPlot.SetupAxes("u (m)", "v (m)")
        if !isempty(uv)
            for ax in (ImPlot.ImAxis_X1, ImPlot.ImAxis_Y1)
                ImPlot.SetupAxisLimits(ax, -ctx.uv_R, ctx.uv_R, ImPlot.ImPlotCond_Once)
                ImPlot.SetupAxisLimitsConstraints(ax, -1.1 * ctx.uv_R, 1.1 * ctx.uv_R)
            end
            lim = ImPlot.GetPlotLimits(); sz = ImPlot.GetPlotSize()
            region = (ClosedInterval(lim.X.Min, lim.X.Max), ClosedInterval(lim.Y.Min, lim.Y.Max))
            mirror = (ClosedInterval(-lim.X.Max, -lim.X.Min), ClosedInterval(-lim.Y.Max, -lim.Y.Min))
            nbins  = (round(Int, sz.x), round(Int, sz.y))
            grey = setproperties(_DOT_SPEC; MarkerSize = 1.5f0, MarkerFillColor = _ST_UVGREY)
            k  = cached!(dc.uv,  (region, nbins)) do; uv[grid_decimate(uv, region, nbins)] end
            kc = cached!(dc.uvc, (mirror, nbins)) do; uv[grid_decimate(uv, mirror, nbins)] end
            ImPlot.PlotScatter("##uv",   k.u,   k.v;  spec = grey)
            ImPlot.PlotScatter("##uvc", -kc.u, -kc.v; spec = grey)
            sel = view.sel[]
            if sel !== nothing
                hot = setproperties(_DOT_SPEC; MarkerSize = 2.5f0, MarkerFillColor = _ST_UVSEL)
                s  = cached!(dc.uv_sel,  (sel, region, nbins)) do
                    sub = uv[findall(∈(sel), ctx.ds.rows.datetime)]; sub[grid_decimate(sub, region, nbins)]
                end
                sc = cached!(dc.uv_selc, (sel, mirror, nbins)) do
                    sub = uv[findall(∈(sel), ctx.ds.rows.datetime)]; sub[grid_decimate(sub, mirror, nbins)]
                end
                ImPlot.PlotScatter("##uvsel",   s.u,   s.v;  spec = hot)
                ImPlot.PlotScatter("##uvselc", -sc.u, -sc.v; spec = hot)
            end
        end
        ImPlot.EndPlot()
    end
end

function _stat_map(ctx)
    ll = ctx.lonlat
    if ImPlot.BeginPlot("antenna map", ig.ImVec2(-1, -1), ImPlot.ImPlotFlags_Equal | ImPlot.ImPlotFlags_NoLegend)
        ImPlot.SetupAxes("lon (°)", "lat (°)")
        for a in eachindex(ll)
            col = _antcolor(a)
            lon, lat = rad2deg(ll.lon[a]), rad2deg(ll.lat[a])
            ImPlot.PlotScatter("##ant$a", Cdouble[lon], Cdouble[lat];
                spec = ImPlot.ImPlotSpec(Marker = ImPlot.ImPlotMarker_Square, MarkerSize = 4f0,
                                         LineColor = col, FillColor = col))
            ImPlot.PlotText(ll.name[a], lon, lat, ig.ImVec2(0f0, -10f0))
        end
        ImPlot.EndPlot()
    end
end

function _stat_chanfreq(state)
    ctx, view = state.ctx, state.view
    cd = _chan_display(ctx, view)
    if ImPlot.BeginPlot("channel → ν", ig.ImVec2(-1, -1), ImPlot.ImPlotFlags_NoLegend)
        ImPlot.SetupAxes("channel", "ν (GHz)")
        setup_linked_axis!(ImPlot.ImAxis_X1, view.axes.chan)
        setup_linked_axis!(ImPlot.ImAxis_Y1, view.axes.freq)
        _setup_unit_axis(ImPlot.ImAxis_Y1, 1e-9)
        ImPlot.SetupAxisFormat(ImPlot.ImAxis_X1, "%.0f")
        ticks = Cdouble[c + 0.45 for c in cd.if_bounds]
        isempty(ticks) || ImPlot.SetupAxisTicks(ImPlot.ImAxis_X1, ticks, length(ticks), C_NULL, false)
        n = ctx.chan.n
        if n > 0
            chans = Float64.(1:n); ν = cd.ν
            ImPlot.PlotLine("##cf", chans, ν; spec = ImPlot.ImPlotSpec(LineWeight = 1f0))
            ImPlot.PlotScatter("##cf", chans, ν;
                spec = ImPlot.ImPlotSpec(Marker = ImPlot.ImPlotMarker_Circle, MarkerSize = 2.5f0))
        end
        axis_crosshair!(view.axes.chan)
        ImPlot.EndPlot()
    end
end

const _LOG_ERRCOL = ig.ImVec4(0.8627f0, 0.196f0, 0.184f0, 1f0)

function _stage_picker(view, snaps)
    if ig.RadioButton("Current", view.version[] === :current)
        view.version[] = :current
    end
    for k in eachindex(snaps.stage_name)
        ig.SameLine()
        ig.PushID(k)
        ig.RadioButton("$(snaps.stage_name[k]) #$k", view.version[] === k) && (view.version[] = k)
        ig.PopID()
    end
end

function _stage_progress(state)
    ctx, solve, view = state.ctx, state.solve, state.view
    if ig.Begin("Stage progress")
        if !ctx.solvable
            ig.TextUnformatted(ctx.sol0 === nothing ? "no solution — data viewer only" :
                               isempty(ctx.components) ? "solution has no components" :
                               "no schedule — nothing to solve")
        elseif solve.task[] === nothing
            ig.Button("Solve") && _start_solve!(state)
        else
            lat, logln, snap_names, snap_wall, nst, failed = lock(solve.lock) do
                (solve.latest[], copy(solve.log), copy(solve.snapshots.stage_name),
                 copy(solve.snapshots.wall), solve.nstages[], solve.failed[])
            end
            _stage_picker(view, (; stage_name = snap_names))
            ig.SameLine(); ig.Button("↻") && (view.refresh[] += 1)
            ig.SameLine(); ig.Button("Stop") && stop!(state)
            ig.Separator()

            ig.BeginChild("##solvelog", ig.ImVec2(0f0, ig.GetTextLineHeightWithSpacing() * 3 + 6f0),
                          ig.lib.ImGuiChildFlags_Borders)
            for (lvl, txt) in logln
                lvl >= Logging.Error ? ig.TextColored(_LOG_ERRCOL, txt) : ig.TextUnformatted(txt)
            end
            ig.GetScrollY() >= ig.GetScrollMaxY() && ig.SetScrollHereY(1f0)
            ig.EndChild()

            task = solve.task[]
            done = istaskdone(task); stopped = solve.stop[]
            if lat !== nothing && lat.running && !done
                lib.igProgressBar(-Float32(ig.GetTime() % 1.0), ig.ImVec2(-1, 0), C_NULL)
                ig.Text(@sprintf("running %s (%d/%d)", lat.stage_name, lat.stage_ix, nst))
            end
            lat === nothing || ig.Text(@sprintf("stage=%s (%d/%d)", lat.stage_name, lat.stage_ix, nst))
            ig.Text(@sprintf("done=%s  stopped=%s  failed=%s", done, stopped, failed))
            if !isempty(snap_wall)
                if ImPlot.BeginPlot("##stagewall", ig.ImVec2(-1, -1), ImPlot.ImPlotFlags_NoLegend)
                    ImPlot.SetupAxes("stage", "wall (s)",
                                     ImPlot.ImPlotAxisFlags_AutoFit, ImPlot.ImPlotAxisFlags_AutoFit)
                    labels = ["$(snap_names[k]) #$k" for k in eachindex(snap_names)]
                    ImPlot.SetupAxisTicks(ImPlot.ImAxis_X1, Cdouble.(1:length(labels)), length(labels), labels, false)
                    ImPlot.PlotBars("wall", Cdouble.(1:length(snap_wall)), snap_wall; bar_size = 0.6)
                    ImPlot.EndPlot()
                end
            end
        end
    end
    ig.End()
end

"""
    _grid(state)

Baseline grid: antenna×antenna matrix, each cell the weighted coherent-average phase vs global channel per
enabled correlation slot, DATA (+ optional MODEL). Heavy `compute_grid_{data,model}` run via the per-frame
coalescing DeriveSlots keyed on `grid_sig`; the selected `Solution` + enabled terms + sel are captured into
the thunks. Display toggles (slots/model/marker/full_matrix/only_active) filter an already-computed grid.
"""
function _grid(state)
    ctx, view, cache = state.ctx, state.view, state.cache
    if ig.Begin("Baseline grid")
        if ctx.sol0 === nothing
            ig.TextUnformatted("no solution — baseline grid unavailable")
        else
            sol = _active_solution(state)
            _grid_controls(view)
            if view.sel[] === nothing
                ig.Text("select a time interval on the timeline")
            else
                enabled = enabled_apply(view.grid.apply); sel = view.sel[]; sig = grid_sig(view)
                griddata, busy = derive!(cache.grid, sig, () -> compute_grid_data(sol, enabled, sel))
                modeldata = nothing; mbusy = false
                if view.grid.show_model[]
                    modeldata, mbusy = derive!(cache.model, sig, () -> compute_grid_model(sol, enabled, sel))
                end
                _grid_body(ctx, view, griddata, modeldata, busy || mbusy)
            end
        end
    end
    ig.End()
end

function _grid_controls(view)
    g = view.grid
    ig.Checkbox("full matrix", g.full_matrix); ig.SameLine()
    ig.Checkbox("only active", g.only_active); ig.SameLine()
    ig.Checkbox("model", g.show_model); ig.SameLine()
    ig.SetNextItemWidth(150f0); ig.SliderFloat("##marker", g.marker, 0.1f0, 2f0, "marker %.1f")
    ig.SameLine()
    ig.PushID("slots")
    foreach(eachindex(g.slots)) do k
        k == firstindex(g.slots) || ig.SameLine()
        ig.PushID(k)
        ig.PushStyleColor(ig.ImGuiCol_Text, _slotcolor(k))
        b = Ref(g.slots.enabled[k]); ig.Checkbox(g.slots.label[k], b); g.slots.enabled[k] = b[]
        ig.PopStyleColor(); ig.PopID()
    end
    ig.PopID()
    ig.PushID("apply")
    foreach(eachindex(g.apply)) do k
        k == firstindex(g.apply) || ig.SameLine()
        ig.PushID(k)
        b = Ref(g.apply.enabled[k]); ig.Checkbox(g.apply.label[k], b); g.apply.enabled[k] = b[]
        ig.PopID()
    end
    ig.PopID()
end

function _grid_body(ctx, view, griddata, modeldata, dim)
    dim && ig.PushStyleVar(ig.ImGuiStyleVar_Alpha, 0.4f0)
    (dim || griddata === nothing) && (ig.SameLine(); ig.Text("computing…"))
    griddata === nothing || _grid_matrix(ctx, view, griddata, modeldata)
    dim && ig.PopStyleVar()
end

function _grid_matrix(ctx, view, griddata, modeldata)
    blrow = Dict(bl.antennas => b for (b, bl) in enumerate(griddata.baseline))
    dense = ctx.antenna_ix
    antennas = ctx.antennas
    if view.grid.only_active[]
        active = Set{Int}()
        for (b, bl) in enumerate(griddata.baseline)
            if any(w -> sum(w) > 0, griddata.ΣW[b])
                push!(active, bl.antennas[1]); push!(active, bl.antennas[2])
            end
        end
        antennas = filter(a -> dense[a] in active, antennas)
    end
    na = length(antennas)
    na == 0 && return ig.Text("no active antennas in the interval")
    cd = _chan_display(ctx, view)
    ticks = Cdouble[c + 0.5 for c in cd.if_bounds]
    avail = ig.GetContentRegionAvail(); lh = ig.GetTextLineHeightWithSpacing()
    lw = maximum(a -> ig.CalcTextSize(String(a.name)).x, antennas) + 6f0
    ph = (avail.y - 2 * lh) / na
    vmid = max(0f0, (ph - ig.GetTextLineHeight()) / 2)
    af = ImPlot.ImPlotAxisFlags_NoLabel | ImPlot.ImPlotAxisFlags_NoTickLabels
    cellflags = ImPlot.ImPlotFlags_NoLegend | ImPlot.ImPlotFlags_NoMenus | ImPlot.ImPlotFlags_NoMouseText | ImPlot.ImPlotFlags_NoTitle

    ig.PushStyleVar(ig.ImGuiStyleVar_CellPadding, ig.ImVec2(0, 0))
    ImPlot.PushStyleVar(ImPlot.ImPlotStyleVar_PlotPadding, ig.ImVec2(2, 2))
    tblflags = ig.ImGuiTableFlags_NoPadInnerX | ig.ImGuiTableFlags_NoPadOuterX | ig.ImGuiTableFlags_NoSavedSettings
    if ig.BeginTable("##grid", na + 2, tblflags, avail)
        ig.TableSetupColumn("##l", ig.ImGuiTableColumnFlags_WidthFixed, lw)
        foreach(c -> ig.TableSetupColumn("##c$c", ig.ImGuiTableColumnFlags_WidthStretch), 1:na)
        ig.TableSetupColumn("##r", ig.ImGuiTableColumnFlags_WidthFixed, lw)
        ig.TableNextRow()
        foreach(ci -> (ig.TableSetColumnIndex(ci); _cell_label(String(antennas[ci].name))), 1:na)
        for ri in 1:na
            ig.TableNextRow()
            ig.TableSetColumnIndex(0); _cell_label(String(antennas[ri].name); voff = vmid)
            for ci in 1:na
                ig.TableSetColumnIndex(ci)
                A = dense[antennas[ri]]; B = dense[antennas[ci]]
                b, conj = if haskey(blrow, (A, B))
                    blrow[(A, B)], false
                elseif view.grid.full_matrix[] && haskey(blrow, (B, A))
                    blrow[(B, A)], true
                else
                    nothing, false
                end
                ig.PushID((ri - 1) * na + ci)
                if ImPlot.BeginPlot("##b$(ri)_$(ci)", ig.ImVec2(-1, ph), cellflags)
                    ImPlot.SetupAxes(C_NULL, C_NULL, af, af)
                    setup_linked_axis!(ImPlot.ImAxis_X1, view.axes.chan)
                    isempty(ticks) || ImPlot.SetupAxisTicks(ImPlot.ImAxis_X1, ticks, length(ticks), C_NULL, false)
                    ImPlot.SetupAxisLimits(ImPlot.ImAxis_Y1, -π, π, ImPlot.ImPlotCond_Always)
                    b === nothing || _grid_cell(view, griddata.phase[b],
                                                modeldata === nothing ? nothing : modeldata.model[b], conj, cd.perm)
                    axis_crosshair!(view.axes.chan)
                    ImPlot.IsPlotHovered() && ig.IsMouseClicked(0) && (view.plane.pair[] = (A, B))
                    ImPlot.EndPlot()
                end
                ig.PopID()
            end
            ig.TableSetColumnIndex(na + 1); _cell_label(String(antennas[ri].name); voff = vmid)
        end
        ig.TableNextRow()
        foreach(ci -> (ig.TableSetColumnIndex(ci); _cell_label(String(antennas[ci].name))), 1:na)
        ig.EndTable()
    end
    ImPlot.PopStyleVar()
    ig.PopStyleVar()
end

function _grid_cell(view, dphase, mphase, conj, perm)
    chans = Float64.(1:length(perm))
    foreach(eachindex(view.grid.slots)) do k
        view.grid.slots.enabled[k] || return
        i, j = view.grid.slots.ij[k]; col = _slotcolor(k); ms = view.grid.marker[]
        φd = [conj ? -dphase[c][j, i] : dphase[c][i, j] for c in perm]
        fin = findall(!isnan, φd)
        isempty(fin) || ImPlot.PlotScatter("d$k", chans[fin], φd[fin];
            spec = setproperties(_DOT_SPEC; MarkerSize = ms, MarkerFillColor = col))
        if mphase !== nothing
            φm = [conj ? -mphase[c][j, i] : mphase[c][i, j] for c in perm]
            finm = findall(!isnan, φm)
            isempty(finm) || ImPlot.PlotLine("m$k", chans[finm], φm[finm];
                spec = ImPlot.ImPlotSpec(LineColor = col, LineWeight = 1f0))
        end
    end
end

function _radio!(label, ref, value)
    ig.RadioButton(label, ref[] === value) && (ref[] = value)
end

function _uv_profile_controls(ctx, view)
    uv = view.uvprof
    ig.TextUnformatted("source")
    for s in eachindex(ctx.ds.sources.name)
        s == firstindex(ctx.ds.sources.name) || ig.SameLine()
        ig.PushID(s)
        _radio!(String(ctx.ds.sources.name[s]), uv.source, s)
        ig.PopID()
    end
    _radio!("amplitude", uv.ymode, :amplitude); ig.SameLine()
    _radio!("SNR", uv.ymode, :snr)
    if uv.ymode[] === :amplitude
        ig.TextUnformatted("time")
        _radio!("integration", uv.time_mode, UVIntegration()); ig.SameLine()
        _radio!("scan", uv.time_mode, UVScan()); ig.SameLine()
        _radio!("whole", uv.time_mode, UVWhole()); ig.SameLine()
        _radio!("custom", uv.time_mode, UVCustom())
        if uv.time_mode[] === UVCustom()
            ig.SameLine(); ig.SetNextItemWidth(160f0)
            ig.SliderFloat("##uvcustomtime", uv.custom_time, 1f0, 3600f0, "custom %.0f s")
        end
        ig.TextUnformatted("freq")
        _radio!("channel", uv.freq_mode, UVChannel()); ig.SameLine()
        _radio!("IF", uv.freq_mode, UVIF()); ig.SameLine()
        _radio!("band", uv.freq_mode, UVBand())
    else
        ig.TextUnformatted("SNR reads the captured Fringes tiles")
    end
end

function _uv_profile_body(ctx, view, data, dim, sig)
    dim && ig.PushStyleVar(ig.ImGuiStyleVar_Alpha, 0.4f0)
    (dim || data === nothing) && (ig.SameLine(); ig.Text("computing…"))
    if data !== nothing
        data.status === nothing || ig.TextUnformatted(data.status)
        ylabel = view.uvprof.ymode[] === :amplitude ? "amplitude |V|" : "fringe SNR"
        if ImPlot.BeginPlot("##uvprofile", ig.ImVec2(-1, -1), ImPlot.ImPlotFlags_NoMenus)
            ImPlot.SetupAxes("UV distance (wavelengths)", ylabel)
            view.uvprof.ymode[] === :snr && ImPlotExtra.setup_axis_scale!(ImPlot.ImAxis_Y1, _SNR_SYMLOG)
            fitsig = (sig, length(data.x))
            if !isempty(data.x) && view.uvprof.fit_sig[] != fitsig
                xlo, xhi = extrema(data.x); ylo, yhi = extrema(data.y)
                xpad = xhi > xlo ? 0.05 * (xhi - xlo) : max(abs(xhi), 1.0) * 0.05
                ypad = yhi > ylo ? 0.05 * (yhi - ylo) : max(abs(yhi), 1.0) * 0.05
                ImPlot.SetupAxisLimits(ImPlot.ImAxis_X1, xlo - xpad, xhi + xpad, ImPlot.ImPlotCond_Always)
                ImPlot.SetupAxisLimits(ImPlot.ImAxis_Y1, ylo - ypad, yhi + ypad, ImPlot.ImPlotCond_Always)
                view.uvprof.fit_sig[] = fitsig
            end
            for k in eachindex(view.grid.slots)
                slot = view.grid.slots.ij[k]
                idx = findall(==(slot), data.correlation_product)
                isempty(idx) && continue
                ImPlot.PlotScatter(view.grid.slots.label[k], data.x[idx], data.y[idx];
                    spec = setproperties(_DOT_SPEC; MarkerSize = view.grid.marker[], MarkerFillColor = _slotcolor(k)))
            end
            ImPlot.EndPlot()
        end
    end
    dim && ig.PopStyleVar()
end

function _uv_profile(state)
    ctx, view, cache = state.ctx, state.view, state.cache
    if ig.Begin("UV profile")
        if ctx.sol0 === nothing
            ig.TextUnformatted("no solution — UV profile unavailable")
        else
            _uv_profile_controls(ctx, view)
            slots = view.grid.slots.ij[view.grid.slots.enabled]
            if isempty(slots)
                ig.Text("enable at least one correlation slot")
            else
                src = ctx.ds.sources.name[view.uvprof.source[]]
                sig = uvprof_sig(view)
                data, busy = if view.uvprof.ymode[] === :amplitude
                    sol = _active_solution(state); enabled = enabled_apply(view.grid.apply)
                    tmode, fmode, ctime = view.uvprof.time_mode[], view.uvprof.freq_mode[], view.uvprof.custom_time[]
                    derive!(cache.uvprof, sig, () -> compute_uv_amplitude(ctx, sol, enabled, src, tmode, fmode, ctime, slots))
                else
                    fringes = _captured_fringes(state)
                    sig2 = (sig, fringes === nothing ? 0 : length(fringes))
                    derive!(cache.uvprof, sig2, () -> compute_uv_snr(ctx, fringes, src, slots))
                end
                _uv_profile_body(ctx, view, data, busy, sig)
            end
        end
    end
    ig.End()
end

const _NOFILL = ig.ImVec4(0f0, 0f0, 0f0, 0f0)
const _PK_SPEC  = ImPlot.ImPlotSpec(Marker = ImPlot.ImPlotMarker_Circle, MarkerSize = 13f0,
                                    MarkerLineColor = ig.ImVec4(0f0, 0.85f0, 0.9f0, 1f0), MarkerFillColor = _NOFILL, LineWeight = 2f0)
const _CUR_SPEC = ImPlot.ImPlotSpec(Marker = ImPlot.ImPlotMarker_Circle, MarkerSize = 8f0,
                                    MarkerLineColor = ig.ImVec4(0.86f0, 0.40f0, 0.10f0, 1f0), MarkerFillColor = _NOFILL, LineWeight = 2f0)

function _plane_nyq(slot)
    slot.shown === nothing && return nothing
    i = findfirst(pr -> !pr.empty, slot.shown.result)
    i === nothing ? nothing : (slot.shown.result[i].nyq_rate, slot.shown.result[i].nyq_delay)
end

function _plane_controls(ctx, view, nyq)
    pl = view.plane; nif = ctx.nif
    rfmt = nyq === nothing ? @sprintf("rate win %.0f%% Nyq", pl.rate_win[] * 100) : @sprintf("rate win %.0f mHz", pl.rate_win[] * nyq[1] * 1e3)
    dfmt = nyq === nothing ? @sprintf("delay win %.0f%% Nyq", pl.delay_win[] * 100) : @sprintf("delay win %.0f ns", pl.delay_win[] * nyq[2] * 1e9)
    ig.SetNextItemWidth(110f0)
    if ig.BeginCombo("freq", pl.if_sel[] == 0 ? "whole band" : "IF $(pl.if_sel[])")
        ig.Selectable("whole band", pl.if_sel[] == 0) && (pl.if_sel[] = 0)
        foreach(k -> ig.Selectable("IF $k", pl.if_sel[] == k) && (pl.if_sel[] = k), 1:nif)
        ig.EndCombo()
    end
    ig.SameLine(); ig.SetNextItemWidth(150f0); ig.SliderFloat("##ros",  pl.rate_os,   1f0, 8f0, "time pad ×%.1f")
    ig.SameLine(); ig.SetNextItemWidth(150f0); ig.SliderFloat("##dos",  pl.delay_os,  1f0, 8f0, "freq pad ×%.1f")
    ig.SameLine(); ig.SetNextItemWidth(150f0); ig.SliderFloat("##rwin", pl.rate_win,  0.05f0, 1f0, rfmt)
    ig.SameLine(); ig.SetNextItemWidth(150f0); ig.SliderFloat("##dwin", pl.delay_win, 0.05f0, 1f0, dfmt)
end

function _plane_marks(state, pq, slots, sel, chans)
    ctx, cache = state.ctx, state.cache
    sol = _active_solution(state)
    t0u = datetime2unix(ctx.ds.time0)
    tmid = (datetime2unix(leftendpoint(sel)) + datetime2unix(rightendpoint(sel))) / 2 - t0u
    νset = chans === Colon() ? ctx.ds.freq.ν : ctx.ds.freq.ν[chans]
    νmid = (minimum(νset) + maximum(νset)) / 2
    sig = (pq, sel, chans, state.view.version[], state.view.refresh[])
    cached!(cache.plane_marks, sig) do
        Dict((i, j) => pair_dr(sol, pq[1], i, pq[2], j, tmid, νmid) for (i, j) in slots)
    end
end

function _plane_body(planes, marks, dim, do_fit)
    planes === nothing && return
    dim && ig.PushStyleVar(ig.ImGuiStyleVar_Alpha, 0.4f0)
    n = length(planes); rows, cols = n ≤ 1 ? (1, 1) : n == 2 ? (1, 2) : (2, 2)
    ext = findfirst(pr -> !pr.empty, planes)
    spflags = ImPlot.ImPlotSubplotFlags_LinkAllX | ImPlot.ImPlotSubplotFlags_LinkAllY
    if ImPlot.BeginSubplots("##planes", rows, cols, ig.ImVec2(-1, -1), spflags)
        for pr in planes
            lab = pr.empty ? pr.label : "$(pr.label)  SNR=$(round(pr.peak.snr; digits = 1))"
            if ImPlot.BeginPlot(lab, ig.ImVec2(-1, -1), ImPlot.ImPlotFlags_NoLegend | ImPlot.ImPlotFlags_NoMenus)
                ImPlot.SetupAxes("rate", "delay")
                _setup_unit_axis(ImPlot.ImAxis_X1, 1e3)
                _setup_unit_axis(ImPlot.ImAxis_Y1, 1e9)
                if ext !== nothing
                    e = planes[ext]
                    do_fit && ImPlot.SetupAxisLimits(ImPlot.ImAxis_X1, e.rate[1],  e.rate[end],  ImPlot.ImPlotCond_Always)
                    do_fit && ImPlot.SetupAxisLimits(ImPlot.ImAxis_Y1, e.delay[1], e.delay[end], ImPlot.ImPlotCond_Always)
                    ImPlot.SetupAxisLimitsConstraints(ImPlot.ImAxis_X1, e.rate[1],  e.rate[end])
                    ImPlot.SetupAxisLimitsConstraints(ImPlot.ImAxis_Y1, e.delay[1], e.delay[end])
                end
                if !pr.empty
                    ImPlotExtra.image!("##h", pr.rate[1]..pr.rate[end], pr.delay[1]..pr.delay[end], pr.snr;
                                       colormap = :viridis, colorrange = (0.0, pr.scale_max), colorscale = _SNR_SYMLOG)
                    ImPlot.PlotScatter("peak", Cdouble[pr.peak.rate], Cdouble[pr.peak.delay]; spec = _PK_SPEC)
                    m = get(marks, pr.correlation_product, nothing)
                    if m !== nothing && m.delay !== nothing && m.rate !== nothing
                        ImPlot.PlotScatter("installed", Cdouble[m.rate], Cdouble[m.delay]; spec = _CUR_SPEC)
                    end
                else
                    ig.Text("no data for this slot")
                end
                ImPlot.EndPlot()
            end
        end
        ImPlot.EndSubplots()
    end
    dim && ig.PopStyleVar()
end

"""
    _plane(state)

Fringe-plane panel: FFT |delay–rate| heatmap for the pair selected by clicking a Baseline-grid cell.
Slots + calibration follow the Baseline grid. Controls: IF/band + oversample + window-fraction sliders.
Overlays: argmax peak (cyan) + the installed-solution (delay,rate) at the tile (orange).
"""
function _plane(state)
    ctx, view, cache = state.ctx, state.view, state.cache
    if ig.Begin("Fringe plane")
        clk = view.plane.pair[]
        pq = clk === nothing ? nothing : resolve_pair(ctx.baselines, clk...)
        if ctx.sol0 === nothing
            ig.TextUnformatted("no solution — fringe plane unavailable")
        elseif pq === nothing
            ig.Text("click a baseline-grid cell to select a pair")
        else
            pl = view.plane
            dim = isbusy(cache.plane)
            ig.Text(@sprintf("%s × %s", ctx.ds.antennas[pq[1]].name, ctx.ds.antennas[pq[2]].name))
            dim && (ig.SameLine(); ig.Text("computing…"))
            _plane_controls(ctx, view, _plane_nyq(cache.plane))
            if view.sel[] === nothing
                ig.Text("select a time interval on the timeline")
            else
                sol = _active_solution(state)
                enabled = enabled_apply(view.grid.apply)
                slots = view.grid.slots.ij[view.grid.slots.enabled]
                sel = view.sel[]; chans = _plane_chans(ctx.ds, view.plane.if_sel[])
                ros, dos, rfr, dfr = pl.rate_os[], pl.delay_os[], pl.rate_win[], pl.delay_win[]
                planes, _ = derive!(cache.plane, plane_sig(view),
                    () -> compute_fft_planes(ctx, sol, enabled, pq, slots, sel, chans, ros, dos, rfr, dfr))
                mslots = planes === nothing ? slots : [pr.correlation_product for pr in planes]
                marks = _plane_marks(state, pq, mslots, sel, chans)
                do_fit = planes !== pl.fit_planes[]; pl.fit_planes[] = planes
                _plane_body(planes, marks, dim, do_fit)
            end
        end
    end
    ig.End()
end

"""
    inspect(ds, sol=nothing; schedule=Step[], autosolve=false, window_size=(1600,1000)) -> state

Open the inspector for `ds` and (optional) seed `sol`. Blocks on the main thread in the render loop;
returns the `state` NamedTuple. `schedule` is a `Vector{<:Fringy.Step}`. `autosolve` (opt-in; requires a
solvable state) spawns the background solve immediately; otherwise use the Stage-progress panel's Solve
button. `stop!(state)` cancels.

Launch julia with an interactive thread (`julia -t auto,1`) so the render loop is isolated from the
`:default` compute pool.
"""
function inspect(ds::Dataset, sol::Union{Nothing,Solution} = nothing;
                 schedule::AbstractVector = Step[], autosolve::Bool = false, window_size = (1600, 1000))
    (Threads.nthreads(:interactive) ≥ 1 && Threads.nthreads(:default) ≥ 1) || @warn(
        "FringyUI runs the render loop on the main thread and spawns compute on the :default pool; for " *
        "the render thread to be isolated, launch julia with an interactive thread, e.g. `julia -t auto,1`.",
        default_threads = Threads.nthreads(:default), interactive_threads = Threads.nthreads(:interactive))
    state = _build_state(ds, sol, schedule)
    autosolve && state.ctx.solvable && _start_solve!(state)
    _renderloop!(state; window_size)
    return state
end
