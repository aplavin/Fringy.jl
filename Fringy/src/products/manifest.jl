
export write_manifest

"""
    write_manifest(path, cfg, facts, sets, written; source_names_path, rfc_provenance, revision = "", extra = (;)) -> path

The session's provenance manifest: identity, code, inputs, model files, and the SHA-256 of every
product `written` (the registered FITS included — they are the deliverable the repository does not
commit).

`written` is the list of paths the products step has produced so far; the manifest is the last file
written and does not describe itself.

`rfc_provenance` is the application's statement of where its RFC catalogue came from — key–value
pairs written into `[inputs.rfc]` before the release `version` the configuration names.
"""
function write_manifest(path, cfg, facts, sets::ModelSets, written; source_names_path,
                        rfc_provenance = Pair{String, String}[], revision = "", extra = (;))
    results = dirname(path)
    io = IOBuffer()
    println(io, MANIFEST_HEADER)

    println(io, "[session]")
    _kv(io, "name", facts.name)
    _kv(io, "date_obs", string(facts.date))
    _kv(io, "start", string(facts.start))
    _kv(io, "stop", string(facts.stop))
    _kv(io, "epoch_obs", facts.epoch_obs)
    _kv(io, "doy", facts.doy)
    _kv(io, "nu_eff_hz", facts.ν_eff)
    _kv(io, "n_channels", length(facts.ν))
    _kv(io, "n_ifs", maximum(facts.if_of))
    _kv(io, "channel_width_hz", facts.Δν)
    for (k, v) in pairs(extra)
        _kv(io, string(k), v)
    end
    println(io)

    println(io, "[pipeline]")
    _kv(io, "package", "Fringy")
    _kv(io, "revision", revision)
    _kv(io, "julia", string(VERSION))
    println(io)

    println(io, "[configuration]")
    _kv(io, "reference_antenna", String(cfg.reference_antenna))
    _kv(io, "reference_receptor_slot", cfg.reference_receptor_slot)
    println(io)

    println(io, "# The visibility file: identified, not hashed — see the header.")
    println(io, "[inputs.visibilities]")
    _file_entry(io, cfg.data; hash = false)
    println(io)
    for (name, p) in ("antenna_geometry" => cfg.antenna_geometry, "eop" => cfg.eop,
                      "source_names" => source_names_path)
        println(io, "[inputs.$name]")
        _file_entry(io, p)
        println(io)
    end
    println(io, "[inputs.rfc]")
    for (k, v) in rfc_provenance
        _kv(io, k, v)
    end
    _kv(io, "version", cfg.rfc_version)
    println(io)
    for (product, files) in pairs(cfg.ionex_files), f in files
        println(io, "[[inputs.ionex]]")
        _kv(io, "product", string(product))
        _file_entry(io, joinpath(cfg.ionex_dir, f))
        println(io)
    end

    for (label, set) in pairs(sets)
        files = get(set.provenance, :files, nothing)
        (isnothing(files) || isempty(files)) && continue
        println(io, "[[model_files]]")
        _kv(io, "model_set", label)
        _kv(io, "kind", String(set.kind))
        _kv(io, "n_sources", length(files))
        println(io, "files = [")
        for (src, rec) in pairs(files)
            for role in (:model, :image)
                r = get(rec, role, nothing)
                isnothing(r) && continue
                @printf(io, "  { source = %s, role = %s, path = %s, bytes = %d, mtime = %s, sha256 = %s },\n",
                        _t(String(src)), _t(String(role)), _t(r.path), r.bytes, _t(string(r.mtime)),
                        _t(r.sha256))
            end
        end
        println(io, "]")
        println(io)
    end

    println(io, "# Every file the products step wrote, hashed. The registered images are here and are")
    println(io, "# NOT committed: this is what says the repository knows them.")
    for p in sort(collect(String.(written)))
        println(io, "[[products]]")
        _kv(io, "file", relpath(p, results))
        _kv(io, "bytes", filesize(p))
        _kv(io, "sha256", _sha256_of(p))
        println(io)
    end

    write(path, take!(io))
    path
end

const MANIFEST_HEADER = """
# The provenance of this session's deliverables: what went in, what came out, and which code made
# one from the other. Written by Fringy's products step; nothing here is typed by hand.
#
# The visibility file is identified by path, size and mtime rather than by SHA-256: hashing 47 GB
# costs minutes on a step that takes seconds, and `[session]` already carries the facts the pipeline
# DERIVED from it — the observation code, its date, its channel comb — which are read out of the file
# on every run.
#
# There is no run timestamp. A generated file that lives in a repository must be a function of its
# inputs alone, or every re-run is a diff that means nothing; when a run happened is what the commit
# records. `revision` names the code, and a `+dirty` suffix means it was never committed.
#
# `[[products]]` lists everything the PRODUCTS STEP wrote, hashed — the registered FITS images
# included. They are the one deliverable this repository does not commit (they are bulk and
# reproducible), so their hashes here are what makes that omission checkable rather than merely
# claimed. The figures are not listed: they are drawn by a separate entry point and are committed
# like any other text, so git already records them.
"""


_t(s::AbstractString) = '"' * replace(String(s), '\\' => "\\\\", '"' => "\\\"") * '"'

_v(x::AbstractString) = _t(x)
_v(x::Integer) = string(x)
_v(x::AbstractFloat) = isfinite(x) ?
    (s = @sprintf("%.17g", x); any(c -> c in ".eE", s) ? s : s * ".0") : _t(_fmt(x))
_v(x) = _t(string(x))

_kv(io, k, v) = println(io, k, " = ", _v(v))

"One input file, as the four things that identify it. `hash = false` states path, size and mtime only."
function _file_entry(io, path; hash = true)
    _kv(io, "path", abspath(path))
    if isfile(path)
        _kv(io, "bytes", filesize(path))
        _kv(io, "mtime", string(unix2datetime(mtime(path))))
        hash && _kv(io, "sha256", _sha256_of(path))
    else
        _kv(io, "absent_at_write", true)
    end
end

"SHA-256 of a file, streamed: the products include ~150 MB of registered FITS and are not read whole."
function _sha256_of(path)
    open(path) do file
        bytes2hex(SHA.sha256(file))
    end
end
