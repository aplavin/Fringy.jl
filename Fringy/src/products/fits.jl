
export write_registered_image, write_registered_copy

const FITS_BLOCK = 2880
const FITS_CARDS_PER_BLOCK = 36
const CARDLEN = 80
const MAS_PER_DEG = 3.6e6


"""
    fits_float(x) -> String

A float as a FITS card value: [`pyfloat`](@ref) text, `e` → `E`, and — the one lossy step in the whole
file — truncated to the 20 columns a card's value field has, keeping the exponent. `CDELT =
−0.1/3.6e6` is the case that hits it: 17 significant digits do not fit, 14 do.
"""
function fits_float(x::Real)
    s = replace(pyfloat(x), 'e' => 'E')
    length(s) <= 20 && return s
    i = findfirst('E', s)
    isnothing(i) ? s[1:20] : s[1:20-(length(s)-(i-1))] * s[i:end]
end

fits_value(v::Bool) = lpad(v ? "T" : "F", 20)
fits_value(v::Integer) = lpad(string(v), 20)
fits_value(v::Real) = lpad(fits_float(v), 20)
fits_value(v::AbstractString) = rpad("'" * rpad(replace(v, "'" => "''"), 8) * "'", 20)

"Pad or truncate to exactly 80 BYTES — a card is a fixed-width record, not a fixed-width string."
function pad80(s::AbstractString)
    b = codeunits(s)
    String(length(b) >= CARDLEN ? b[1:CARDLEN] : vcat(b, fill(UInt8(' '), CARDLEN - length(b))))
end

"""
    card(key, value, comment = "") -> String

One 80-character FITS card. Keyword in columns 1–8, `= ` in 9–10, the value field in 11–30 (numbers
right-justified, strings quoted and left-justified), then ` / comment` if there is one.
"""
function card(key::AbstractString, value, comment::AbstractString = "")
    ncodeunits(key) <= 8 || error("fits: keyword $key is longer than 8 characters")
    pad80(rpad(key, 8) * "= " * fits_value(value) * (isempty(comment) ? "" : " / " * comment))
end

"""
    commentary(key, text) -> Vector{String}

A HISTORY/COMMENT entry as the cards that carry it: the text cut into 72-character pieces, in order,
each on its own card. No word wrapping — the cut falls where column 80 falls, which is what the
shipped headers show.
"""
function commentary(key::AbstractString, text::AbstractString)
    b = codeunits(text)
    isempty(b) && return [pad80(rpad(key, 8))]
    [pad80(rpad(key, 8) * String(b[i:min(i + 71, end)])) for i in 1:72:length(b)]
end

"The cards of a header as the 2880-byte blocks it occupies: `END`, then blanks to the block boundary."
function header_bytes(cards::Vector{String})
    all = vcat(cards, [rpad("END", CARDLEN)])
    n = cld(length(all), FITS_CARDS_PER_BLOCK) * FITS_CARDS_PER_BLOCK
    append!(all, fill(" "^CARDLEN, n - length(all)))
    codeunits(join(all))
end

"Pad `n` bytes of data to the FITS block size."
fits_padding(n::Integer) = zeros(UInt8, mod(-n, FITS_BLOCK))


"""
    read_fits_bytes(path) -> Vector{UInt8}

A published image as bytes. An archive mirror may carry each map both plainly and gzipped; the plain
file is used when it is there and the gzipped one is decompressed when it is not.
"""
function read_fits_bytes(path)
    isfile(path) && return read(path)
    isfile(path * ".gz") || error("fits: neither $path nor $path.gz exists")
    read(`gzip -dc $(path * ".gz")`)
end

"The 80-character cards of the header at byte offset `pos` (1-based), and the offset after it."
function read_cards(b::Vector{UInt8}, pos::Int)
    cards = String[]
    while true
        pos + FITS_BLOCK - 1 <= length(b) || error("fits: header runs past the end of the file")
        for k in 0:FITS_CARDS_PER_BLOCK-1
            c = String(b[pos+80k : pos+80k+79])
            startswith(c, "END     ") && return (cards, pos + FITS_BLOCK)
            push!(cards, c)
        end
        pos += FITS_BLOCK
    end
end

"""
    split_card(c) -> (value, comment)

Everything before the first unquoted `/`, and after it. Sliced by BYTE, because an archival header may
carry bytes that are not valid UTF-8.
"""
function split_card(c::AbstractString)
    b = codeunits(c)
    inq = false
    for i in 11:length(b)
        ch = Char(b[i])
        ch == '\'' && (inq = !inq)
        !inq && ch == '/' && return (strip(String(b[11:i-1])), strip(String(b[i+1:end])))
    end
    (strip(String(b[11:end])), "")
end

"Index of the card with keyword `key`, or `nothing`."
find_card(cards, key) = findfirst(c -> strip(String(codeunits(c)[1:8])) == key, cards)

"""
    hdu_data_bytes(cards) -> Int

The size of this HDU's data unit, from its own header: `|BITPIX|/8 × GCOUNT × (PCOUNT + Π NAXISi)`,
with the random-groups convention that a zero NAXIS1 drops out of the product.
"""
function hdu_data_bytes(cards)
    geti(key, default) = (i = find_card(cards, key); isnothing(i) ? default :
                          parse(Int, first(split_card(cards[i]))))
    naxis = geti("NAXIS", 0)
    naxis == 0 && return 0
    n = 1
    for a in 1:naxis
        v = geti("NAXIS$a", 1)
        (a == 1 && v == 0) || (n *= v)
    end
    abs(geti("BITPIX", 8)) ÷ 8 * geti("GCOUNT", 1) * (geti("PCOUNT", 0) + n)
end


"""
    wcs_history(model_set, sigma_ra, sigma_dec; session, date_obs) -> Vector{String}

The HISTORY block both products carry: what CRVAL now means, which model set produced it, what its
uncertainty is, and in what frame — plus, first, the framing line that the catalogue's metadata also
carries. A reader who has only the file must be able to tell that these are absolute registered
coordinates and that the registered map is the thing this pipeline delivers.
"""
wcs_history(model_set, sigma_ra, sigma_dec; session, date_obs) = [
    "THIS FILE IS THE PRODUCT: registered coordinates define an image WCS; " *
    "the registered maps, not the position list, are what is delivered.",
    "ABSOLUTE ASTROMETRY: CRVAL1/CRVAL2 are the registered absolute position",
    "of this map's reference pixel, from the $(uppercase(session)) ($date_obs)",
    "group-delay solve with this model set's structure delay subtracted.",
    "Model set: $model_set.",
    @sprintf("sigma_final %.3f / %.3f mas (RA*, Dec), floor-dominated;", sigma_ra, sigma_dec),
    "frame: ITRF2020 + IERS finals, no rotation to an external catalogue removed.",
]


"""
    write_registered_image(path, source, si::SourceImage, crval, sigma;
                           model_set, session, date_obs)

Our restored map with the registered WCS. The array is `[x = East, y = North]` in column-major order
with the model origin at pixel `npix ÷ 2 + 1` of both axes. FITS wants axis 1 = RA and, by universal
convention, RA DECREASING with column index — so axis 1 is the x axis reversed, which is a
`reverse(dims = 1)` and nothing else (the array is already in the column-major order FITS writes), and
CRPIX1 follows it.
"""
function write_registered_image(path, source, si, crval, sigma; model_set, session, date_obs)
    npix, pixel = si.grid.npix, si.grid.pixel
    cards = [card("SIMPLE", true, "conforms to FITS standard"),
             card("BITPIX", -32, "array data type"),
             card("NAXIS", 2, "number of array dimensions"),
             card("NAXIS1", npix),
             card("NAXIS2", npix),
             card("BUNIT", "JY/BEAM", "Unit of measurement"),
             card("OBJECT", String(source), "B1950 name"),
             card("TELESCOP", "VLBA", "Telescope used"),
             card("DATE-OBS", date_obs, "Observation date ($(uppercase(session)))"),
             card("CTYPE1", "RA---SIN", "Axis name"),
             card("CRPIX1", float(npix - npix ÷ 2), "Reference pixel (the model coordinate origin)"),
             card("CRVAL1", crval[1], "REGISTERED absolute RA of the reference pixel"),
             card("CDELT1", -pixel / MAS_PER_DEG, "Pixel increment"),
             card("CTYPE2", "DEC--SIN", "Axis name"),
             card("CRPIX2", float(npix ÷ 2 + 1), "Reference pixel"),
             card("CRVAL2", crval[2], "REGISTERED absolute Dec of the reference pixel"),
             card("CDELT2", pixel / MAS_PER_DEG, "Pixel increment"),
             card("EQUINOX", 2000.0),
             card("BMAJ", fwhm_max(si.beam) / MAS_PER_DEG, "Clean beam major axis (deg)"),
             card("BMIN", fwhm_min(si.beam) / MAS_PER_DEG, "Clean beam minor axis (deg)"),
             card("BPA", rad2deg(position_angle(si.beam)), "Clean beam position angle (deg)")]
    for line in wcs_history(model_set, sigma[1], sigma[2]; session, date_obs)
        append!(cards, commentary("HISTORY", line))
    end
    img = si.restored
    px = [hton(reinterpret(UInt32, Float32(img[npix + 1 - i, j]))) for j in 1:npix for i in 1:npix]
    open(path, "w") do io
        write(io, header_bytes(cards))
        write(io, px)
        write(io, fits_padding(4 * length(px)))
    end
    path
end


"""
    write_registered_copy(path, src, crval, sigma;
                          model_set, session, date_obs) -> (crval1, crval2)

The published image with CRVAL replaced by the registered absolute position, written as a BYTE COPY of
the original with two cards rewritten and a HISTORY block appended. Every other card of the primary
header, the whole pixel array and the whole AIPS CC binary table are copied through unread and
unaltered; the only other card that changes is a pre-standard `XTENSION = 'A3DTABLE'`, normalized to
the standard spelling of the same type (see the file header).

Returns the CRVAL the source file carried — the phase-centre label its self-calibration frame is not
tied to — which the manifest records beside ours.
"""
function write_registered_copy(path, src, crval, sigma; model_set, session, date_obs)
    b = read_fits_bytes(src)
    cards, pos = read_cards(b, 1)
    old = map(("CRVAL1", "CRVAL2")) do key
        i = find_card(cards, key)
        isnothing(i) && error("fits: $src has no $key card")
        parse(Float64, first(split_card(cards[i])))
    end
    for (key, v) in (("CRVAL1", crval[1]), ("CRVAL2", crval[2]))
        i = find_card(cards, key)
        cards[i] = card(key, v, last(split_card(cards[i])))
    end
    for line in wcs_history(model_set, sigma[1], sigma[2]; session, date_obs)
        append!(cards, commentary("HISTORY", line))
    end
    append!(cards, commentary("HISTORY", "Pixel data and every other keyword are the published file's."))
    append!(cards, commentary("HISTORY",
                              @sprintf("Header CRVAL was %.9f %.9f (phase-centre label).", old...)))
    open(path, "w") do io
        write(io, header_bytes(cards))
        ndata = cld(hdu_data_bytes(cards), FITS_BLOCK) * FITS_BLOCK
        write(io, @view b[pos:min(pos + ndata - 1, length(b))])
        pos += ndata
        while pos + FITS_BLOCK - 1 <= length(b)
            ext, next = read_cards(b, pos)
            i = find_card(ext, "XTENSION")
            if !isnothing(i) && first(split_card(ext[i])) == "'A3DTABLE'"
                ext[i] = card("XTENSION", "BINTABLE", "binary table extension")
            end
            nd = cld(hdu_data_bytes(ext), FITS_BLOCK) * FITS_BLOCK
            write(io, header_bytes(ext))
            write(io, @view b[next:min(next + nd - 1, length(b))])
            pos = next + nd
        end
    end
    (old[1], old[2])
end
