"""Paper ingestion: identity, query building, extraction, reading, rendering.

Everything here is offline. The PDFs are synthesised in the fixture, so the
suite asserts this module's behaviour rather than arXiv's availability, and it
runs the same on a machine with no network. The one thing that genuinely needs
the network - that arXiv accepts the query grammar we build - is asserted
against recorded shapes here and exercised for real in test_live_mcp.
"""

from __future__ import annotations

import json

import pytest

from icn import papers

pymupdf = pytest.importorskip("pymupdf", reason="PDF fixtures and rendering need PyMuPDF")


@pytest.fixture
def sample_pdf(tmp_path):
    """A three-page PDF with headings, a body, and real bookmarks."""
    document = pymupdf.open()
    bodies = [
        ("1. Introduction", "We study speculative refresh batching under contention.\n"
                            "The baseline serialises every refresh."),
        ("2. Method", "We batch refreshes and verify them in one pass.\n"
                      "The acceptance rate governs the observed speedup."),
        ("3. Results", "Throughput improved by 2.4x at batch size 8.\n"
                       "At batch size 1 there is no measurable gain."),
    ]
    for heading, body in bodies:
        page = document.new_page()
        page.insert_text((72, 96), heading, fontsize=16)
        page.insert_text((72, 140), body, fontsize=11)

    document.set_toc([[1, "Introduction", 1], [1, "Method", 2], [1, "Results", 3]])
    target = tmp_path / "sample.pdf"
    document.save(str(target))
    document.close()
    return target


# ---------------------------------------------------------------- identity


@pytest.mark.parametrize("raw,expected", [
    ("2401.12345", "2401.12345"),
    ("2401.12345v2", "2401.12345v2"),
    ("arXiv:2401.12345", "2401.12345"),
    ("arxiv: 2401.12345v3", "2401.12345v3"),
    ("https://arxiv.org/abs/2401.12345", "2401.12345"),
    ("https://arxiv.org/pdf/2401.12345v2", "2401.12345v2"),
    ("math/0309136", "math/0309136"),
    ("cond-mat.stat-mech/0309136v1", "cond-mat.stat-mech/0309136v1"),
    ("not an identifier", None),
    ("", None),
])
def test_arxiv_ids_are_recognised_in_every_form_an_agent_meets_them(raw, expected):
    assert papers.normalize_arxiv_id(raw) == expected


def test_the_version_suffix_survives_normalisation():
    """v1 and v3 can make different claims; a cache that conflates them lies."""
    assert papers.normalize_arxiv_id("2401.12345v1") != papers.normalize_arxiv_id("2401.12345v3")


# ---------------------------------------------------------------- queries


def test_a_plain_query_is_anded_not_quoted_as_one_phrase():
    """The bug this guards: arXiv reads a quoted string as an exact phrase, so
    quoting a whole question matched zero papers where ANDing its terms matches
    hundreds."""
    built = papers._build_search_query("speculative decoding language model inference", "")
    assert '"speculative decoding language model inference"' not in built
    assert built.count(" AND ") == 4
    assert "all:speculative" in built and "all:inference" in built


def test_function_words_are_dropped_so_an_anded_question_still_matches():
    built = papers._build_search_query("how should I cache LLM completions across requests?", "")
    for stopword in ("all:how", "all:should", "all:the"):
        assert stopword not in built
    assert "all:cache" in built and "all:completions" in built


def test_a_quoted_phrase_is_preserved_as_a_phrase():
    built = papers._build_search_query('"speculative decoding" transformer', "")
    assert 'all:"speculative decoding"' in built
    assert "all:transformer" in built


def test_fielded_arxiv_grammar_is_passed_through_untouched():
    """A caller who wrote arXiv syntax meant it; escaping it would break it."""
    original = 'ti:"attention is all you need" AND au:vaswani'
    assert papers._build_search_query(original, "") == original


def test_categories_restrict_the_query_and_work_without_one():
    with_query = papers._build_search_query("caching", "cs.DC, cs.DB")
    assert "cat:cs.DC OR cat:cs.DB" in with_query and "all:caching" in with_query
    assert papers._build_search_query("", "cs.DC") == "(cat:cs.DC)"


def test_an_empty_search_is_refused_rather_than_sent():
    with pytest.raises(papers.PaperError):
        papers._build_search_query("", "")


def test_the_or_fallback_differs_from_the_and_query():
    """search() widens to OR on an empty result set; the two must not be equal
    or the retry would repeat the failed request."""
    text = "speculative decoding inference"
    assert (papers._build_search_query(text, "", connector="AND")
            != papers._build_search_query(text, "", connector="OR"))


# ---------------------------------------------------------------- pages


@pytest.mark.parametrize("spec,total,expected", [
    ("1", 10, [1]),
    ("7-9", 10, [7, 8, 9]),
    ("1,4,9-11", 12, [1, 4, 9, 10, 11]),
    ("9-7", 10, [7, 8, 9]),          # reversed range is a typo, not an error
    ("3, 3, 3", 10, [3]),            # duplicates collapse
    ("8-99", 10, [8, 9, 10]),        # clamped to the document
])
def test_page_specs_parse_the_way_a_reader_writes_them(spec, total, expected):
    assert papers.parse_pages(spec, total) == expected


def test_a_page_spec_selecting_nothing_is_an_error_not_an_empty_read():
    with pytest.raises(papers.PaperError):
        papers.parse_pages("50-60", 10)


def test_an_unparseable_page_spec_names_the_bad_chunk():
    with pytest.raises(papers.PaperError, match="seven"):
        papers.parse_pages("seven", 10)


# ---------------------------------------------------------------- outline


def test_real_bookmarks_win_over_the_heading_scan():
    toc = [{"level": 1, "title": "Method", "page": 4}]
    outline = papers.build_outline(["1. Introduction\nbody"], toc)
    assert [s["source"] for s in outline] == ["bookmarks"]
    assert outline[0]["title"] == "Method"


def test_headings_are_detected_when_the_pdf_carries_no_bookmarks():
    """arXiv PDFs are usually produced without bookmarks, so this is the common
    path, not the fallback."""
    pages = ["1. Introduction\nsome text\n2.1 Draft Strategies\nmore text",
             "References\nbibliography"]
    outline = papers.build_outline(pages, [])
    titles = [s["title"] for s in outline]
    assert "1. Introduction" in titles
    assert "2.1 Draft Strategies" in titles
    assert "References" in titles
    assert all(s["source"] == "heading-scan" for s in outline)
    assert outline[1]["level"] == 2       # 2.1 is a subsection of 2


def test_a_section_is_found_by_number_by_name_and_by_position():
    outline = [
        {"title": "Introduction", "number": "", "page": 1, "level": 1},
        {"title": "Method", "number": "", "page": 2, "level": 1},
        {"title": "Draft Strategies", "number": "2.1", "page": 2, "level": 2},
        {"title": "Results", "number": "", "page": 3, "level": 1},
    ]
    assert papers._match_section(outline, "2.1") == 2          # by number
    assert papers._match_section(outline, "results") == 3      # by name
    assert papers._match_section(outline, "3") == 3            # positional: 3rd top level
    assert papers._match_section(outline, "nonexistent") is None


# ---------------------------------------------------------------- extraction


def test_a_pdf_round_trips_into_pages_text_and_an_outline(sample_pdf):
    extracted = papers.extract_pdf(sample_pdf)
    assert extracted["engine"] == "pymupdf"
    assert len(extracted["pages"]) == 3
    assert "speculative refresh batching" in extracted["pages"][0]
    assert [item["title"] for item in extracted["toc"]] == ["Introduction", "Method", "Results"]


def test_fetching_a_local_pdf_caches_it_without_touching_the_network(sample_pdf):
    meta = papers.fetch(path=str(sample_pdf))
    assert meta["ok"] and meta["kind"] == "file"
    assert meta["pages"] == 3 and meta["characters"] > 0
    assert meta["cached"] is False
    assert json.loads((papers.entry_dir(meta["key"]) / "outline.json").read_text())


def test_a_second_fetch_is_served_from_cache(sample_pdf):
    first = papers.fetch(path=str(sample_pdf))
    second = papers.fetch(path=str(sample_pdf))
    assert first["cached"] is False and second["cached"] is True
    assert second["key"] == first["key"]


def test_refresh_re_extracts_rather_than_serving_the_cache(sample_pdf):
    papers.fetch(path=str(sample_pdf))
    again = papers.fetch(path=str(sample_pdf), refresh=True)
    assert again["cached"] is False


def test_an_unrecognisable_source_says_what_it_expected():
    with pytest.raises(papers.PaperError, match="arXiv id"):
        papers.fetch(paper_id="this is not anything")


def test_fetch_without_any_source_is_refused():
    with pytest.raises(papers.PaperError):
        papers.fetch()


# ---------------------------------------------------------------- reading


def test_the_default_read_is_the_outline_because_it_is_cheapest(sample_pdf):
    result = papers.read(path=str(sample_pdf))
    assert result["mode"] == "outline"
    assert [s["title"] for s in result["sections"]] == ["Introduction", "Method", "Results"]
    assert "full" in result["next"]


def test_reading_the_full_text_returns_every_page(sample_pdf):
    result = papers.read(path=str(sample_pdf), mode="full", max_chars=0)
    assert result["truncated"] is False
    for expected in ("speculative refresh batching", "acceptance rate", "2.4x"):
        assert expected in result["text"]


def test_a_long_read_pages_through_offsets_without_losing_text(sample_pdf):
    whole = papers.read(path=str(sample_pdf), mode="full", max_chars=0)["text"]
    first = papers.read(path=str(sample_pdf), mode="full", max_chars=100)
    assert first["truncated"] is True and first["next_offset"] == 100
    second = papers.read(path=str(sample_pdf), mode="full", max_chars=100,
                         offset=first["next_offset"])
    assert whole.startswith(first["text"] + second["text"])


def test_reading_one_section_stops_at_the_next_section_of_equal_rank(sample_pdf):
    result = papers.read(path=str(sample_pdf), section="method", max_chars=0)
    assert result["section"] == "Method"
    assert "acceptance rate" in result["text"]
    assert "2.4x" not in result["text"]        # that belongs to Results


def test_a_section_that_does_not_exist_lists_the_ones_that_do(sample_pdf):
    with pytest.raises(papers.PaperError, match="Introduction"):
        papers.read(path=str(sample_pdf), section="nonexistent")


def test_reading_a_page_range_selects_only_those_pages(sample_pdf):
    result = papers.read(path=str(sample_pdf), pages="3", max_chars=0)
    assert result["selected_pages"] == [3]
    assert "2.4x" in result["text"] and "speculative refresh batching" not in result["text"]


def test_the_abstract_mode_warns_that_it_is_not_the_paper(sample_pdf):
    result = papers.read(path=str(sample_pdf), mode="abstract")
    assert "not the paper" in result["warning"]


# ---------------------------------------------------------------- grep


def test_grep_finds_a_claim_and_reports_its_page(sample_pdf):
    result = papers.grep(path=str(sample_pdf), pattern=r"2\.4x")
    assert result["hits"] == 1
    assert result["matches"][0]["page"] == 3
    assert "Throughput" in result["matches"][0]["context"]


def test_grep_is_case_insensitive_by_default_and_can_be_made_strict(sample_pdf):
    assert papers.grep(path=str(sample_pdf), pattern="THROUGHPUT")["hits"] == 1
    assert papers.grep(path=str(sample_pdf), pattern="THROUGHPUT",
                       ignore_case=False)["hits"] == 0


def test_an_invalid_regex_is_reported_not_raised_as_a_crash(sample_pdf):
    with pytest.raises(papers.PaperError, match="invalid regex"):
        papers.grep(path=str(sample_pdf), pattern="(unclosed")


def test_grep_without_a_pattern_is_refused(sample_pdf):
    with pytest.raises(papers.PaperError):
        papers.grep(path=str(sample_pdf), pattern="  ")


# ---------------------------------------------------------------- vision


def test_rendering_produces_real_png_bytes_and_writes_them(sample_pdf):
    result = papers.render(path=str(sample_pdf), pages="1,3", dpi=72)
    assert [item["page"] for item in result["rendered"]] == [1, 3]
    for item in result["rendered"]:
        assert item["png"].startswith(b"\x89PNG\r\n\x1a\n")
        assert item["width"] > 0 and item["height"] > 0


def test_rendering_refuses_more_pages_than_the_context_can_carry(tmp_path):
    """The cap is on pages actually rendered, so it needs a document long
    enough to exceed it - asking for nine pages of a three-page paper is a
    clamp, not an overrun."""
    document = pymupdf.open()
    for number in range(12):
        document.new_page().insert_text((72, 96), f"page {number + 1}", fontsize=14)
    long_pdf = tmp_path / "long.pdf"
    document.save(str(long_pdf))
    document.close()

    with pytest.raises(papers.PaperError, match="per-call limit"):
        papers.render(path=str(long_pdf), pages=f"1-{papers.MAX_RENDER_PAGES + 1}", dpi=72)
    assert len(papers.render(path=str(long_pdf), pages="1-3", dpi=72)["rendered"]) == 3


def test_rendering_can_target_a_directory_the_caller_names(sample_pdf, tmp_path):
    destination = tmp_path / "shots"
    result = papers.render(path=str(sample_pdf), pages="2", dpi=72, save_to=str(destination))
    written = list(destination.glob("*.png"))
    assert len(written) == 1
    assert result["saved_to"] == str(destination.resolve())


def test_figure_extraction_reports_honestly_when_there_are_no_rasters(sample_pdf):
    """A text-only PDF has no embedded images. Saying so beats an empty list
    the caller has to interpret."""
    result = papers.figures(path=str(sample_pdf))
    assert result["count"] == 0
    assert "vector" in result["note"]


# ---------------------------------------------------------------- disk


def test_download_writes_the_pdf_where_the_caller_asked(sample_pdf, tmp_path):
    destination = tmp_path / "library"
    result = papers.download(path=str(sample_pdf), dest=str(destination), with_text=True)
    assert len(result["written"]) == 2
    pdfs = list(destination.glob("*.pdf"))
    texts = list(destination.glob("*.txt"))
    assert len(pdfs) == 1 and len(texts) == 1
    assert pdfs[0].read_bytes().startswith(b"%PDF")
    assert "acceptance rate" in texts[0].read_text(encoding="utf-8")


def test_download_without_a_destination_is_refused(sample_pdf):
    with pytest.raises(papers.PaperError, match="dest"):
        papers.download(path=str(sample_pdf), dest="")


def test_a_custom_filename_is_honoured(sample_pdf, tmp_path):
    result = papers.download(path=str(sample_pdf), dest=str(tmp_path), filename="mine.pdf")
    assert result["written"][0].endswith("mine.pdf")


def test_listing_and_forgetting_the_cache(sample_pdf):
    papers.fetch(path=str(sample_pdf))
    listed = papers.cached()
    assert listed["count"] == 1

    dropped = papers.forget(path=str(sample_pdf))
    assert dropped["removed"] is True
    assert papers.cached()["count"] == 0

    again = papers.forget(path=str(sample_pdf))
    assert again["removed"] is False      # forgetting twice is not an error


# ---------------------------------------------------------------- html


def test_the_html_extractor_keeps_text_drops_chrome_and_lifts_latex():
    html = """
    <html><head><style>p { color: red }</style></head>
    <body><nav>skip me</nav>
    <p>The acceptance rate governs the speedup.</p>
    <math alttext="\\alpha + \\beta"><mi>a</mi></math>
    <img alt="Figure 1: throughput" src="x.png">
    <script>alert('no')</script>
    </body></html>
    """
    parser = papers._HTMLText()
    parser.feed(html)
    text = parser.text()
    assert "acceptance rate governs" in text
    assert "$\\alpha + \\beta$" in text
    assert "[figure: Figure 1: throughput" in text
    for dropped in ("skip me", "alert(", "color: red"):
        assert dropped not in text


def test_a_landing_page_yields_the_pdf_it_names():
    """Publishers serve HTML where a human copies the URL and name the real PDF
    in a citation_pdf_url meta tag."""
    html = '<meta name="citation_pdf_url" content="https://example.invalid/paper.pdf">'
    assert (papers._pdf_url_from_page(html, "https://example.invalid/forum")
            == "https://example.invalid/paper.pdf")

    relative = '<a href="/downloads/paper.pdf">PDF</a>'
    assert (papers._pdf_url_from_page(relative, "https://example.invalid/forum")
            == "https://example.invalid/downloads/paper.pdf")

    assert papers._pdf_url_from_page("<p>nothing here</p>", "https://example.invalid") is None


# ---------------------------------------------------------------- citation


def test_a_citation_is_pasteable_and_abbreviates_long_author_lists():
    line = papers.citation({
        "authors": ["A One", "B Two", "C Three", "D Four"],
        "published": "2022-03-30T17:27:09Z",
        "title": "Speculative Decoding",
        "arxiv_id": "2203.16487v6",
    })
    assert line == "A One et al. (2022). Speculative Decoding. 2203.16487v6"


def test_a_citation_survives_missing_metadata():
    line = papers.citation({"key": "url-abc123"})
    assert "unknown authors" in line and "url-abc123" in line


# ---------------------------------------------------------------- scanned PDFs


@pytest.fixture
def scanned_pdf(tmp_path):
    """A genuinely image-only PDF: pages rendered, then pasted back as pixels.

    This is what a scanned paper actually is - the glyphs are gone, so the text
    layer extracts as nothing at all.
    """
    def build(page_count: int, name: str = "scanned.pdf"):
        source = pymupdf.open()
        for number in range(page_count):
            page = source.new_page()
            page.insert_text((72, 96), f"Scanned page {number + 1}", fontsize=18)
        flattened = pymupdf.open()
        for page in source:
            pixmap = page.get_pixmap(dpi=60)
            blank = flattened.new_page(width=page.rect.width, height=page.rect.height)
            blank.insert_image(blank.rect, stream=pixmap.tobytes("png"))
        target = tmp_path / name
        flattened.save(str(target))
        flattened.close()
        source.close()
        return target
    return build


def test_a_scanned_page_counts_as_blank_even_though_it_is_not_an_empty_string(scanned_pdf):
    path = scanned_pdf(3)
    papers.fetch(path=str(path))
    _, _, _, page_texts = papers._load_entry("", "", str(path))
    assert papers.blank_pages(page_texts) == [1, 2, 3]


def test_a_scanned_paper_is_flagged_rather_than_reported_as_a_blank_paper(scanned_pdf):
    path = scanned_pdf(2)
    meta = papers.fetch(path=str(path))
    assert meta["characters"] == 0
    assert "scanned or image-only" in meta["extraction_warning"]


def test_every_read_mode_carries_the_extraction_warning_not_just_the_outline(scanned_pdf):
    """The bug this guards: mode='full' returned '[page 1]\n\n[page 2]\n' with
    no explanation, which reads as 'this paper is blank' rather than
    'this paper is pixels'."""
    path = scanned_pdf(2)
    assert "extraction_warning" in papers.read(path=str(path), mode="full")
    assert "extraction_warning" in papers.read(path=str(path), pages="1")


def test_grep_says_a_scan_is_unsearchable_instead_of_reporting_zero_hits(scanned_pdf):
    """Zero hits against a paper with no text layer is not evidence of absence,
    and a plain 0 invites exactly that reading."""
    path = scanned_pdf(2)
    result = papers.grep(path=str(path), pattern="Scanned")
    assert result["hits"] == 0
    assert result["searchable"] is False
    assert "not searchable" in result["warning"]


def test_grep_on_a_readable_paper_does_not_claim_to_be_unsearchable(sample_pdf):
    result = papers.grep(path=str(sample_pdf), pattern="acceptance")
    assert result["hits"] >= 1
    assert "searchable" not in result


def test_a_scanned_read_comes_back_as_images_through_the_tool(scanned_pdf):
    from icn import server

    path = scanned_pdf(3)
    result = server.paper(action="read", path=str(path), mode="full", dpi=60)
    summary, images = result[0], result[1:]
    assert summary["served_as"] == "images"
    assert summary["rendered_pages"] == [1, 2, 3]
    assert len(images) == 3
    assert all(image.data.startswith(b"\x89PNG\r\n\x1a\n") for image in images)


def test_a_long_scan_is_capped_and_says_how_to_ask_for_the_rest(scanned_pdf):
    from icn import server

    path = scanned_pdf(20, "long.pdf")
    summary = server.paper(action="read", path=str(path), mode="full", dpi=60)[0]
    assert len(summary["rendered_pages"]) == papers.MAX_RENDER_PAGES
    assert "9-16" in summary["more"]


def test_vision_can_be_declined_in_favour_of_the_honest_empty_result(scanned_pdf):
    from icn import server

    path = scanned_pdf(2)
    result = server.paper(action="read", path=str(path), mode="full", vision=False)
    assert isinstance(result, dict)
    assert "extraction_warning" in result


def test_a_readable_paper_is_never_turned_into_images(sample_pdf):
    """The vision fallback must trigger on an absent text layer only; a normal
    paper coming back as pictures would be a large silent regression."""
    from icn import server

    result = server.paper(action="read", path=str(sample_pdf), mode="full")
    assert isinstance(result, dict)
    assert "acceptance rate" in result["text"]
