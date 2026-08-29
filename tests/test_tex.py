"""LaTeX e-print handling.

Every case here is drawn from a real failure seen against real arXiv papers,
not from imagined TeX. The comments name the paper where that shape bit.
"""

from __future__ import annotations

import gzip
import io
import tarfile

import pytest

from icn import tex


def _tarball(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, body in files.items():
            raw = body.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    return buffer.getvalue()


# ---------------------------------------------------------------- unpacking


def test_a_submission_tarball_yields_its_tex_files():
    blob = _tarball({"main.tex": "\\documentclass{article}", "fig1.png": "not tex",
                     "sections/intro.tex": "hello"})
    files = tex.unpack_eprint(blob)
    assert set(files) == {"main.tex", "sections/intro.tex"}


def test_a_single_gzipped_tex_is_the_1990s_submission_shape():
    """hep-th/9301001 is one gzipped .tex, not a tarball."""
    blob = gzip.compress(b"\\documentstyle[12pt]{article}\\begin{document}hi\\end{document}")
    assert "documentstyle" in tex.unpack_eprint(blob)["main.tex"]


def test_a_targz_is_not_mistaken_for_a_gzipped_tex():
    """A tarball is also valid gzip, so testing gzip first hands back the tar
    header as if it were TeX."""
    files = tex.unpack_eprint(_tarball({"main.tex": "\\begin{document}real\\end{document}"}))
    assert files["main.tex"].startswith("\\begin{document}")


def test_a_pdf_only_submission_says_so_instead_of_returning_garbage():
    with pytest.raises(tex.TexError, match="PDF-only"):
        tex.unpack_eprint(b"%PDF-1.5\nnot tex at all")
    with pytest.raises(tex.TexError, match="PDF-only"):
        tex.unpack_eprint(gzip.compress(b"%PDF-1.5\nstill a pdf"))


def test_an_empty_or_unrecognisable_payload_is_an_error_not_an_empty_paper():
    with pytest.raises(tex.TexError):
        tex.unpack_eprint(b"")
    with pytest.raises(tex.TexError):
        tex.unpack_eprint(b"just some prose with no tex markers in it at all")


# ---------------------------------------------------------------- comments


def test_comments_are_dropped_but_an_escaped_percent_survives():
    source = "keep this % drop this\n50\\% of runs\n% whole line\n"
    stripped = tex.strip_comments(source)
    assert "drop this" not in stripped
    assert "50\\% of runs" in stripped
    assert "keep this" in stripped


# ---------------------------------------------------------------- main file


def test_the_main_file_is_the_one_that_opens_the_document():
    files = {
        "sample.tex": "\\documentclass{aps}\n% a style package's example file",
        "paper.tex": "\\documentclass{article}\\begin{document}\\section{A}\\end{document}",
    }
    assert tex.pick_main(files) == "paper.tex"


def test_a_bundle_with_no_document_is_reported_not_guessed():
    with pytest.raises(tex.TexError):
        tex.pick_main({"notes.bib": "@article{x}"})


# ---------------------------------------------------------------- flattening


def test_input_files_are_inlined_so_their_sections_are_seen():
    """The bug this guards: a multi-file submission outlines as zero sections
    because every \\section lives in a file main.tex only references."""
    files = {
        "main.tex": "\\begin{document}\\input{sections/intro}\\input{sections/method}\\end{document}",
        "sections/intro.tex": "\\section{Introduction}text",
        "sections/method.tex": "\\section{Method}more",
    }
    flat = tex.flatten(files, "main.tex")
    assert [s["title"] for s in tex.outline(flat)] == ["Introduction", "Method"]


def test_an_include_resolves_relative_to_the_including_file():
    """Two directories each holding intro.tex: resolving by basename alone
    splices in whichever the dict happens to yield first."""
    files = {
        "main.tex": "\\input{body/chapter}",
        "body/chapter.tex": "\\input{intro}",
        "body/intro.tex": "\\section{Right}",
        "intro.tex": "\\section{Wrong}",
    }
    assert [s["title"] for s in tex.outline(tex.flatten(files, "main.tex"))] == ["Right"]


def test_the_import_package_two_argument_form_is_resolved():
    files = {"main.tex": "\\import{parts/}{body}", "parts/body.tex": "\\section{Imported}"}
    assert [s["title"] for s in tex.outline(tex.flatten(files, "main.tex"))] == ["Imported"]


def test_a_mutually_including_bundle_terminates():
    """Malformed but real; without the seen-set this recurses until the stack ends."""
    files = {"a.tex": "\\section{A}\\input{b}", "b.tex": "\\section{B}\\input{a}"}
    assert "A" in tex.flatten(files, "a.tex")


def test_an_include_escaping_the_bundle_is_refused():
    files = {"main.tex": "\\input{../../../etc/passwd}\\section{Only}"}
    flat = tex.flatten(files, "main.tex")
    assert [s["title"] for s in tex.outline(flat)] == ["Only"]


# ---------------------------------------------------------------- macros


def test_a_macro_defined_by_the_author_is_expanded_in_a_section_title():
    """Papers define \\newcommand{\\ourmethod}{SpecDec} then write
    \\section{\\ourmethod{} in practice}; unexpanded that title reads
    ' in practice', which cannot be found by name."""
    source = ("\\newcommand{\\ourmethod}{SpecDec}\n"
              "\\section{\\ourmethod{} in practice}")
    assert tex.outline(tex.flatten({"m.tex": source}, "m.tex"))[0]["title"] == \
        "SpecDec in practice"


def test_a_macro_taking_arguments_is_left_alone():
    """Half-expanding \\vec{x} produces text that looks right and is wrong."""
    assert "#1" not in tex.collect_macros("\\newcommand{\\vec}[1]{\\mathbf{#1}}").get("vec", "")
    assert "vec" not in tex.collect_macros("\\newcommand{\\vec}[1]{\\mathbf{#1}}")


def test_texorpdfstring_titles_use_the_readable_half():
    source = "\\section{\\texorpdfstring{$\\mathcal{L}_2$}{L2} bounds}"
    assert tex.outline(source)[0]["title"] == "L2 bounds"


# ---------------------------------------------------------------- outline


def test_section_numbers_follow_the_hierarchy_not_a_flat_counter():
    source = ("\\section{One}\\subsection{First}\\subsection{Second}"
              "\\section{Two}\\subsection{Only}")
    numbered = [(s["number"], s["title"]) for s in tex.outline(source)]
    assert numbered == [("1", "One"), ("1.1", "First"), ("1.2", "Second"),
                        ("2", "Two"), ("2.1", "Only")]


def test_a_title_containing_braces_is_not_truncated():
    """A non-greedy [^}]* stops inside the nested group and cuts the title."""
    source = "\\section{The $\\mathcal{O}(n)$ bound}"
    assert "bound" in tex.outline(source)[0]["title"]


def test_a_starred_section_and_an_optional_short_title_are_both_handled():
    source = "\\section*{Acknowledgements}\\section[Short]{The full long title}"
    assert [s["title"] for s in tex.outline(source)] == \
        ["Acknowledgements", "The full long title"]


def test_each_section_body_ends_where_the_next_same_level_section_starts():
    source = "\\section{A}alpha\\subsection{A1}beta\\section{B}gamma"
    entries = tex.outline(source)
    body = source[entries[0]["body_start"]:entries[2]["start"]]
    assert "alpha" in body and "beta" in body and "gamma" not in body


# ---------------------------------------------------------------- detex


def test_math_survives_detex_because_the_reader_reads_tex_math():
    r"""The regression this guards: the generic \[a-zA-Z]+ macro sweep cannot
    tell \emph from \nabla, and turned Perelman's functional into
    '$ = _M(R + | f|^2)$' - which still looks like math and is silently wrong."""
    out = tex.detex(r"Consider $\mathcal{F}=\int_M{(R+|\nabla f|^2)e^{-f}dV}$ now.")
    assert "\\nabla" in out and "\\int_M" in out


def test_display_math_and_equation_environments_are_kept_whole():
    assert "\\alpha" in tex.detex(r"\begin{equation}\alpha = \beta\end{equation}")
    assert "\\gamma" in tex.detex(r"\[ \gamma^2 \]")


def test_citations_and_labels_are_dropped_but_their_sentence_is_not():
    out = tex.detex("Prior work~\\cite{smith2020} shows\\label{sec:x} this.")
    assert "smith2020" not in out and "Prior work" in out and "this." in out


def test_text_style_commands_unwrap_to_their_content():
    assert "important" in tex.detex("\\textbf{important}")


# ---------------------------------------------------------------- frontmatter


def test_title_and_authors_come_from_the_preamble():
    source = ("\\title{A Survey of Things}\\author{Ada Lovelace \\and Alan Turing}"
              "\\begin{abstract}We survey things.\\end{abstract}")
    found = tex.title_and_authors(source)
    assert found["title"] == "A Survey of Things"
    assert found["authors"] == ["Ada Lovelace", "Alan Turing"]
    assert found["abstract"] == "We survey things."


def test_affiliations_are_not_mistaken_for_authors():
    source = "\\title{T}\\author{Ada Lovelace \\and Institute for Advanced Study}"
    assert tex.title_and_authors(source)["authors"] == ["Ada Lovelace"]


def test_a_1993_paper_with_no_title_command_still_yields_its_frontmatter():
    """hep-th/9301001 exactly: no \\title, no \\author, no abstract environment -
    the front matter is a run of centered blocks. There is genuinely no
    sectioning to recover, but the title and byline are plainly readable."""
    source = (
        "\\documentstyle[12pt]{article}\\begin{document}\n"
        "\\begin{center} Gonihedric String and Asymptotic Freedom\\end{center}\n"
        "\\begin{center}G.K.Savvidy\\end{center}\n"
        "\\begin{center}Institut fur Theoretische Physik, Frankfurt\\end{center}\n"
        "\\begin{center}and\\end{center}\n"
        "\\begin{center}K.G.Savvidy\\end{center}\n"
        "\\begin{center}Abstract\\end{center}\n"
        "Few natural basic principles allow the extension.\n\\newpage\n"
    )
    found = tex.title_and_authors(source)
    assert found["title"] == "Gonihedric String and Asymptotic Freedom"
    assert found["authors"] == ["G.K.Savvidy", "K.G.Savvidy"]
    assert "Few natural basic principles" in found["abstract"]


def test_a_paper_with_a_real_title_ignores_the_centered_fallback():
    source = "\\title{Declared}\\begin{center}Some centered banner\\end{center}"
    assert tex.title_and_authors(source)["title"] == "Declared"
