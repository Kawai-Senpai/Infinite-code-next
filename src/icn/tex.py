"""LaTeX source handling for arXiv e-print submissions.

Why this exists: extracting a paper from its PDF is a reconstruction. Section
boundaries come from a heading heuristic that misreads a sentence starting with
a numeral as a heading, two-column layouts interleave, ligatures come back as
"fl" inside "flow", and every formula degrades into unspaced glyphs. None of
that is a bug that can be fixed - it is what reading a rendered document back
into structure costs.

arXiv also serves the thing the PDF was *made from*, at /e-print/<id>. For the
large majority of papers that is the original TeX. Section boundaries stop
being a guess because \\section{...} is literally in the file, math survives
because it was never rasterised, and the title and authors are readable even
when the metadata API refuses to answer.

Everything here is pure: bytes and strings in, structure out. The network and
the cache live in papers.py, so all of this is testable offline.
"""

from __future__ import annotations

import gzip
import io
import posixpath
import re
import tarfile
from typing import Any

# Extensions worth keeping out of a submission bundle. Everything else in a
# typical tarball is figures, which the PDF path already handles better.
TEXT_SUFFIXES = (".tex", ".bbl", ".bib", ".sty", ".cls", ".ltx")
MAX_MEMBER_BYTES = 12 * 1024 * 1024
MAX_MEMBERS = 400

# \section -> 1 keeps these comparable with the PDF outline's levels, so
# _match_section and the section-boundary walk work against either source.
_SECTION_LEVELS = {
    "part": 1, "chapter": 1, "section": 1,
    "subsection": 2, "subsubsection": 3, "paragraph": 4, "subparagraph": 5,
}
_SECTION_RE = re.compile(
    r"\\(part|chapter|section|subsection|subsubsection|paragraph|subparagraph)\s*\*?\s*(?=[\[{])"
)


class TexError(RuntimeError):
    """The e-print is not usable TeX - typically a PDF-only submission."""


def _decode(raw: bytes) -> str:
    """TeX predates the UTF-8 consensus; latin-1 is the usual second guess and
    cannot itself fail, so this never raises."""
    for encoding in ("utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def _wanted(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith(TEXT_SUFFIXES) and not lowered.startswith("__macosx")


def unpack_eprint(blob: bytes) -> dict[str, str]:
    """Turn an e-print payload into {filename: source}.

    arXiv serves three shapes from one endpoint and does not distinguish them
    in the content type: a tarball of the whole submission, a single gzipped
    .tex (the common 1990s case), or a bare PDF for submissions that never had
    source. The third is not an error worth hiding - it just means this paper
    has no TeX and the caller should stay on the PDF path.
    """
    if not blob:
        raise TexError("the e-print response was empty")
    if blob[:5] == b"%PDF-":
        raise TexError("this submission is PDF-only; arXiv has no TeX source for it")

    # Tarball first: a .tar.gz is also valid gzip, so testing gzip first would
    # hand back the tar header as if it were TeX.
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as archive:
            files: dict[str, str] = {}
            for member in archive.getmembers()[:MAX_MEMBERS]:
                if not member.isfile() or not _wanted(member.name):
                    continue
                if member.size > MAX_MEMBER_BYTES:
                    continue
                handle = archive.extractfile(member)
                if handle is not None:
                    files[member.name.lstrip("./")] = _decode(handle.read())
            if files:
                return files
            raise TexError("the submission tarball contains no .tex source")
    except tarfile.TarError:
        pass

    if blob[:2] == b"\x1f\x8b":
        try:
            plain = gzip.decompress(blob)
        except OSError as err:
            raise TexError(f"the e-print gzip stream is corrupt: {err}") from err
        if plain[:5] == b"%PDF-":
            raise TexError("this submission is PDF-only; arXiv has no TeX source for it")
        return {"main.tex": _decode(plain)}

    text = _decode(blob)
    if "\\documentclass" in text or "\\begin{document}" in text or "\\documentstyle" in text:
        return {"main.tex": text}
    raise TexError("the e-print payload is not TeX source")


def strip_comments(source: str) -> str:
    r"""Drop TeX comments, honouring the escaped \% that means a literal sign."""
    out = []
    for line in source.split("\n"):
        cut = None
        index = 0
        while index < len(line):
            if line[index] == "\\":
                index += 2
                continue
            if line[index] == "%":
                cut = index
                break
            index += 1
        out.append(line if cut is None else line[:cut])
    return "\n".join(out)


def pick_main(files: dict[str, str]) -> str:
    """Find the file that actually starts the document.

    A submission routinely carries a dozen .tex files - one per section, plus
    the journal's class examples. The main one is the one that opens the
    document environment; \\documentclass alone is not enough, because sample
    files shipped by a style package have it too.
    """
    if not files:
        raise TexError("no TeX files in this submission")

    scored: list[tuple[int, int, str]] = []
    for name, source in files.items():
        body = strip_comments(source)
        score = 0
        if "\\begin{document}" in body:
            score += 100
        if "\\documentclass" in body or "\\documentstyle" in body:
            score += 50
        if "\\title" in body:
            score += 10
        if _SECTION_RE.search(body):
            score += 5
        if not score:
            # No document evidence at all. The tiebreakers below must not run:
            # a lone notes.bib would score for sitting at the bundle root and
            # then be handed back as the main TeX file.
            continue
        # A file living at the root of the bundle beats one buried in a
        # subdirectory when both otherwise look like the entry point.
        if "/" not in name:
            score += 3
        if name.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower() in {
                "main", "paper", "article", "manuscript", "ms", "root"}:
            score += 20
        scored.append((score, len(body), name))

    if not scored:
        raise TexError("no TeX file in this submission opens a document")
    scored.sort(key=lambda item: (-item[0], -item[1]))
    return scored[0][2]


def _safe_join(*parts: str) -> str | None:
    """Join archive path parts, refusing anything that escapes the bundle.

    A submission is an untrusted tarball, so `\\input{../../etc/passwd}` is a
    thing that can appear in one. Nothing here writes to disk by that name, but
    resolving it would still let a crafted bundle pull in a file the caller did
    not intend to read.
    """
    pieces = []
    for part in parts:
        cleaned = part.strip().replace("\\", "/")
        if cleaned.startswith("/"):
            return None
        if cleaned:
            pieces.append(cleaned)
    if not pieces:
        return None
    candidate = posixpath.normpath(posixpath.join(*pieces))
    if candidate in {".", ".."} or candidate.startswith("../"):
        return None
    return candidate


def _resolve(files: dict[str, str], target: str, current: str = "") -> str | None:
    r"""Match an \input target against the bundle's real filenames.

    Resolution is relative to the *including* file first, which matters
    whenever a submission has sections/intro.tex and appendix/intro.tex: a
    global search by basename picks whichever happens to come first and
    silently splices in the wrong section.
    """
    target = target.strip().strip('"')
    if not target:
        return None

    bases = [posixpath.dirname(current), ""] if current else [""]
    for base in bases:
        for suffix in ("", ".tex"):
            candidate = _safe_join(base, target + suffix)
            if candidate and candidate in files:
                return candidate

    lowered = {name.lower(): name for name in files}
    for base in bases:
        for suffix in ("", ".tex"):
            candidate = _safe_join(base, target + suffix)
            if candidate and candidate.lower() in lowered:
                return lowered[candidate.lower()]

    # Last resort: a submission flattened on upload, where \input{sec/intro}
    # refers to a file that now sits at the root as intro.tex.
    tail = target.rsplit("/", 1)[-1]
    for candidate in (tail, tail + ".tex"):
        for name in files:
            if name.rsplit("/", 1)[-1].lower() == candidate.lower():
                return name
    return None


# \import{dir}{file} and friends take the directory as a separate argument, so
# a one-argument pattern silently drops the path and resolves nothing.
_INPUT_RE = re.compile(
    r"\\(?P<cmd>subimport|subinputfrom|subincludefrom|import|inputfrom|includefrom"
    r"|input|include|subfile)\s*\*?\s*\{(?P<first>[^{}]*)\}"
    r"(?:\s*\{(?P<second>[^{}]*)\})?"
)
_TWO_ARG = {"import", "inputfrom", "includefrom", "subimport", "subinputfrom",
            "subincludefrom"}


def flatten(files: dict[str, str], main: str, depth: int = 0,
            seen: frozenset[str] = frozenset()) -> str:
    r"""Inline \input/\include/\import so the document reads as one stream.

    Without this a multi-file submission yields an outline of two sections,
    because every real \section lives in a file the main one only references.
    The seen-set is not optional: a bundle where two files include each other
    is malformed but real, and would otherwise recurse until the stack ends.
    """
    if depth > 20 or main in seen:
        return ""
    source = strip_comments(files.get(main, ""))
    source = _expand_macros(source)
    seen = seen | {main}

    def substitute(match: re.Match[str]) -> str:
        command = match.group("cmd")
        first, second = match.group("first"), match.group("second")
        if command in _TWO_ARG and second is not None:
            base = posixpath.dirname(main) if command.startswith("sub") else ""
            joined = _safe_join(base, first, second)
            target = joined if joined else second
            resolved = _resolve(files, target, main)
        else:
            resolved = _resolve(files, first, main)
        if resolved is None:
            return ""
        return "\n" + flatten(files, resolved, depth + 1, seen) + "\n"

    return _INPUT_RE.sub(substitute, source)


_MACRO_DEF_RE = re.compile(
    r"\\(?:newcommand|renewcommand|providecommand|DeclareRobustCommand)\s*\*?"
    r"\s*(?:\{\\([A-Za-z@]+)\}|\\([A-Za-z@]+))\s*(?:\[(\d+)\])?(?:\[[^\]]*\])?\s*\{"
)
_DEF_RE = re.compile(r"\\(?:g)?def\s*\\([A-Za-z@]+)\s*\{")
MAX_MACRO_ROUNDS = 6


def collect_macros(source: str) -> dict[str, str]:
    r"""Zero-argument \newcommand/\def bodies, by name.

    Only the no-argument forms. A macro taking arguments needs real TeX
    expansion semantics to substitute correctly, and a half-done job on
    \newcommand{\vec}[1]{\mathbf{#1}} produces text that looks expanded and
    is wrong - worse than leaving the macro visible.
    """
    macros: dict[str, str] = {}
    for pattern in (_MACRO_DEF_RE, _DEF_RE):
        for match in pattern.finditer(source):
            groups = match.groups()
            name = groups[0] or (groups[1] if len(groups) > 1 else None)
            arity = groups[2] if len(groups) > 2 else None
            if not name or (arity and arity != "0"):
                continue
            body, _ = _balanced(source, match.end() - 1)
            if body and "#" not in body:
                macros.setdefault(name, body)
    return macros


def _expand_macros(source: str, macros: dict[str, str] | None = None) -> str:
    r"""Substitute zero-argument macros, repeatedly, until it settles.

    Papers routinely define \newcommand{\ourmethod}{SpecDec} and then write
    \section{\ourmethod{} in practice}. Without expansion that section's title
    reads " in practice", which is useless for finding it by name.
    """
    macros = collect_macros(source) if macros is None else macros
    if not macros:
        return source
    pattern = re.compile(
        r"\\(" + "|".join(re.escape(n) for n in sorted(macros, key=len, reverse=True))
        + r")(?![A-Za-z@])\s*(?:\{\})?")
    text = source
    for _ in range(MAX_MACRO_ROUNDS):
        expanded = pattern.sub(lambda m: macros.get(m.group(1), m.group(0)), text)
        if expanded == text:
            break
        text = expanded
    return text


def _balanced(source: str, start: int) -> tuple[str, int]:
    """Read one brace group starting at `start`, returning (body, end index).

    A regex cannot do this: section titles contain braces of their own, as in
    \\section{The $\\mathcal{O}(n)$ bound}, and a non-greedy [^}]* stops inside
    the nested group and truncates the title.
    """
    if start >= len(source) or source[start] != "{":
        return "", start
    depth = 0
    index = start
    while index < len(source):
        char = source[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start + 1:index], index + 1
        index += 1
    return source[start + 1:], len(source)


def _skip_optional(source: str, index: int) -> int:
    """Step over \\section[short form]{...}'s bracketed argument."""
    if index < len(source) and source[index] == "[":
        depth = 0
        while index < len(source):
            if source[index] == "[":
                depth += 1
            elif source[index] == "]":
                depth -= 1
                if depth == 0:
                    return index + 1
            index += 1
    return index


def detex(source: str) -> str:
    """Make TeX readable without pretending to typeset it.

    Deliberately conservative. Math is left in TeX form, because $O(n\\log n)$
    is precise and any attempt to flatten it to text loses the meaning; the
    reader here is a language model, which reads TeX math perfectly well.
    """
    text = source
    text = re.sub(r"\\begin\{(figure|table|tabular|thebibliography)\*?\}.*?"
                  r"\\end\{\1\*?\}", " ", text, flags=re.S)

    # Lift math out before the macro sweep below, or the sweep guts it: the
    # generic \[a-zA-Z]+ rule cannot tell \emph from \nabla, and stripping the
    # second turns "$\mathcal{F} = \int_M (R + |\nabla f|^2)$" into "$ = _M(R +
    # | f|^2)$" - which still looks like math and is silently wrong.
    math: list[str] = []

    def keep(match: re.Match[str]) -> str:
        math.append(match.group(0))
        return f"\x00MATH{len(math) - 1}\x00"

    text = re.sub(
        r"\$\$.*?\$\$|\\\[.*?\\\]|\$(?:\\.|[^$\\])*\$|"
        r"\\begin\{(equation|align|gather|multline|eqnarray|displaymath)\*?\}"
        r".*?\\end\{\1\*?\}",
        keep, text, flags=re.S)
    text = re.sub(r"\\(?:label|cite[a-z]*|ref|eqref|index|vspace|hspace|bibliography"
                  r"|bibliographystyle|newcommand|renewcommand|usepackage|documentclass)"
                  r"\s*(\[[^\]]*\])?\s*(\{[^{}]*\})*", " ", text)
    text = re.sub(r"\\(?:emph|textbf|textit|texttt|textsc|mbox|text)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:begin|end)\s*\{[^}]*\}", "\n", text)
    text = re.sub(r"\\\\", "\n", text)
    text = re.sub(r"\\[a-zA-Z@]+\s*", " ", text)
    text = text.replace("~", " ").replace("\\&", "&")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    text = re.sub(r"\x00MATH(\d+)\x00", lambda m: math[int(m.group(1))], text)
    return text.strip()


def _plain_title(raw: str) -> str:
    r"""Take the readable half of \texorpdfstring{$\mathcal{L}$}{L}.

    The author wrote the second argument precisely because the first does not
    survive outside math mode - it is the bookmark text, which is exactly what
    a section list wants.
    """
    title = raw.strip()
    marker = "\\texorpdfstring"
    while True:
        at = title.find(marker)
        if at < 0:
            return title
        index = at + len(marker)
        while index < len(title) and title[index] in " \t\r\n":
            index += 1
        _, after = _balanced(title, index)
        while after < len(title) and title[after] in " \t\r\n":
            after += 1
        second, end = _balanced(title, after)
        if end <= after:
            return title
        # Splice, rather than return the argument: the construct is usually one
        # phrase inside a longer heading, and returning it alone drops the rest
        # of the title ("L2" for "\texorpdfstring{$\mathcal{L}_2$}{L2} bounds").
        title = (title[:at] + second + title[end:]).strip()


def outline(source: str) -> list[dict[str, Any]]:
    r"""Every sectioning command in the flattened document, in order.

    Unlike the PDF heading scan this is exact: \section{Results} is a section
    because the author declared one, not because a line looked like a heading.
    Each entry carries the character span of its body, so a section read is a
    slice rather than a page range that overshoots.
    """
    found: list[dict[str, Any]] = []
    for match in _SECTION_RE.finditer(source):
        command = match.group(1)
        index = _skip_optional(source, match.end())
        title, after = _balanced(source, index)
        title = detex(_plain_title(title)).strip()
        if not title:
            continue
        found.append({
            "title": title,
            "level": _SECTION_LEVELS.get(command, 1),
            "command": command,
            "start": match.start(),
            "body_start": after,
        })

    for position, entry in enumerate(found):
        entry["end"] = found[position + 1]["start"] if position + 1 < len(found) else len(source)

    # Number only what the author left unnumbered-looking, and only within a
    # level, so "4.2" means the same thing it does in the rendered PDF.
    counters: list[int] = []
    for entry in found:
        level = entry["level"]
        if level > len(counters):
            counters.extend([0] * (level - len(counters)))
        del counters[level:]
        counters[level - 1] += 1
        entry["number"] = ".".join(str(c) for c in counters)
    return found


def title_and_authors(source: str) -> dict[str, Any]:
    """Pull \\title and \\author out of the preamble.

    This is the fallback that makes old papers usable at all: arXiv's metadata
    API answers "Rate exceeded." for pre-2007 ids far more readily than for
    modern ones, and the PDF's own title field is usually the arXiv stamp.
    """
    result: dict[str, Any] = {}
    match = re.search(r"\\title\s*\*?\s*(\[[^\]]*\])?\s*\{", source)
    if match:
        body, _ = _balanced(source, match.end() - 1)
        title = " ".join(detex(_plain_title(body)).split())
        if title:
            result["title"] = title

    authors: list[str] = []
    for match in re.finditer(r"\\author\s*(\[[^\]]*\])?\s*\{", source):
        body, _ = _balanced(source, match.end() - 1)
        cleaned = re.sub(r"\\(?:thanks|footnote|affil[a-z]*|inst|orcid)\s*\{[^{}]*\}", " ", body)
        cleaned = re.sub(r"\\(?:and|AND)\b", "|", cleaned)
        cleaned = re.sub(r"\$\^?\{?[^$]*\}?\$", " ", cleaned)   # affiliation markers
        for part in re.split(r"\||\\\\|,(?![^{]*\})", cleaned):
            name = " ".join(detex(part).split()).strip(" ,;")
            # Affiliation lines survive the split; a real byline is short and
            # has no institutional giveaway in it.
            if 2 <= len(name) <= 60 and not re.search(
                    r"(?i)univ|institut|laborator|depart|school|college|academy|@|\d", name):
                authors.append(name)
    if authors:
        seen: set[str] = set()
        result["authors"] = [a for a in authors if not (a in seen or seen.add(a))]

    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", source, flags=re.S)
    if match:
        abstract = " ".join(detex(match.group(1)).split())
        if abstract:
            result["abstract"] = abstract

    if not result.get("title"):
        result.update(_centered_frontmatter(source))
    return result


def _centered_frontmatter(source: str) -> dict[str, Any]:
    r"""Recover the front matter of a paper that never used \title.

    Pre-LaTeX2e submissions routinely typeset the title and byline as a run of
    \begin{center} blocks and have no \title, \author or abstract environment
    at all. There is no sectioning to recover in such a paper - that part is a
    real limit - but the title and authors are still plainly there, and
    returning nothing for them when they are readable is a choice, not a
    constraint.
    """
    blocks = [" ".join(detex(b).split())
              for b in re.findall(r"\\begin\{center\}(.*?)\\end\{center\}", source, flags=re.S)]
    blocks = [b for b in blocks if b]
    if not blocks:
        return {}

    result: dict[str, Any] = {"title": blocks[0], "frontmatter_source": "centered blocks"}
    authors: list[str] = []
    for block in blocks[1:]:
        if re.fullmatch(r"(?i)abstract", block):
            break
        if re.fullmatch(r"(?i)and", block):
            continue
        if re.search(r"(?i)univ|institut|laborator|depart|school|college|academy|@|\d", block):
            continue    # an affiliation line, not a byline
        if 2 <= len(block) <= 60:
            authors.append(block)
    if authors:
        result["authors"] = authors

    match = re.search(r"\\begin\{center\}\s*(?i:abstract)\s*\\end\{center\}(.*?)"
                      r"(?:\\newpage|\\section|$)", source, flags=re.S)
    if match:
        abstract = " ".join(detex(match.group(1)).split())
        if abstract:
            result["abstract"] = abstract
    return result
