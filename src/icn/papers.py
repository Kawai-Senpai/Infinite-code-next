"""Paper ingestion: search arXiv, fetch any PDF, read all of it, render pages.

The abstract is not the paper. A tool that stops at the abstract forces the same
rediscovery later, so this module is built around the full text: fetch once into
a content-addressed cache, then serve slices of it - a page range, a section, a
regex hit with context - so a forty-page paper can inform a decision without
arriving in one indigestible block.

Sources, in the order they are tried:
    arXiv id      2401.12345, math/0309136, with or without a version suffix
    any http(s)   a direct PDF URL, or a landing page that names one
    local path    a .pdf already on disk

Layout under storage_root()/cache/papers/<key>/:
    meta.json     identity, title, authors, abstract, source, fetch time
    paper.pdf     the bytes as downloaded
    text.json     per-page extracted text, the canonical read surface
    outline.json  sections resolved to pages
    pages/        rendered page PNGs, produced on demand

The cache is rebuildable by construction: deleting it costs a re-download and
nothing else, which is why it lives under cache/ rather than data/.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from . import paths
from . import tex

USER_AGENT = "infinite-code-next/0.1 (+https://ranitbhowmick.com)"
ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
ARX = "{http://arxiv.org/schemas/atom}"
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"

HTTP_TIMEOUT = 45.0
MAX_BYTES = 120 * 1024 * 1024        # a paper past this is a mirror error, not a paper
COURTESY_SECONDS = 3.0               # arXiv asks for one request at a time
DEFAULT_READ_CHARS = 24_000
MAX_RENDER_PAGES = 8                 # per call; images are expensive in context
DEFAULT_DPI = 140

_ARXIV_LOCK = threading.Lock()
_LAST_ARXIV_CALL = 0.0

# 2401.12345, optionally versioned, plus the pre-2007 archive/YYMMNNN form.
_NEW_ID = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b")
_OLD_ID = re.compile(r"\b([a-z-]+(?:\.[A-Za-z][A-Za-z-]+)?/\d{7})(v\d+)?\b")


class PaperError(RuntimeError):
    """Anything the caller can act on: a bad id, a dead URL, a missing engine."""


# ---------------------------------------------------------------- identity


def normalize_arxiv_id(raw: str) -> str | None:
    """Pull a bare arXiv id out of an id, a URL, or a citation string.

    The version suffix is preserved when present, because v1 and v3 of a paper
    can make different claims and a cache that conflates them is worse than no
    cache at all.
    """
    if not raw:
        return None
    text = re.sub(r"(?i)^arxiv[:\s]+", "", raw.strip())
    for pattern in (_NEW_ID, _OLD_ID):
        found = pattern.search(text)
        if found:
            return found.group(1) + (found.group(2) or "")
    return None


def _key_for(kind: str, ident: str) -> str:
    if kind == "arxiv":
        return "arxiv-" + ident.replace("/", "_")
    digest = hashlib.sha1(ident.encode("utf-8", "replace")).hexdigest()[:16]
    return f"{kind}-{digest}"


def papers_dir() -> Path:
    return paths.storage_root() / "cache" / "papers"


def entry_dir(key: str) -> Path:
    return papers_dir() / key


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- http


def _http_get(url: str, accept: str | None = None, retries: int = 3) -> tuple[bytes, str, str]:
    """GET with a real User-Agent and backoff. Returns (body, final_url, content_type).

    arXiv rejects the stdlib default agent outright, and answers 429/503 with a
    Retry-After under load, so both are handled here rather than at each caller.
    """
    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    last: Exception | None = None

    for attempt in range(retries):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
                declared = response.headers.get("Content-Length")
                if declared and declared.isdigit() and int(declared) > MAX_BYTES:
                    raise PaperError(f"{url} declares {declared} bytes, over the {MAX_BYTES} limit")
                body = response.read(MAX_BYTES + 1)
                if len(body) > MAX_BYTES:
                    raise PaperError(f"{url} exceeded the {MAX_BYTES} byte limit")
                return body, response.geturl(), response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as err:
            last = err
            if err.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                header = err.headers.get("Retry-After") if err.headers else None
                if (header or "").replace(".", "", 1).isdigit():
                    delay = float(header)
                elif err.code == 429:
                    # arXiv's penalty window is measured in seconds, not
                    # milliseconds: retrying after 2s just earns another 429.
                    delay = max(COURTESY_SECONDS, 5.0) * (attempt + 1)
                else:
                    delay = 2.0 * (attempt + 1)
                time.sleep(min(delay, 30.0))
                continue
            raise PaperError(f"{url} returned HTTP {err.code} {err.reason}") from err
        except urllib.error.URLError as err:
            last = err
            if attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise PaperError(f"{url} could not be reached: {err.reason}") from err
        except (TimeoutError, OSError) as err:
            # A socket timeout mid-body raises TimeoutError, not URLError, so it
            # would otherwise escape the retry loop and abort the whole fetch on
            # a slow arXiv day. Big PDFs hit this far more often than small ones.
            last = err
            if attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise PaperError(f"{url} timed out after {retries} attempts: {err}") from err

    raise PaperError(f"{url} failed after {retries} attempts: {last}")


def _arxiv_get(url: str) -> bytes:
    """arXiv API access, serialised with the courtesy delay their terms ask for."""
    global _LAST_ARXIV_CALL
    with _ARXIV_LOCK:
        elapsed = time.monotonic() - _LAST_ARXIV_CALL
        if elapsed < COURTESY_SECONDS:
            time.sleep(COURTESY_SECONDS - elapsed)
        try:
            body, _, _ = _http_get(url, accept="application/atom+xml")
        finally:
            _LAST_ARXIV_CALL = time.monotonic()
    return body


# ---------------------------------------------------------------- search


# Function words that make an ANDed query return nothing. Deliberately short:
# every entry here is a word no paper is ever indexed *about*.
_STOPWORDS = frozenset("""
a an the of for to in on at is are was were be been being do does did how what
why which when where who whom this that these those it its as by with without
from into over under can could should would will shall may might must i we you
they he she my our your their and or not no if then than so such use using used
best better way ways new novel paper study approach method
""".split())


def _query_terms(text: str) -> tuple[list[str], list[str]]:
    """Split a plain query into quoted phrases and bare content words."""
    phrases = [p.strip() for p in re.findall(r'"([^"]+)"', text) if p.strip()]
    remainder = re.sub(r'"[^"]*"', " ", text)
    words = [w for w in re.findall(r"[A-Za-z0-9][\w.+-]*", remainder) if len(w) > 1]
    content = [w for w in words if w.lower() not in _STOPWORDS]
    return phrases, (content or words)


def _build_search_query(query: str, category: str, connector: str = "AND") -> str:
    """Build an arXiv search_query.

    A plain query becomes its content terms joined by `connector`, with any
    quoted phrase preserved as a phrase. Quoting the *whole* query is what a
    naive implementation does and it is measurably wrong: arXiv treats it as an
    exact phrase, so a six-word question matches zero papers. ANDing the terms
    of that same question matches hundreds.

    Fielded queries (ti:, au:, abs:, cat:, AND/OR) are passed through untouched,
    because a caller who wrote arXiv grammar meant it.
    """
    text = (query or "").strip()
    fielded = re.search(r"\b(ti|au|abs|co|jr|cat|rn|id|all)\s*:", text)

    if not text:
        clause = ""
    elif fielded:
        clause = text
    else:
        phrases, words = _query_terms(text)
        parts = [f'all:"{p}"' for p in phrases] + [f"all:{w}" for w in words]
        if not parts:
            raise PaperError(f"no searchable terms in {query!r}")
        clause = f" {connector} ".join(parts)
        if len(parts) > 1:
            clause = f"({clause})"

    cats = [c.strip() for c in (category or "").replace(",", " ").split() if c.strip()]
    if cats:
        joined = " OR ".join(f"cat:{c}" for c in cats)
        return f"{clause} AND ({joined})" if clause else f"({joined})"
    if not clause:
        raise PaperError("search needs a query, a category, or both")
    return clause


def _entry_to_dict(entry: ET.Element) -> dict[str, Any]:
    def text_of(tag: str) -> str:
        node = entry.find(ATOM + tag)
        return " ".join(node.text.split()) if node is not None and node.text else ""

    raw_id = text_of("id")
    arxiv_id = normalize_arxiv_id(raw_id) or raw_id

    links: dict[str, str] = {}
    for link in entry.findall(ATOM + "link"):
        href = link.get("href") or ""
        if link.get("title") == "pdf" or link.get("type") == "application/pdf":
            links["pdf"] = href
        elif link.get("rel") == "alternate":
            links["abs"] = href

    def arx_text(tag: str) -> str:
        node = entry.find(ARX + tag)
        return " ".join(node.text.split()) if node is not None and node.text else ""

    primary = entry.find(ARX + "primary_category")
    authors = []
    for author in entry.findall(ATOM + "author"):
        name = author.find(ATOM + "name")
        if name is not None and name.text:
            authors.append(" ".join(name.text.split()))

    return {
        "arxiv_id": arxiv_id,
        "title": text_of("title"),
        "authors": authors,
        "abstract": text_of("summary"),
        "published": text_of("published"),
        "updated": text_of("updated"),
        "categories": [c.get("term", "") for c in entry.findall(ATOM + "category") if c.get("term")],
        "primary_category": primary.get("term") if primary is not None else "",
        "comment": arx_text("comment"),
        "journal_ref": arx_text("journal_ref"),
        "doi": arx_text("doi"),
        "abs_url": links.get("abs", f"https://arxiv.org/abs/{arxiv_id}"),
        "pdf_url": links.get("pdf", f"https://arxiv.org/pdf/{arxiv_id}"),
    }


def search(query: str = "", category: str = "", max_results: int = 10,
           sort: str = "relevance", start: int = 0,
           abstract_chars: int = 900) -> dict[str, Any]:
    """Search arXiv and return real metadata for every hit.

    Abstracts are truncated by default because a search is for choosing what to
    read, not for reading. fetch() gets the whole paper.
    """
    sort_by = {
        "relevance": "relevance",
        "recent": "submittedDate",
        "submitted": "submittedDate",
        "updated": "lastUpdatedDate",
    }.get((sort or "relevance").lower(), "relevance")

    def run(search_query: str) -> tuple[list[dict[str, Any]], int | None]:
        params = {
            "search_query": search_query,
            "start": max(0, int(start)),
            "max_results": max(1, min(int(max_results or 10), 100)),
            "sortBy": sort_by,
            "sortOrder": "descending",
        }
        try:
            root = ET.fromstring(_arxiv_get(f"{ARXIV_API}?{urllib.parse.urlencode(params)}"))
        except ET.ParseError as err:
            raise PaperError(f"arXiv returned unparseable Atom: {err}") from err

        total_node = root.find(OPENSEARCH + "totalResults")
        found = []
        for entry in root.findall(ATOM + "entry"):
            item = _entry_to_dict(entry)
            if abstract_chars and len(item["abstract"]) > abstract_chars:
                item["abstract"] = item["abstract"][:abstract_chars].rstrip() + " [...]"
                item["abstract_truncated"] = True
            found.append(item)
        total = int(total_node.text) if total_node is not None and total_node.text else None
        return found, total

    # Precision first, recall second. ANDed terms are the right answer when the
    # query has one, and silently returning nothing is the worst outcome here,
    # so a miss widens to OR rather than reporting an empty shelf.
    strict = _build_search_query(query, category, connector="AND")
    results, total = run(strict)
    used, strategy = strict, "all terms (AND)"

    if not results:
        loose = _build_search_query(query, category, connector="OR")
        if loose != strict:
            results, total = run(loose)
            used, strategy = loose, "any term (OR), after the AND query matched nothing"

    return {
        "ok": True,
        "query": used,
        "strategy": strategy,
        "sort": sort_by,
        "total_available": total,
        "returned": len(results),
        "start": max(0, int(start)),
        "results": results,
        "next": "paper(action='fetch', paper_id=...) downloads and extracts the full text",
    }


def arxiv_metadata(arxiv_id: str) -> dict[str, Any]:
    """Metadata for one known id, via the same Atom feed."""
    # The slash in a pre-2007 id (cond-mat/0309136) must stay a slash: percent
    # encoding it makes arXiv answer 429/empty for every old-style paper, which
    # is why they all came back titleless while new-style ids worked.
    ident = urllib.parse.quote(arxiv_id, safe="/.")
    url = f"{ARXIV_API}?id_list={ident}&max_results=1"
    try:
        root = ET.fromstring(_arxiv_get(url))
    except ET.ParseError as err:
        raise PaperError(f"arXiv returned unparseable Atom for {arxiv_id}: {err}") from err
    entries = root.findall(ATOM + "entry")
    if not entries:
        raise PaperError(f"arXiv has no entry for {arxiv_id}")
    item = _entry_to_dict(entries[0])
    if not item["title"]:
        raise PaperError(f"arXiv returned an empty entry for {arxiv_id}")
    return item


# ---------------------------------------------------------------- latex source


ARXIV_EPRINT = "https://arxiv.org/e-print/"


def eprint_source(arxiv_id: str) -> dict[str, str]:
    """Download and unpack one submission's original TeX.

    Measured, not assumed: /e-print/ answers 200 for ids whose metadata query
    arXiv refuses with "Rate exceeded.", so this is the more reliable of the
    two endpoints for old papers, not merely a nicer one.
    """
    ident = urllib.parse.quote(arxiv_id, safe="/.")
    blob, _, _ = _http_get(ARXIV_EPRINT + ident)
    try:
        return tex.unpack_eprint(blob)
    except tex.TexError as err:
        raise PaperError(f"no usable TeX source for {arxiv_id}: {err}") from err


def latex_document(arxiv_id: str) -> dict[str, Any]:
    """The submission as one flattened TeX stream, with its outline."""
    files = eprint_source(arxiv_id)
    try:
        main = tex.pick_main(files)
        source = tex.flatten(files, main)
    except tex.TexError as err:
        raise PaperError(f"could not assemble the TeX for {arxiv_id}: {err}") from err
    return {
        "main": main,
        "files": sorted(files),
        "source": source,
        "outline": tex.outline(source),
        **tex.title_and_authors(source),
    }


# ---------------------------------------------------------------- html


class _HTMLText(HTMLParser):
    """Minimal readable-text extractor.

    Deliberately not a browser. It drops chrome, keeps block structure, and
    lifts LaTeX out of MathML alttext, which is the one thing that makes an
    arXiv HTML render more useful than the PDF on a formula-heavy paper.
    """

    SKIP = {"script", "style", "noscript", "nav", "header", "footer", "svg", "form", "button"}
    BLOCK = {"p", "div", "section", "article", "br", "li", "tr", "h1", "h2", "h3",
             "h4", "h5", "h6", "table", "blockquote", "pre", "figcaption"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self._skip_depth += 1
            return
        mapping = dict(attrs)
        if tag == "math" and mapping.get("alttext"):
            self.parts.append(" $" + str(mapping["alttext"]) + "$ ")
        elif tag == "img" and mapping.get("alt"):
            self.parts.append("\n[figure: " + str(mapping["alt"]) + "]\n")
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self.parts.append(data)

    def text(self) -> str:
        joined = "".join(self.parts)
        joined = re.sub(r"[ \t\r\f\v]+", " ", joined)
        joined = re.sub(r"\n\s*\n\s*\n+", "\n\n", joined)
        return joined.strip()


def html_fulltext(arxiv_id: str) -> tuple[str, str]:
    """Full text from arXiv's own HTML, falling back to ar5iv.

    Returns (text, source_url). Recent submissions have native HTML; older ones
    exist only on ar5iv, and some exist as neither - which is why every caller
    treats this as an enhancement over the PDF, never a replacement for it.
    """
    bare = re.sub(r"v\d+$", "", arxiv_id)
    problems = []
    for url in (f"https://arxiv.org/html/{arxiv_id}",
                f"https://arxiv.org/html/{bare}",
                f"https://ar5iv.labs.arxiv.org/html/{bare}"):
        try:
            body, final, ctype = _http_get(url, accept="text/html", retries=2)
        except PaperError as err:
            problems.append(str(err))
            continue
        if "html" not in ctype.lower():
            problems.append(f"{url} served {ctype or 'an unknown type'}")
            continue
        parser = _HTMLText()
        parser.feed(body.decode("utf-8", "replace"))
        text = parser.text()
        if len(text) > 4000:      # a stub or an error page is far shorter
            return text, final
        problems.append(f"{url} yielded only {len(text)} characters")
    raise PaperError("no HTML rendering available: " + "; ".join(problems))


def _pdf_url_from_page(html: str, base_url: str) -> str | None:
    """Find the PDF a landing page points at.

    Publishers serve HTML at the URL a human copies, and name the real PDF in a
    citation_pdf_url meta tag. Checking that first is what makes an OpenReview
    or ACL Anthology link work without the caller hunting for the PDF href.
    """
    for pattern in (r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)',
                    r'href=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']',
                    r'href=["\']([^"\']*/pdf[^"\']*)["\']'):
        found = re.search(pattern, html, re.I)
        if found:
            return urllib.parse.urljoin(base_url, found.group(1))
    return None


# ---------------------------------------------------------------- pdf


def _load_pymupdf():
    try:
        import pymupdf
        return pymupdf
    except ImportError:
        try:
            import fitz
            return fitz
        except ImportError:
            return None


def extract_pdf(pdf_path: Path) -> dict[str, Any]:
    """Per-page text plus whatever outline the file carries.

    PyMuPDF is preferred because it is the only engine here that also renders,
    so text and pixels come from one parse of one file. pypdf is a text-only
    fallback: a machine without PyMuPDF still reads papers, just without vision.
    """
    engine = _load_pymupdf()
    if engine is not None:
        with engine.open(pdf_path) as document:
            pages = [page.get_text("text") for page in document]
            try:
                toc = [{"level": level, "title": " ".join(str(title).split()), "page": page}
                       for level, title, page in (document.get_toc() or [])]
            except Exception:
                toc = []                     # a malformed outline must not lose the text
            meta = {k: str(v) for k, v in (document.metadata or {}).items() if v}
        return {"engine": "pymupdf", "pages": pages, "toc": toc, "pdf_metadata": meta}

    try:
        from pypdf import PdfReader
    except ImportError as err:
        raise PaperError(
            "no PDF engine available. Install one with: pip install pymupdf "
            "(pymupdf also enables page rendering; pypdf gives text only)"
        ) from err

    reader = PdfReader(str(pdf_path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    meta = {k.lstrip("/"): str(v) for k, v in dict(reader.metadata or {}).items()}
    return {"engine": "pypdf", "pages": pages, "toc": [], "pdf_metadata": meta}


_HEADING_NUMBERED = re.compile(r"^\s{0,6}((?:\d{1,2})(?:\.\d{1,2}){0,3})\.?\s+([A-Z][^\n]{2,90})\s*$")
_HEADING_NAMED = re.compile(
    r"^\s{0,6}((?:[IVX]+\.\s+)?(?:Abstract|Introduction|Background|Related Work|"
    r"Preliminaries|Method(?:s|ology)?|Approach|Model|Architecture|Experiments?|"
    r"Evaluation|Results|Analysis|Ablations?|Discussion|Limitations|Future Work|"
    r"Conclusions?|Acknowledg(?:e)?ments|References|Bibliography|"
    r"Appendix(?:\s+[A-Z])?))\s*$", re.I)


def build_outline(pages: list[str], toc: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Section map, from real bookmarks when the PDF has them, from heading
    shapes when it does not.

    arXiv PDFs are usually produced without bookmarks, so the heuristic path is
    the common one here, not the fallback.
    """
    if toc:
        return [{"title": item["title"], "page": item["page"], "level": item["level"],
                 "number": "", "source": "bookmarks"}
                for item in toc if item.get("title")]

    sections: list[dict[str, Any]] = []
    for page_number, text in enumerate(pages, start=1):
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or len(stripped) > 100:
                continue
            numbered = _HEADING_NUMBERED.match(stripped)
            if numbered:
                sections.append({"title": stripped, "number": numbered.group(1),
                                 "page": page_number,
                                 "level": numbered.group(1).count(".") + 1,
                                 "source": "heading-scan"})
            elif _HEADING_NAMED.match(stripped):
                sections.append({"title": stripped, "number": "", "page": page_number,
                                 "level": 1, "source": "heading-scan"})

    deduped: list[dict[str, Any]] = []
    for section in sections:
        if deduped and section["title"].lower() == deduped[-1]["title"].lower():
            continue
        deduped.append(section)
    return deduped


# ---------------------------------------------------------------- fetch


def _resolve_source(paper_id: str, url: str, path: str) -> tuple[str, str, str]:
    """Work out what the caller means. Returns (kind, identifier, cache key).

    One free-text argument is accepted in any of the three slots because an
    agent that has just read a citation has a string, not a taxonomy.
    """
    for candidate in (paper_id, url, path):
        text = (candidate or "").strip()
        if not text:
            continue

        local = Path(text).expanduser()
        if text.lower().endswith(".pdf") and local.exists():
            resolved = str(local.resolve())
            return "file", resolved, _key_for("file", resolved.lower())

        if text.lower().startswith(("http://", "https://")):
            found = normalize_arxiv_id(text) if "arxiv.org" in text.lower() else None
            if found:
                return "arxiv", found, _key_for("arxiv", found)
            return "url", text, _key_for("url", text)

        found = normalize_arxiv_id(text)
        if found:
            return "arxiv", found, _key_for("arxiv", found)

        if local.exists():
            resolved = str(local.resolve())
            return "file", resolved, _key_for("file", resolved.lower())

        raise PaperError(
            f"could not tell what {text!r} refers to. Expected an arXiv id "
            "(2401.12345), an http(s) URL, or a path to an existing .pdf"
        )
    raise PaperError("fetch needs one of paper_id, url, or path")


def _download_pdf(kind: str, ident: str, target: Path) -> dict[str, Any]:
    """Put the PDF bytes at target. Returns what was actually retrieved."""
    if kind == "file":
        source = Path(ident)
        if not source.exists():
            raise PaperError(f"{ident} no longer exists")
        shutil.copyfile(source, target)
        return {"source_url": None, "source_path": ident, "bytes": target.stat().st_size}

    if kind == "arxiv":
        bare = re.sub(r"v\d+$", "", ident)
        attempts = [f"https://arxiv.org/pdf/{ident}", f"https://arxiv.org/pdf/{bare}"]
    else:
        attempts = [ident]

    problems = []
    for candidate in attempts:
        try:
            body, final, ctype = _http_get(candidate, accept="application/pdf")
        except PaperError as err:
            problems.append(str(err))
            continue

        if not body.startswith(b"%PDF") and "pdf" not in ctype.lower():
            # A landing page rather than the file. Follow the PDF it names.
            nested = _pdf_url_from_page(body.decode("utf-8", "replace"), final)
            if not nested:
                problems.append(f"{candidate} served {ctype or 'non-PDF content'} and names no PDF")
                continue
            body, final, ctype = _http_get(nested, accept="application/pdf")

        if not body.startswith(b"%PDF"):
            problems.append(f"{final} did not return a PDF (content-type {ctype or 'unknown'})")
            continue

        target.write_bytes(body)
        return {"source_url": final, "source_path": None, "bytes": len(body)}

    raise PaperError("could not download a PDF: " + "; ".join(problems))


def fetch(paper_id: str = "", url: str = "", path: str = "", refresh: bool = False,
          with_html: bool = False, with_latex: bool = False) -> dict[str, Any]:
    """Resolve, download, extract, and cache one paper. Idempotent.

    Returns identity plus the outline and page count, not the text: knowing the
    paper is 34 pages with a section 4 called Method is what tells the caller
    which read() to make next.
    """
    kind, ident, key = _resolve_source(paper_id, url, path)
    directory = entry_dir(key)
    meta_path = directory / "meta.json"
    text_path = directory / "text.json"
    pdf_path = directory / "paper.pdf"

    if meta_path.exists() and text_path.exists() and not refresh:
        meta = _read_json(meta_path)
        meta["cached"] = True
        return {"ok": True, **meta}

    directory.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {"key": key, "kind": kind, "identifier": ident}
    if kind == "arxiv":
        try:
            meta.update(arxiv_metadata(ident))
        except PaperError as err:
            # Metadata is a nicety; the PDF is the point. Do not lose the paper
            # because the Atom feed hiccuped.
            meta["metadata_warning"] = str(err)

    retrieved = _download_pdf(kind, ident, pdf_path)
    meta.update(retrieved)

    extracted = extract_pdf(pdf_path)
    pages = extracted["pages"]
    outline = build_outline(pages, extracted["toc"])

    _write_json(text_path, {"engine": extracted["engine"], "pages": pages})
    _write_json(directory / "outline.json", outline)

    characters = sum(len(p) for p in pages)
    meta.update({
        "pages": len(pages),
        "characters": characters,
        "engine": extracted["engine"],
        "pdf_metadata": extracted["pdf_metadata"],
        "sections": len(outline),
        "cache_dir": str(directory),
        "pdf_path": str(pdf_path),
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "can_render": _load_pymupdf() is not None,
    })

    if not meta.get("title"):
        candidate = (extracted["pdf_metadata"].get("title") or "").strip()
        # Old arXiv PDFs carry the sidebar stamp ("arXiv:hep-th/9301001v1 1 Jan
        # 1993") as their document title. That is an id, not a title, and
        # passing it off as one is worse than admitting we have none.
        if candidate and not re.match(r"(?i)^arxiv[:\s]", candidate):
            meta["title"] = candidate

    if characters < 200 * max(1, len(pages)) // 10:
        meta["extraction_warning"] = (
            f"only {characters} characters across {len(pages)} pages. This is likely a "
            "scanned or image-only PDF; use action='render' to read it visually."
        )

    # The PDF path failing and the metadata API refusing are the same papers:
    # old submissions. TeX answers both at once - a real title, real authors,
    # and section boundaries that are declared rather than guessed - so it is
    # pulled automatically exactly when the cheap path came back thin.
    degraded = bool(meta.get("extraction_warning")) or not meta.get("title") or len(outline) < 2
    if kind == "arxiv" and (with_latex or degraded):
        try:
            document = ensure_latex(directory, {**meta, "kind": kind, "arxiv_id": ident})
            meta["latex_sections"] = len(document["outline"])
            meta["latex_source"] = ARXIV_EPRINT + ident
            for field in ("title", "authors", "abstract"):
                if not meta.get(field) and document.get(field):
                    meta[field] = document[field]
                    meta.setdefault("recovered_from_latex", []).append(field)
            if len(document["outline"]) >= 2:
                # Declared sections beat a heading scan outright - never by
                # count. The heading scan *over*-detects: on Perelman it called
                # 55 things sections where the author declared 15, so "more
                # sections wins" picks the guesswork every time it is worst.
                # Keep the PDF outline under its own key rather than dropping
                # it, because it is what maps sections onto page numbers.
                meta["outline_source"] = "latex"
                meta["pdf_sections"] = len(outline)
                meta["sections"] = len(document["outline"])
        except PaperError as err:
            meta["latex_warning"] = str(err)

    if with_html and kind == "arxiv":
        try:
            html_text, html_url = html_fulltext(ident)
            (directory / "fulltext.html.txt").write_text(html_text, encoding="utf-8")
            meta["html_source"] = html_url
            meta["html_characters"] = len(html_text)
        except PaperError as err:
            meta["html_warning"] = str(err)

    _write_json(meta_path, meta)
    meta["cached"] = False
    return {"ok": True, **meta}


def ensure_latex(directory: Path, meta: dict[str, Any]) -> dict[str, Any]:
    """The paper's TeX, downloaded once and cached beside the PDF.

    Lazy on purpose. The e-print tarball is a second download of the same
    paper, so paying for it on every fetch would double the cost of the common
    case where the PDF extracted cleanly. It is fetched when the PDF path came
    back degraded, or when a caller explicitly asks to read the source.
    """
    source_path = directory / "source.tex"
    outline_path = directory / "latex_outline.json"
    if source_path.exists() and outline_path.exists():
        source = source_path.read_text(encoding="utf-8")
        # Re-derive title/authors rather than returning only source+outline: on
        # a cache hit the caller still needs them, and a fetch(refresh=True)
        # that reuses the cached TeX would otherwise silently lose the very
        # metadata the TeX was fetched to recover.
        return {"source": source, "outline": _read_json(outline_path),
                "cached": True, **tex.title_and_authors(source)}

    if meta.get("kind") != "arxiv":
        raise PaperError("TeX source is only available for arXiv papers")

    document = latex_document(meta.get("arxiv_id") or meta.get("identifier", ""))
    source_path.write_text(document["source"], encoding="utf-8")
    _write_json(outline_path, document["outline"])
    return {**document, "cached": False}


def _load_entry(paper_id: str, url: str, path: str) -> tuple[str, Path, dict[str, Any], list[str]]:
    """Cache accessor that fetches on a miss, so read/grep/render never fail
    merely because the caller skipped fetch()."""
    _, _, key = _resolve_source(paper_id, url, path)
    directory = entry_dir(key)
    if not (directory / "text.json").exists():
        fetch(paper_id=paper_id, url=url, path=path)
    meta = _read_json(directory / "meta.json")
    pages = _read_json(directory / "text.json")["pages"]
    return key, directory, meta, pages


# ---------------------------------------------------------------- read


def parse_pages(spec: str, total: int) -> list[int]:
    """Parse '1,3,7-12' into one-based page numbers, clamped to the document."""
    if not spec or not spec.strip():
        return []
    wanted: list[int] = []
    for chunk in spec.replace(" ", "").split(","):
        if not chunk:
            continue
        if "-" in chunk:
            start_text, _, end_text = chunk.partition("-")
            try:
                start, end = int(start_text), int(end_text or total)
            except ValueError as err:
                raise PaperError(f"unparseable page range {chunk!r}") from err
            if start > end:
                start, end = end, start
            wanted.extend(range(start, end + 1))
        else:
            try:
                wanted.append(int(chunk))
            except ValueError as err:
                raise PaperError(f"unparseable page number {chunk!r}") from err

    seen: list[int] = []
    for number in wanted:
        if 1 <= number <= total and number not in seen:
            seen.append(number)
    if not seen:
        raise PaperError(f"page spec {spec!r} selects nothing in a {total} page document")
    return seen


def _match_section(outline: list[dict[str, Any]], wanted: str) -> int | None:
    """Find a section by number ('4', '4.2'), by name ('method'), or by position.

    Plenty of papers typeset section numbers in a way that never reaches the
    text layer, so an outline of thirty correctly-detected headings can carry no
    numbers at all. When that happens a caller asking for '4' means the fourth
    section, and refusing them on a technicality helps nobody.
    """
    target = wanted.strip().lower()
    if not target:
        return None

    for index, section in enumerate(outline):
        if (section.get("number") or "").lower() == target:
            return index

    for index, section in enumerate(outline):
        if target in section["title"].lower():
            return index

    if re.fullmatch(r"\d{1,2}", target):
        top_level = [i for i, s in enumerate(outline) if s.get("level", 1) == 1]
        ordinal = int(target)
        if 1 <= ordinal <= len(top_level):
            return top_level[ordinal - 1]
    return None


def _clip(text: str, max_chars: int, offset: int) -> dict[str, Any]:
    total = len(text)
    start = max(0, min(offset, total))
    if max_chars <= 0:
        return {"text": text[start:], "truncated": False, "returned_chars": total - start}
    piece = text[start:start + max_chars]
    truncated = start + len(piece) < total
    result = {"text": piece, "truncated": truncated, "returned_chars": len(piece),
              "total_chars": total}
    if truncated:
        result["next_offset"] = start + len(piece)
        result["hint"] = f"call again with offset={start + len(piece)} for the next slice"
    return result


def blank_pages(page_texts: list[str], wanted: list[int] | None = None) -> list[int]:
    """Which of these pages carry no readable text.

    A page of a scanned paper extracts as whitespace, so "did this read return
    anything" cannot be answered by checking for an empty string alone.
    """
    numbers = wanted if wanted else range(1, len(page_texts) + 1)
    return [n for n in numbers
            if 1 <= n <= len(page_texts) and len(page_texts[n - 1].strip()) < 20]


def read(paper_id: str = "", url: str = "", path: str = "", mode: str = "outline",
         pages: str = "", section: str = "", max_chars: int = DEFAULT_READ_CHARS,
         offset: int = 0) -> dict[str, Any]:
    """Read the cached paper: an outline, a page range, a section, or all of it.

    Defaults to the outline because that is the cheapest thing that tells the
    caller what to ask for next. Nothing here re-downloads.
    """
    key, directory, meta, page_texts = _load_entry(paper_id, url, path)
    outline = _read_json(directory / "outline.json") if (directory / "outline.json").exists() else []
    total = len(page_texts)
    head = {"ok": True, "key": key, "title": meta.get("title", ""),
            "arxiv_id": meta.get("arxiv_id", ""), "pages": total,
            "characters": meta.get("characters", 0)}
    # Every mode carries this, not just the outline. A read that returns
    # "[page 1]\n\n[page 2]\n" with no explanation reads as "this paper is
    # blank" when the truth is "this paper is pixels".
    if meta.get("extraction_warning"):
        head["extraction_warning"] = meta["extraction_warning"]

    if pages:
        wanted = parse_pages(pages, total)
        joined = "\n\n".join(f"[page {n}]\n{page_texts[n - 1]}" for n in wanted)
        return {**head, "mode": "pages", "selected_pages": wanted, **_clip(joined, max_chars, offset)}

    # A section read is served from the TeX whenever the TeX is the better
    # witness: its boundaries are declared, not inferred, so it cannot overshoot
    # into the next section the way a page-granular answer must.
    if (mode or "").lower() == "latex" or (section and meta.get("outline_source") == "latex"):
        document = ensure_latex(directory, meta)
        tex_outline = document["outline"]
        source = document["source"]
        if section:
            index = _match_section(tex_outline, section)
            if index is None:
                titles = [s["title"] for s in tex_outline][:40]
                raise PaperError(f"no section matching {section!r}. Detected: {titles}")
            entry = tex_outline[index]
            level = entry.get("level", 1)
            end = len(source)
            for later in tex_outline[index + 1:]:
                if later.get("level", 1) <= level:
                    end = later["start"]
                    break
            body = tex.detex(source[entry["body_start"]:end])
            return {**head, "mode": "latex", "source": "latex", "boundaries": "declared",
                    "section": entry["title"], "section_number": entry.get("number", ""),
                    **_clip(body, max_chars, offset)}
        if not section and not pages:
            return {**head, "mode": "latex", "source": "latex",
                    "sections": [{"number": s.get("number", ""), "title": s["title"],
                                  "level": s.get("level", 1)} for s in tex_outline],
                    **_clip(tex.detex(source), max_chars, offset)}

    if section:
        if not outline:
            raise PaperError("this paper has no detected sections; use pages= or mode='full'")
        index = _match_section(outline, section)
        if index is None:
            titles = [s["title"] for s in outline][:40]
            raise PaperError(f"no section matching {section!r}. Detected: {titles}")
        # A section ends where the next section of the same or higher rank
        # begins, not at the next outline entry: stopping at the first
        # subsection would return one page of a twelve-page Results.
        start_page = outline[index]["page"]
        level = outline[index].get("level", 1)
        end_page = total
        successor = None
        for later in outline[index + 1:]:
            if later.get("level", 1) <= level:
                successor = later
                end_page = later["page"]
                break
        end_page = max(start_page, min(end_page, total))
        parts = [f"[page {n}]\n{page_texts[n - 1]}"
                 for n in range(start_page, end_page + 1)]
        # The next section usually starts partway down the page this one ends
        # on, so that page is kept - but cut at its heading where the heading
        # is findable in the text layer, rather than handing back a slab of
        # the following section.
        if successor is not None and len(parts) > 1:
            tail = parts[-1]
            marker, _, body = tail.partition("\n")
            cut = body.find(successor["title"])
            if cut > 0:
                parts[-1] = marker + "\n" + body[:cut].rstrip()
            elif cut == 0:
                parts.pop()
                end_page -= 1
        joined = "\n\n".join(parts)
        return {**head, "mode": "section", "section": outline[index]["title"],
                "section_pages": [start_page, end_page],
                "note": "section boundaries come from a heading scan and may overshoot by a page",
                **_clip(joined, max_chars, offset)}

    normalized = (mode or "outline").lower()

    if normalized == "full":
        joined = "\n\n".join(f"[page {n}]\n{text}" for n, text in enumerate(page_texts, start=1))
        return {**head, "mode": "full", **_clip(joined, max_chars, offset)}

    if normalized == "html":
        html_file = directory / "fulltext.html.txt"
        if not html_file.exists():
            arxiv_id = meta.get("arxiv_id") or meta.get("identifier", "")
            if meta.get("kind") != "arxiv":
                raise PaperError("HTML full text is only available for arXiv papers")
            text, source = html_fulltext(arxiv_id)
            html_file.write_text(text, encoding="utf-8")
            meta["html_source"] = source
            _write_json(directory / "meta.json", meta)
        text = html_file.read_text(encoding="utf-8")
        return {**head, "mode": "html", "html_source": meta.get("html_source", ""),
                **_clip(text, max_chars, offset)}

    if normalized == "abstract":
        abstract = meta.get("abstract") or ""
        if not abstract and page_texts:
            first = page_texts[0]
            found = re.search(r"(?is)abstract\b(.{200,2500}?)(?:\n\s*\n|introduction\b)", first)
            abstract = " ".join(found.group(1).split()) if found else first[:1500]
        return {**head, "mode": "abstract", "text": abstract,
                "warning": "the abstract is a pointer, not the paper. mode='full' reads all of it"}

    return {
        **head,
        "mode": "outline",
        "engine": meta.get("engine"),
        "sections": outline,
        "abstract": (meta.get("abstract") or "")[:1200],
        "extraction_warning": meta.get("extraction_warning"),
        "next": ("read with mode='full' for everything, pages='7-12' for a range, "
                 "section='4' or section='method' for one section, or action='grep' "
                 "to search inside it"),
    }


def grep(paper_id: str = "", url: str = "", path: str = "", pattern: str = "",
         ignore_case: bool = True, context: int = 320,
         max_hits: int = 40) -> dict[str, Any]:
    """Regex search inside the cached full text, with page-anchored context.

    This is how a specific question gets answered - what batch size, which
    baseline, does it mention a failure mode - without reading the whole paper.
    """
    if not pattern.strip():
        raise PaperError("grep needs a pattern")
    key, _, meta, page_texts = _load_entry(paper_id, url, path)
    try:
        compiled = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as err:
        raise PaperError(f"invalid regex {pattern!r}: {err}") from err

    hits = []
    for number, text in enumerate(page_texts, start=1):
        for found in compiled.finditer(text):
            start = max(0, found.start() - context // 2)
            end = min(len(text), found.end() + context // 2)
            hits.append({
                "page": number,
                "match": found.group(0)[:200],
                "context": " ".join(text[start:end].split()),
            })
            if len(hits) >= max_hits:
                break
        if len(hits) >= max_hits:
            break

    result = {"ok": True, "key": key, "title": meta.get("title", ""), "pattern": pattern,
              "hits": len(hits), "truncated": len(hits) >= max_hits, "matches": hits}
    if not hits and len(blank_pages(page_texts)) == len(page_texts) and page_texts:
        # Zero hits against a paper with no text layer is not evidence of
        # absence, and reporting it as a plain 0 invites exactly that reading.
        result["searchable"] = False
        result["warning"] = (
            "this paper has no text layer, so a text search cannot answer "
            "anything about it - zero hits here means 'not searchable', not "
            "'not present'. Use action='render' to read the pages as images."
        )
    return result


# ---------------------------------------------------------------- vision


def render(paper_id: str = "", url: str = "", path: str = "", pages: str = "1",
           dpi: int = DEFAULT_DPI, save_to: str = "") -> dict[str, Any]:
    """Rasterise pages to PNG so they can actually be looked at.

    This is the answer to a figure, a table, an architecture diagram, or a
    scanned PDF whose text layer is empty - everything the extracted text drops
    on the floor. Returns PNG bytes; the caller wraps them for the transport.
    """
    engine = _load_pymupdf()
    if engine is None:
        raise PaperError("page rendering needs PyMuPDF. Install it with: pip install pymupdf")

    key, directory, meta, page_texts = _load_entry(paper_id, url, path)
    total = len(page_texts)
    wanted = parse_pages(pages or "1", total)
    if len(wanted) > MAX_RENDER_PAGES:
        raise PaperError(
            f"{len(wanted)} pages requested; the per-call limit is {MAX_RENDER_PAGES} "
            "because images are expensive in context. Ask for fewer, or call again."
        )

    dpi = max(50, min(int(dpi or DEFAULT_DPI), 400))
    destination = Path(save_to).expanduser().resolve() if save_to else (directory / "pages")
    destination.mkdir(parents=True, exist_ok=True)

    rendered = []
    with engine.open(directory / "paper.pdf") as document:
        for number in wanted:
            pixmap = document[number - 1].get_pixmap(dpi=dpi)
            png = pixmap.tobytes("png")
            out = destination / f"p{number:04d}_{dpi}dpi.png"
            out.write_bytes(png)
            rendered.append({"page": number, "png": png, "file": str(out),
                             "width": pixmap.width, "height": pixmap.height,
                             "bytes": len(png)})

    return {"ok": True, "key": key, "title": meta.get("title", ""),
            "dpi": dpi, "saved_to": str(destination), "rendered": rendered}


def figures(paper_id: str = "", url: str = "", path: str = "", pages: str = "",
            min_width: int = 120, min_height: int = 120,
            save_to: str = "", max_images: int = 12) -> dict[str, Any]:
    """Pull the embedded raster images out of a paper.

    Distinct from render(): this extracts the figure as the author embedded it,
    at its own resolution, rather than a picture of the page around it. Vector
    figures are invisible to this - use render() for those.
    """
    engine = _load_pymupdf()
    if engine is None:
        raise PaperError("figure extraction needs PyMuPDF. Install it with: pip install pymupdf")

    key, directory, meta, page_texts = _load_entry(paper_id, url, path)
    total = len(page_texts)
    wanted = parse_pages(pages, total) if pages else list(range(1, total + 1))
    destination = Path(save_to).expanduser().resolve() if save_to else (directory / "figures")
    destination.mkdir(parents=True, exist_ok=True)

    found = []
    seen: set[int] = set()
    with engine.open(directory / "paper.pdf") as document:
        for number in wanted:
            for entry in document[number - 1].get_images(full=True):
                xref = entry[0]
                if xref in seen:
                    continue                 # one figure repeated across pages
                seen.add(xref)
                try:
                    raw = document.extract_image(xref)
                except Exception:
                    continue                 # a broken xref must not abort the rest
                if raw["width"] < min_width or raw["height"] < min_height:
                    continue                 # rules, logos, and math glyphs
                out = destination / f"p{number:04d}_x{xref}.{raw['ext']}"
                out.write_bytes(raw["image"])
                found.append({"page": number, "xref": xref, "file": str(out),
                              "width": raw["width"], "height": raw["height"],
                              "ext": raw["ext"], "bytes": len(raw["image"])})
                if len(found) >= max_images:
                    break
            if len(found) >= max_images:
                break

    return {"ok": True, "key": key, "title": meta.get("title", ""),
            "saved_to": str(destination), "count": len(found), "figures": found,
            "note": "vector figures are not embedded rasters; use action='render' to see those"}


# ---------------------------------------------------------------- disk


def download(paper_id: str = "", url: str = "", path: str = "", dest: str = "",
             filename: str = "", with_text: bool = False) -> dict[str, Any]:
    """Copy the PDF out of the cache into a directory the caller names."""
    if not dest.strip():
        raise PaperError("download needs dest, the directory to write into")
    key, directory, meta, _ = _load_entry(paper_id, url, path)

    target_dir = Path(dest).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    if filename.strip():
        stem = Path(filename).stem
    else:
        title = re.sub(r"[^\w\s-]", "", meta.get("title", "") or key).strip()
        title = re.sub(r"\s+", "-", title)[:80]
        identifier = meta.get("arxiv_id") or key
        stem = f"{identifier}-{title}".strip("-") if title else identifier
        stem = re.sub(r"[^\w.-]", "_", stem)

    pdf_target = target_dir / f"{stem}.pdf"
    shutil.copyfile(directory / "paper.pdf", pdf_target)
    written = [str(pdf_target)]

    if with_text:
        pages = _read_json(directory / "text.json")["pages"]
        text_target = target_dir / f"{stem}.txt"
        text_target.write_text(
            "\n\n".join(f"[page {n}]\n{t}" for n, t in enumerate(pages, start=1)),
            encoding="utf-8")
        written.append(str(text_target))

    return {"ok": True, "key": key, "title": meta.get("title", ""),
            "dest": str(target_dir), "written": written}


def cached(limit: int = 50) -> dict[str, Any]:
    """What is already on disk, newest first."""
    root = papers_dir()
    if not root.exists():
        return {"ok": True, "cache_dir": str(root), "count": 0, "papers": []}

    entries = []
    for directory in root.iterdir():
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = _read_json(meta_path)
        except (OSError, json.JSONDecodeError):
            continue
        entries.append({
            "key": meta.get("key", directory.name),
            "title": meta.get("title", ""),
            "arxiv_id": meta.get("arxiv_id", ""),
            "pages": meta.get("pages"),
            "characters": meta.get("characters"),
            "fetched_at": meta.get("fetched_at", ""),
            "cache_dir": str(directory),
        })

    entries.sort(key=lambda item: item.get("fetched_at", ""), reverse=True)
    return {"ok": True, "cache_dir": str(root), "count": len(entries),
            "papers": entries[:max(1, limit)]}


def forget(paper_id: str = "", url: str = "", path: str = "") -> dict[str, Any]:
    """Drop one paper from the cache. The bytes are re-downloadable by design."""
    _, _, key = _resolve_source(paper_id, url, path)
    directory = entry_dir(key)
    if not directory.exists():
        return {"ok": True, "key": key, "removed": False, "note": "was not cached"}
    shutil.rmtree(directory)
    return {"ok": True, "key": key, "removed": True, "cache_dir": str(directory)}


def citation(meta: dict[str, Any]) -> str:
    """A one-line citation an agent can paste into a memory or a comment."""
    authors = meta.get("authors") or []
    if len(authors) > 3:
        who = f"{authors[0]} et al."
    elif authors:
        who = ", ".join(authors)
    else:
        who = "unknown authors"
    year = (meta.get("published") or "")[:4]
    identifier = meta.get("arxiv_id") or meta.get("source_url") or meta.get("key", "")
    title = meta.get("title") or "untitled"
    return f"{who} ({year}). {title}. {identifier}"
