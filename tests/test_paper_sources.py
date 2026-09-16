"""Literature search beyond arXiv: alphaXiv, OpenAlex, bioRxiv.

Offline: the HTTP layer is replaced with recorded response shapes, so these
assert how requests are built and how answers are normalised, not whether the
services are up today.
"""

from __future__ import annotations

import json
import urllib.parse

import pytest

from icn import papers

ALPHAXIV_HITS = [
    {
        "paperId": "2603.03251",
        "title": "Speculative  Speculative Decoding",
        "abstract": "Autoregressive decoding is bottlenecked by its sequential nature.",
        "publicationDate": "2026-03-03T18:41:32.000Z",
        "votes": 220,
        "snippets": [{"pageNumber": 1, "snippet": "Speculative\nSpeculative Decoding"}],
    },
    {"paperId": "2211.17192", "title": "Fast Inference", "abstract": "", "votes": 3, "snippets": []},
]

OPENALEX_PAGE = {
    "meta": {"count": 26014},
    "results": [
        {
            "id": "https://openalex.org/W4405717632",
            "doi": "https://doi.org/10.1109/tmc.2024.3513457",
            "title": "EdgeLLM: Fast On-Device LLM Inference",
            "publication_date": "2024-12-23",
            "cited_by_count": 47,
            "abstract_inverted_index": {"decoding": [2], "Speculative": [0, 3], "fast": [1]},
            "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
            "best_oa_location": None,
            "primary_location": {"landing_page_url": "https://ieeexplore.ieee.org/x", "pdf_url": None},
        },
        {
            "id": "https://openalex.org/W1",
            "doi": "https://doi.org/10.48550/arxiv.2211.17192",
            "title": "Fast Inference from Transformers via Speculative Decoding",
            "publication_date": "2022-11-30",
            "cited_by_count": 900,
            "abstract_inverted_index": None,
            "authorships": [],
            "best_oa_location": {"pdf_url": "https://arxiv.org/pdf/2211.17192",
                                 "landing_page_url": "https://arxiv.org/abs/2211.17192"},
            "primary_location": {},
        },
    ],
}


@pytest.fixture
def http(monkeypatch):
    """Capture every URL requested and answer with the queued JSON body."""
    calls: list[str] = []
    answers: list[object] = []

    def fake_get(url, accept=None, retries=3):
        calls.append(url)
        return json.dumps(answers.pop(0)).encode("utf-8"), url, "application/json"

    monkeypatch.setattr(papers, "_http_get", fake_get)
    return calls, answers


def _params(url: str) -> dict[str, str]:
    return {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlparse(url).query).items()}


def test_unknown_source_is_refused_with_the_valid_list():
    with pytest.raises(papers.PaperError, match="alphaxiv_semantic"):
        papers.search("anything", source="scholar")


def test_alphaxiv_keyword_search_normalises_hits_and_keeps_snippets(http):
    calls, answers = http
    answers.append(ALPHAXIV_HITS)

    result = papers.search("speculative decoding", source="alphaxiv", max_results=1, sort="popular")

    assert "/search/v2/paper/discover/keyword" in calls[0]
    assert _params(calls[0]) == {"q": "speculative decoding", "prioritize": "popular"}
    assert result["returned"] == 1
    hit = result["results"][0]
    assert hit["arxiv_id"] == "2603.03251"
    assert hit["title"] == "Speculative Speculative Decoding"
    assert hit["snippets"] == [{"page": 1, "text": "Speculative Speculative Decoding"}]
    assert hit["pdf_url"] == "https://arxiv.org/pdf/2603.03251"


def test_alphaxiv_semantic_uses_the_embedding_strategy(http):
    calls, answers = http
    answers.append(ALPHAXIV_HITS)
    result = papers.search("make inference faster", source="alphaxiv-semantic")
    assert "/discover/embedding" in calls[0] and result["strategy"] == "embedding"


def test_openalex_rebuilds_abstracts_and_says_how_to_fetch(http):
    calls, answers = http
    answers.append(OPENALEX_PAGE)

    result = papers.search("speculative decoding", source="openalex", max_results=2, start=4)

    params = _params(calls[0])
    assert params["search"] == "speculative decoding"
    assert "sort" not in params
    assert params["page"] == "3" and params["per_page"] == "2"
    assert "filter" not in params and "mailto" not in params
    assert result["total_available"] == 26014

    closed, arxiv = result["results"]
    assert closed["abstract"] == "Speculative fast decoding Speculative"
    assert closed["authors"] == ["Ada Lovelace"]
    assert closed["pdf_url"] is None and "no open-access PDF" in closed["fetch"]
    assert arxiv["arxiv_id"] == "2211.17192"
    assert arxiv["fetch"] == "paper(action='fetch', paper_id='2211.17192')"


def test_popular_reorders_a_relevance_page_instead_of_the_whole_corpus(http):
    """The bug this guards: sorted by citations over full-text matches, OpenAlex
    answered "speculative decoding" with off-topic papers, measured live.
    Titles and abstracts pick the candidates; citations only reorder them."""
    calls, answers = http
    answers.append(OPENALEX_PAGE)

    result = papers.search("speculative decoding", source="openalex", max_results=1, sort="popular")

    params = _params(calls[0])
    assert "sort" not in params and "search" not in params and params["per_page"] == "50"
    assert params["filter"] == "title_and_abstract.search:speculative decoding"
    assert result["returned"] == 1
    assert result["results"][0]["cited_by"] == 900


def test_biorxiv_is_openalex_filtered_to_the_biorxiv_source(http):
    calls, answers = http
    answers.append({"meta": {"count": 0}, "results": []})
    result = papers.search("crispr", source="biorxiv")
    assert _params(calls[0])["filter"] == f"primary_location.source.id:{papers.BIORXIV_SOURCE_ID}"
    assert result["source"] == "biorxiv"

    answers.append({"meta": {"count": 0}, "results": []})
    papers.search("crispr", source="biorxiv", sort="recent")
    assert _params(calls[1])["filter"] == (
        f"title_and_abstract.search:crispr,primary_location.source.id:{papers.BIORXIV_SOURCE_ID}")


def test_openalex_contact_address_is_only_sent_when_configured(http, monkeypatch):
    calls, answers = http
    monkeypatch.setenv("ICN_OPENALEX_MAILTO", "someone@example.invalid")
    answers.append({"meta": {"count": 0}, "results": []})
    papers.search("crispr", source="openalex")
    assert _params(calls[0])["mailto"] == "someone@example.invalid"


def test_a_non_json_answer_is_a_paper_error(monkeypatch):
    monkeypatch.setattr(papers, "_http_get", lambda url, accept=None, retries=3: (b"<html>", url, "text/html"))
    with pytest.raises(papers.PaperError, match="not JSON"):
        papers.search("x", source="openalex")
