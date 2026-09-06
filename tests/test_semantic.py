"""Hybrid retrieval: semantic vectors beside lexical search.

The property under test throughout is that semantics are an *enhancement*.
With no model, a model still downloading, or a broken model, ICN must retrieve
exactly as it did before. Every test here that turns the encoder off is really
asserting that the feature cannot take the tool down with it.
"""

from __future__ import annotations

import pathlib

import pytest

from icn import db, embed, ids, vectors

pytestmark = pytest.mark.usefixtures("no_model_env")


@pytest.fixture()
def no_model_env(monkeypatch):
    """Semantic search off unless a test explicitly asks for it."""
    monkeypatch.setenv(embed.ENV_VAR, "none")
    embed.reset_cache()
    yield
    embed.reset_cache()


class FakeEncoder:
    """A deterministic stand-in, so retrieval logic is testable without a model.

    Encodes each text as a bag of characters. Crude, but it gives stable,
    genuinely content-dependent vectors with no download and no dependency on
    a specific model's behaviour.
    """

    encoder_id = "fake-encoder-v1"
    dimensions = 26

    def encode(self, texts):
        import numpy as np

        out = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for char in (text or "").lower():
                index = ord(char) - 97
                if 0 <= index < self.dimensions:
                    out[row, index] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


@pytest.fixture()
def store(tmp_path: pathlib.Path):
    conn = db.init_repo_store(tmp_path / "s.db")
    specs = [
        ("verify_anchor", "anchors.verify_anchor", "recompute fingerprint downgrade stale"),
        ("parse_latex", "tex.parse_latex", "tokenize tex sections environments"),
        ("connect", "db.connect", "open sqlite connection wal busy timeout"),
    ]
    for name, path, body in specs:
        symbol_id = ids.new_id("sym")
        conn.execute("INSERT INTO symbols (symbol_id, file_id, name, symbol_path, kind, status)"
                     " VALUES (?,?,?,?,?, 'ACTIVE')", (symbol_id, "f1", name, path, "function"))
        conn.execute("INSERT INTO fts_symbols (symbol_id, symbol_path, name, signature, body)"
                     " VALUES (?,?,?,?,?)", (symbol_id, path, name, f"def {name}()", body))
    conn.commit()
    yield conn
    conn.close()


# --------------------------------------------------------------- the off switch

def test_semantic_is_off_when_the_env_says_none():
    assert embed.configured_model() == ""
    assert embed.load_encoder() is None
    assert embed.status()["state"] == embed.DISABLED


def test_it_is_on_by_default(monkeypatch):
    """A zero-config tool should not need configuring to get its best search."""
    monkeypatch.delenv(embed.ENV_VAR, raising=False)
    assert embed.configured_model() == embed.DEFAULT_MODEL


def test_a_friendly_alias_resolves_to_a_real_model(monkeypatch):
    monkeypatch.setenv(embed.ENV_VAR, "potion-code")
    assert embed.configured_model() == embed.DEFAULT_MODEL


def test_an_unknown_name_is_passed_through_untouched(monkeypatch):
    """Custom models must stay possible without editing the alias table."""
    monkeypatch.setenv(embed.ENV_VAR, "some-org/some-model")
    assert embed.configured_model() == "some-org/some-model"


def test_a_model_that_cannot_load_degrades_instead_of_raising(monkeypatch):
    monkeypatch.setenv(embed.ENV_VAR, "definitely/not-a-real-model-xyz")
    embed.prewarm(blocking=True)
    assert embed.load_encoder() is None
    state = embed.status()
    assert state["state"] == embed.UNAVAILABLE
    assert "lexical" in state["hint"]


# ------------------------------------------------------------- vector storage

def test_refresh_without_an_encoder_is_a_no_op(store):
    assert vectors.refresh(store, None)["encoded"] == 0
    assert vectors.stats(store)["symbols"] == 0


def test_refresh_encodes_then_skips_unchanged_content(store):
    encoder = FakeEncoder()
    first = vectors.refresh(store, encoder, scope=vectors.SYMBOL)
    assert first["encoded"] == 3

    second = vectors.refresh(store, encoder, scope=vectors.SYMBOL)
    assert second["encoded"] == 0, "unchanged content must not be re-encoded"
    assert second["skipped"] == 3


def test_changing_the_body_re_encodes_only_that_symbol(store):
    encoder = FakeEncoder()
    vectors.refresh(store, encoder, scope=vectors.SYMBOL)
    row = store.execute("SELECT symbol_id FROM symbols LIMIT 1").fetchone()
    store.execute("UPDATE fts_symbols SET body='completely different words now'"
                  " WHERE symbol_id = ?", (row["symbol_id"],))
    store.commit()

    result = vectors.refresh(store, encoder, scope=vectors.SYMBOL)
    assert result["encoded"] == 1
    assert result["skipped"] == 2


def test_switching_encoder_discards_the_old_vectors(store):
    """Two models produce incomparable vectors; mixing them returns nonsense."""
    vectors.refresh(store, FakeEncoder(), scope=vectors.SYMBOL)
    assert vectors.stats(store)["encoder"] == "fake-encoder-v1"

    class OtherEncoder(FakeEncoder):
        encoder_id = "other-encoder-v1"
        dimensions = 12

        def encode(self, texts):
            import numpy as np
            out = np.ones((len(texts), self.dimensions), dtype=np.float32)
            return out / np.linalg.norm(out, axis=1, keepdims=True)

    vectors.refresh(store, OtherEncoder(), scope=vectors.SYMBOL)
    stats = vectors.stats(store)
    assert stats["encoder"] == "other-encoder-v1"
    assert stats["dimensions"] == 12
    assert stats["symbols"] == 3, "every symbol re-encoded in the new space"


def test_a_deleted_symbol_loses_its_vector(store):
    """Otherwise search resurrects code that no longer exists."""
    encoder = FakeEncoder()
    vectors.refresh(store, encoder, scope=vectors.SYMBOL)
    row = store.execute("SELECT symbol_id FROM symbols LIMIT 1").fetchone()
    store.execute("UPDATE symbols SET status='DELETED' WHERE symbol_id=?", (row["symbol_id"],))
    store.commit()

    vectors.refresh(store, encoder, scope=vectors.SYMBOL)
    assert vectors.stats(store)["symbols"] == 2
    found = {item for item, _ in vectors.search(store, encoder, "anything", vectors.SYMBOL)}
    assert row["symbol_id"] not in found


def test_search_returns_similarity_ordered_results(store):
    encoder = FakeEncoder()
    vectors.refresh(store, encoder, scope=vectors.SYMBOL)
    hits = vectors.search(store, encoder, "tokenize tex sections", vectors.SYMBOL)

    assert hits, "a populated store must return something"
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True), "results must be best first"

    best = store.execute("SELECT symbol_path FROM symbols WHERE symbol_id=?",
                         (hits[0][0],)).fetchone()["symbol_path"]
    assert best == "tex.parse_latex"


def test_search_without_an_encoder_or_query_is_empty(store):
    vectors.refresh(store, FakeEncoder(), scope=vectors.SYMBOL)
    assert vectors.search(store, None, "anything", vectors.SYMBOL) == []
    assert vectors.search(store, FakeEncoder(), "   ", vectors.SYMBOL) == []


def test_limit_larger_than_the_corpus_is_not_an_error(store):
    """argpartition on k >= n is the classic off-by-one in this code."""
    encoder = FakeEncoder()
    vectors.refresh(store, encoder, scope=vectors.SYMBOL)
    assert len(vectors.search(store, encoder, "sqlite", vectors.SYMBOL, limit=500)) == 3


# ---------------------------------------------------------------- rank fusion

def test_rank_score_decays_from_one():
    from icn.search import _rank_score

    assert _rank_score(0) == 1.0
    assert _rank_score(1) < 1.0
    assert _rank_score(50) < _rank_score(10)
    assert _rank_score(10_000) > 0.0, "never negative, never zero"


def test_fusion_without_semantic_hits_changes_nothing():
    """The no-model path must be byte-identical to the old behaviour."""
    from icn.search import _fuse_memory_hits

    lexical = {"mem_a": 1.0, "mem_b": 0.4}
    assert _fuse_memory_hits(lexical, []) == lexical


def test_fusion_keeps_the_zero_to_one_range():
    """MEMORY_FLOOR is calibrated against this range; moving it breaks quoting."""
    from icn.search import _fuse_memory_hits

    fused = _fuse_memory_hits({"mem_a": 1.0, "mem_b": 0.4},
                              [("mem_c", 0.9), ("mem_a", 0.7)])
    assert max(fused.values()) == pytest.approx(1.0)
    assert min(fused.values()) > 0.0
    assert all(0.0 < v <= 1.0 for v in fused.values())


def test_fusion_rewards_agreement_between_the_two_rankers():
    """A hit both rankers like should beat one that only appears in either."""
    from icn.search import _fuse_memory_hits

    fused = _fuse_memory_hits({"both": 1.0, "lexical_only": 0.9},
                              [("both", 0.5), ("semantic_only", 0.4)])
    assert fused["both"] > fused["lexical_only"]
    assert fused["both"] > fused["semantic_only"]


def test_semantic_seeds_reach_symbols_lexical_search_never_found(store):
    """The whole point of the feature, stated as a test."""
    from icn.search import _seed_semantic

    encoder = FakeEncoder()
    vectors.refresh(store, encoder, scope=vectors.SYMBOL)

    seeds: dict = {}
    added = _seed_semantic(store, encoder, "tokenize tex sections", seeds)
    assert added == 3
    assert seeds, "semantic search must be able to seed on its own"
    assert all("sem" in seed for seed in seeds.values())
    assert max(seed["sem"] for seed in seeds.values()) == 1.0


def test_semantic_seeding_never_lowers_an_existing_lexical_score(store):
    """Adding a signal must not remove one."""
    from icn.search import _seed_semantic

    encoder = FakeEncoder()
    vectors.refresh(store, encoder, scope=vectors.SYMBOL)
    symbol_id = store.execute("SELECT symbol_id FROM symbols LIMIT 1").fetchone()["symbol_id"]

    seeds = {symbol_id: {"lex": 1.0, "sym": 1.0}}
    _seed_semantic(store, encoder, "tokenize tex sections", seeds)
    assert seeds[symbol_id]["lex"] == 1.0
    assert seeds[symbol_id]["sym"] == 1.0


def test_seeding_with_no_encoder_adds_nothing(store):
    from icn.search import _seed_semantic

    seeds: dict = {}
    assert _seed_semantic(store, None, "anything", seeds) == 0
    assert seeds == {}


def test_a_broken_vector_store_does_not_break_retrieval(store):
    """Retrieval degrading is acceptable; retrieval raising is not."""
    from icn.search import _seed_semantic

    class ExplodingEncoder(FakeEncoder):
        def encode(self, texts):
            raise RuntimeError("model exploded")

    vectors.refresh(store, FakeEncoder(), scope=vectors.SYMBOL)
    seeds: dict = {}
    assert _seed_semantic(store, ExplodingEncoder(), "query", seeds) == 0
    assert seeds == {}


def test_every_intent_weights_semantics_below_lexical():
    """Dense retrieval scores below BM25 alone on CoIR; it must not outrank it."""
    from icn.search import INTENT_WEIGHTS

    for intent, weights in INTENT_WEIGHTS.items():
        assert "sem" in weights, intent
        assert 0 < weights["sem"] < weights["lex"], intent
