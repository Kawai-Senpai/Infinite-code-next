"""Optional semantic embeddings for hybrid retrieval.

Lexical search finds text that shares words with the query. It cannot find
`verify_anchor` from "how do we detect that stored knowledge went stale",
because the two share no words at all. Embeddings close that gap: they turn
text into a vector whose direction carries meaning, so related ideas sit near
each other even when they are worded differently.

This module is deliberately the only place that knows an embedding model
exists. Everything else asks for an `Encoder` and gets `None` when the model
is off, still downloading, or unavailable, so ICN keeps working exactly as
before. That is the load-bearing property here: semantic search is an
enhancement to lexical retrieval, never a prerequisite for it.

Published CoIR results for this model class put dense retrieval BELOW BM25
alone (39.1 vs 42.3) and the hybrid above both (43.4). So the vectors are
wired in as an additional seed source, never as a replacement ranker.

The model downloads itself, once, in the background. ICN is a zero-config
tool, so requiring a separate install step would be a worse failure than
going without semantic search: a blocking download on first call would stall
the very first question a user asks, and an error would break a tool that was
working fine a minute earlier.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Protocol

# 33MB of static vectors, distilled specifically for natural-language to code
# retrieval, 256 dimensions. Chosen over a general-English model of the same
# size because the corpus here is overwhelmingly code and prose about code.
DEFAULT_MODEL = "minishlab/potion-code-16M-v2"

# Friendly names so a user never has to type a Hugging Face path. Anything not
# in this map is passed through untouched, so a custom model still works.
ALIASES = {
    "potion-code": DEFAULT_MODEL,
    "potion": "minishlab/potion-base-8M",
    "default": DEFAULT_MODEL,
}

ENV_VAR = "ICN_EMBED_MODEL"

# State names reported by status(). "loading" is the one that matters: it is
# not an error, and the caller should simply carry on lexically and try again
# on the next call.
DISABLED, LOADING, READY, UNAVAILABLE = "disabled", "loading", "ready", "unavailable"

_lock = threading.Lock()
_encoders: dict[str, Any] = {}
_started: set[str] = set()
_errors: dict[str, str] = {}


class Encoder(Protocol):
    """The whole contract. Swapping models is a config change, not a rewrite."""

    encoder_id: str
    dimensions: int

    def encode(self, texts: list[str]) -> Any:
        """Return an (len(texts), dimensions) float32 array of unit vectors."""


class _StaticEncoder:
    """model2vec static embeddings: a lookup table plus pooling.

    No transformer runs at query time, which is the entire reason this is
    viable inside an MCP server that answers interactively. Measured on a
    Tiger Lake laptop: 1.6ms to encode a query, against tens of milliseconds
    for a 400M-parameter model.
    """

    def __init__(self, model: Any, encoder_id: str) -> None:
        self._model = model
        self.encoder_id = encoder_id
        self.dimensions = int(model.dim)

    def encode(self, texts: list[str]) -> Any:
        import numpy as np

        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        vectors = np.asarray(self._model.encode(texts), dtype=np.float32)

        # Normalise once, here, so every consumer can use a plain dot product
        # as cosine similarity. A zero vector (empty or unknown text) would
        # divide by zero, so it keeps a norm of 1 and stays orthogonal to
        # everything, which is the correct "matches nothing" behaviour.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


def configured_model() -> str:
    """The model to use, or '' when the user has switched semantics off.

    On by default. `ICN_EMBED_MODEL=none` disables it.
    """
    raw = (os.environ.get(ENV_VAR) or "").strip()
    if not raw:
        return DEFAULT_MODEL
    if raw.lower() in ("none", "off", "0", "false", "disabled", "no"):
        return ""
    return ALIASES.get(raw.lower(), raw)


def _load(name: str) -> None:
    """Import, download and construct. Runs on the prewarm thread."""
    # Hugging Face caches by symlinking blobs into snapshot directories. On
    # Windows that needs developer mode or admin rights, and without them the
    # download dies with WinError 1314 partway through. Copying instead costs
    # a little disk and always works.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    # stdout is the MCP transport. A progress bar or a chatty logger writing
    # there would corrupt the protocol framing, not merely look untidy.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        from model2vec import StaticModel

        encoder: Any = _StaticEncoder(StaticModel.from_pretrained(name), name)
        error = ""
    except Exception as exc:         # noqa: BLE001 - never raise, see module docstring
        encoder, error = None, f"{type(exc).__name__}: {exc}"[:300]

    with _lock:
        _encoders[name] = encoder
        if error:
            _errors[name] = error


def prewarm(model: str | None = None, blocking: bool = False) -> None:
    """Start downloading and loading the model, once per process.

    Called on workspace open so the 33MB fetch happens while the agent is
    reading the briefing, not in the middle of its first question. Returns
    immediately unless `blocking`, which only tests should need.
    """
    name = model if model is not None else configured_model()
    if not name:
        return

    with _lock:
        if name in _started:
            already = name in _encoders
        else:
            _started.add(name)
            already = False
    if already:
        return

    if blocking:
        _load(name)
        return

    # Daemon: a half-finished model download must never hold the server open
    # at shutdown.
    threading.Thread(target=_load, args=(name,), daemon=True,
                     name="icn-embed-prewarm").start()


def load_encoder(model: str | None = None) -> Encoder | None:
    """The encoder if it is loaded and ready, else None.

    Never blocks and never raises. While the model is still downloading this
    returns None and the caller falls back to lexical search, which is the
    behaviour ICN had before embeddings existed.
    """
    name = model if model is not None else configured_model()
    if not name:
        return None
    with _lock:
        if name in _encoders:
            return _encoders[name]
    prewarm(name)
    return None


def status(model: str | None = None) -> dict[str, Any]:
    """What the semantic layer is doing, for workspace status and diagnostics.

    Users need to be able to tell "still downloading" from "quietly broken";
    without this the two look identical from the outside.
    """
    name = model if model is not None else configured_model()
    if not name:
        return {"state": DISABLED, "model": None,
                "detail": f"set {ENV_VAR} to enable semantic search"}

    with _lock:
        loaded = _encoders.get(name, "missing")
        started = name in _started
        error = _errors.get(name)

    if loaded == "missing":
        return ({"state": LOADING, "model": name,
                 "detail": "downloading model; lexical search is unaffected"}
                if started else {"state": LOADING, "model": name,
                                 "detail": "not started"})
    if loaded is None:
        return {"state": UNAVAILABLE, "model": name,
                "detail": error or "model could not be loaded",
                "hint": "lexical search is unaffected; "
                        f"set {ENV_VAR}=none to stop retrying"}
    return {"state": READY, "model": name, "dimensions": loaded.dimensions}


def reset_cache() -> None:
    """Drop loaded models and state. For tests that switch models mid-process."""
    with _lock:
        _encoders.clear()
        _started.clear()
        _errors.clear()
