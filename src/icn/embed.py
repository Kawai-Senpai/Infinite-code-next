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

# One event per model, created by whichever caller wins the race to start the
# load. Its presence in this dict IS the "a load is already in flight" flag,
# and it is what a blocking caller waits on instead of starting a download of
# its own. `_started` cannot serve that purpose: it stays set after the load
# finishes, so it cannot distinguish in-flight from done.
_events: dict[str, threading.Event] = {}

# A blocking waiter is always a background thread, never a tool call, but it
# must not outlive a fetch that has genuinely wedged.
LOAD_WAIT_SECONDS = 300.0


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


def _configure_environment() -> None:
    """Environment the model stack needs, set before anything imports it."""
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


# Every module in the load path that carries a native extension, imported on
# the main thread before any worker exists. See preload_native.
#
# `import model2vec` alone is not enough, and that gap is what cost the hour.
# It pulls in numpy but NOT huggingface_hub, and huggingface_hub downloads
# through hf_xet.pyd, a Rust extension that neither `import huggingface_hub`
# nor `import huggingface_hub.file_download` loads: it arrives lazily, at the
# moment the first real download runs, on the prewarm thread. safetensors is a
# second Rust extension, loaded when the weights are actually read.
#
# Derived by measurement, not by guesswork: with this list preloaded, a cold
# load on a background thread imports zero further .pyd files. Anything missing
# here is a DLL that would load on a worker thread, which is the bug.
_NATIVE_MODULES = (
    "model2vec",                 # pulls numpy and its OpenBLAS DLL
    "huggingface_hub",
    "hf_xet",                    # Rust downloader
    "safetensors", "safetensors.numpy",   # Rust weight reader
    "yaml",                      # C extension (_yaml)
    "sqlite3",
    "brotli", "urllib3", "requests",      # transport, _brotli is native
)


def preload_native() -> None:
    """Load the numpy/model2vec native extensions NOW, on the importing thread.

    Deadlock, Windows, every cold MCP start, no timeout and no error: the tool
    call simply never returns.

    numpy's `_multiarray_umath` extension and the OpenBLAS DLL behind it do
    real work in their DllMain-time initialisation. Loading them from a thread
    while the stdio server owns the process's standard handles wedges inside
    the Windows loader: the thread parks in an Executive wait and never
    resumes. Measured on a bare FastMCP server whose only tool body was
    `import numpy` - it hung indefinitely, and moving that same import to
    module scope returned in 1.0s. So this is not an ICN bug, but ICN has to
    dodge it.

    ICN reaches that import lazily, from the prewarm thread, on the first
    workspace(action='open'). That is precisely the deadlocking shape, and it
    is why open() hung forever with semantic search on (never returning in
    300s) yet finished in 2.0s with ICN_EMBED_MODEL=none.

    Calling this at server import time - before mcp.run() takes over stdio and
    before any worker thread exists - loads the DLLs on the main thread, where
    the loader behaves. Every later import is then a sys.modules hit. Costs a
    few hundred milliseconds of startup and never raises: if the model stack
    is missing or broken, semantic search degrades to lexical exactly as it
    did before, and _load records the real error.
    """
    _configure_environment()
    try:
        import model2vec  # noqa: F401  - imported for its side effect
    except Exception:     # noqa: BLE001 - see module docstring
        pass

    for module in _NATIVE_MODULES:
        try:
            __import__(module)
        except Exception:  # noqa: BLE001 - absent or broken; see module docstring
            pass


def _cached_snapshot(name: str) -> str | None:
    """The model's directory in the shared Hugging Face cache, if already there.

    model2vec's `StaticModel.from_pretrained` defaults to `force_download=True`
    (0.9.0), so passing it a repo id re-fetches all 33MB on EVERY load and
    ignores a cache that already holds the file. That is why the model appeared
    to download again on every server start, and why a flaky or slow link
    turned a load into an open-ended stall instead of a cache hit.

    Handing it a local directory sidesteps the hub entirely: measured on this
    machine, 0.19s from cache against 1.44s through the network path, and no
    network dependency at all. Downloaded once per machine, reused for as long
    as the cache lives.

    Returns None when the model has genuinely never been fetched here, which is
    the one case that does need the network.
    """
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(name, local_files_only=True)
    except Exception:    # noqa: BLE001 - not cached, or no hub; fall back to the network
        return None


def _load(name: str) -> None:
    """Import, download and construct. Runs on the prewarm thread.

    The heavy native import is normally already done by preload_native at
    server startup, so `from model2vec import ...` here is a sys.modules
    lookup rather than a DLL load. See preload_native for why that matters.
    """
    _configure_environment()

    try:
        from model2vec import StaticModel

        # A local snapshot path when the machine already has it, the repo id
        # only on the genuine first fetch. See _cached_snapshot.
        source = _cached_snapshot(name) or name

        # encoder_id stays the model NAME, never the resolved path: it is what
        # vectors.refresh compares against the stored encoder to decide whether
        # existing vectors are still valid. A path here would differ from what
        # earlier runs recorded and silently re-encode the whole repository.
        encoder: Any = _StaticEncoder(StaticModel.from_pretrained(source), name)
        error = ""
    except Exception as exc:         # noqa: BLE001 - never raise, see module docstring
        encoder, error = None, f"{type(exc).__name__}: {exc}"[:300]

    with _lock:
        _encoders[name] = encoder
        if error:
            _errors[name] = error
        event = _events.get(name)

    # After the result is published, never before: a waiter that woke early
    # would look up an encoder that is not in the dict yet and conclude the
    # model was unavailable.
    if event is not None:
        event.set()


def prewarm(model: str | None = None, blocking: bool = False) -> None:
    """Start downloading and loading the model, once per process.

    Called on workspace open so the 33MB fetch happens while the agent is
    reading the briefing, not in the middle of its first question. Returns
    immediately unless `blocking`, which only tests should need.

    "Once per process" is load-bearing and was previously not enforced. The
    guard tested `name in _encoders`, which is only true once the load has
    FINISHED, so every call arriving while the model was still downloading
    fell through and started another one. load_encoder() calls this on every
    miss, so a cold start spawned a fresh 33MB download per investigate() -
    all of them contending on the same huggingface_hub blob lock, each
    re-fetching the same file. That is what turned a 2s cold start into an
    hour under an agent, while a single manual call stayed fast.
    """
    name = model if model is not None else configured_model()
    if not name:
        return

    with _lock:
        if name in _encoders:      # already loaded, or failed and recorded
            return
        event = _events.get(name)
        if event is None:
            event = _events[name] = threading.Event()
            _started.add(name)
            ours = True
        else:
            ours = False

    if not ours:
        # Someone is already fetching this model. Joining their download is
        # the entire point; starting a second one is the bug above.
        if blocking:
            event.wait(LOAD_WAIT_SECONDS)
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
        # Any in-flight loader is released, so a test that switches models is
        # never left waiting on an event nothing will ever set.
        for event in _events.values():
            event.set()
        _events.clear()
