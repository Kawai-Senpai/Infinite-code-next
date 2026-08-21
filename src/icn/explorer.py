"""The knowledge explorer: a local, self-contained graph UI.

Renders the whole graph - repository, files, symbols, memories and every edge
between them - as one HTML file with the data inlined, then serves it on
localhost and opens a browser.

Two constraints shape it:

  * No network, no build step, no npm. The visualisation is hand-written
    canvas plus a force layout in about two hundred lines. A knowledge tool
    that needs a toolchain to look at its own knowledge will not get looked at.

  * The file is self-contained. Saving it, mailing it, or committing it all
    work, because the data is embedded rather than fetched.

Canvas rather than SVG: a repository with a few thousand symbols is tens of
thousands of DOM nodes in SVG, and the browser stops being interactive long
before the graph stops being useful.
"""

from __future__ import annotations

import http.server
import json
import socketserver
import threading
import webbrowser
from pathlib import Path
from typing import Any

# Node palettes are picked for meaning, not decoration: memories are warm so
# knowledge stands out against the cool structural nodes, and severity shifts
# hue rather than size so a critical warning reads at any zoom.
# The UI lives in web/ as real .html, .css and .js rather than string
# literals, so an editor treats it as what it is. They are inlined at render
# time: the published page must stay a single self-contained file, because
# saving it, mailing it or committing it all have to keep working.
WEB = Path(__file__).parent / "web"
_CACHE: dict[str, str] = {}


def _asset(name: str) -> str:
    if name not in _CACHE:
        _CACHE[name] = (WEB / name).read_text(encoding="utf-8")
    return _CACHE[name]


def page_template() -> str:
    """The shell with its stylesheet and script inlined."""
    html = _asset("explorer.html")
    html = html.replace(
        '<link rel="stylesheet" href="explorer.css">',
        "<style>\n" + _asset("explorer.css") + "\n</style>")
    # The data block is a separate <script>, so the app script must land after
    # it - replacing the tag in place preserves that order.
    html = html.replace(
        '<script src="explorer.js"></script>',
        "<script>\n" + _asset("explorer.js") + "\n</script>")
    return html


def render(graph: dict[str, Any]) -> str:
    """Inline the graph into the page. Self-contained, no fetch at runtime."""
    payload = json.dumps(graph, ensure_ascii=False).replace("</", "<\\/")
    title = graph.get("repo_name") or graph.get("repo_id") or "Knowledge"
    return page_template().replace("__DATA__", payload).replace("__TITLE__", _escape(title))


def _escape(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def write_html(graph: dict[str, Any], target: Path) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render(graph), encoding="utf-8")
    return target


def serve(html: Path, port: int = 0, open_browser: bool = True) -> None:
    """Serve one file on localhost until interrupted.

    A file:// URL would avoid the server entirely, but browsers treat local
    files inconsistently and a real origin keeps behaviour predictable. Port 0
    lets the OS pick, so two explorers never collide.
    """
    html = Path(html).resolve()
    directory = str(html.parent)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

        def log_message(self, *args):    # keep the terminal for the user
            pass

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as server:
        actual = server.server_address[1]
        url = f"http://127.0.0.1:{actual}/{html.name}"
        print(f"  Knowledge explorer: {url}")
        print("  Press Ctrl+C to stop.")
        if open_browser:
            threading.Timer(0.6, lambda: webbrowser.open(url)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n  Stopped.")
