"""Entry point. `python -m icn`, `uvx infinite-code-next`, or the `icn` script.

stdio only. No port, no daemon, no listening socket - the client owns the
process lifetime and restarts it if it dies.
"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in ("--version", "-V"):
        from . import __version__
        print(__version__)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--where":
        from . import paths
        print(paths.storage_root())
        return

    from .server import mcp
    mcp.run("stdio")


if __name__ == "__main__":
    main()
