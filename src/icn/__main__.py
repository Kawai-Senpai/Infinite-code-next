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
    if len(sys.argv) > 1 and sys.argv[1] == "hook":
        # Before anything heavy is imported: this runs before every tool call.
        from .hooks import run as hook_run
        raise SystemExit(hook_run(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "rules":
        from .rules_cli import main as rules_main
        raise SystemExit(rules_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "trace":
        from .trace_cli import main as trace_main
        raise SystemExit(trace_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "doctor":
        from .doctor import main as doctor_main
        raise SystemExit(doctor_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "install":
        from .install import main as install_main
        raise SystemExit(install_main(sys.argv[2:]))

    # server.main(), not mcp.run() directly: the native preload it does first
    # is what keeps workspace(action='open') from deadlocking on Windows, and
    # this is the entry point the installed console script actually uses.
    from .server import main as serve
    serve()


if __name__ == "__main__":
    main()
