"""Starts ICN's execution recorder in any Python process launched by `icn trace`.

`icn trace` puts this directory first on PYTHONPATH, so every Python
interpreter the traced command starts (the script itself, pytest, a worker
process) imports it at startup. It loads flowtrace.py by file path, never the
icn package, then hands over to the environment's own sitecustomize if one
exists, so a virtualenv's customisation still runs.
"""

import os
import sys

if os.environ.get("ICN_TRACE_DIR"):
    try:
        import importlib.util

        _here = os.path.dirname(os.path.abspath(__file__))
        _spec = importlib.util.spec_from_file_location(
            "icn_flowtrace", os.path.join(os.path.dirname(_here), "flowtrace.py"))
        _module = importlib.util.module_from_spec(_spec)
        sys.modules["icn_flowtrace"] = _module
        _spec.loader.exec_module(_module)
        _module.start()
    except Exception as _err:  # noqa: BLE001 - tracing must never stop the program
        sys.stderr.write(f"icn trace: recorder did not start: {_err}\n")

    # Chain to the sitecustomize this one shadowed, if any.
    _here = os.path.dirname(os.path.abspath(__file__))
    _saved = list(sys.path)
    try:
        sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]
        _self = sys.modules.pop("sitecustomize", None)
        try:
            import sitecustomize  # noqa: F401
        except ImportError:
            if _self is not None:
                sys.modules["sitecustomize"] = _self
    except Exception:  # noqa: BLE001
        pass
    finally:
        sys.path[:] = _saved
