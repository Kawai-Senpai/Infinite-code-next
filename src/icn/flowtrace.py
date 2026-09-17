"""In-process execution recorder for Python: call flow, values, time, errors.

Loaded into the traced interpreter by tracehook/sitecustomize.py, by FILE PATH,
so it imports nothing but the standard library. Importing the icn package here
would drag its native extensions into the user's process, and late native
imports are known to wedge on Windows (the same rule labrun.py follows).

What it records, for code in the project (not the standard library or
installed packages, unless ICN_TRACE_LIBS=1):

    calls       a calling context tree: one node per distinct call path, with
                call count, total and self time. Repeated calls in a loop fold
                into one node, so a million iterations stay one line.
    values      arguments and return values of the first calls of each
                function, and every change to a local variable, detected by
                comparing value representations line by line. Comparing
                representations rather than identities is what catches
                `items.append(x)`, which mutates without assigning.
    library     which library functions project code called, with counts.
    errors      exceptions raised, where they were handled or escaped.
    memory      peak resident memory at exit; allocation sites with
                ICN_TRACE_MEMORY=1 (tracemalloc, slower).

Python 3.12+ uses sys.monitoring (PEP 669), which lets uninteresting code be
switched off per location so the standard library runs at full speed. Older
interpreters fall back to sys.settrace, which is slower.

Everything is bounded. Past ICN_TRACE_MAX_EVENTS recorded values the tracer
stops recording values and keeps only timing and counts, and says so, rather
than making a long run unusable.

Output: <ICN_TRACE_DIR>/py-<pid>.json, written at interpreter exit, plus
py-<pid>.live.json refreshed every second while the program runs.
"""

from __future__ import annotations

import atexit
import json
import os
import reprlib
import sys
import threading
import time
import types

THIS_FILE = os.path.normcase(os.path.abspath(__file__))
LIBRARY_MARKERS = ("site-packages", "dist-packages", f"{os.sep}lib{os.sep}python",
                   f"{os.sep}Lib{os.sep}", "node_modules", f"{os.sep}.venv{os.sep}",
                   f"{os.sep}venv{os.sep}")
SKIPPED_TYPES = (types.ModuleType, types.FunctionType, types.BuiltinFunctionType,
                 types.MethodType, type)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class Settings:
    def __init__(self) -> None:
        self.out_dir = os.environ.get("ICN_TRACE_DIR") or os.getcwd()
        roots = os.environ.get("ICN_TRACE_ROOTS") or os.getcwd()
        self.roots = [os.path.normcase(os.path.abspath(r)) for r in roots.split(os.pathsep) if r]
        self.libs = os.environ.get("ICN_TRACE_LIBS") == "1"
        self.values = os.environ.get("ICN_TRACE_VALUES", "1") != "0"
        self.memory = os.environ.get("ICN_TRACE_MEMORY") == "1"
        self.max_events = _env_int("ICN_TRACE_MAX_EVENTS", 50_000)
        self.value_calls = _env_int("ICN_TRACE_VALUE_CALLS", 3)
        self.max_nodes = _env_int("ICN_TRACE_MAX_NODES", 20_000)


CO_OPTIMIZED, CO_NEWLOCALS = 0x1, 0x2
CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR = 0x20, 0x100, 0x200
VALUES_PER_NAME = 6          # first values kept for each variable in one call
LINES_PER_CALL = 400         # lines diffed in one call before only the end state is kept
CPU_CLOCK_DEPTH = 12         # stack depth beyond which CPU time is not measured (see Frame)


def code_kind(code: types.CodeType) -> str:
    if code.co_name == "<module>":
        return "module"
    if not code.co_flags & (CO_OPTIMIZED | CO_NEWLOCALS):
        return "class"
    if code.co_flags & (CO_GENERATOR | CO_ASYNC_GENERATOR):
        return "generator"
    if code.co_flags & CO_COROUTINE:
        return "coroutine"
    return "function"


class Node:
    __slots__ = ("key", "kind", "children", "count", "total_ns", "self_ns", "cpu_ns", "self_cpu_ns",
                 "calls", "errors", "library", "thread")

    def __init__(self, key: tuple[str, str, int], kind: str = "function") -> None:
        self.key = key                      # (file, qualname, first line)
        self.kind = kind
        self.children: dict[tuple[str, str, int], Node] = {}
        self.count = 0
        self.total_ns = 0
        self.self_ns = 0
        self.cpu_ns = 0
        self.self_cpu_ns = 0
        self.calls: list[dict] = []         # samples: args, value changes, return
        self.errors: dict[str, int] = {}
        self.library: dict[str, int] = {}
        self.thread: str | None = None

    def to_json(self) -> dict:
        data = {"file": self.key[0], "function": self.key[1], "line": self.key[2], "kind": self.kind,
                "count": self.count, "total_ms": round(self.total_ns / 1e6, 3),
                "self_ms": round(self.self_ns / 1e6, 3), "cpu_ms": round(self.cpu_ns / 1e6, 3),
                "self_cpu_ms": round(self.self_cpu_ns / 1e6, 3), "calls": self.calls,
                "errors": self.errors, "library": self.library,
                "children": [c.to_json() for c in self.children.values()]}
        if self.thread:
            data["thread"] = self.thread
        return data


class Frame:
    __slots__ = ("node", "code", "start", "cpu_start", "child_ns", "child_cpu_ns", "sample", "last_line",
                 "snapshot", "lines", "counts")

    def __init__(self, node: Node, sample: dict | None, code: object = None,
                 with_cpu: bool = True) -> None:
        self.node = node
        self.code = code
        self.start = time.perf_counter_ns()
        # The CPU clock costs 125 ns against 53 ns for the wall clock and is
        # read on entry and exit of every call (~90 ms over 180k calls), for a
        # column the report only shows above 20 ms of self time. Deep frames
        # are the numerous, short ones, so only shallow frames are clocked;
        # see Recorder.leave.
        self.cpu_start = time.thread_time_ns() if with_cpu else 0
        self.child_ns = 0
        self.child_cpu_ns = 0
        self.sample = sample
        self.last_line = 0
        self.snapshot: dict[str, str] = {}
        self.lines = 0
        self.counts: dict[str, int] = {}


class FieldRepr(reprlib.Repr):
    """reprlib, but a plain object shows its fields instead of its address.

    `<pricing.Order object at 0x1d09>` tells a reader nothing, and reprlib
    formats the items of a list itself, so overriding only the top-level call
    left every Order inside a list unreadable.
    """

    def repr_instance(self, x: object, level: int) -> str:
        if type(x).__repr__ is object.__repr__ and hasattr(x, "__dict__"):
            if level <= 0:
                return f"{type(x).__name__}(...)"
            fields = [(k, v) for k, v in vars(x).items() if not k.startswith("_")]
            inner = ", ".join(f"{k}={self.repr1(v, level - 1)}" for k, v in fields[:5])
            text = f"{type(x).__name__}({inner}{', ...' if len(fields) > 5 else ''})"
            return text if len(text) <= 120 else text[:117] + "..."
        return super().repr_instance(x, level)


class Recorder:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.started = time.perf_counter_ns()
        self.wall_started = time.time()
        self.root = Node(("", "<process>", 0), "process")
        self.stacks: dict[int, list[Frame]] = {}
        self.busy = threading.local()
        self.interesting: dict[object, bool] = {}
        self.events = 0
        self.nodes = 0
        self.values_dropped = False
        self.errors: list[dict] = []
        self.repr = FieldRepr()
        self.repr.maxstring = 80
        self.repr.maxother = 80
        self.repr.maxlist = self.repr.maxtuple = self.repr.maxset = self.repr.maxdict = 6
        self.repr.maxlevel = 3
        self.lock = threading.Lock()
        self.finished = False

    # -------------------------------------------------------------- selection

    def wanted(self, code: types.CodeType) -> bool:
        known = self.interesting.get(code)
        if known is not None:
            return known
        filename = code.co_filename or ""
        path = os.path.normcase(os.path.abspath(filename)) if not filename.startswith("<") else ""
        if not path or path == THIS_FILE or "tracehook" in path or code.co_name == "__annotate__":
            # __annotate__ is synthesized by Python 3.14 for annotations and
            # probed by tools with NotImplementedError: noise, not program flow.
            answer = False
        elif self.s.libs:
            answer = True
        else:
            inside = any(path == r or path.startswith(r + os.sep) for r in self.s.roots)
            answer = inside and not any(marker in path for marker in LIBRARY_MARKERS)
        self.interesting[code] = answer
        return answer

    def show(self, value: object) -> str:
        try:
            return self.repr.repr(value)
        except Exception as err:  # noqa: BLE001 - a user __repr__ may raise anything
            return f"<{type(value).__name__}: repr failed: {type(err).__name__}>"

    def snapshot(self, frame: types.FrameType) -> dict[str, str]:
        out = {}
        for name, value in list(frame.f_locals.items())[:40]:
            if name.startswith("__") or isinstance(value, SKIPPED_TYPES):
                continue
            out[name] = self.show(value)
        return out

    # ------------------------------------------------------------------ calls

    def stack(self) -> list[Frame]:
        ident = threading.get_ident()
        stack = self.stacks.get(ident)
        if stack is None:
            stack = self.stacks[ident] = []
        return stack

    def enter(self, code: types.CodeType, frame: types.FrameType | None, resumed: bool = False) -> None:
        stack = self.stack()
        parent = stack[-1].node if stack else self.root
        key = (code.co_filename, getattr(code, "co_qualname", code.co_name), code.co_firstlineno)
        node = parent.children.get(key)
        if node is None:
            if self.nodes >= self.s.max_nodes:
                node = parent            # fold the overflow into the caller
            else:
                node = parent.children[key] = Node(key, code_kind(code))
                if parent is self.root:
                    current = threading.current_thread()
                    if current is not threading.main_thread():
                        node.thread = current.name
                self.nodes += 1
        if not resumed:
            node.count += 1
        sample = None
        if (self.s.values and not resumed and len(node.calls) < self.s.value_calls
                and not self.values_dropped and frame is not None and node is not parent
                and node.kind != "module"):
            args = self.snapshot(frame)
            sample = {"at_ms": round((time.perf_counter_ns() - self.started) / 1e6, 3),
                      "args": args, "changes": []}
            node.calls.append(sample)
            self.count(len(args))
        record = Frame(node, sample, code, with_cpu=len(stack) < CPU_CLOCK_DEPTH)
        if sample is not None:
            record.snapshot = dict(sample["args"])
        stack.append(record)

    def leave(self, value: object = None, returned: bool = True,
              frame: types.FrameType | None = None, yielded: bool = False,
              code: object = None) -> None:
        stack = self.stack()
        if not stack:
            return
        if code is not None and stack[-1].code is not code:
            # An exit for a frame that is not on top. Measured: a generator
            # closed by the garbage collector reports PY_THROW then PY_UNWIND
            # while other code is running, and popping blindly removed that
            # code's frame, so its callees were credited to its caller. Close
            # the frames above the matching one; ignore an exit with no match.
            depth = next((i for i in range(len(stack) - 1, -1, -1) if stack[i].code is code), None)
            if depth is None:
                return
            while len(stack) > depth + 1:
                self.leave(None, False, None)
        record = stack.pop()
        if record.sample is not None and frame is not None:
            self.diff(record, frame, record.last_line, final=True)
            if returned:
                record.sample["yield" if yielded else "return"] = self.show(value)
                # When the value was produced, so a diff can order a differing
                # return against the variable changes it then causes.
                record.sample["returned_at_ms"] = round(
                    (time.perf_counter_ns() - self.started) / 1e6, 3)
            self.close_sample(record)
        elapsed = time.perf_counter_ns() - record.start
        # Only frames that were clocked on entry (see Frame) are clocked here.
        cpu = time.thread_time_ns() - record.cpu_start if record.cpu_start else 0
        if record.node is not self.root:
            record.node.total_ns += elapsed
            record.node.self_ns += elapsed - record.child_ns
            if cpu:
                record.node.cpu_ns += cpu
                record.node.self_cpu_ns += max(0, cpu - record.child_cpu_ns)
        if stack:
            stack[-1].child_ns += elapsed
            stack[-1].child_cpu_ns += cpu

    def line(self, frame: types.FrameType, line: int) -> None:
        stack = self.stack()
        if not stack:
            return
        record = stack[-1]
        if record.sample is not None and record.lines < LINES_PER_CALL:
            record.lines += 1
            self.diff(record, frame, record.last_line)
            if record.lines == LINES_PER_CALL:
                record.sample["lines_capped"] = LINES_PER_CALL
        record.last_line = line

    def diff(self, record: Frame, frame: types.FrameType, line: int, final: bool = False) -> None:
        if self.values_dropped or not line:
            return
        now = self.snapshot(frame)
        changes = record.sample["changes"]
        for name, shown in now.items():
            if record.snapshot.get(name) == shown:
                continue
            seen = record.counts.get(name, 0) + 1
            record.counts[name] = seen
            if seen <= VALUES_PER_NAME:
                changes.append({"line": line, "name": name, "value": shown,
                                "at_ms": round((time.perf_counter_ns() - self.started) / 1e6, 3)})
                self.count(1)
            else:
                # A loop variable changes thousands of times. Keep the first few
                # and the final value, and count the rest.
                last = record.sample.setdefault("last", {})
                last[name] = {"line": line, "value": shown}
        record.snapshot = now

    def close_sample(self, record: Frame) -> None:
        extra = {name: n - VALUES_PER_NAME for name, n in record.counts.items() if n > VALUES_PER_NAME}
        if extra:
            record.sample["more_changes"] = extra

    def count(self, n: int) -> None:
        self.events += n
        if self.events > self.s.max_events and not self.values_dropped:
            self.values_dropped = True

    def library_call(self, target: object) -> None:
        stack = self.stack()
        if not stack:
            return
        module = getattr(target, "__module__", None) or type(target).__module__
        name = getattr(target, "__qualname__", None) or type(target).__qualname__
        code = getattr(target, "__code__", None)
        if code is not None and self.wanted(code):
            return                              # a project call is its own node
        if isinstance(target, type):
            source = getattr(sys.modules.get(module or ""), "__file__", None)
            init = getattr(getattr(target, "__init__", None), "__code__", None)
            if (init is not None and self.wanted(init)) or (
                    source and any(os.path.normcase(os.path.abspath(source)).startswith(r + os.sep)
                                   for r in self.s.roots)
                    and not any(m in source for m in LIBRARY_MARKERS)):
                return                          # a project class: its methods are nodes
        label = f"{module}.{name}" if module and module != "builtins" else str(name)
        library = stack[-1].node.library
        if label in library or len(library) < 40:
            library[label] = library.get(label, 0) + 1

    def error(self, code: types.CodeType, exc: BaseException, line: int | None, kind: str) -> None:
        if isinstance(exc, (GeneratorExit, StopIteration, StopAsyncIteration)):
            return          # how generators and iteration end, not something that went wrong
        stack = self.stack()
        name = type(exc).__name__
        if stack:
            errors = stack[-1].node.errors
            errors[f"{kind}:{name}"] = errors.get(f"{kind}:{name}", 0) + 1
        if len(self.errors) < 300:
            self.errors.append({"kind": kind, "type": name, "message": self.show(str(exc)),
                                "exception": id(exc), "file": code.co_filename,
                                "function": getattr(code, "co_qualname", code.co_name), "line": line,
                                "at_ms": round((time.perf_counter_ns() - self.started) / 1e6, 3)})

    # ----------------------------------------------------------------- output

    def rss_peak_bytes(self) -> int | None:
        try:
            if os.name == "nt":
                import ctypes
                from ctypes import wintypes

                class Counters(ctypes.Structure):
                    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                                ("PeakWorkingSetSize", ctypes.c_size_t),
                                ("WorkingSetSize", ctypes.c_size_t),
                                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                                ("PagefileUsage", ctypes.c_size_t),
                                ("PeakPagefileUsage", ctypes.c_size_t)]
                counters = Counters()
                counters.cb = ctypes.sizeof(Counters)
                # Without explicit types ctypes passes the pseudo-handle -1 as a
                # 32-bit int, and the call fails silently on 64-bit Windows.
                kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
                kernel32.GetCurrentProcess.restype = wintypes.HANDLE
                psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
                process = kernel32.GetCurrentProcess()
                if psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
                    return int(counters.PeakWorkingSetSize)
                return None
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return int(peak if sys.platform == "darwin" else peak * 1024)
        except Exception:  # noqa: BLE001 - memory is a nice-to-have
            return None

    def payload(self, final: bool) -> dict:
        cpu = time.process_time()
        data = {
            "pid": os.getpid(), "argv": sys.argv, "python": sys.version.split()[0],
            "executable": sys.executable, "started_at": self.wall_started,
            "wall_ms": round((time.perf_counter_ns() - self.started) / 1e6, 3),
            "cpu_ms": round(cpu * 1000, 3), "peak_rss_bytes": self.rss_peak_bytes(),
            "engine": "sys.monitoring" if hasattr(sys, "monitoring") else "sys.settrace",
            "roots": self.s.roots, "libraries_traced": self.s.libs,
            "values_recorded": self.s.values, "values_dropped_after_events": (
                self.s.max_events if self.values_dropped else None),
            "node_limit_reached": self.nodes >= self.s.max_nodes,
            "threads": len(self.stacks), "final": final,
            "errors": self.errors, "tree": self.root.to_json(),
        }
        if not final:
            data["now"] = {str(ident): [f"{r.node.key[1]} ({os.path.basename(r.node.key[0])}:{r.node.key[2]})"
                                        for r in stack] for ident, stack in list(self.stacks.items())}
        if self.s.memory and final:
            try:
                import tracemalloc
                if tracemalloc.is_tracing():
                    snap = tracemalloc.take_snapshot()
                    data["allocations"] = [
                        {"where": f"{s.traceback[0].filename}:{s.traceback[0].lineno}",
                         "kib": round(s.size / 1024, 1), "count": s.count}
                        for s in snap.statistics("lineno")[:15]]
                    data["traced_peak_bytes"] = tracemalloc.get_traced_memory()[1]
            except Exception:  # noqa: BLE001
                pass
        return data

    def write(self, final: bool) -> None:
        name = f"py-{os.getpid()}.json" if final else f"py-{os.getpid()}.live.json"
        path = os.path.join(self.s.out_dir, name)
        try:
            with self.lock:
                data = self.payload(final)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(data, handle)
            os.replace(tmp, path)
        except Exception:  # noqa: BLE001 - never break the traced program
            pass


# ------------------------------------------------------------------ engines

def _start_monitoring(rec: Recorder) -> None:
    mon = sys.monitoring
    tool = None
    for candidate in (mon.PROFILER_ID, 3, 4):
        try:
            mon.use_tool_id(candidate, "icn-trace")
            tool = candidate
            break
        except ValueError:
            continue
    if tool is None:
        raise RuntimeError("no free sys.monitoring tool id")
    E = mon.events
    DISABLE = mon.DISABLE
    busy = rec.busy

    def guard() -> bool:
        if getattr(busy, "on", False):
            return True
        busy.on = True
        return False

    def on_start(code, offset):
        if not rec.wanted(code):
            return DISABLE
        if guard():
            return None
        try:
            frame = sys._getframe(1)
            rec.enter(code, frame)
            if rec.s.values and not rec.values_dropped:
                mon.set_local_events(tool, code, E.LINE | E.CALL)
            else:
                mon.set_local_events(tool, code, E.CALL)
        finally:
            busy.on = False
        return None

    def on_resume(code, offset):
        if not rec.wanted(code):
            return DISABLE
        if guard():
            return None
        try:
            rec.enter(code, sys._getframe(1), resumed=True)
        finally:
            busy.on = False
        return None

    def on_throw(code, offset, exc):
        # A generator resumed by throw() or close(); not a local event, so it
        # can never return DISABLE.
        if not rec.wanted(code) or guard():
            return None
        try:
            rec.enter(code, sys._getframe(1), resumed=True)
        finally:
            busy.on = False
        return None

    def on_return(code, offset, value):
        if not rec.wanted(code):
            return DISABLE
        if guard():
            return None
        try:
            rec.leave(value, True, sys._getframe(1), code=code)
        finally:
            busy.on = False
        return None

    def on_yield(code, offset, value):
        if not rec.wanted(code):
            return DISABLE
        if guard():
            return None
        try:
            rec.leave(value, True, sys._getframe(1), yielded=True, code=code)
        finally:
            busy.on = False
        return None

    def on_unwind(code, offset, exc):
        if not rec.wanted(code) or guard():
            return None
        try:
            rec.error(code, exc, sys._getframe(1).f_lineno, "escaped")
            rec.leave(None, False, sys._getframe(1), code=code)
        finally:
            busy.on = False
        return None

    def on_raise(code, offset, exc):
        if not rec.wanted(code) or guard():
            return None
        try:
            frame = sys._getframe(1)
            rec.error(code, exc, frame.f_lineno, "raised")
        finally:
            busy.on = False
        return None

    def on_handled(code, offset, exc):
        if not rec.wanted(code) or guard():
            return None
        try:
            rec.error(code, exc, sys._getframe(1).f_lineno, "handled")
        finally:
            busy.on = False
        return None

    def on_line(code, line):
        if rec.values_dropped:
            return DISABLE
        if guard():
            return None
        try:
            rec.line(sys._getframe(1), line)
        finally:
            busy.on = False
        return None

    def on_call(code, offset, target, arg0):
        if guard():
            return None
        try:
            rec.library_call(target)
        finally:
            busy.on = False
        return None

    for event, callback in ((E.PY_START, on_start), (E.PY_RESUME, on_resume), (E.PY_THROW, on_throw),
                            (E.PY_RETURN, on_return), (E.PY_YIELD, on_yield),
                            (E.PY_UNWIND, on_unwind), (E.RAISE, on_raise),
                            (E.EXCEPTION_HANDLED, on_handled), (E.LINE, on_line),
                            (E.CALL, on_call)):
        mon.register_callback(tool, event, callback)
    mon.set_events(tool, E.PY_START | E.PY_RESUME | E.PY_THROW | E.PY_RETURN | E.PY_YIELD
                   | E.PY_UNWIND | E.RAISE | E.EXCEPTION_HANDLED)

    def stop():
        try:
            mon.set_events(tool, 0)
            mon.free_tool_id(tool)
        except Exception:  # noqa: BLE001
            pass
    rec.stop = stop  # type: ignore[attr-defined]


def _start_settrace(rec: Recorder) -> None:
    def local(frame, event, arg):
        if getattr(rec.busy, "on", False):
            return local
        rec.busy.on = True
        try:
            if event == "line":
                rec.line(frame, frame.f_lineno)
            elif event == "return":
                rec.leave(arg, True, frame, code=frame.f_code)
            elif event == "exception":
                rec.error(frame.f_code, arg[1], frame.f_lineno, "raised")
        finally:
            rec.busy.on = False
        return local

    def global_trace(frame, event, arg):
        if event != "call" or not rec.wanted(frame.f_code) or getattr(rec.busy, "on", False):
            return None
        rec.busy.on = True
        try:
            rec.enter(frame.f_code, frame)
        finally:
            rec.busy.on = False
        return local

    sys.settrace(global_trace)
    threading.settrace(global_trace)

    def stop():
        sys.settrace(None)
        threading.settrace(None)
    rec.stop = stop  # type: ignore[attr-defined]


_RECORDER: Recorder | None = None


def start() -> Recorder:
    """Begin recording this interpreter. Idempotent."""
    global _RECORDER
    if _RECORDER is not None:
        return _RECORDER
    settings = Settings()
    os.makedirs(settings.out_dir, exist_ok=True)
    rec = _RECORDER = Recorder(settings)
    if settings.memory:
        import tracemalloc
        tracemalloc.start(1)
    if hasattr(sys, "monitoring"):
        _start_monitoring(rec)
    else:
        _start_settrace(rec)

    def live():
        while not rec.finished:
            time.sleep(1.0)
            if not rec.finished:
                rec.write(final=False)
    threading.Thread(target=live, name="icn-trace-live", daemon=True).start()

    previous_hook = sys.excepthook

    def excepthook(kind, value, tb):
        rec.errors.append({"kind": "uncaught", "type": kind.__name__, "message": rec.show(str(value)),
                           "file": tb.tb_frame.f_code.co_filename if tb else None,
                           "function": None, "line": tb.tb_lineno if tb else None,
                           "at_ms": round((time.perf_counter_ns() - rec.started) / 1e6, 3)})
        previous_hook(kind, value, tb)
    sys.excepthook = excepthook

    atexit.register(finish)
    return rec


def finish() -> None:
    rec = _RECORDER
    if rec is None or rec.finished:
        return
    rec.finished = True
    try:
        rec.stop()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    # Frames still open at exit (the module body, a sys.exit deep in a call)
    # are closed now so their time is counted.
    for stack in list(rec.stacks.values()):
        while stack:
            rec.leave(None, False, None)
    rec.write(final=True)
    try:
        os.remove(os.path.join(rec.s.out_dir, f"py-{os.getpid()}.live.json"))
    except OSError:
        pass
