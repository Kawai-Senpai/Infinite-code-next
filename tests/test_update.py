"""The updater stages a pinned revision and keeps launch usable offline."""

import subprocess

from icn import update

OLD = "3" * 40
NEW = "a" * 40


def test_only_a_full_revision_of_the_fixed_branch_is_accepted(monkeypatch):
    def remote(command, **kwargs):
        assert command[-2:] == [update.UPSTREAM, update.BRANCH]
        assert kwargs["stdin"] == subprocess.DEVNULL
        return f"{NEW}\t{update.BRANCH}\n"

    monkeypatch.setattr(subprocess, "check_output", remote)
    assert update._latest("git") == NEW
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "main")
    assert update._latest("git") is None


def test_prepare_keeps_previous_revision_until_candidate_works_offline(monkeypatch):
    calls = []

    def probe(command, timeout):
        calls.append(command)
        return OLD in " ".join(command) or "--offline" not in command

    monkeypatch.setattr(update, "_latest", lambda git: OLD)
    monkeypatch.setattr(update, "_probe", probe)
    update.prepare("uv", "git", "preview")
    assert update._prepared("preview") == OLD
    assert update._prepared("") is None

    monkeypatch.setattr(update, "_latest", lambda git: NEW)
    update.prepare("uv", "git", "preview")

    assert update._prepared("preview") == OLD
    assert "--offline" in calls[-1]
    assert f"infinite-code-next[preview] @ git+{update.UPSTREAM}@{NEW}" in calls[-1]


def test_launch_uses_the_revision_available_before_starting_worker(monkeypatch):
    order = []
    monkeypatch.setattr(update.shutil, "which", lambda executable: executable)
    monkeypatch.setattr(update, "_prepared", lambda extras: order.append(extras) or OLD)
    monkeypatch.setattr(update, "_start_worker", lambda flags: order.append(flags))
    monkeypatch.setattr(update, "_probe", lambda command, timeout: True)

    def serve(command, **kwargs):
        assert "stdin" not in kwargs and "stdout" not in kwargs
        assert command[-2:] == ["-m", "icn"]
        assert "--offline" in command
        assert f"infinite-code-next[semantic] @ git+{update.UPSTREAM}@{OLD}" in command
        return 0

    monkeypatch.setattr(subprocess, "call", serve)
    assert update.main(["--semantic"]) == 0
    assert order == ["semantic", ["--semantic"]]


def test_launch_falls_back_when_cached_candidate_cannot_load(monkeypatch):
    from icn import server

    started = []
    monkeypatch.setattr(update.shutil, "which", lambda executable: executable)
    monkeypatch.setattr(update, "_prepared", lambda extras: OLD)
    monkeypatch.setattr(update, "_start_worker", lambda flags: None)
    monkeypatch.setattr(update, "_probe", lambda command, timeout: False)
    monkeypatch.setattr(server, "main", lambda: started.append("installed"))

    assert update.main([]) == 0
    assert started == ["installed"]
