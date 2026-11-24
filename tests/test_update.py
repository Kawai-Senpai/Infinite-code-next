"""The updater stages a pinned revision and keeps launch usable offline."""

import subprocess
import sys

from icn import update

OLD = "3" * 40
NEW = "a" * 40
OLD_SLOT = "oldslot1"


def test_only_a_full_revision_of_the_fixed_branch_is_accepted(monkeypatch):
    def remote(command, **kwargs):
        assert command[-2:] == [update.UPSTREAM, update.BRANCH]
        assert kwargs["stdin"] == subprocess.DEVNULL
        return f"{NEW}\t{update.BRANCH}\n"

    monkeypatch.setattr(subprocess, "check_output", remote)
    assert update._latest("git") == NEW
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "main")
    assert update._latest("git") is None


def test_prepare_keeps_previous_revision_when_new_candidate_fails(monkeypatch):
    monkeypatch.setattr(update, "_latest", lambda git: OLD)
    monkeypatch.setattr(update, "_stage", lambda uv, revision, extras: OLD_SLOT)
    update.prepare("uv", "git", "preview")
    assert update._prepared("preview") == (OLD, OLD_SLOT)
    assert update._prepared("") is None

    monkeypatch.setattr(update, "_latest", lambda git: NEW)
    monkeypatch.setattr(update, "_stage", lambda uv, revision, extras: None)
    update.prepare("uv", "git", "preview")

    assert update._prepared("preview") == (OLD, OLD_SLOT)


def test_stage_installs_a_pinned_commit_and_discards_a_failed_environment(monkeypatch):
    commands = []
    monkeypatch.setattr(update, "_run", lambda args, timeout: commands.append(args) or True)
    monkeypatch.setattr(update, "_probe", lambda python: False)

    assert update._stage("uv", NEW, "preview") is None
    assert f"infinite-code-next[preview] @ git+{update.UPSTREAM}@{NEW}" in commands[-1]
    assert not list(update._environments().iterdir())


def test_launch_uses_the_revision_available_before_starting_worker(monkeypatch):
    order = []
    monkeypatch.delenv("UV_OFFLINE", raising=False)
    monkeypatch.setattr(update.shutil, "which", lambda executable: executable)
    monkeypatch.setattr(update, "_prepared", lambda extras: order.append(extras) or (OLD, OLD_SLOT))
    monkeypatch.setattr(update, "_start_worker", lambda flags: order.append(flags))
    monkeypatch.setattr(update, "_probe", lambda python: True)

    def serve(command, **kwargs):
        assert kwargs["stdin"] is sys.stdin and kwargs["stdout"] is sys.stdout
        assert command[-2:] == ["-m", "icn"]
        assert command[0] == str(update._python(update._environments() / OLD_SLOT))
        return 0

    monkeypatch.setattr(subprocess, "call", serve)
    assert update.main(["--semantic"]) == 0
    assert order == ["semantic", ["--semantic"]]


def test_launch_falls_back_when_cached_candidate_cannot_load(monkeypatch):
    from icn import server

    started = []
    monkeypatch.setattr(update.shutil, "which", lambda executable: executable)
    monkeypatch.setattr(update, "_prepared", lambda extras: (OLD, OLD_SLOT))
    monkeypatch.setattr(update, "_start_worker", lambda flags: None)
    monkeypatch.setattr(update, "_probe", lambda python: False)
    monkeypatch.setattr(server, "main", lambda: started.append("installed"))

    assert update.main([]) == 0
    assert started == ["installed"]


def test_offline_environment_skips_background_network(monkeypatch):
    from icn import server

    def fail_if_started(flags):
        raise AssertionError("offline startup tried to spawn the update worker")

    monkeypatch.setenv("UV_OFFLINE", "1")
    monkeypatch.setattr(update.shutil, "which", lambda executable: executable)
    monkeypatch.setattr(update, "_start_worker", fail_if_started)
    monkeypatch.setattr(server, "main", lambda: None)
    assert update.main([]) == 0
