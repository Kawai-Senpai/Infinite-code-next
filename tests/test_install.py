"""The skill and hook installer.

Most of these tests are about what the installer must NOT do. It edits a file
the user owns, so the failure that matters is not "the hook was missing" but
"their other settings are gone".
"""

from __future__ import annotations

import json

import pytest

from icn import install as install_mod


def test_install_writes_the_skill_and_the_hook(tmp_path):
    report = install_mod.install(tmp_path)
    assert report["ok"] is True

    skill = tmp_path / install_mod.SKILL_DIR / "SKILL.md"
    assert skill.exists()
    assert skill.read_text(encoding="utf-8").startswith("---\nname: icn-workflow")

    settings = json.loads((tmp_path / install_mod.SETTINGS_PATH).read_text(encoding="utf-8"))
    entries = settings["hooks"]["SessionStart"]
    assert any(h.get("_source") == install_mod.HOOK_MARKER
               for e in entries for h in e["hooks"])


def test_installing_twice_changes_nothing_the_second_time(tmp_path):
    install_mod.install(tmp_path)
    second = install_mod.install(tmp_path)
    assert second["written"] == []
    assert len(second["unchanged"]) == 2

    settings = json.loads((tmp_path / install_mod.SETTINGS_PATH).read_text(encoding="utf-8"))
    assert len(settings["hooks"]["SessionStart"]) == 1, "re-install duplicated the hook"


def test_existing_settings_are_preserved(tmp_path):
    """The installer merges. It must never replace the file."""
    path = tmp_path / install_mod.SETTINGS_PATH
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "model": "opus",
        "env": {"FOO": "bar"},
        "hooks": {
            "SessionStart": [
                {"matcher": "*", "hooks": [{"type": "command", "command": "echo mine"}]}
            ],
            "PreToolUse": [
                {"matcher": "Bash", "hooks": [{"type": "command", "command": "echo pre"}]}
            ],
        },
    }, indent=2), encoding="utf-8")

    install_mod.install(tmp_path)
    settings = json.loads(path.read_text(encoding="utf-8"))

    assert settings["model"] == "opus"
    assert settings["env"] == {"FOO": "bar"}
    assert settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"] == "echo pre"

    commands = [h["command"] for e in settings["hooks"]["SessionStart"] for h in e["hooks"]]
    assert "echo mine" in commands, "the user's own SessionStart hook was dropped"
    assert len(settings["hooks"]["SessionStart"]) == 2


def test_malformed_settings_are_refused_not_overwritten(tmp_path):
    """A file we cannot parse may still hold the user's configuration."""
    path = tmp_path / install_mod.SETTINGS_PATH
    path.parent.mkdir(parents=True)
    path.write_text("{ this is not json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        install_mod.install(tmp_path)

    assert path.read_text(encoding="utf-8") == "{ this is not json"


def test_no_hook_leaves_settings_untouched(tmp_path):
    report = install_mod.install(tmp_path, with_hook=False)
    assert report["ok"] is True
    assert not (tmp_path / install_mod.SETTINGS_PATH).exists()
    assert (tmp_path / install_mod.SKILL_DIR / "SKILL.md").exists()


def test_a_stale_icn_hook_is_replaced_not_duplicated(tmp_path):
    path = tmp_path / install_mod.SETTINGS_PATH
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"hooks": {"SessionStart": [
        {"matcher": "*", "hooks": [{
            "type": "command",
            "command": "echo an older reminder",
            "_source": install_mod.HOOK_MARKER,
        }]}
    ]}}, indent=2), encoding="utf-8")

    install_mod.install(tmp_path)
    settings = json.loads(path.read_text(encoding="utf-8"))
    entries = settings["hooks"]["SessionStart"]
    assert len(entries) == 1
    assert "an older reminder" not in entries[0]["hooks"][0]["command"]


def test_install_reports_a_missing_directory(tmp_path):
    report = install_mod.install(tmp_path / "nope")
    assert report["ok"] is False
    assert "not a directory" in report["error"]
