"""Preview regression tests, including image blocks over MCP stdio."""

import base64
import io
import json
import subprocess
from pathlib import Path

import pytest

from icn import preview as preview_mod
from icn.server import preview
from mcp_client import McpStdioClient


def _pillow():
    return pytest.importorskip("PIL.Image")


def _pdf_engine():
    engine = preview_mod._load_pymupdf()
    if engine is None:
        pytest.skip("PyMuPDF is not installed")
    return engine


def _pdf(path):
    engine = _pdf_engine()
    with engine.open() as document:
        for color in ((1, 0, 0), (0, 0, 1)):
            page = document.new_page(width=480, height=240)
            page.draw_rect(page.rect, fill=color)
        document.save(path)


def _decode(result, index=0):
    Image = _pillow()
    return Image.open(io.BytesIO(result[index + 1].data))


def test_image_preview_preserves_orientation_and_scales_to_fit(tmp_path):
    Image = _pillow()
    source = Image.new("RGB", (1600, 800), "red")
    exif = source.getexif()
    exif[274] = 6
    source.save(tmp_path / "photo.jpg", exif=exif)

    result = preview("photo.jpg", root=str(tmp_path), max_side=640)

    assert result[0]["path"] == str(tmp_path / "photo.jpg")
    assert result[0]["kind"] == "image"
    assert _decode(result).size == (320, 640)


def test_gif_frames_have_a_resume_page(tmp_path):
    Image = _pillow()
    first = Image.new("RGB", (30, 25), "red")
    second = Image.new("RGB", first.size, "blue")
    first.save(tmp_path / "motion.gif", save_all=True, append_images=[second], duration=100)

    assert preview("motion.gif", root=str(tmp_path))[0]["next_page"] == 2
    result = preview("motion.gif", root=str(tmp_path), page=2, count=4)
    assert result[0]["next_page"] is None
    assert result[0]["images"][0]["label"] == "Frame 2"
    assert _decode(result).getpixel((15, 10)) == (0, 0, 255)


def test_pdf_pages_render_without_a_repository_or_cache(tmp_path, isolated_store):
    _pillow()
    source = tmp_path / "scanned report.pdf"
    _pdf(source)
    result = preview(str(source), page=2, max_side=640, root=str(tmp_path))

    assert result[0]["total_pages"] == 2
    assert result[0]["images"][0]["label"] == "Page 2"
    image = _decode(result)
    assert image.size == (639, 320)
    assert image.getpixel((100, 100)) == (0, 0, 255)
    assert list(tmp_path.iterdir()) == [source]
    assert not isolated_store.exists()


def test_svg_detection_uses_vector_rendering(tmp_path):
    _pillow()
    _pdf_engine()
    (tmp_path / "diagram.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="96" height="48">'
        '<rect width="96" height="48" fill="red"/></svg>', encoding="utf-8",
    )
    result = preview("diagram.svg", root=str(tmp_path), max_side=640)
    assert result[0]["kind"] == "document"
    assert _decode(result).getpixel((100, 100)) == (255, 0, 0)


def test_office_uses_isolated_profile_and_cleans_conversion_output(tmp_path, monkeypatch):
    _pillow()
    _pdf_engine()
    source = tmp_path / "office file.docx"
    source.touch()
    monkeypatch.setattr(preview_mod, "_office_executable", lambda: "test-soffice")
    output_dirs = []

    def convert(args, *, stdin, capture_output, timeout, creationflags):
        assert args[0] == "test-soffice"
        assert stdin == subprocess.DEVNULL and capture_output and timeout > 0
        assert args[-1] == str(source)
        outdir = Path(args[args.index("--outdir") + 1])
        assert args[1] == f"-env:UserInstallation={(outdir / 'profile').as_uri()}"
        assert outdir != tmp_path
        output_dirs.append(outdir)
        _pdf(outdir / "office file.pdf")
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(preview_mod.subprocess, "run", convert)
    result = preview(str(source), count=2, root=str(tmp_path))
    assert result[0]["kind"] == "office"
    assert result[0]["shown"] == 2
    assert _decode(result, 1).getpixel((100, 100)) == (0, 0, 255)
    assert len(output_dirs) == 1 and not output_dirs[0].exists()


def test_text_is_wrapped_and_pageable_at_minimum_size(tmp_path):
    _pillow()
    value = "a" * 7000
    source = tmp_path / "long.log"
    source.write_text(value, encoding="utf-16")
    first = preview("long.log", root=str(tmp_path), max_side=640)
    second = preview("long.log", root=str(tmp_path), max_side=640, page=2)
    inline = preview(text=value, root=str(tmp_path), max_side=640)

    assert first[0]["next_page"] == 2
    assert second[0]["next_page"] == 3
    assert first[1].data == inline[1].data
    assert first[1].data != second[1].data
    pixels = _decode(first).convert("L").point(lambda value: 255 - value)
    assert pixels.getbbox()[3] < pixels.height - 10
    assert max(pixels.size) <= 640


def test_video_samples_a_requested_interval_and_emits_a_filmstrip(tmp_path):
    Image = _pillow()
    av = pytest.importorskip("av")
    path = tmp_path / "short clip.mp4"
    with av.open(str(path), mode="w") as output:
        stream = output.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 160, 90, "yuv420p"
        for index in range(40):
            color = "red" if index < 20 else "blue"
            frame = av.VideoFrame.from_image(Image.new("RGB", (160, 90), color))
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)

    result = preview(str(path), start_seconds=2, end_seconds=3, count=3,
                     max_side=640, root=str(tmp_path))
    summary = result[0]
    assert summary["kind"] == "video"
    assert summary["duration_seconds"] == 4
    assert len(summary["timestamps_seconds"]) == 3
    assert all(2 <= timestamp < 3 for timestamp in summary["timestamps_seconds"])
    assert summary["timestamps_seconds"] == sorted(summary["timestamps_seconds"])
    image = _decode(result)
    assert image.format == "JPEG"
    red, _, blue = image.getpixel((150, 35))
    assert blue > 200 and red < 35
    assert max(image.size) <= 640


def test_preview_reports_input_failures(tmp_path):
    _pillow()
    (tmp_path / "unsupported.bin").write_bytes(b"binary\0payload")
    (tmp_path / "bad.png").write_bytes(b"broken")
    root = str(tmp_path)
    for arguments, message in [
        ({}, "exactly one"),
        ({"path": "bad.png", "text": ""}, "exactly one"),
        ({"path": "missing.png"}, "not found"),
        ({"path": "unsupported.bin"}, "binary"),
        ({"path": "bad.png"}, "Could not preview"),
        ({"text": "hello", "page": 2}, "out of range"),
        ({"text": "hello", "count": 100}, "between 1 and 4"),
        ({"text": "hello", "end_seconds": 2}, "videos only"),
    ]:
        result = preview(root=root, **arguments)
        assert result["ok"] is False
        assert message in result["error"]


def test_stdio_preview_returns_an_mcp_image_and_no_source_text(tmp_path):
    Image = _pillow()
    (tmp_path / "notes.txt").write_text("verbatim source stays in pixels", encoding="utf-8")

    with McpStdioClient(tmp_path) as client:
        result = client.request("tools/call", {
            "name": "preview", "arguments": {"path": "notes.txt", "max_side": 640}
        })
        assert not result.get("isError"), result
        metadata, image = result["content"]

    assert metadata["type"] == "text"
    summary = json.loads(metadata["text"])
    assert summary["ok"] and summary["kind"] == "text"
    assert "verbatim source" not in metadata["text"]
    assert image["type"] == "image" and image["mimeType"] == "image/png"
    with Image.open(io.BytesIO(base64.b64decode(image["data"], validate=True))) as decoded:
        assert decoded.format == "PNG"
        assert decoded.width == 640
