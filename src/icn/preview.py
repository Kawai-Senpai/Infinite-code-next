"""Render local files to bounded image previews without indexing them."""

from __future__ import annotations

import codecs
import io
import math
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING

from .papers import _load_pymupdf

if TYPE_CHECKING:
    from PIL.Image import Image as PillowImage
    from PIL.ImageFont import FreeTypeFont

IMAGE_FORMATS = ("PNG", "JPEG", "GIF", "WEBP", "TIFF", "BMP", "ICO", "AVIF",
                 "JPEG2000", "PPM", "TGA", "PSD", "DDS", "QOI")
IMAGE_SUFFIXES = set("""
    .png .apng .jpg .jpeg .gif .webp .tif .tiff .bmp .ico .avif .jp2
    .pbm .pgm .ppm .tga .psd .dds .qoi
""".split())
DOCUMENT_SUFFIXES = {".pdf", ".svg", ".xps", ".oxps", ".epub", ".mobi", ".fb2", ".cbz"}
OFFICE_SUFFIXES = set("""
    .doc .docx .odt .rtf .ppt .pptx .odp .xls .xlsx .ods
""".split())
VIDEO_SUFFIXES = set("""
    .mp4 .m4v .mov .webm .mkv .avi .mpg .mpeg .ts .mts .m2ts .flv .wmv .ogv .mxf
""".split())
VIDEO_FORMATS = "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,avi,mpeg,mpegts,flv,asf,ogg,mxf"
MAX_TEXT_BYTES = 16 * 1024 * 1024
MAX_SOURCE_PIXELS = 40_000_000
MAX_IMAGE_BYTES = 1_500_000
MAX_DOCUMENT_BYTES = 256 * 1024 * 1024
MAX_FRAMES = 12
MAX_PAGES = 4
TEXT_ROWS = 38


class PreviewError(Exception):
    """An input or renderer failure the caller can act on."""


@dataclass
class Rendered:
    label: str
    image: PillowImage


def _pillow():
    try:
        from PIL import Image
    except ImportError as exc:
        raise PreviewError("Image rendering needs Pillow. Install infinite-code-next[preview].") from exc
    return Image


def _open_raster(path: Path):
    Image = _pillow()
    # Optional codecs (such as AVIF) vary between Pillow releases and builds.
    registered = set(Image.registered_extensions().values())
    formats = [image_format for image_format in IMAGE_FORMATS if image_format in registered]
    return Image.open(path, formats=formats)


def _bounded_size(width: int, height: int, limit: int) -> tuple[int, int]:
    scale = min(1, limit / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def _page_range(page: int, count: int, total: int) -> range:
    if page > total:
        raise PreviewError(f"Page or frame {page} is out of range ({total} available).")
    return range(page, min(page + count, total + 1))


def _page_info(page: int, shown: int, total: int | None, more: bool) -> dict:
    return {"page": page, "shown": shown, "total_pages": total,
            "next_page": page + shown if more else None}


def _encode(rendered: Rendered, *, photographic: bool) -> tuple[dict, bytes]:
    image = rendered.image
    fmt = "jpeg" if photographic else "png"
    while True:
        buffer = io.BytesIO()
        options = {"quality": 87} if fmt == "jpeg" else {}
        image.save(buffer, format=fmt.upper(), **options)
        if buffer.tell() <= MAX_IMAGE_BYTES:
            return ({"label": rendered.label, "width": image.width, "height": image.height,
                     "mime_type": f"image/{fmt}"}, buffer.getvalue())
        if fmt == "png":
            fmt = "jpeg"
        else:
            image = image.resize(_bounded_size(*image.size, int(max(image.size) * 0.8)),
                                 _pillow().Resampling.LANCZOS)


def _raster(path: Path, page: int, count: int, size: int) -> tuple[list[Rendered], dict]:
    from PIL import ImageOps

    Image = _pillow()
    with _open_raster(path) as source:
        total = getattr(source, "n_frames", 1)
        shown = _page_range(page, count, total)
        rendered = []
        for number in shown:
            source.seek(number - 1)
            if source.width * source.height > MAX_SOURCE_PIXELS:
                raise PreviewError("Image frame exceeds the 40 megapixel preview limit.")
            image = ImageOps.exif_transpose(source).convert("RGBA")
            image.thumbnail((size, size), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", image.size, "white")
            canvas.paste(image, mask=image.getchannel("A"))
            rendered.append(Rendered(f"Frame {number}" if total > 1 else "Image", canvas))
    return rendered, _page_info(page, len(shown), total, shown.stop <= total)


def _document(path: Path, page: int, count: int, size: int) -> tuple[list[Rendered], dict]:
    engine = _load_pymupdf()
    if engine is None:
        raise PreviewError("Document rendering needs PyMuPDF. Install infinite-code-next[preview].")
    Image = _pillow()
    with engine.open(path) as document:
        if document.needs_pass:
            raise PreviewError("This document requires a password. Preview an unlocked copy.")
        total = len(document)
        shown = _page_range(page, count, total)
        rendered = []
        for number in shown:
            source = document[number - 1]
            width, height = source.rect.width, source.rect.height
            if min(width, height) <= 0 or not math.isfinite(width + height):
                raise PreviewError(f"Page {number} has invalid dimensions.")
            scale = (size - 1) / max(width, height)
            pixmap = source.get_pixmap(matrix=engine.Matrix(scale, scale),
                                       colorspace=engine.csRGB, alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            rendered.append(Rendered(f"Page {number}", image))
    return rendered, _page_info(page, len(shown), total, shown.stop <= total)


def _office_executable() -> str:
    for name in ("soffice.com", "soffice", "libreoffice"):
        if executable := shutil.which(name):
            return executable
    for directory in filter(None, (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"))):
        executable = Path(directory) / "LibreOffice" / "program" / "soffice.com"
        if executable.is_file():
            return str(executable)
    macos = Path("/Applications/LibreOffice.app/Contents/MacOS/soffice")
    if macos.is_file():
        return str(macos)
    raise PreviewError("Office preview needs LibreOffice. Install it and make soffice available on PATH.")


def _office(path: Path, page: int, count: int, size: int) -> tuple[list[Rendered], dict]:
    executable = _office_executable()
    with tempfile.TemporaryDirectory(prefix="icn-preview-") as directory:
        output = Path(directory)
        profile = (output / "profile").as_uri()
        pdf = output / path.with_suffix(".pdf").name
        try:
            process = subprocess.run(
                [executable, f"-env:UserInstallation={profile}", "--headless",
                 "--convert-to", "pdf", "--outdir", str(output), str(path)],
                stdin=subprocess.DEVNULL, capture_output=True, timeout=45,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as exc:
            raise PreviewError("LibreOffice did not convert the document within 45 seconds.") from exc
        if process.returncode or not pdf.is_file():
            detail = (process.stderr or process.stdout).decode(errors="replace").strip()[:300]
            raise PreviewError(f"LibreOffice could not convert this document. {detail}".strip())
        return _document(pdf, page, count, size)


def _font(size: int):
    from PIL import ImageFont

    for name in ("DejaVuSansMono.ttf", "consola.ttf", "Menlo.ttc", "Courier New.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _text_rows(value: str, font: FreeTypeFont, width: int) -> Iterator[tuple[str, str]]:
    @lru_cache(maxsize=2048)
    def advance(char: str) -> float:
        return font.getlength(char)

    for number, line in enumerate(value.splitlines() or [""], 1):
        line = line.expandtabs(4)
        start = 0
        pixels = 0.0
        label = str(number)
        for index, char in enumerate(line):
            length = advance(char)
            if pixels + length > width and index > start:
                yield label, line[start:index]
                start, pixels, label = index, 0.0, ""
            pixels += length
        yield label, line[start:]


def _text(value: str, page: int, count: int, size: int) -> tuple[list[Rendered], dict]:
    Image = _pillow()
    from PIL import ImageDraw

    margin = size // 32
    font = _font(max(12, size // 62))
    row_height = sum(font.getmetrics()) + 2
    gutter = math.ceil(font.getlength("00000000"))
    rows_per_page = min(TEXT_ROWS, (size - 2 * margin) // row_height)
    rows = _text_rows(value, font, size - 2 * margin - gutter)
    offset = (page - 1) * rows_per_page
    capacity = count * rows_per_page
    selected = list(islice(rows, offset, offset + capacity + 1))
    if not selected:
        raise PreviewError(f"Text page {page} is out of range.")
    more = len(selected) > capacity
    rendered = []
    for index in range(0, min(len(selected), capacity), rows_per_page):
        chunk = selected[index:min(index + rows_per_page, capacity)]
        canvas = Image.new("RGB", (size, 2 * margin + row_height * len(chunk)), "white")
        draw = ImageDraw.Draw(canvas)
        for row, (number, content) in enumerate(chunk):
            y = margin + row * row_height
            draw.text((margin, y), number, font=font, fill="#52525b")
            draw.text((margin + gutter, y), content, font=font, fill="#18181b")
        rendered.append(Rendered(f"Text page {page + index // rows_per_page}", canvas))
    total = None if more else page + len(rendered) - 1
    return rendered, _page_info(page, len(rendered), total, more)


def _read_text(path: Path) -> str:
    with path.open("rb") as source:
        data = source.read(MAX_TEXT_BYTES + 1)
    if len(data) > MAX_TEXT_BYTES:
        raise PreviewError("Text file exceeds the 16 MiB text preview limit.")
    encoding = "utf-8-sig"
    if data.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        encoding = "utf-32"
    elif data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        encoding = "utf-16"
    try:
        value = data.decode(encoding)
    except UnicodeDecodeError as exc:
        raise PreviewError("Unsupported file format or text encoding. Use UTF-8 or BOM-marked UTF-16/32.") from exc
    if "\0" in value or any(ord(char) < 32 and char not in "\t\r\n\f" for char in value):
        raise PreviewError("This file contains binary data and could not be previewed as text.")
    return value


def _video_sample(container, stream, seconds: float, start: float, stop: float,
                  origin: int, deadline: float):
    ticks = origin + round(seconds / stream.time_base)
    container.seek(ticks, stream=stream)
    last = None
    for frame in container.decode(stream):
        if time.monotonic() >= deadline:
            raise PreviewError("Video sampling exceeded its time limit. Request fewer frames.")
        if frame.pts is None:
            continue
        timestamp = float((frame.pts - origin) * stream.time_base)
        if timestamp >= stop:
            break
        if timestamp >= start:
            last = (timestamp, frame)
        if frame.pts >= ticks:
            break
    return last


def _video_tile(frame, width: int, height: int) -> PillowImage:
    Image = _pillow()
    rotation = getattr(frame, "rotation", 0) or 0
    if abs(rotation) % 180 == 90:
        width, height = height, width
    scale = min(1, width / frame.width, height / frame.height)
    image = frame.reformat(width=max(1, round(frame.width * scale)),
                           height=max(1, round(frame.height * scale)), format="rgb24").to_image()
    if rotation:
        image = image.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True)
        image.thumbnail((height, width) if abs(rotation) % 180 == 90 else (width, height))
    return image


def _video(path: Path, count: int, size: int, start: float, end: float | None) -> tuple[list[Rendered], dict]:
    try:
        import av
    except ImportError as exc:
        raise PreviewError("Video preview needs PyAV. Install infinite-code-next[preview].") from exc

    Image = _pillow()
    from PIL import ImageDraw

    columns = min(count, 3 if count > 4 else 2)
    rows = math.ceil(count / columns)
    tile = size // max(columns, rows)
    label_height = 25
    font = _font(13)
    deadline = time.monotonic() + 30
    frames = {}
    try:
        with path.open("rb") as source, av.open(
            source, container_options={"format_whitelist": VIDEO_FORMATS,
                                       "protocol_whitelist": "file"},
            timeout=(10, 10),
        ) as container:
            if not container.streams.video:
                raise PreviewError("The file does not contain a video stream.")
            stream = container.streams.video[0]
            if not stream.time_base:
                raise PreviewError("The video stream does not provide a time base.")
            if stream.width * stream.height > MAX_SOURCE_PIXELS:
                raise PreviewError("Video frames exceed the 40 megapixel preview limit.")
            origin = stream.start_time
            if origin is None:
                origin = round((container.start_time or 0) / av.time_base / stream.time_base)
            duration = (float(stream.duration * stream.time_base) if stream.duration
                        else (container.duration or 0) / av.time_base)
            stop = end if end is not None else duration
            if stop <= start or (duration and stop > duration + 0.001):
                if not duration and end is None:
                    raise PreviewError("Video duration is unavailable. Specify end_seconds.")
                raise PreviewError(f"Invalid video time range for duration {duration:.3f}s.")
            for index in range(count):
                if time.monotonic() >= deadline:
                    raise PreviewError("Video sampling exceeded its time limit. Request fewer frames.")
                target = start + (stop - start) * (index + 0.5) / count
                if sample := _video_sample(container, stream, target, start, stop, origin, deadline):
                    timestamp, frame = sample
                    if timestamp not in frames:
                        if frame.width * frame.height > MAX_SOURCE_PIXELS:
                            raise PreviewError("A video frame exceeds the 40 megapixel preview limit.")
                        frames[timestamp] = _video_tile(frame, tile, tile - label_height)
    except av.FFmpegError as exc:
        raise PreviewError(f"Could not decode or seek in this video: {exc}") from exc
    if not frames:
        raise PreviewError("No timestamped video frames could be decoded in that range.")

    timestamps = sorted(frames)
    height = max(image.height for image in frames.values()) + label_height
    filmstrip = Image.new("RGB", (min(columns, len(frames)) * tile,
                                  math.ceil(len(frames) / columns) * height), "#18181b")
    draw = ImageDraw.Draw(filmstrip)
    for index, timestamp in enumerate(timestamps):
        image = frames[timestamp]
        row, column = divmod(index, columns)
        x, y = column * tile, row * height
        filmstrip.paste(image, (x + (tile - image.width) // 2, y))
        draw.text((x + 8, y + height - label_height + 3), f"{timestamp:.3f}s",
                  font=font, fill="white")
    return [Rendered("Video filmstrip", filmstrip)], {
        "duration_seconds": round(duration, 6) if duration else None,
        "timestamps_seconds": [round(timestamp, 6) for timestamp in timestamps],
    }


def _detect(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in OFFICE_SUFFIXES:
        return "office"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    if suffix in DOCUMENT_SUFFIXES:
        return "document"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    with path.open("rb") as source:
        if b"%PDF-" in source.read(1024):
            return "document"
    from PIL import UnidentifiedImageError

    try:
        with _open_raster(path):
            return "image"
    except UnidentifiedImageError:
        return "text"


def render(root: Path, *, path: str | None, text: str | None, kind: str, page: int,
           count: int | None, max_side: int, start_seconds: float | None,
           end_seconds: float | None) -> tuple[dict, list[bytes]]:
    """Render one source and return metadata alongside aligned encoded images."""
    if (path is None) == (text is None):
        raise PreviewError("Provide exactly one of path or text.")
    if kind not in {"auto", "image", "document", "office", "text", "video"}:
        raise PreviewError("kind must be auto, image, document, office, text, or video.")
    if not 640 <= max_side <= 2048:
        raise PreviewError("max_side must be between 640 and 2048 pixels.")
    if not 1 <= page <= 1_000_000:
        raise PreviewError("page must be between 1 and 1000000.")

    # Optional imports stay out of the default MCP server startup path.
    Image = _pillow()
    source = None
    try:
        if path is not None:
            if not path.strip():
                raise PreviewError("path must name a local file.")
            source = (root / Path(path).expanduser()).resolve()
            if not source.is_file():
                raise PreviewError(f"Local file not found: {source}")
            kind = _detect(source) if kind == "auto" else kind
        elif kind in {"auto", "text"}:
            kind = "text"
        else:
            raise PreviewError("Inline text requires kind='text' or 'auto'.")

        limit = MAX_FRAMES if kind == "video" else MAX_PAGES
        if count is None:
            count = 6 if kind == "video" else 1
        if not 1 <= count <= limit:
            raise PreviewError(f"count must be between 1 and {limit} for {kind} previews.")
        if kind == "video":
            if page != 1:
                raise PreviewError("Use start_seconds and end_seconds to select a video range.")
            start = 0.0 if start_seconds is None else start_seconds
            if not math.isfinite(start) or start < 0 or (
                end_seconds is not None and (not math.isfinite(end_seconds) or end_seconds <= start)
            ):
                raise PreviewError("Video times must be finite and end_seconds must exceed start_seconds >= 0.")
            rendered, info = _video(source, count, max_side, start, end_seconds)
        elif start_seconds is not None or end_seconds is not None:
            raise PreviewError("start_seconds and end_seconds apply to videos only.")
        elif kind == "text":
            value = _read_text(source) if source is not None else text
            if source is None and len(value) > MAX_TEXT_BYTES:
                raise PreviewError("Inline text is too large. Pass a local file path.")
            rendered, info = _text(value, page, count, max_side)
        else:
            if source.stat().st_size > MAX_DOCUMENT_BYTES:
                raise PreviewError("File exceeds the 256 MiB image and document preview limit.")
            renderer = {"image": _raster, "document": _document, "office": _office}[kind]
            rendered, info = renderer(source, page, count, max_side)
        images, payloads = zip(*(_encode(item, photographic=kind == "video") for item in rendered))
    except (OSError, RuntimeError, ValueError, OverflowError, Image.DecompressionBombError) as exc:
        raise PreviewError(f"Could not preview this {kind} input: {exc}") from exc

    return ({"ok": True, "kind": kind, "path": str(source) if source is not None else None,
             "resolved_root": str(root), **info, "images": list(images)}, list(payloads))
