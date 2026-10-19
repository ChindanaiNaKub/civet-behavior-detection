"""Portable launcher and HTTP media contracts; no GPU or third-party runtime."""

import io
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import review
import review_launcher


def packet(tmp_path):
    root = tmp_path / "review folder with spaces"
    (root / "scripts").mkdir(parents=True)
    (root / "data/raw").mkdir(parents=True)
    source = Path(review.__file__).parent
    for name in ("review.py", "review_page.html", "review_i18n.json", "labels.py"):
        shutil.copyfile(source / name, root / "scripts" / name)
    shutil.copyfile(source / "review_launcher.py", root / "start_review.py")
    (root / "data/clips.csv").write_text("clip_id,relative_path,duration_s\nc,c.mp4,8\n", encoding="utf-8")
    (root / "data/gold-clips.txt").write_text("c\n", encoding="utf-8")
    (root / "data/raw/c.mp4").write_bytes(b"test video only")
    return root


def test_launcher_checks_packet_from_another_working_directory_without_writing(tmp_path):
    root = packet(tmp_path)
    protected = root / "data/annotations.csv"
    protected.write_bytes(b"friend's original records")
    command = [sys.executable, str(root / "start_review.py"), "--check", "--no-browser"]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Ready: 1 blind clips" in result.stdout
    assert protected.read_bytes() == b"friend's original records"
    assert not (root / "data/annotations-second.csv").exists()
    (root / "data/raw/c.mp4").unlink()
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0 and "Missing video files" in result.stderr


def test_launcher_uses_absolute_paths_and_only_secondary_output(tmp_path):
    root = packet(tmp_path)
    args = review_launcher.review_arguments(root, None, False)
    assert args[args.index("--out") + 1] == str(root / "data/annotations-second.csv")
    assert "--auto-port" in args and "--open-browser" in args
    assert str(root / "data/annotations.csv") in args
    explicit = review_launcher.review_arguments(root, 9000, True)
    assert explicit[explicit.index("--port") + 1] == "9000"
    assert "--auto-port" not in explicit and "--open-browser" not in explicit


def test_utf8_bom_ledgers_and_gold_list_work_under_any_locale(tmp_path):
    root = packet(tmp_path)
    ledger = root / "data/clips.csv"
    ledger.write_text("clip_id,relative_path,duration_s\nc,c.mp4,8\n", encoding="utf-8-sig")
    assert review.load_clips(ledger, root / "data/raw")[0].clip_id == "c"
    gold = root / "data/gold-clips.txt"
    gold.write_text("c\n", encoding="utf-8-sig")
    assert review.id_set([], gold) == {"c"}


@pytest.mark.parametrize("header,status,expected", [
    ("bytes=2-4", 206, b"234"), ("bytes=7-", 206, b"789"),
    ("bytes=-3", 206, b"789"), ("bytes=50-", 416, b""),
    ("bytes=bad", 416, b""), ("bytes=0-1,3-4", 416, b""),
])
def test_browser_seek_ranges(tmp_path, header, status, expected):
    path = tmp_path / "video.mp4"
    path.write_bytes(b"0123456789")
    handler = object.__new__(review.Handler)
    handler.headers = {"Range": header}
    codes, headers = [], {}
    handler.send_response = codes.append
    handler.send_header = lambda key, value: headers.update({key: value})
    handler.end_headers = lambda: None
    handler.wfile = io.BytesIO()
    handler._serve_video(path)
    assert codes == [status]
    assert handler.wfile.getvalue() == expected
    assert int(headers["Content-Length"]) == len(expected)


def test_busy_preferred_port_falls_back_without_stopping_other_process(tmp_path, monkeypatch, capsys):
    root = packet(tmp_path)
    calls = []
    closed = []

    class Server:
        server_port = 43210

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            closed.append(True)

    def create(address, handler):
        calls.append(address)
        if len(calls) == 1:
            raise OSError("address already in use")
        return Server()

    monkeypatch.setattr(review, "ThreadingHTTPServer", create)
    args = review_launcher.review_arguments(root, None, True)
    assert review.main(args) == 0
    assert calls == [("127.0.0.1", 8000), ("127.0.0.1", 0)]
    assert closed == [True]
    assert "http://127.0.0.1:43210/" in capsys.readouterr().out


def test_both_languages_cover_ui_and_preserve_format_parameters():
    source = Path(review.__file__).parent
    translations = json.loads((source / "review_i18n.json").read_text(encoding="utf-8"))
    assert translations.keys() == {"en", "th"}
    assert translations["en"].keys() == translations["th"].keys()
    for key, english in translations["en"].items():
        assert english.strip() and translations["th"][key].strip()
        assert set(re.findall(r"\{(\w+)\}", english)) == set(re.findall(r"\{(\w+)\}", translations["th"][key]))
    html = (source / "review_page.html").read_text(encoding="utf-8")
    referenced = set(re.findall(r'data-i18n(?:-placeholder|-aria)?="([^"]+)"', html))
    referenced.update(re.findall(r"\b(?:message|saved|setStep|t)\('([^']+)'", html))
    assert referenced <= translations["en"].keys()
    assert set(review.BEHAVIORS) <= translations["en"].keys()
    assert '<html lang="en">' in html
    assert "let language='en'" in html
    assert "__TRANSLATIONS__" not in review.PAGE
