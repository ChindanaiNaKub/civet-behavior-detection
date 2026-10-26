#!/usr/bin/env python3
"""Build the CMZ Phase 2 presentation deck and its two evidence figures.

Run from the repository root:

    python3 scripts/make_deck.py

Canvas is 16:9 at 1280x720 px, matching the Phase 1 design spec.
Every number on the slides is copied from docs/TASKS-2-AND-3.md,
docs/EVALUATION-REPORT.md or outputs/features/features.md.
"""

from pathlib import Path
import csv

import cv2
import numpy as np
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
OUT = ROOT / "docs" / "CMZ-Phase-2-Presentation.pptx"

# ---------------------------------------------------------------- palette

BG = RGBColor(0xF3, 0xEE, 0xDF)
PANEL = RGBColor(0xE1, 0xE6, 0xDA)
FOREST = RGBColor(0x28, 0x42, 0x36)
RUST = RGBColor(0xC4, 0x5A, 0x3B)
MOSS = RGBColor(0x87, 0x96, 0x6F)
CHAR = RGBColor(0x22, 0x24, 0x1F)
MUTED = RGBColor(0x64, 0x67, 0x5E)
RULE = RGBColor(0x64, 0x67, 0x5E)

SERIF = "Georgia"
SANS = "Arial"
MONO = "DejaVu Sans Mono"

LEFT, CENTER, RIGHT = PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.RIGHT


def PX(v):
    """Design pixels (96 per inch) to EMU."""
    return Emu(int(round(v * 9525)))


# ---------------------------------------------------------------- text

def _write(tf, paras, size, font, color, align, spacing):
    """paras: str | list[str | list[(text, overrides)]]."""
    if isinstance(paras, str):
        paras = [paras]
    tf.clear()
    for i, para_def in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.line_spacing = spacing
        if isinstance(para_def, str):
            para_def = [(para_def, {})]
        for chunk, over in para_def:
            run = para.add_run()
            run.text = chunk
            f = run.font
            f.name = over.get("font", font)
            f.size = Pt(over.get("size", size) * 0.75)
            f.bold = over.get("bold", False)
            f.italic = over.get("italic", False)
            f.color.rgb = over.get("color", color)
            spc = over.get("spc")
            if spc:
                run._r.get_or_add_rPr().set("spc", str(int(spc * 100)))


def txt(slide, x, y, w, h, paras, size=16, font=SANS, color=CHAR,
        align=LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.2, spc=None):
    box = slide.shapes.add_textbox(PX(x), PX(y), PX(w), PX(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    if spc:
        paras = _apply_spc(paras, spc)
    _write(tf, paras, size, font, color, align, spacing)
    return box


def _apply_spc(paras, spc):
    if isinstance(paras, str):
        return [[(paras, {"spc": spc})]]
    out = []
    for p in paras:
        if isinstance(p, str):
            out.append([(p, {"spc": spc})])
        else:
            out.append([(t, {**o, "spc": o.get("spc", spc)}) for t, o in p])
    return out


def _bare(shp):
    """Drop the theme shape style so no inherited shadow or outline survives."""
    style = shp._element.find(qn("p:style"))
    if style is not None:
        shp._element.remove(style)
    shp.shadow.inherit = False


def rule(slide, x, y, w, color=RULE, h=1):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, PX(x), PX(y), PX(w), PX(h))
    _bare(shp)
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    return shp


def arrow(slide, x, y, color=MOSS, size=12):
    return txt(slide, x, y, 28, 16, "→", size=size, color=color,
               align=CENTER, spacing=1.0)


# ---------------------------------------------------------------- tables

def _border(cell, edge, color, w=9525):
    """Explicit cell border, kept in schema order (L, R, T, B)."""
    tag = {"L": "a:lnL", "R": "a:lnR", "T": "a:lnT", "B": "a:lnB"}[edge]
    rank = {"lnL": 0, "lnR": 1, "lnT": 2, "lnB": 3}
    tcPr = cell._tc.get_or_add_tcPr()
    for old in tcPr.findall(qn(tag)):
        tcPr.remove(old)
    ln = etree.Element(qn(tag), w=str(w), cap="flat", cmpd="sng", algn="ctr")
    fill = etree.SubElement(ln, qn("a:solidFill"))
    etree.SubElement(fill, qn("a:srgbClr"), val=str(color))
    etree.SubElement(ln, qn("a:prstDash"), val="solid")

    mine = rank[tag.split(":")[1]]
    pos = 0
    for child in tcPr:
        name = etree.QName(child).localname
        if name in rank and rank[name] < mine:
            pos += 1
        else:
            break
    tcPr.insert(pos, ln)


def table(slide, x, y, widths, heights):
    shape = slide.shapes.add_table(
        len(heights), len(widths), PX(x), PX(y), PX(sum(widths)), PX(sum(heights))
    )
    tbl = shape.table
    tbl.first_row = False
    tbl.horz_banding = False
    tbl.vert_banding = False
    style = tbl._tbl.tblPr.find(qn("a:tableStyleId"))
    if style is not None:
        tbl._tbl.tblPr.remove(style)
    for i, w in enumerate(widths):
        tbl.columns[i].width = PX(w)
    for i, h in enumerate(heights):
        tbl.rows[i].height = PX(h)
    for row in tbl.rows:
        for cell in row.cells:
            cell.fill.solid()
            cell.fill.fore_color.rgb = BG
            cell.margin_left = 0
            cell.margin_right = PX(10)
            cell.margin_top = PX(4)
            cell.margin_bottom = PX(4)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            for edge in ("L", "R", "T", "B"):
                _border(cell, edge, BG, 12700)
    return tbl


def cell(tbl, r, c, paras, size=11, font=SANS, color=CHAR, align=LEFT,
         spacing=1.1, bottom=RULE, bold=False):
    tc = tbl.cell(r, c)
    if bottom is not None:
        _border(tc, "B", bottom, 9525)
    if bold:
        if isinstance(paras, str):
            paras = [[(paras, {"bold": True})]]
        else:
            paras = [[(t, {**o, "bold": o.get("bold", True)}) for t, o in
                      ([(p, {})] if isinstance(p, str) else p)] for p in paras]
    _write(tc.text_frame, paras, size, font, color, align, spacing)


# ---------------------------------------------------------------- page frame

def page(prs, n, kicker, title, title_y=64, title_size=36, title_w=1152):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, PX(1280), PX(720))
    _bare(bg)
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG
    bg.line.fill.background()

    rule(slide, 64, 48, 1152)
    txt(slide, 64, 24, 900, 16, kicker, size=10, color=FOREST, spc=1.6)
    txt(slide, 64, title_y, title_w, 48, title, size=title_size,
        font=SERIF, color=FOREST, spacing=1.05)

    txt(slide, 64, 690, 400, 14, "CMZ / PHASE 2", size=9, color=MUTED, spc=1.2)
    txt(slide, 1056, 690, 160, 14, f"{n:02d} / 10", size=9, color=MUTED,
        align=RIGHT, spc=1.2)
    return slide


def kicker_line(slide, x, y, w, text, color=RUST, size=9):
    txt(slide, x, y, w, 14, text, size=size, color=color, spc=1.4)


# ---------------------------------------------------------------- figures

def _read_frame(clip, t):
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {clip}")
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read {clip} at {t}s")
    return frame


def _muted(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= 0.55
    hsv[..., 2] = np.clip(hsv[..., 2] * 1.03, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    paper = np.array([223, 238, 243], np.float32)  # BGR of #F3EEDF
    return (out * 0.93 + paper * 0.07).astype(np.uint8)


def build_figures():
    FIG.mkdir(parents=True, exist_ok=True)

    cover = _read_frame(
        ROOT / "data/raw/Eating/25690519193422801.mp4", 25.5
    )
    cover = _muted(cover[0:1080, 0:900])
    cv2.imwrite(str(FIG / "cover.jpg"), cover, [cv2.IMWRITE_JPEG_QUALITY, 88])

    clip_id = "25690519194200758"
    t = 7.98
    box = None
    with open(ROOT / "outputs/detection/heldout.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["clip_id"] == clip_id and abs(float(row["time_s"]) - t) < 0.1:
                box = {k: float(row[k]) for k in ("x1", "y1", "x2", "y2", "conf")}
                break
    if box is None:
        raise RuntimeError("held-out box row not found")

    det = _read_frame(ROOT / "data/raw/Resting" / f"{clip_id}.mp4", t)
    det = _muted(det[0:560, 700:1540])
    x1, y1 = int(box["x1"]) - 700, int(box["y1"])
    x2, y2 = int(box["x2"]) - 700, int(box["y2"])
    cv2.rectangle(det, (x1, y1), (x2, y2), (59, 90, 196), 4)  # rust in BGR
    cv2.imwrite(str(FIG / "detection.jpg"), det, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return box["conf"]


# ---------------------------------------------------------------- slides

def slide_cover(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, PX(1280), PX(720))
    _bare(bg)
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG
    bg.line.fill.background()

    rule(s, 64, 48, 1152)
    kicker_line(s, 64, 24, 800, "CMZ PROJECT · PHASE 2 · OCTOBER 2026", FOREST)

    txt(s, 64, 128, 610, 200, ["Three Tasks,", "One Animal,", "One Pipeline"],
        size=46, font=SERIF, color=FOREST, spacing=1.06)
    rule(s, 64, 330, 96, RUST, 2)
    txt(s, 64, 352, 560, 80,
        "Short-clip behavior classification, detection and feature analysis "
        "for a zoo civet", size=16, color=CHAR, spacing=1.35)
    txt(s, 64, 650, 400, 16, "Project team, CMU", size=10, color=MUTED, spc=1.2)
    txt(s, 1056, 690, 160, 14, "01 / 10", size=9, color=MUTED,
        align=RIGHT, spc=1.2)

    rule(s, 706, 88, 1, RUST, 562)
    s.shapes.add_picture(str(FIG / "cover.jpg"), PX(748), PX(88), PX(468), PX(561.6))
    txt(s, 748, 660, 468, 14, "CAMERA 02 · 19 MAY 2026", size=8, color=MUTED, spc=1.2)
    return s


def slide_brief(prs):
    s = page(prs, 2, "02 / SCOPE", "What the brief asked, what we delivered")

    widths = [168, 360, 318, 306]
    tbl = table(s, 64, 150, widths, [32, 114, 114, 114])
    heads = ["TASK", "BRIEF ASKED", "WHAT WE BUILT", "MEASURED RESULT"]
    for c, head in enumerate(heads):
        cell(tbl, 0, c, head, size=8, color=MUTED, bottom=FOREST)

    rows = [
        (
            [("TASK 1", {"size": 8, "color": RUST, "spc": 1.2}),
             ("Image-based classification", {"size": 13, "color": FOREST})],
            "Walking · Eating · Resting · Climbing",
            "Five-second window CNN, three classes",
            [("Macro F1 ", {}), ("0.949", {"bold": True, "color": RUST}),
             (" on 15 held-out windows", {})],
        ),
        (
            [("TASK 2", {"size": 8, "color": RUST, "spc": 1.2}),
             ("YOLO-based detection", {"size": 13, "color": FOREST})],
            "Video → frames → YOLO annotation → training → detection",
            "Auto-labeled detector, no hand-drawn boxes",
            [("mAP50 ", {}), ("0.537", {"bold": True, "color": RUST}),
             (" on validation; animal found in 59% of held-out frames", {})],
        ),
        (
            [("TASK 3", {"size": 8, "color": RUST, "spc": 1.2}),
             ("Keypoint feature analysis", {"size": 13, "color": FOREST})],
            "Body Angle · Head Angle · Movement Speed · Limb Distance",
            "Silhouette geometry with the same four names",
            [("Macro F1 ", {}), ("0.422", {"bold": True, "color": RUST}),
             (" on 15 test windows", {})],
        ),
    ]
    for r, (task, asked, built, result) in enumerate(rows, start=1):
        cell(tbl, r, 0, [[task[0]], [task[1]]], size=9.5, spacing=1.35)
        cell(tbl, r, 1, asked, size=12.5, spacing=1.3)
        cell(tbl, r, 2, built, size=12.5, spacing=1.3)
        cell(tbl, r, 3, [result], size=12.5, spacing=1.3)

    txt(s, 64, 552, 1152, 60,
        [[("Climbing dropped with instructor confirmation (ADR 0007). ", {"color": RUST}),
          ("Task 3 is a silhouette proxy, not keypoints (ADR 0009).", {"color": RUST})]],
        size=11, spacing=1.35)
    return s


def slide_task1_data(prs):
    s = page(prs, 3, "03 / TASK 1 · DATA", "Task 1: the evidence base")

    numbers = [
        ("22", "source clips", 42),
        ("201", "five-second windows", 42),
        ("138", "labelled windows", 42),
        ("93 · 30 · 15", "train · val · test", 34),
    ]
    for i, (num, label, size) in enumerate(numbers):
        x = 64 + i * 280
        txt(s, x, 126, 250, 56, num, size=size, font=SERIF, color=FOREST, spacing=1.0)
        txt(s, x, 194, 250, 16, label, size=9, color=MUTED, spc=1.2)

    rule(s, 64, 244, 1152)

    steps = [
        ("01", "Source clips", "22 fixed-camera MP4s"),
        ("02", "Verified intervals", "one class per whole span"),
        ("03", "Five-second windows", "a fixed grid over each clip"),
        ("04", "Split by whole clip", "no frame-level leakage"),
    ]
    for i, (num, title, desc) in enumerate(steps):
        x = 64 + i * 284
        txt(s, x, 288, 40, 14, num, size=9, color=MOSS, spc=1.2)
        txt(s, x, 308, 230, 26, title, size=15, font=SERIF, color=FOREST)
        txt(s, x, 338, 220, 34, desc, size=10.5, color=CHAR, spacing=1.25)
        if i < 3:
            arrow(s, x + 238, 312)

    rule(s, 64, 430, 1152)
    txt(s, 64, 448, 1152, 24, "THE TEST SPLIT", size=9, color=MUTED, spc=1.4)
    txt(s, 64, 474, 1100, 66, [[
        ("The eight-clip gold set: two labeling passes, interval Dice ", {}),
        ("0.597", {"bold": True, "color": RUST}),
        (", reconciled, and locked on 2026-10-20 before the test was read.", {}),
    ]], size=14, color=CHAR, spacing=1.4)
    txt(s, 64, 560, 1100, 20,
        "63 of 201 windows are dropped: 46 not covered completely by one interval, "
        "17 touching a Transition or Uncertain span.", size=10, color=MUTED)
    return s


def slide_task1_result(prs):
    s = page(prs, 4, "04 / TASK 1 · RESULT", "Task 1: 0.949 held-out macro F1")

    txt(s, 64, 136, 430, 120, "0.949", size=96, font=SERIF, color=RUST, spacing=0.95)
    txt(s, 64, 268, 430, 14, "MACRO F1 · 15 HELD-OUT WINDOWS", size=9,
        color=MUTED, spc=1.4)
    rule(s, 64, 300, 430)
    txt(s, 64, 316, 430, 14, "ONE ERROR", size=9, color=RUST, spc=1.4)
    txt(s, 64, 336, 430, 60,
        ["Clip 25690519193808204, 40–45 s.", "Labelled Eating; predicted Resting at 0.847."],
        size=12, color=CHAR, spacing=1.3)

    txt(s, 600, 118, 550, 14, "CONFUSION MATRIX · ROWS TRUE / COLUMNS PREDICTED",
        size=8, color=MUTED, spc=1.2)
    widths = [170, 92, 108, 92, 88]
    tbl = table(s, 600, 144, widths, [32, 46, 46, 46, 46])
    for c, head in enumerate(["", "EATING", "LOCOMOTION", "RESTING", "TOTAL"]):
        cell(tbl, 0, c, head, size=8, color=MUTED,
             align=LEFT if c == 0 else CENTER, bottom=FOREST)
    matrix = [
        ("Eating", ("6", "0", "1", "7")),
        ("Locomotion", ("0", "2", "0", "2")),
        ("Resting", ("0", "0", "6", "6")),
    ]
    for r, (label, vals) in enumerate(matrix, start=1):
        cell(tbl, r, 0, label, size=12, color=CHAR, align=RIGHT)
        for c, val in enumerate(vals, start=1):
            color = RUST if (r == 1 and c == 3) else CHAR
            bold = r == 1 and c == 3
            cell(tbl, r, c, val, size=15, font=SERIF, color=color,
                 align=CENTER, bold=bold)
    cell(tbl, 4, 0, "TOTAL", size=8, color=MUTED, bold=True, bottom=None,
         align=RIGHT)
    for c, val in enumerate(("6", "2", "7", "15"), start=1):
        cell(tbl, 4, c, val, size=15, font=SERIF, color=FOREST,
             align=CENTER, bold=True, bottom=None)
    txt(s, 600, 386, 550, 16, "Abstained: 0 of 15. Accuracy over decided windows: 0.933.",
        size=9, color=MUTED)

    panel = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, PX(64), PX(420), PX(1152), PX(150))
    _bare(panel)
    panel.fill.solid()
    panel.fill.fore_color.rgb = PANEL
    panel.line.fill.background()
    rule(s, 64, 420, 1152, RUST, 2)
    txt(s, 84, 438, 1100, 20, "WHAT THIS NUMBER DOES NOT ESTABLISH",
        size=9, color=RUST, spc=1.4)
    txt(s, 84, 462, 1100, 40,
        "Fifteen windows: Locomotion has 2, Resting takes all 6 from one clip, and "
        "camera and class travel together. 0.949 is an upper bound on a new recording, "
        "not an estimate of it.", size=13, color=CHAR, spacing=1.4)
    txt(s, 84, 512, 1100, 40,
        "The one error may be a model miss on a partly hidden animal or an annotation "
        "error; the label was not changed after the test was read.",
        size=10.5, color=MUTED, spacing=1.35)
    return s


def slide_task2_methods(prs, conf):
    s = page(prs, 5, "05 / TASK 2 · LABELS", "Task 2: nobody drew a box")

    widths = [252, 448]
    tbl = table(s, 64, 140, widths, [30, 72, 72, 72, 72])
    cell(tbl, 0, 0, "METHOD TRIED", size=8, color=MUTED, bottom=FOREST)
    cell(tbl, 0, 1, "WHAT IT RETURNED", size=8, color=MUTED, bottom=FOREST)

    rows = [
        ("COCO YOLO11n, full frame",
         "Zero detections of the animal. One potted plant at 0.344."),
        ("Frame differencing, largest blob",
         "Locks onto swaying foliage, not the animal."),
        ("Grounding DINO, open vocabulary",
         "40+ boxes at 0.12; top score 0.535 on empty floor."),
        ("Per-pixel median background",
         "408 boxes kept after a one-second persistence filter."),
    ]
    for r, (method, result) in enumerate(rows, start=1):
        cell(tbl, r, 0, method, size=11.5, color=FOREST)
        cell(tbl, r, 1, result, size=11.5, spacing=1.3)

    s.shapes.add_picture(str(FIG / "detection.jpg"), PX(800), PX(140),
                         PX(396), PX(264))
    txt(s, 800, 412, 396, 34,
        [f"YOLO BOX {conf:.2f} · CAMERA 03 · 19 MAY 2026",
         [("Source: ", {}), ("outputs/detection/heldout.csv", {"font": MONO, "size": 6.5})]],
        size=7.5, color=MUTED, spacing=1.4, spc=0.8)

    txt(s, 64, 500, 1100, 60,
        "The cameras never move, so each clip supplies its own background.",
        size=21, font=SERIF, color=FOREST, spacing=1.25)
    txt(s, 64, 566, 1050, 40,
        "Median frame per clip · per-pixel noise threshold · components in an animal-size "
        "band · a box must persist for about a second.",
        size=10.5, color=CHAR, spacing=1.4)
    return s


def slide_task2_loop(prs):
    s = page(prs, 6, "06 / TASK 2 · RESULTS",
             "Task 2: the loop that learns its own labels")

    steps = [
        ("01", "1061", "frames sampled at 1 fps across 22 clips"),
        ("02", "470", "candidate boxes from the background model"),
        ("03", "408", "kept after the one-second persistence filter"),
        ("04", "116", "boxes audited; 113 were on the animal"),
        ("05", "0.537", "YOLO round 1 mAP50; round 2 self-trains"),
    ]
    rule(s, 64, 176, 1152)
    for i, (num, big, label) in enumerate(steps):
        x = 64 + i * 235
        rule(s, x, 176, 1, MOSS, 6)
        txt(s, x, 190, 40, 12, num, size=8, color=MOSS, spc=1.4)
        txt(s, x, 206, 214, 36, big, size=30, font=SERIF, color=FOREST)
        txt(s, x, 246, 214, 46, label, size=10, color=CHAR, spacing=1.3)

    rule(s, 64, 320, 1152)
    results = [
        ("0.499", "mAP50 on untouched validation after self-training"),
        ("59%", "of held-out frames fire at confidence 0.25"),
        ("5 / 10", "audited held-out boxes were on the animal"),
    ]
    for i, (big, label) in enumerate(results):
        x = 64 + i * 392
        txt(s, x, 342, 340, 52, big, size=40, font=SERIF, color=RUST)
        txt(s, x, 398, 320, 44, label, size=11, color=CHAR, spacing=1.3)

    txt(s, 64, 500, 1152, 16, "ROUND 2 ON ITS OWN VALIDATION: 0.774", size=8,
        color=MUTED, spc=1.4)
    txt(s, 64, 522, 1152, 18,
        "That number is not comparable: 115 of round 2's validation labels were partly "
        "written by round 1.", size=10, color=MUTED)

    rule(s, 64, 576, 72, RUST, 2)
    txt(s, 64, 594, 1100, 40, "No person drew a bounding box at any point.",
        size=22, font=SERIF, color=RUST, spacing=1.2)
    return s


def slide_task3(prs):
    s = page(prs, 7, "07 / TASK 3 · FEATURES",
             "Task 3: four features from a silhouette")

    widths = [250, 140, 175, 175]
    tbl = table(s, 64, 150, widths, [34, 54, 54, 54, 54])
    heads = ["FEATURE · MEDIAN PER CLASS", "EATING", "LOCOMOTION", "RESTING"]
    for c, head in enumerate(heads):
        cell(tbl, 0, c, head, size=8, color=MUTED,
             align=LEFT if c == 0 else CENTER, bottom=FOREST)
    rows = [
        ("Body angle (deg)", "115°", "88°", "154°"),
        ("Head angle (deg)", "113°", "49°", "96°"),
        ("Movement speed (px/s)", "17", "134", "2"),
        ("Limb extent (px)", "56", "148", "98"),
    ]
    for r, row in enumerate(rows, start=1):
        cell(tbl, r, 0, row[0], size=11.5, color=CHAR)
        for c, val in enumerate(row[1:], start=1):
            cell(tbl, r, c, val, size=16, font=SERIF, color=FOREST, align=CENTER)

    txt(s, 64, 418, 760, 16,
        "Medians over training and validation windows with at least one silhouette.",
        size=8.5, color=MUTED)

    rule(s, 860, 150, 1, RULE, 250)
    txt(s, 896, 150, 300, 18, "VALIDATION MACRO F1", size=8, color=MUTED, spc=1.4)
    txt(s, 896, 170, 300, 52, "0.479", size=44, font=SERIF, color=FOREST)
    txt(s, 896, 236, 300, 18, "TEST MACRO F1", size=8, color=MUTED, spc=1.4)
    txt(s, 896, 256, 300, 52, "0.422", size=44, font=SERIF, color=RUST)

    rule(s, 64, 452, 1152, RUST, 2)
    txt(s, 64, 470, 1100, 18, "PROXY, NOT POSE ESTIMATION", size=9, color=RUST, spc=1.4)
    txt(s, 64, 492, 1080, 62,
        "No keypoints, no pose model. Resting is invisible: 58 of 93 training windows "
        "have no silhouette, so Resting scores 0 on test.", size=15, color=CHAR, spacing=1.4)
    txt(s, 64, 545, 1080, 40,
        "Per-class test F1: Eating 0.267 · Locomotion 1.000 · Resting 0.000. "
        "Locomotion separates by an order of magnitude in speed.",
        size=10, color=MUTED, spacing=1.35)
    return s


def slide_demo(prs):
    s = page(prs, 8, "08 / ONE COMMAND", "All three tasks in one command")

    txt(s, 64, 120, 1152, 18,
        [[("$ ", {"color": MOSS}),
          ("python3 scripts/demo_all.py data/raw/Eating/25690519193422801.mp4",
           {"color": FOREST})]], size=10, font=MONO)

    panel = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, PX(64), PX(156), PX(1152), PX(274))
    _bare(panel)
    panel.fill.solid()
    panel.fill.fore_color.rgb = PANEL
    panel.line.fill.background()

    code = [
        "data/raw/Eating/25690519193422801.mp4  (50.0 s, 10 windows, abstain below 0.00)",
        "  start     end  T1           conf  T2 box               via     conf  crop         body  head   speed  limb",
        "   0.00    5.00  Eating       0.73  408,462,606,528      motion  0.00  Eating      6 161       1    63",
        "   5.00   10.00  Eating       0.69  606,510,674,564      motion  0.37  Eating      25 16      23    52",
        "  10.00   15.00  Eating       0.69  632,472,700,550      motion  0.00  Eating      43 24      20    68",
        "  15.00   20.00  Eating       0.67  341,365,624,534      yolo    0.58  Locomotion  119 19      24    51",
        "",
        "detection: 10/10 windows found the animal directly (6 by the trained detector, 4 by the per-clip model)",
        "full-frame and crop labels agree on 4/10 windows",
    ]
    txt(s, 88, 180, 1104, 230, code, size=13, font=MONO, color=CHAR, spacing=1.75)

    txt(s, 64, 462, 1100, 40,
        "Task 1 behavior, Task 2 detector box and crop, Task 3 features — same windows, "
        "same timecodes.", size=14, color=CHAR, spacing=1.35)
    txt(s, 64, 502, 1100, 32,
        "Validation clip. Output trimmed to the header, four data rows and the two summary "
        "lines; the per-class features are medians over the window's sampled frames.",
        size=9, color=MUTED, spacing=1.35)
    return s


def slide_limits(prs):
    s = page(prs, 9, "09 / BOUNDARIES", "What this does not prove")

    limits = [
        ("15 test windows from 5 clips",
         "Locomotion has 2 and Resting takes 6 from one clip. A per-class score on that support is a baseline, not a stable estimate."),
        ("Camera and class travel together",
         "Every Resting window is Camera 03 and nearly all Eating is Camera 02, so a score can come from the recording rather than the behavior."),
        ("The detector does not generalize",
         "It fails on the 2026-05-20 recording day, absent from training, and fires on a bright plant."),
        ("Task 3 is a proxy",
         "No keypoints and no pose model. A still animal joins the median background, so most Resting windows have no features at all."),
        ("One training seed",
         "Seed 0 only. No run-to-run spread is reported, so the split between score and seed is unknown."),
    ]
    y = 118
    for i, (claim, reason) in enumerate(limits, start=1):
        txt(s, 64, y + 18, 36, 16, f"0{i}", size=9, color=MOSS, spc=1.0)
        txt(s, 108, y + 12, 300, 26, claim, size=15, font=SERIF, color=FOREST)
        txt(s, 420, y + 12, 796, 52, reason, size=11, color=CHAR, spacing=1.35)
        rule(s, 64, y + 66, 1152)
        y += 72

    rule(s, 64, 520, 1152, RUST, 2)
    txt(s, 64, 538, 1100, 44,
        "And no human ground truth for detection: its quality is a counted visual audit, "
        "not mAP against truth.", size=12, color=RUST, spacing=1.35)
    return s


def slide_next(prs):
    s = page(prs, 10, "10 / NEXT", "Next")

    txt(s, 64, 108, 900, 22,
        "Four changes that would turn these numbers into evidence.",
        size=12, color=MUTED, spacing=1.3)

    steps = [
        ("01", "Hold out a recording group",
         "Collect footage from sessions absent from training and repeat the evaluation, so the upper bound becomes an estimate."),
        ("02", "Widen per-class support",
         "Add Locomotion and Resting test clips from more than one clip per class before reading per-class differences."),
        ("03", "Calibrate abstention",
         "Choose the abstain threshold on a larger validation set or a separate calibration set, never on the test windows."),
        ("04", "Revisit Climbing and pose",
         "Only if verified footage and keypoint labels exist. Until then, three classes and silhouette features are the scope."),
    ]
    for i, (num, claim, body) in enumerate(steps):
        x = 64 + (i % 2) * 592
        y = 152 + (i // 2) * 168
        txt(s, x, y, 40, 34, num, size=26, font=SERIF, color=MOSS)
        txt(s, x + 52, y + 4, 460, 26, claim, size=17, font=SERIF, color=FOREST)
        txt(s, x + 52, y + 36, 470, 60, body, size=11, color=CHAR, spacing=1.35)

    rule(s, 64, 512, 1152, RUST, 2)
    txt(s, 64, 534, 1080, 80,
        "A reproducible baseline that states what it measured and what it did not.",
        size=24, font=SERIF, color=FOREST, spacing=1.2)
    return s


# ---------------------------------------------------------------- notes

NOTES = {
1: "We built all three tasks of the brief on the same 22 clips, and everything you see comes from one pipeline. The scores are only half the story: the other half is what each score covers. I will take the tasks in order, then show the single command that produces all three at once.",
2: "This page is the whole brief and the whole delivery side by side. Task 1 is the strongest result, Task 2 works but does not survive a new recording day, and Task 3 is the honest weak one. The footnote matters: Climbing was dropped with the instructor's agreement, and Task 3 measures a silhouette rather than keypoints.",
3: "Task 1 rests on 22 clips cut into 201 windows, of which 138 carry one verified label. The flow is deliberately boring: clips, verified intervals, five-second windows, then a split by whole clip. The test set is the eight-clip gold set, locked on the 20th before anyone read a test score.",
4: "0.949 is fifteen windows, so treat it as a headline rather than a property of the animal. There is one error: an Eating window predicted Resting at 0.847. Read the bottom line before you read the number, because camera and class travel together and that makes this an upper bound.",
5: "We tried four ways to get a box without a person drawing one, and three failed in instructive ways. A COCO model does not know what a civet is, differencing sees the plant, and Grounding DINO puts boxes on the floor. What worked is the fact that these cameras never move, so every clip carries its own background.",
6: "From 1061 frames the background model proposes 470 boxes, and persistence cuts that to 408. We audited 116 by eye and 113 were on the animal, which is what made training possible. Round 1 reached 0.537; round 2 looks better on its own split, but the comparable number is 0.499 on the untouched seed labels. No person drew a box at any point.",
7: "The four names are the brief's four names, but the measurement is silhouette geometry. Locomotion separates by an order of magnitude in speed, which is why it scores 1.000 on test. Resting is the failure: a still animal becomes the background, so most training windows have no silhouette and Resting scores zero.",
8: "One command prints all three tasks for the same five-second windows, so nothing here is stitched together afterwards. Task 1 gives the behavior, Task 2 gives the box and the crop classifier, Task 3 gives the four features. On this clip it found the animal in all ten windows, six by the detector and four by the per-clip model.",
9: "These are the limits we would rather say ourselves. The test set is small and the camera tracks the class, the detector does not survive a new recording day, and Task 3 is a proxy. Detection has no human ground truth, so its quality is a counted audit, not a mAP against truth.",
10: "The next version needs footage from recording sessions that are absent from training, because that is the only way to turn an upper bound into an estimate. Then more support per class and an abstain threshold chosen off the test set. Climbing and pose come back only if verified footage exists. What we have now is a baseline that is reproducible and honest about its coverage.",
}


# ---------------------------------------------------------------- main

def build():
    conf = build_figures()

    prs = Presentation()
    prs.slide_width = PX(1280)
    prs.slide_height = PX(720)

    slide_cover(prs)
    slide_brief(prs)
    slide_task1_data(prs)
    slide_task1_result(prs)
    slide_task2_methods(prs, conf)
    slide_task2_loop(prs)
    slide_task3(prs)
    slide_demo(prs)
    slide_limits(prs)
    slide_next(prs)

    for i, slide in enumerate(prs.slides, start=1):
        slide.notes_slide.notes_text_frame.text = NOTES[i]

    cp = prs.core_properties
    cp.title = "Three Tasks, One Animal, One Pipeline"
    cp.author = "Project team, CMU"
    cp.subject = "CMZ project, Phase 2 presentation"
    cp.comments = "Generated by scripts/make_deck.py"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB), "
          f"{len(prs.slides)} slides")


if __name__ == "__main__":
    build()
