"""The behavior label domain, and the column order of every CSV that carries it.

One source of truth for which labels exist, which of them a model may predict,
and the shape of each table, so the rules in docs/ANNOTATION-GUIDE.md are not
restated per script. The glossary in GLOSSARY.md defines what each label means;
this module only records which ones the pipeline may use where.
"""

# What a model may predict. Climbing is a valid annotation but has no verified
# source clip (docs/FOOTAGE-AUDIT.md), so there is nothing to train or score it
# on. Add it here when a verified climb exists.
MODEL_CLASSES = ("Eating", "Locomotion", "Resting")

# Valid annotations that are not model classes. A window touching either one is
# excluded from training and evaluation rather than forced into a class.
NON_CLASS_LABELS = ("Transition", "Uncertain")

ALL_LABELS = MODEL_CLASSES + ("Climbing",) + NON_CLASS_LABELS

# One window is five seconds by design; prelabel.py, the window builder and the
# demo all share these, so the pre-labels and the demo cover the same grid.
WINDOW_S = 5.0
# The pre-labeller may end a proposed span with a trailing window at least this
# long. The classifier's own windows are always exactly WINDOW_S, because a
# shorter one is a differently shaped input.
MIN_WINDOW_S = 2.0

SOURCE_FIELDS = (
    "clip_id",
    "relative_path",
    "folder_label_weak",
    "duration_s",
    "capture_datetime_overlay",
    "camera_id_overlay",
    "session_id",
    "room_id",
)

ANNOTATION_FIELDS = (
    "clip_id",
    "start_s",
    "end_s",
    "behavior",
    "visibility",
    "lighting",
    "camera_view",
    "occlusion",
    "uncertain_reason",
    "annotator",
    "notes",
)

SPLIT_FIELDS = (
    "clip_id",
    "split",
    "grouping_key",
    "grouping_evidence",
    "assignment_reason",
    "test_locked_by",
    "test_locked_at",
)

WINDOW_FIELDS = ("clip_id", "start_s", "end_s", "behavior", "split")

SPLITS = ("train", "validation", "test")
