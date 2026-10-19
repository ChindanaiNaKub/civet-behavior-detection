# Finish the independent gold review (issue #4)

The first annotator has labeled all 22 source clips (62 intervals), including
blind passes on all eight gold clips. The second person's pass, reconciliation,
and human test lock are still pending. This handoff does not close #4. It builds
on PR #19 and PR #20; merge those in order before merging this follow-up.

## Give the second reviewer a separate folder

Reviewer-facing instructions, including Fedora/Linux startup, guided time marking,
browser drafts and codec troubleshooting: [Reviewer quickstart](REVIEWER-QUICKSTART.md).

From the repository root, run:

```powershell
python scripts/prepare_gold_review.py
```

Give `outputs/gold-review/` to a person who has not seen the first-pass answers
or machine proposals. It contains eight videos under neutral filenames, a
minimal source ledger without folder labels, the label definitions, a local
review tool, and instructions. It contains no first-pass annotations, proposals,
split results, or agreement results. The tool uses Python's standard library;
Python 3.10 or newer is sufficient. Keep the folder in team-approved private
storage because it contains the supplied footage.

The reviewer runs `python start_review.py` inside that folder and opens
http://127.0.0.1:8000/. They enter their name on the page and click Start, use
Add record, choose the behavior, edit start/end seconds, then Save clip. There
are also buttons to take start/end from the current video position. Keyboard
shortcuts remain optional; no browser prompt is needed. They use one consistent personal identifier, watch all
eight clips, and return `data/annotations-second.csv` along with confirmation
that they worked independently without viewing existing answers. A script cannot
establish independence merely from different names. If the reviewer has already
seen those answers, choose another person and disclose that exposure.

The generator refuses an existing output directory to protect work in progress.
Use `--out outputs/gold-review-new` for a fresh packet. Generated videos and
review packets remain ignored by Git; commit the returned CSV only after review.

Keep the same output location when resuming. The launcher writes only
`data/annotations-second.csv` inside the packet and protects the repository's
first-pass `data/annotations.csv` when the packet is under `outputs/`. Saves use
an atomic replacement and keep the previous second-pass file as `.csv.bak`.
Invalid or overlapping intervals are rejected before writing. A browser cancelling
a video request while seeking or closing the page no longer produces a traceback.
The page keeps a local browser draft scoped to the output file, annotator and
clip, and warns before leaving unsaved work. Browser drafts are not CSV submissions.
Use the same browser, origin/port and reviewer name to restore them. Saved CSV
records remain available even when changing browsers or operating systems.
The reviewer opens in English, with a live English/Thai switch. Display language
does not change the stored behavior codes, interval times or reviewer-entered text.
To load upgraded code in an existing packet, copy `scripts/review.py`,
`scripts/review_page.html`, `scripts/review_i18n.json` and update the launcher; preserve its entire `data/`
directory. Stop and restart the server after saving current work, then refresh
the page. Do not regenerate or replace an in-progress review folder.

## Receive and compare the independent pass

1. Put the returned file at `data/annotations-second.csv`. Preserve it as the
   original independent submission; do not copy the first pass over it or merge
   both annotators' rows into `annotations.csv`.
2. Run `python scripts/agreement.py`. It requires all eight gold clips in both
   files, one consistent non-empty identity per pass, different identities
   between passes, valid non-overlapping intervals, and no non-gold clips in the
   second file. The first file can still contain the other 14 source clips.
   These are file checks, not a certificate that humans reviewed the whole video.
3. Record the printed Dice agreement and inspect the per-clip ledger, including
   boundary matches. Discuss label differences, unmatched spans, gaps, and
   boundary shifts against the glossary and annotation guide. A label match with
   a boundary shift beyond 0.5 seconds still needs review even if `disagreements`
   is zero. Check full-clip coverage manually, including Uncertain spans.
4. Fill `resolved_labels`, `resolution_reason`, `reviewed_by`, and `reviewed_at`
   for every gold clip in `data/agreement.csv`, including clips where the reviewers
   agree. Record the final labels **and their time ranges**, the reason for each
   change (or that both passes agree), the reviewers, and an ISO date/time.
5. Apply the reconciled intervals to the gold clips in `data/annotations.csv` as
   one final set of rows per clip. Preserve the original independent submissions
   in Git history before editing; retain `annotations-second.csv` unchanged.
   Do not concatenate the two passes: overlapping rows make windows unusable.
   Re-running agreement against reconciled answers is not the original independent
   agreement score. The tool refuses to overwrite filled reconciliation columns;
   use a different `--out` only when intentionally starting another comparison.

## Regenerate and arrange the human lock

After reconciliation, from the repository root:

```powershell
python scripts/lighting.py
python scripts/split.py
python scripts/windows.py
```

Check that the gold list still matches the test split and all three classes have
verified windows in every split. Counts before the second pass, in Eating /
Locomotion / Resting order, are train 14 / 5 / 74, validation 10 / 1 / 19, and test
7 / 5 / 6. Final labels may change these counts; report the final support rather
than forcing the old numbers. If a class is lost, follow ADR 0008's fallback;
do not invent labels to retain the split.

A person who is not tuning the model then reviews the final ledger and fills
`test_locked_by` and `test_locked_at` on every test row in `data/splits.csv`.
Do this **after** the last `split.py` run: it writes a new ledger with blank
lock fields. The current evaluation script does not enforce this lock, so do
not run it until the reviewer has signed. Training and evaluation are #5 and #6.

The PR completing #4 should include both passes, measured agreement, completed
reconciliation, final intervals, source conditions, and the signed split ledger.
Disclose the small class support, recording/class confounding, and the first
annotator's earlier exposure on the two clips moved into test (ADR 0008).
