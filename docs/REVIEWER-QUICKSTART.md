# Behavior review quickstart: Windows and Fedora Linux

Use Python 3.10 or newer and a browser. The review tool does not require PyTorch
or model-training packages. Keep the prepared review packet together as a folder;
do not move the video files out of it.

## Start the tool

### Add observation conditions to saved answers

The review page has four required dropdowns for each interval. Select an interval
row, then choose a value in every menu before saving. The page checks every
interval, and the server rejects missing or unsupported values as well.

| Field | Choices |
|---|---|
| `visibility` | Clear; partial; fully hidden behind an obstruction; out of view; cannot determine |
| `lighting` | Daylight/color; infrared/grayscale; cannot determine |
| `camera_view` | Elevated oblique (looking down at an angle); overhead (straight down); animal-level side view (horizontal); cannot determine |
| `occlusion` | None; tree/plant/planter; fence/mesh; enclosure structure; multiple obstructions; cannot determine |

These choices were informed by eight sampled frames from each of the 22 clips;
see [Observation condition options](OBSERVATION-CONDITIONS.md) for the evidence
and definitions. Camera view describes the viewing angle, not camera identity.
Record conditions for the selected interval from the video. If the relevant
condition changes, split the interval as needed rather than guessing one value.

Every menu starts with **Select an option…**. **Cannot determine from the video**
is a deliberate choice stored as `unknown`, not an automatic default. When
editing existing answers, legacy `unknown` or unsupported values are left
unselected so the reviewer must assess them. Known supported values are retained.
These fields are included in browser drafts and written to the CSV when you click
**Save clip and continue**. Old drafts with automatic defaults must be checked
again; newly selected choices, including cannot determine, survive draft restore.

To add conditions to an existing second-pass submission, preserve the original
in Git or as a separate copy first. Then run this command from the project root
(not from the review packet sent to the reviewer):

```powershell
python scripts/review.py --edit-existing --out data/annotations-second.csv --only-file data/gold-clips.txt --protect-file data/annotations.csv --port 8001
```

Open `http://127.0.0.1:8001/` and enter the exact annotator name used in the
original CSV. The tool loads that person's saved behaviors, time ranges, reasons,
notes, and conditions for editing. It does not show machine proposals or another
person's answers. Saving replaces only the rows for that clip and annotator.
The previous file is retained as `.csv.bak`; this is a backup of the latest save,
not a complete revision history. A mismatched name or a clip with no saved
answers produces an error.

This edits an existing submission; it is not a new blind pass. It cannot be
combined with `--blind` or `--blind-file`. Preserve the original independent
submission, and leave behaviors and time ranges unchanged when only adding
conditions. If a label or time range needs correction, record the change during
reconciliation as well. Normal blind mode still starts with empty answers and
does not load saved answers.

For a packet prepared before this change, save your work and stop the server.
Update `review.py`, `review_page.html`, and `review_i18n.json` in the packet's
`scripts/` folder, then restart the server and refresh the page. Preserve the
entire existing `data/` folder.

### Start an independent review

Open a terminal in the review packet folder containing `start_review.py`:

| System | Command |
|---|---|
| Windows / PowerShell | `python start_review.py` |
| Fedora / Linux | `python3 start_review.py` |

Your default browser opens automatically. If it does not, open the URL printed
in the terminal. The tool tries port 8000 first and chooses a free port if another
program is using it. Use the URL actually printed; it may use a different port.
Keep the terminal open during review. Stop the tool with Ctrl+C.

For Fedora, a machine without a desktop, or a developer choosing a port:

```sh
python3 start_review.py --no-browser --port 9000
python3 start_review.py --check
```

`--check` validates the packet without starting a server or writing answers.
If a port you selected is busy, use `--port 0` to let the system choose one.
The server listens only on your own machine and is not accessible to other
machines on the network.

## Review one clip

The page starts in English. Use the language selector at the top to switch
between English and Thai before or during review. Switching languages does not
change the annotator name, time ranges, behavior codes, or reasons you entered.
Opening or refreshing the page resets the display language to English; eligible
browser drafts can still be restored.

1. Enter the same annotator name each time and click **Start review**. Watch the
   entire clip once before labeling.
2. Seek to the start of a behavior and click **1. Mark start**.
3. Play or seek to its end and click **2. Mark end**. The video pauses.
4. Choose a behavior. The reason field is enabled and required only for Uncertain.
   For other behaviors it is disabled and blank; switching away from Uncertain
   clears the reason. For Uncertain, explain why you cannot identify it. Fill in
   the observation conditions for the selected interval. Add optional **Notes**
   for other observations or boundary decisions. Notes are saved as entered,
   including Thai text and line breaks; they do not replace an Uncertain reason.
5. Repeat for the remaining intervals. Check that the table has no overlapping
   times. Click a row to edit its times, behavior, or conditions.
6. When the entire clip is complete, click **Save clip and continue** to write
   the answers to the file.

Behavior choices include descriptions in the selected language. Resting requires
at least five continuous seconds of inactivity. Once confirmed, mark the interval
from the start of inactivity. If the animal or behavior cannot be seen clearly,
use Uncertain with a reason. Do not infer behavior from the clip name or the
first annotator's answers.

Playback speed can be changed without changing source timestamps. To type times
manually, click **+ Add interval manually**. Keyboard shortcuts are optional;
the full workflow can be completed with the mouse.

## Pause and resume

Drafts are stored automatically in the same browser, scoped to the review
submission, annotator name, and clip. Refreshing or returning to the same clip
restores an available draft for checking. **A draft is not a CSV submission**;
save the entire clip before handing in your work.

Resume with the same browser, URL/port, and annotator name. Changing browsers or
ports, clearing site data, or using private browsing may make the original draft
unavailable. If the terminal closes or the tool stops, start it again with the
same command. In normal review mode, clips already saved to the file are skipped.

Your answers are in `data/annotations-second.csv` **inside your review packet**.
The backup from the latest save is `data/annotations-second.csv.bak`.
Do not overwrite another person's `data/annotations.csv`. Return only your own
CSV when all clips are complete.

## If video does not play on Fedora / Firefox

Firefox uses the system's video support, so MP4/H.264 playback depends on the
installed codecs. Having OpenH264 for video calls alone does not guarantee MP4
playback. See [Mozilla's audio and video guide](https://support.mozilla.org/en-US/kb/audio-and-video-firefox)
and [Fedora's Firefox video guidance](https://fedoraproject.org/wiki/Firefox_Hardware_acceleration)
when checking the developer machine's setup.

Alternatively, open the URL printed in the terminal in a browser that can play
the supplied MP4. Save the current clip before switching browsers, because drafts
are stored only in the original browser. Saved CSV files work on both Windows
and Linux without modifying the source videos.

## For developers

Prepare a new packet from the repository root with
`python3 scripts/prepare_gold_review.py` on Fedora or
`python scripts/prepare_gold_review.py` on Windows. Do not regenerate a packet
that already has review work in progress. When upgrading it, update only the code
and launcher while preserving the entire `data/` folder. Update
`review_i18n.json` together with `review.py` and `review_page.html`. Save pending
work before restarting the server and refreshing the browser.

Test the reviewer without a GPU or model downloads:

```sh
python3 -m pytest scripts/test_review.py scripts/test_review_launcher.py scripts/test_gold_review.py scripts/test_agreement.py -q
```

Only the test command requires pytest. The `Reviewer compatibility` workflow
checks Fedora containers and Windows separately. Python/HTTP checks do not
confirm codec availability or actual browser playback on every machine;
video playback must also be checked on a Fedora desktop.
