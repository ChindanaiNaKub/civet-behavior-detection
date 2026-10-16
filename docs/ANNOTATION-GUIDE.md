# Civet behavior interval annotation guide

Use this guide to assign one behavior label to each clear interval in a source clip. The behavior terms follow [`GLOSSARY.md`](../GLOSSARY.md). Folder names are search hints only; verify behavior from the video and record start/end timecodes.

## Labels

| Label | Apply when | Do not use when |
|---|---|---|
| **Eating** | The civet performs visible food-directed action, such as taking or chewing food. | The civet only lowers its head or investigates an object and food-directed action is not clear. |
| **Locomotion** | The civet moves so its location changes. This includes horizontal travel on a structure. | It only shifts posture or moves a limb without changing location. |
| **Climbing** | The civet moves up or down a sloped or vertical structure, or actively climbs along it. | It travels horizontally across a structure; use Locomotion. |
| **Resting** | The civet remains continuously inactive for at least five seconds. Once the five-second duration is confirmed, mark the interval from the start of that inactive state. | A brief pause, alert observation, or short stop that does not reach five seconds. |
| **Transition** | The video shows a change from one behavior to another. Record the interval boundaries when they are visible. | A stable behavior interval. Transition is not one of the four behavior classes. |
| **Uncertain** | The view does not provide enough evidence to choose one behavior, including when the civet is obscured or too small to read. Add a short reason. | A clear behavior that merely needs a more specific note. |

## Interval and overlap rules

1. Watch the full clip once before marking intervals. Use seconds from the start of the source clip; keep the original source file unchanged.
2. Start an interval at the first frame where the behavior is visible. End it at the last frame before it changes or becomes unobservable. Split at behavior changes; do not label an entire clip from its folder name.
3. Assign one behavior label to an interval. If actions overlap, choose the most specific behavior that is clearly visible. If no single behavior is clear, use Uncertain and note the ambiguity.
4. Mark Transition separately when a change is visible. Exclude any five-second model window that crosses a transition or an Uncertain interval.
5. For Resting, confirm at least five continuous seconds of inactivity, then timestamp the interval from the inactivity onset. If the clip ends before five seconds can be confirmed, do not label that span Resting.
6. Keep observation conditions separate from behavior. A dim/infrared view, camera angle, visibility, and occlusion are context fields, never behavior labels.

## Record for each interval

| Field | What to enter |
|---|---|
| `clip_id` | Stable source filename or team-assigned ID. |
| `start_s`, `end_s` | Start and end time in seconds from the source clip; end must be greater than start. |
| `behavior` | Eating, Locomotion, Resting, Climbing, Transition, or Uncertain. |
| `visibility` | Clear, partial, or out of view for the interval. |
| `lighting` | Observed condition (for example, daylight or infrared/grayscale) only when visible; otherwise unknown. |
| `camera_view` | A short description of angle/view, or unknown. Do not infer camera ID. |
| `occlusion` | None or visible obstruction and its source (for example, fence, plant, or structure). |
| `uncertain_reason` | Required when behavior is Uncertain; otherwise blank. |
| `annotator`, `notes` | Annotator ID and concise evidence or boundary note. |

Record room, camera ID, session, and capture time only when a source record confirms the value. Use `unknown` instead of inferring them from folder names, numeric filenames, or the proposed four-room/four-camera setting.

## Machine pre-labels

Human effort is verification, not first-pass labeling. The pipeline proposes behavior intervals, deriving Resting from sustained low motion (at least five seconds) and Locomotion from location change, and proposing Eating and Climbing with a zero-shot model. For each proposed interval, the reviewer accepts it, edits its boundaries or label, or marks it Transition or Uncertain; label from scratch only where the pipeline proposes nothing or the proposal is wrong.

Run `python3 scripts/review.py` to do this against the video. The page serves each clip with its proposed intervals under the player, and writes accepted intervals to `data/annotations.csv`. Add `--blind <clip_id>` for a gold-set clip; the page then hides every machine proposal, which is required for the gold set. See the tool's `--counts` flag to see what is left.

The gold set used for reported metrics must be labeled by a human who has not seen the machine proposal. Otherwise the score measures agreement with the model, not the model's accuracy. See [ADR 0003](adr/0003-automate-prelabeling-with-human-gold-set.md).

The second annotator runs the same tool with `--out data/annotations-second.csv` on the shared sample only, then `python3 scripts/agreement.py` compares the two files and writes `data/agreement.csv` with the reconciliation columns left for the two annotators.

## Review procedure

- Two annotators independently label the same 20% of clips, with the sample spread across available source folders and conditions. If a class has too few examples for a useful shared sample, review every candidate clip for that class together.
- Compare interval boundaries and labels, discuss disagreements against these rules, and record the resolved label and reason. Clarify this guide before labeling the remaining clips if disagreements reveal an ambiguous rule.
- Have one annotator complete the remaining clips and a second reviewer check uncertain intervals and behavior boundaries. Report agreement and resolved disagreements before training.
- Keep source clips as the unit for train/validation/test splits. Create five-second windows only after splitting so neighboring windows from one clip cannot leak across sets.
