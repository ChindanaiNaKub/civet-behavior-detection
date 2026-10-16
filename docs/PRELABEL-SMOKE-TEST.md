# Behavior pre-labeler smoke test

**Date:** 2026-10-16
**Issue:** [#9 auto pre-label behavior timelines](https://github.com/ChindanaiNaKub/civet-behavior-detection/issues/9)
**Purpose:** the one smoke test the plan budgets before committing to the pre-labeler. It records where pre-labeling works and where a human must label from scratch, so the fallback in the [Phase 2 plan](PHASE-2-PLAN.md#risks-and-responses) can be applied to the right classes.

**Answer:** Resting is pre-labeled reliably and Locomotion partly. **The pre-labeler fails on Eating and Climbing**: no Eating interval was proposed anywhere, and every Climbing proposal is a false positive. Eating and Climbing must be labeled from scratch; Climbing is the weakest class, as the plan expected (and there is still no verified climbing example to check against).

## What was run

```
python3 -m pytest scripts                       # 11 logic checks, no model
python3 scripts/prelabel.py --clips data/raw --out outputs/prelabel/timeline.csv
```

- All 22 tracked clips, 1,051.8 s of footage → **75 proposed intervals**, in 2 min 41 s on the audit workstation (RTX 3050 Laptop, 4 GiB; 1.04 GiB peak VRAM, about 0.35 s per five-second window).
- Model: `microsoft/xclip-base-patch16-zero-shot`, revision `6746d6a5202bf24300cd9bf36f457f174483d41b`, 32 frames at 224×224, four text prompts.
- The run is deterministic and was repeated: both runs produced the same 75 intervals. `outputs/` is git-ignored generated output; re-run the command to reproduce it.

Resting and Locomotion come from motion only, so they need no model:

| Signal | Meaning | Threshold |
|---|---|---|
| motion | fraction of pixels whose grey value changed by >8 grey levels since the previous sample, at 8 samples/s | mean over a 5 s window < 0.002 → Resting |
| disp | distance the centroid of those moving pixels travelled in the last 1 s, in frame widths | ≥ 0.08 → Locomotion |

Eating and Climbing come from X-CLIP scoring every five-second window against four prompts ("a video of a civet eating food from a bowl" / "walking across the floor" / "lying still on the ground" / "climbing up a vertical structure"). A window is proposed only when one of the two learned classes wins with probability ≥ 0.35; otherwise the row is `Uncertain` and names the model's top class.

## Where pre-labeling works

Motion labels against the **weak folder names** (folder names are hints, not verified intervals, so this is a sanity check and not an accuracy measurement):

| weak folder | clips | Resting | Locomotion | other |
|---|---:|---|---|---|
| Resting | 7 | 72–98 % of samples | 0–10 % | 2–22 % |
| Locomotion | 10 | 0–13 % | 0–78 % | 22–100 % |
| Eating | 5 | 0–69 % | 0–11 % | 31–100 % |

- **Resting works.** 574 of the 659 s of Resting-folder footage is proposed as Resting, all seven clips land at 72–98 % Resting, the long runs carry confidence 0.93–0.98, and the five-second rule is enforced by construction: no Resting interval can start before five seconds of continuous low motion.
- **Locomotion partly works.** 48 of 139 s of Locomotion-folder footage is proposed as Locomotion, concentrated in the clips where the animal clearly crosses the enclosure (up to 78 % of a clip). Its confidence is low by design, 0.00–0.21, because most samples sit just above the displacement threshold; a reviewer should treat every Locomotion row as a hint. The remainder is `Uncertain`, and the two clips where the animal is small or mostly stationary yield almost no Locomotion.

## Where it fails

| weak folder | proposed Eating | proposed Locomotion | proposed Resting | proposed Climbing | Uncertain |
|---|---:|---:|---:|---:|---:|
| Eating (254 s) | 0 (0 s) | 3 (10 s) | 7 (75 s) | 0 | 10 (160 s) |
| Locomotion (139 s) | 0 (0 s) | 13 (48 s) | 2 (4 s) | 4 (16 s) | 10 (55 s) |
| Resting (659 s) | 0 (0 s) | 2 (4 s) | 12 (574 s) | 0 | 12 (72 s) |

- **Eating is never proposed — 0 of 75 intervals.** No interval's strongest window gave Eating more than 0.33, below the 0.35 abstain threshold, and only one window in the whole run had Eating as its top class at all. 75 s (30 %) of the Eating-folder footage is proposed as Resting instead, because a civet with its head down at a bowl barely moves and never reaches the zero-shot stage, so a reviewer labels Eating from scratch everywhere.
- **Climbing is proposed only as false positives — 4 of 4.** All four sit in weak-Locomotion clips, three as the final interval of their clip where the animal is no longer moving, and the [footage audit](FOOTAGE-AUDIT.md) found no climbing in this set, so none matches a known climb. The class is unusable until real climbing footage exists; nothing here can validate a Climbing proposal.
- `zeroshot_top` over all 75 intervals is Locomotion 65, Climbing 9, Eating 1: the model's default answer for this footage is "walking", which is why so much of the timeline is Uncertain.
- 18 of 22 clips receive no Eating or Climbing proposal at all.

The model itself is not broken, the task is out of its distribution. A control on the same window scored "a video of a civet in a cage" 22.8 against "a busy city street" 17.0 and "a plate of food" 15.6 logits, so the video encoder reads the scene; it just cannot rank the four behavior phrases. A one-off check of six prompt styles over 10 hand-inspected windows (three eating, four walking, three still) scored at most 3 of 10 correct:

| prompt style | correct windows |
|---|---:|
| "a civet with its head down over a feeding bowl" / "moving across the enclosure" / "resting motionless" / "scaling the enclosure wall" | 3/10 |
| "a video of a civet eating food from a bowl" / "walking across the floor" / "lying still on the ground" / "climbing up a vertical structure" (shipped default) | 2/10 |
| "a wildlife video of a civet feeding at a food bowl" / "walking" / "resting on the ground" / "climbing a tree" | 2/10 |
| "the civet is eating food from a bowl" and similar sentences | 0/10 |
| "a civet eating food" and similar noun phrases | 0/10 |
| "a photograph of a civet eating" and similar | 0/10 |

No prompt formulation rescued the ranking. The likely causes are the size of the animal in a 1920×1080 enclosure view, half the footage being infrared/greyscale, and eating-versus-idle being a difference in what the animal is doing with its head rather than a whole-body action.

## Limits

- **This output is weak supervision, never evaluation data.** Folder names are unverified and the smoke set has no reviewed intervals or agreement numbers, so nothing here measures model quality (ADR 0003). Reported metrics still come only from the human gold set in #4.
- Motion thresholds were calibrated on these 22 clips only. A different camera, room, or lighting needs re-calibration; the constants are at the top of `scripts/prelabel.py`.
- A five-second window cut by a behavior boundary is labeled from the full window, and window pieces shorter than two seconds are dropped, which leaves sub-second gaps between some rows.
- No verified Climbing interval exists in the footage, so the four Climbing rows are false positives by the audit, not by a reviewed climb.
- Locomotion is derived from the centroid of moving pixels, so a camera pan, a light change, or a moving animal that never leaves its spot can shift it; an animal that fills the frame gives a weak centroid.

## Next step

Apply the plan's fallback per class: keep Resting and Locomotion auto-derived (they are the pipeline's working half), and label Eating and Climbing by hand until a classifier trained on the human gold set (#4) replaces the zero-shot stage. The zero-shot stage can stay in the loop as an `Uncertain`-only signal, but it must not be given weight in reported metrics.
