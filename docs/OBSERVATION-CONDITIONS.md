# Observation condition options

## Evidence and limits

On 2026-10-19, eight frames were sampled from each of all 22 source clips,
using `duration_s * (k + 0.25) / 8` for k = 0 through 7 (176 frames total).
The three contact sheets are retained locally under `outputs/condition-audit/`.
This is a sampling audit for menu design, not a complete frame-by-frame review
or a replacement for either independent behavior annotation pass. No annotation
CSV was changed by this audit. Reviewers must still watch each interval.

Both camera scenes look down into the enclosure at an oblique angle. They show
color or infrared/grayscale images. Plants/planters, fence/mesh, ledges, logs,
walls and framing provide potential obstructions. Their presence in a scene
does not mean they obstruct the animal during every interval. Select none when
they do not block the relevant view. Scene content and the annotation guide
motivate the categories below; an option is not a claim that every state was
observed in every sampled clip.

Overhead and animal-level side view were added at the user's request as additional
choices for other viewing angles. They were not observed in the 22-clip sampling
audit and must not be assigned to those clips merely because the menu offers them.

## Required menus

| Field | CSV value | Meaning |
|---|---|---|
| visibility | `clear` | Animal and the body parts needed to assess the action are visible. |
| visibility | `partial` | Only part of the animal is visible; assess whether behavior is still identifiable. |
| visibility | `fully occluded` | Animal is hidden behind a visible obstruction, rather than outside the frame. |
| visibility | `out of view` | Animal is outside the image boundaries. |
| lighting | `daylight/color` | The color-image condition used by the existing dataset; this does not establish the physical light source or measured brightness. |
| lighting | `infrared/grayscale` | Infrared/grayscale appearance, consistent with the existing dataset vocabulary. |
| camera_view | `elevated oblique` | Camera looks down at an angle. Both scenes have this angle; camera ID is separate source metadata. |
| camera_view | `overhead` | Camera looks straight down from directly above. |
| camera_view | `animal-level side view` | Camera looks horizontally at the animal from the side, at approximately its body level. |
| occlusion | `none` | No obstruction blocks the relevant view of the animal. |
| occlusion | `tree/plant` | Tree, foliage or planter obstructs the view. |
| occlusion | `fence/mesh` | Fence or mesh obstructs the view. |
| occlusion | `enclosure structure` | Ledge, log, wall, frame or another enclosure structure obstructs the view. |
| occlusion | `multiple obstructions` | More than one obstruction category applies during the interval. |
| all four fields | `unknown` | Reviewer explicitly cannot determine this condition from the video. Never selected automatically. |

All four menus begin unselected. Missing values and values outside these lists
are rejected by the page and server before any CSV is written. The server uses
the same option contract that is embedded in the page. Existing CSV files remain
readable; legacy unknown or unsupported values require a fresh menu choice in
editing mode. The tool does not silently convert old text into new categories.

Do not infer behavior from conditions. If visibility is insufficient to identify
the action, label behavior Uncertain and record a reason. Use unknown for
occlusion if the animal is outside the frame and no obstruction can be assessed.
When conditions change materially, divide the interval where appropriate; do
not change independent behavior labels without documenting reconciliation.

## Sampled clips

The camera identifiers below come from the source ledger's recorded overlays.
Only the angle is offered to reviewers, so this menu does not expose camera IDs
in an independent review packet. Lighting describes the sampled frames, not a
certification of every frame in the clip.

| Clip | Source camera | Sampled image | Sampled view |
|---|---|---|---|
| `25690518092326646` | Camera 02 | Color | Elevated oblique |
| `25690518092408866` | Camera 02 | Color | Elevated oblique |
| `25690519193422801` | Camera 02 | Color | Elevated oblique |
| `25690519193547292` | Camera 02 | Color | Elevated oblique |
| `25690519193808204` | Camera 03 | Color | Elevated oblique |
| `25690518090944462` | Camera 03 | Infrared/grayscale | Elevated oblique |
| `25690518091012230` | Camera 03 | Infrared/grayscale | Elevated oblique |
| `25690518091030781` | Camera 03 | Infrared/grayscale | Elevated oblique |
| `25690518091047366` | Camera 03 | Infrared/grayscale | Elevated oblique |
| `25690518091210873` | Camera 03 | Infrared/grayscale | Elevated oblique |
| `25690518091711167` | Camera 02 | Infrared/grayscale | Elevated oblique |
| `25690518091847591` | Camera 02 | Infrared/grayscale | Elevated oblique |
| `25690518091918610` | Camera 02 | Infrared/grayscale | Elevated oblique |
| `25690518091948946` | Camera 02 | Infrared/grayscale | Elevated oblique |
| `25690520153014458` | Camera 02 | Infrared/grayscale | Elevated oblique |
| `25690519194109670` | Camera 03 | Color | Elevated oblique |
| `25690519194200758` | Camera 03 | Color | Elevated oblique |
| `25690519194418953` | Camera 03 | Color | Elevated oblique |
| `25690519194809256` | Camera 03 | Color | Elevated oblique |
| `25690519195143424` | Camera 03 | Color | Elevated oblique |
| `25690520152850102` | Camera 02 | Infrared/grayscale | Elevated oblique |
| `25690520153258104` | Camera 02 | Infrared/grayscale | Elevated oblique |
