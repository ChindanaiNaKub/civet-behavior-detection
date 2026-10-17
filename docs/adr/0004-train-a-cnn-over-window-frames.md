# Train a CNN over the frames of each window

Phase 2 delivers the professor's Task 1 as a per-frame CNN with mean and max pooling over the eight frames of a five-second window. The backbone is an ImageNet-pretrained ResNet-18 from torchvision; only the small pooling head trains on project data. `scripts/model.py` owns the architecture.

## Why

- The brief names a CNN. Running one CNN over the window's frames keeps that pipeline and adds the smallest possible step for time.
- Twenty-two source clips cannot train a video network from scratch. A pretrained backbone with a trained head is the standard way to learn from a few hundred windows, and it fits the 4 GiB GPU on the audit workstation.
- Mean pooling summarises the whole window, and max pooling keeps a state that appears in only one or two frames. Without the max term, a brief action in an otherwise quiet window is averaged away.
- The pre-labeller's zero-shot video-text model failed on this footage for a domain reason, not a size reason, so a larger video model is not the answer. The control in [`../PRELABEL-SMOKE-TEST.md`](../PRELABEL-SMOKE-TEST.md) scored "a video of a civet in a cage" above "a plate of food" on the same window.

## Consequences

- The model sees 8 frames per five-second window, not all 25 frames per second. Posture, position and approach survive; fast motion detail does not.
- Pooling cannot represent motion direction on its own. If Locomotion needs that, add frame differences or a temporal convolution as a second input and re-measure.
- Changing the backbone invalidates every saved checkpoint, because the checkpoint stores the configuration it was trained with.
- Climbing is not a model class. No Climbing clip was supplied, so there is nothing to train or score it on; see [ADR 0007](0007-drop-climbing-from-phase-2-scope.md). The class list lives in `scripts/labels.py` and grows there.
- This supersedes the earlier "compact pretrained video classifier" wording in the Phase 2 plan. Task 2 and Task 3 of the brief were rejected: Task 2 needs YOLO boxes that do not exist here, and pose estimation would fail on this footage for the same reasons the zero-shot stage failed.
