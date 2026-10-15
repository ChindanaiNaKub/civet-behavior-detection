# Track source clips in the repository

The 22 audited source clips (67 MB, all `.mp4`, largest 11.9 MB) are committed under `data/raw/<folder>/<clip_id>.mp4`, using the `relative_path` values in [`../../data/source-clips.csv`](../../data/source-clips.csv). This supersedes the earlier "raw videos are not stored in this Git repository" rule in the README.

## Why

Collaborators must be able to reproduce the dataset from a plain `git clone`. The team is small, the repository is private, and the total size is modest, so the friction of an out-of-band download is not worth the repository cleanliness.

## Consequences

- History retains these blobs permanently; removing the clips later would require a history rewrite.
- The clips are delivered to anyone with read access to this private repository. Do not republish them elsewhere without team approval.
- The files stay exactly as delivered. They carry folder-level labels only, which are weak and unverified — see [`../FOOTAGE-AUDIT.md`](../FOOTAGE-AUDIT.md).
- Source archives, other footage formats, and model checkpoints remain untracked (`.gitignore`). Only `data/raw/**/*.mp4` is un-ignored.
