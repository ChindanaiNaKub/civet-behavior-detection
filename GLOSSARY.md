# Civet Behavior Detection

This glossary defines the behavior labels used to describe civet footage and the evidence available to the project.

## Behavior labels

**Behavior**:
An observable action or state of the civet over time. Each labeled clip segment has one behavior label; when actions overlap, use the most specific visible behavior or mark it Uncertain if no one behavior is clear.

**Eating**:
Food-directed action by the civet.

**Locomotion**:
Movement in which the civet changes its location.
_Avoid_: Walking (too narrow for the Phase 1 category)

**Climbing**:
Movement up or down a sloped or vertical structure, or active climbing along it. Horizontal travel on a structure is Locomotion.

**Resting**:
A continuously inactive state lasting at least five seconds.

**Transition**:
An interval in which the civet changes from one behavior to another; it is not one of the behavior labels.

**Uncertain**:
An annotation state used when the available evidence is insufficient to assign one behavior label.

## Activity states

These describe footage that has no annotation, where only what a fixed camera can
measure is known. They are not behavior labels: an Active window does not say
whether the animal was eating or walking, and they never appear in
`data/annotations.csv`.

**Active**:
A five-second window in which the animal is in view and moving, either because
its silhouette travels at least 0.05 body widths per second or because a fifth of
it changed in a second.

**Inactive**:
A five-second window in which the animal is in view and neither of those holds.
It is the camera's evidence of rest, not a verified Resting label.

**Absent**:
A five-second window in which the animal is not in view. A state of the table,
never a training class.

**Unmeasured**:
A sampled frame the scan could not read because the whole view changed, as at
dusk or when the infrared gain steps. Counted as neither presence nor absence.
