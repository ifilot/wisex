# 3D Rotations with Manim

This folder contains Python scripts using the
[Manim](https://www.manim.community/) animation engine to visualize **rotations
in 3D**. The scripts showcase how vectors in \(\mathbb{R}^3\) can be rotated
around arbitrary axes using angle-axis representations and Lie algebra concepts.

## Features

- 3D vector visualization using `ThreeDScene`
- Arbitrary axis-angle rotations
- Lie algebra (so(3)) representations
- Clean animations for educational or illustrative purposes

## Requirements

To run these scripts, you'll need:

- Python 3.8+
- [Manim Community Edition](https://docs.manim.community/en/stable/)

You can install Manim with pip:

```bash
pip install manim
```

## How to run

To render a scene (e.g., a file named `rotation_3d.py`), use:

```bash
manim -pql rotation_3d.py RotateVectors3D
```

* `-p` previews the video after rendering
* `-ql` renders at low quality for fast testing
* Replace `RotateVectors3D` with the class name of the scene you'd like to run