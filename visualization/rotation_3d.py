from manim import *
import numpy as np

class RotateVectors3D(ThreeDScene):
    def construct(self):
        # Set up the 3D camera
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        # Add 3D coordinate axes and set their colors manually
        axes = ThreeDAxes(x_range=[-2, 2], y_range=[-2, 2], z_range=[-2, 2],
                          x_length=4, y_length=4, z_length=4)
        axes.x_axis.set_color(BLACK)
        axes.y_axis.set_color(BLACK)
        axes.z_axis.set_color(BLACK)
        self.add(axes)

        # === Read axis and angle from file ===
        try:
            with open("rotation.txt", "r") as f:
                values = list(map(float, f.read().strip().split()))
                axis = np.array(values[:3])
                axis = axis / np.linalg.norm(axis)
                angle = values[3]
        except Exception as e:
            print("Error reading 'rotation.txt', using default values:", e)
            axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
            angle = np.pi / 3

        # Define three basis vectors with dark, high-contrast colors
        v1 = Vector([1, 0, 0], color="#FF0000")   # dark red
        v2 = Vector([0, 1, 0], color="#00FF00")   # dark green
        v3 = Vector([0, 0, 1], color="#0000FF")   # dark blue
        self.add(v1, v2, v3)

        # Show rotation axis as a dashed black line
        axis_length = 2
        axis_start = -axis_length * axis
        axis_end = axis_length * axis
        dashed_axis = DashedLine(start=axis_start, end=axis_end, color=BLACK)
        self.add(dashed_axis)

        # Animate rotation
        self.play(
            Rotate(v1, angle=angle, axis=axis, about_point=ORIGIN),
            Rotate(v2, angle=angle, axis=axis, about_point=ORIGIN),
            Rotate(v3, angle=angle, axis=axis, about_point=ORIGIN),
            run_time=3
        )

        self.wait()