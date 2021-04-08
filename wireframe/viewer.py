#!/usr/bin/env python3
import sys
import json
import math
import pickle
import random
import os.path as osp

import cv2
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from vispy import app, gloo
from vispy.gloo.util import _screenshot
from vispy.util.transforms import rotate, translate

app.use_app("pyqt5")

VERTEX = """
attribute vec3 vertex;

uniform mat4 proj;
uniform mat4 view;
uniform mat4 rotation;

void main()
{
    vec4 pos = proj * view * rotation * vec4(vertex, 1);
    pos.z /= 10;
    gl_Position = pos;
}
"""

FRAGMENT = """
float colormap_red(float x) {
    if (x < 0.5) {
        return -6.0 * x + 67.0 / 32.0;
    } else {
        return 6.0 * x - 79.0 / 16.0;
    }
}

float colormap_green(float x) {
    if (x < 0.4) {
        return 6.0 * x - 3.0 / 32.0;
    } else {
        return -6.0 * x + 79.0 / 16.0;
    }
}

float colormap_blue(float x) {
    if (x < 0.7) {
       return 6.0 * x - 67.0 / 32.0;
    } else {
       return -6.0 * x + 195.0 / 32.0;
    }
}

vec4 colormap(float x) {
    float r = clamp(colormap_red(x), 0.0, 1.0);
    float g = clamp(colormap_green(x), 0.0, 1.0);
    float b = clamp(colormap_blue(x), 0.0, 1.0);
    return vec4(r, g, b, 1.0);
}

void main() {
    gl_FragColor = colormap(gl_FragCoord.w);
}
"""


class Canvas(app.Canvas):
    def __init__(self, prefix, vertices, lines, proj_matrix, view_matrix):
        app.Canvas.__init__(self, config=dict(samples=8), size=(1024, 1024))

        self.prefix = prefix
        self.rotation = np.eye(4, dtype=np.float32)
        self.view = view_matrix

        self.native.setWindowTitle("wireframe")

        self.program = gloo.Program(VERTEX, FRAGMENT)
        self.program["vertex"] = vertices
        self.program["proj"] = proj_matrix.T.astype(np.float32)
        self.index_buffer = gloo.IndexBuffer(lines.astype(np.uint32))
        self.fbo = gloo.FrameBuffer(
            gloo.Texture2D(shape=self.size + (4,)),
            gloo.RenderBuffer(self.size, format="depth"),
        )

        gloo.set_line_width(2)
        gloo.set_viewport(0, 0, self.size[0], self.size[1])
        self.alpha = 0
        self.theta = 0
        self.update_matrix()
        self.show()

    def update_matrix(self):
        rotation_matrix = rotate(self.alpha, [0, 1, 0]) @ rotate(self.theta, [1, 0, 0])
        self.program["rotation"] = rotation_matrix
        self.program["view"] = self.view
        self.update()

    def on_draw(self, event):
        gloo.clear(color="black")
        self.program.draw("lines", self.index_buffer)
        # gloo.set_state(blend=True, depth_test=True, polygon_offset_fill=True)

    def on_resize(self, event):
        gloo.set_viewport(0, 0, self.size[0], self.size[1])

    def on_mouse_wheel(self, event):
        self.view = self.view @ translate((0, 0, -0.2 * event.delta[1]))
        self.update_matrix()

    def on_key_press(self, event):
        # modifiers = [key.name for key in event.modifiers]
        if event.text == "S":
            self.screenshot()

    def screenshot(self):
        with self.fbo:
            gloo.clear(color="black")
            gloo.set_viewport(0, 0, self.size[0], self.size[1])
            self.program.draw("lines", self.index_buffer)
            im = _screenshot((0, 0, self.size[0], self.size[1]))[:, :, :3]
        for i in range(100):
            if not osp.exists(f"{self.prefix}_s{i:02d}.png"):
                print(f"Save to {self.prefix}_s{i:02d}.png")
                cv2.imwrite(f"{self.prefix}_s{i:02d}.png", im[:, :, ::-1])
                break

    def on_mouse_move(self, event):
        if event.button == 1 and event.last_event is not None:
            x0, y0 = event.last_event.pos
            x1, y1 = event.pos
            self.alpha -= 0.3 * (x1 - x0)
            self.theta -= 0.3 * (y1 - y0)
            self.update_matrix()
        elif event.button == 2:
            self.alpha = 0
            self.theta = 0
            self.update_matrix()
        elif event.button == 3 and event.last_event is not None:
            # middle drag, pan
            x0, y0 = event.last_event.pos
            x1, y1 = event.pos
            self.view = self.view @ translate(
                (0.003 * (x1 - x0), 0.003 * (y0 - y1), 0.0)
            )
            self.update_matrix()


def show_wireframe(prefix, vertices, lines, projection_matrix=np.eye(4)):
    vertices = vertices.copy()
    d = np.average(vertices[:, 2])
    vertices[:, 2] -= d
    view_matrix = translate((0, 0, d))
    c = Canvas(  # noqa
        prefix,
        np.array(vertices, dtype=np.float32),
        np.array(lines, dtype=np.uint16),
        projection_matrix,
        view_matrix,
    )
    app.run()
