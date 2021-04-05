#!/usr/bin/env python3
import os
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
PI2 = math.pi * 2

VERTEX = """
attribute vec3 vertex;

uniform mat4 proj;
uniform mat4 view;
uniform mat4 rotation;
uniform float dmin;
uniform float dmax;

varying float depth;

void main()
{
    vec4 rotated = rotation * vec4(vertex, 1);
    vec4 pos = proj * view * rotated;
    depth = (rotated.z / rotated.w - dmin) / (dmax - dmin);
    pos.z /= 100;
    gl_Position = pos;
}
"""

FRAGMENT = """
varying float depth;

float colormap_red(float x) {
    if (x < 0.75) {
        return 8.0 / 9.0 * x - (13.0 + 8.0 / 9.0) / 1000.0;
    } else {
        return (13.0 + 8.0 / 9.0) / 10.0 * x - (3.0 + 8.0 / 9.0) / 10.0;
    }
}

float colormap_green(float x) {
    if (x <= 0.375) {
        return 8.0 / 9.0 * x - (13.0 + 8.0 / 9.0) / 1000.0;
    } else if (x <= 0.75) {
        return (1.0 + 2.0 / 9.0) * x - (13.0 + 8.0 / 9.0) / 100.0;
    } else {
        return 8.0 / 9.0 * x + 1.0 / 9.0;
    }
}

float colormap_blue(float x) {
    if (x <= 0.375) {
        return (1.0 + 2.0 / 9.0) * x - (13.0 + 8.0 / 9.0) / 1000.0;
    } else {
        return 8.0 / 9.0 * x + 1.0 / 9.0;
    }
}

vec4 colormap(float x) {
    float r = clamp(colormap_red(x), 0.0, 1.0);
    float g = clamp(colormap_green(x), 0.0, 1.0);
    float b = clamp(colormap_blue(x), 0.0, 1.0);
    return vec4(r, g, b, 1.0);
}

void main() {
    gl_FragColor = colormap(depth);
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
            gloo.Texture2D(shape=(2048, 2048, 3)),
            gloo.RenderBuffer((2048, 2048), format="depth"),
        )

        self.vertices = np.c_[vertices, np.ones(len(vertices))]
        gloo.set_viewport(0, 0, self.size[0], self.size[1])
        self.alpha = 0
        self.theta = 0
        self.update_matrix()
        self.show()

    def update_matrix(self):
        rotation_matrix = rotate(self.alpha, [0, 1, 0]) @ rotate(self.theta, [1, 0, 0])
        self.program["rotation"] = rotation_matrix
        self.program["view"] = self.view
        depth = (self.vertices @ rotation_matrix)[:, 2]
        self.program["dmin"] = np.min(depth)
        self.program["dmax"] = np.max(depth)
        self.update()

    def on_draw(self, event):
        gloo.set_line_width(13)
        gloo.set_viewport(0, 0, self.size[0], self.size[1])
        gloo.set_state(blend=True, depth_test=True)
        gloo.clear(color="white", depth=True)
        self.program.draw("lines", self.index_buffer)

    def on_resize(self, event):
        gloo.set_viewport(0, 0, self.size[0], self.size[1])

    def on_mouse_wheel(self, event):
        self.view = self.view @ translate((0, 0, -0.005 * event.delta[1]))
        self.update_matrix()

    def on_key_press(self, event):
        # modifiers = [key.name for key in event.modifiers]
        if event.text == "S":
            self.screenshot()
        elif event.text == "A":
            self.videotape()

    def screenshot(self, linewidth=26, start=0):
        with self.fbo:
            gloo.set_line_width(linewidth)
            gloo.set_viewport(0, 0, 2048, 2048)
            gloo.set_state(blend=True, depth_test=True)
            gloo.clear(color="white", depth=True)
            self.program.draw("lines", self.index_buffer)
            im = _screenshot((0, 0, 2048, 2048))[:, :, :3]
        for i in range(start, 9999):
            if not osp.exists(f"{self.prefix}_s{i:04d}.png"):
                print(f"Save to {self.prefix}_s{i:04d}.png")
                cv2.imwrite(f"{self.prefix}_s{i:04d}.png", im[:, :, ::-1])
                break

    def videotape(self):
        oldprefix = self.prefix
        os.makedirs(self.prefix, exist_ok=True)
        self.prefix = osp.join(self.prefix, "video")
        A = np.r_[
            np.linspace(0, 000, 50),
            np.linspace(0, 360, 100),
            np.linspace(0, 0, 20),
            np.linspace(0, 000, 100),
            np.linspace(0, 0, 20),
            np.linspace(0, 360, 100),
            np.linspace(0, 0, 20),
            np.linspace(0, 360, 100),
            np.linspace(0, 0, 20),
        ]
        T = np.r_[
            np.linspace(0, 000, 50),
            np.linspace(0, 000, 100),
            np.linspace(0, 0, 20),
            np.linspace(0, 360, 100),
            np.linspace(0, 0, 20),
            np.linspace(0, 360, 100),
            np.linspace(0, 0, 20),
            np.linspace(0, -360, 100),
            np.linspace(0, 0, 20),
        ]
        for i, (self.alpha, self.theta) in enumerate(zip(A, T)):
            self.update_matrix()
            self.screenshot(linewidth=40)
        self.prefix = oldprefix

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
                (0.001 * (x1 - x0), 0.001 * (y0 - y1), 0.0)
            )
            self.update_matrix()


def show_wireframe(prefix, vertices, lines, projection_matrix=np.eye(4)):
    vertices = vertices.copy()
    d = np.average(vertices, axis=0)
    vertices -= d
    view_matrix = translate(tuple(d))
    print(projection_matrix)
    c = Canvas(  # noqa
        prefix,
        np.array(vertices, dtype=np.float32),
        np.array(lines, dtype=np.uint16),
        projection_matrix,
        view_matrix,
    )
    app.run()
