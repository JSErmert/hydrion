"""
Matplotlib-based spatial renderer for Hydrion.

Pure visualization backend:
- no mechanics
- no mutation
- deterministic
- optional frame saving
"""

import os
import matplotlib.pyplot as plt


class MatplotlibRenderer:
    def __init__(
        self,
        xlim=(-1.0, 1.2),
        ylim=(-1.0, 1.2),
        particle_size=10,
        save_dir: str | None = None,
    ):
        self.xlim = xlim
        self.ylim = ylim
        self.particle_size = particle_size
        self.save_dir = save_dir

        self.fig = None
        self.ax = None
        self.frame_idx = 0

        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

    # ----------------------------
    # Frame lifecycle
    # ----------------------------

    def begin_frame(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))

        self.ax.clear()
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.axis("off")

    def end_frame(self):
        if self.save_dir is not None:
            fname = f"frame_{self.frame_idx:05d}.png"
            path = os.path.join(self.save_dir, fname)
            self.fig.savefig(path, dpi=150)
            self.frame_idx += 1

        plt.pause(0.001)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

    # ----------------------------
    # Dynamic elements
    # ----------------------------

    def draw_particles(self, particles):
        if not particles:
            return

        xs_free, ys_free = [], []
        xs_cap, ys_cap = [], []

        for p in particles:
            if p.captured:
                xs_cap.append(p.x)
                ys_cap.append(p.y)
            else:
                xs_free.append(p.x)
                ys_free.append(p.y)

        if xs_free:
            self.ax.scatter(
                xs_free,
                ys_free,
                s=self.particle_size,
                c="white",
                edgecolors="none",
            )

        if xs_cap:
            self.ax.scatter(
                xs_cap,
                ys_cap,
                s=self.particle_size,
                c="cyan",
                edgecolors="none",
            )

    # ----------------------------
    # Static geometry (Commit 5.3)
    # ----------------------------

    def draw_inlet(self, x_range, y):
        self.ax.plot(
            [x_range[0], x_range[1]],
            [y, y],
            color="gray",
            linewidth=1,
        )

    def draw_capture_region(self, x, y_top, y_bottom):
        self.ax.plot(
            [x, x],
            [y_bottom, y_top],
            color="cyan",
            linewidth=1,
        )

    def draw_chamber(self, x, y, width, height):
        rect = plt.Rectangle(
            (x - width / 2, y - height / 2),
            width,
            height,
            fill=False,
            edgecolor="cyan",
            linewidth=1,
        )
        self.ax.add_patch(rect)
