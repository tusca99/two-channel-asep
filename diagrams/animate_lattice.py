"""Animate the two-channel ASEP lattice using the real simulation model.

Gillespie continuous-time MC: each frame is ONE event (one particle moves).
Bulk hops are rate 1 (one site, only if the next site is empty); the ends are
enter (rate alpha) / exit (rate beta).

Narrow-entrance visualisation (shared bottleneck at each end):
  - position 0  : ch1 ENTERS (blue, top half-box)  <->  ch2 EXITS (orange, square)
  - position L-1 : ch1 EXITS (blue, square)         <->  ch2 ENTERS (orange, half-box)
The two end sites of a lane are drawn as a SHARED bottleneck: a particle can
only be in the bottleneck if the partner lane's end site is empty. When it is
blocked, the entering particle is shown WAITING in a queue outside the entrance,
so you can see it wait for the other type to free the entrance.

Output: diagrams/lattice_animation.mp4 (and .gif).  ~2 events/sec.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle, Circle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from asep.model import TwoChannelASEP

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
L = 20
ALPHA, BETA = 0.9, 0.3

C_CH1 = "#1f77b4"
C_CH2 = "#ff7f0e"
C_EMPTY = "#b0b0b0"
C_ENTRANCE = "#2ca02c"

# Two glued rows (touch at Y_T).
H = 0.9
HALF = H / 2.0
W = 0.82
Y_T = 0.55            # bottom of top (ch1) row == top of bottom (ch2) row
Y_B = Y_T - H         # bottom of bottom (ch2) row

sim = TwoChannelASEP(L=L, alpha=ALPHA, beta=BETA, seed=7)
sim.run(n_steps=4000, sample_every=100)  # warmup into steady state

# One MC step per frame = one event.
n_frames = 60
frames1, frames2 = [], []
for _ in range(n_frames):
    sim.run(n_steps=1, sample_every=1)
    frames1.append(sim.lane1.copy())
    frames2.append(sim.lane2.copy())

fig, ax = plt.subplots(figsize=(14, 4.6))

boxes = []  # (rect, lane, site, is_entrance_half)


def add_box(xc, y, w, h, lane, site, is_entrance):
    r = Rectangle((xc - w / 2, y), w, h, fc=C_EMPTY,
                  ec=C_ENTRANCE if is_entrance else "#333333",
                  lw=2 if is_entrance else 1)
    ax.add_patch(r)
    boxes.append((r, lane, site, is_entrance))


for i in range(L):
    # Top lane (ch1, ->): entrance at i==0 (half-box at bottom of top row,
    # attached to ch2's square below).
    is_ent1 = (i == 0)
    h1 = HALF if is_ent1 else H
    y1 = Y_T if is_ent1 else Y_T
    add_box(i, y1, W, h1, 1, i, is_ent1)

    # Bottom lane (ch2, <-): entrance at i==L-1 (half-box at top of bottom row,
    # attached to ch1's square above).
    is_ent2 = (i == L - 1)
    h2 = HALF if is_ent2 else H
    y2 = Y_B + (H - h2) if is_ent2 else Y_B
    add_box(i, y2, W, h2, 2, i, is_ent2)

# Waiting markers (single dot outside each entrance): at most one particle can
# be waiting, since each entrance is a single site.
QY = Y_T + H / 2          # queue vertical centre (top lane level)
QY2 = Y_B + H / 2         # queue vertical centre (bottom lane level)
blue_wait = Circle((-1.0, QY), 0.16, fc=C_CH1, ec="#000000", lw=0.5)
orange_wait = Circle((L, QY2), 0.16, fc=C_CH2, ec="#000000", lw=0.5)
for c in (blue_wait, orange_wait):
    ax.add_patch(c)

# Hopping-direction arrows.
ax.annotate("", xy=(L - 1.6, Y_T + H + 0.25), xytext=(1.6, Y_T + H + 0.25),
            arrowprops=dict(arrowstyle="->", lw=2.2, color=C_CH1))
ax.annotate("", xy=(1.6, Y_B - 0.25), xytext=(L - 1.6, Y_B - 0.25),
            arrowprops=dict(arrowstyle="->", lw=2.2, color=C_CH2))

# Lane labels (on top of each channel).
ax.text(L / 2, Y_T + H + 0.55, "Channel 1  (→)  enter @0, exit @L-1",
        ha="center", va="center", fontsize=10, color=C_CH1, fontweight="bold")
ax.text(L / 2, Y_B - 0.55, "Channel 2  (←)  enter @L-1, exit @0",
        ha="center", va="center", fontsize=10, color=C_CH2, fontweight="bold")

# Entrance / queue labels.
ax.text(0, Y_B - 0.55, "narrow entrance: ch1 enter ⟂ lane2[0] empty",
        ha="center", fontsize=8, color=C_ENTRANCE)
ax.text(L - 1, Y_B - 0.55, "narrow entrance: ch2 enter ⟂ lane1[L-1] empty",
        ha="center", fontsize=8, color=C_ENTRANCE)
ax.text(-1.0, QY + 0.35, "waiting", ha="center", fontsize=7, color=C_CH1)
ax.text(L, QY2 + 0.35, "waiting", ha="center", fontsize=7, color=C_CH2)
ax.text(0.5, -0.85, "1 frame = 1 event. Bulk hop: rate 1, one site, only if next site empty. "
                    "Ends: enter α=%.2f, exit β=%.2f. A waiting dot = entrance blocked by the other lane's exit."
        % (ALPHA, BETA), ha="center", fontsize=9, transform=ax.transAxes)

ax.set_xlim(-2.2, L + 1.6)
ax.set_ylim(Y_B - 0.9, Y_T + H + 0.6)
ax.set_aspect("equal")
ax.axis("off")


def init():
    for r, _, _, _ in boxes:
        r.set_facecolor(C_EMPTY)
    for c in (blue_wait, orange_wait):
        c.set_visible(False)
    return [r for r, _, _, _ in boxes] + [blue_wait, orange_wait]


def update(frame):
    l1, l2 = frames1[frame], frames2[frame]

    # Interior + exit sites (full boxes) and entrance halves.
    for r, lane, site, is_ent in boxes:
        if lane == 1:
            occ = l1[site]
            # entrance half at pos 0 only shows when the partner exit is free
            if is_ent:
                occ = bool(l1[site]) and not bool(l2[0])
            r.set_facecolor(C_CH1 if occ else C_EMPTY)
        else:
            occ = l2[site]
            if is_ent:
                occ = bool(l2[site]) and not bool(l1[L - 1])
            r.set_facecolor(C_CH2 if occ else C_EMPTY)

    # Waiting markers: shown when the entrance is blocked by the partner's exit.
    blocked1 = bool(l2[0])          # ch1 entrance blocked by ch2 exit at pos 0
    blocked2 = bool(l1[L - 1])      # ch2 entrance blocked by ch1 exit at L-1
    blue_wait.set_visible(blocked1)
    orange_wait.set_visible(blocked2)

    return [r for r, _, _, _ in boxes] + [blue_wait, orange_wait]


anim = animation.FuncAnimation(
    fig, update, frames=len(frames1), init_func=init, blit=True, interval=500
)

mp4 = os.path.join(OUT_DIR, "lattice_animation.mp4")
anim.save(mp4, writer="ffmpeg", fps=2, dpi=120)
gif = os.path.join(OUT_DIR, "lattice_animation.gif")
anim.save(gif, writer="pillow", fps=2, dpi=120)
print("wrote", mp4, "and", gif, "frames=", len(frames1), "duration~", len(frames1) / 2, "s")
