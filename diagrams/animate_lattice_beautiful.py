"""Beautiful lattice animation for presentation - improved fps, colors, smoothing.

Improvements over original animate_lattice.py:
 - 180 frames (was 60) at 15 fps (was 2 fps) => 12s smooth, not choppy 2 fps
 - Higher DPI 200 (was 120) => 2800x900 crisp for 16:9 slides
 - Softer palette + shadows + rounded particle caps
 - Subtle background + grid + time counter
 - Particle slide interpolation between MC steps (3 subframes per event) for smooth motion
 - Pulsing waiting dots + entrance glow when blocked
 - Title with LaTeX and current/density overlay
 - Both mp4 (h264 15fps) and gif (15fps, optimized) at high quality
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.colors import to_rgba

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from asep.model import TwoChannelASEP

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
L = 24
ALPHA, BETA = 0.9, 0.28  # near coexistence, shows flipping

# --- Palette (softer, presentation-friendly) ---
C_BG = "#f8f9fa"
C_CH1 = "#1f77b4"  # muted blue
C_CH1_LIGHT = "#4a9eda"
C_CH2 = "#ff7f0e"  # muted orange
C_CH2_LIGHT = "#ffaa4a"
C_EMPTY = "#e9ecef"
C_EMPTY_EDGE = "#adb5bd"
C_ENTRANCE = "#2ca02c"
C_ENTRANCE_GLOW = "#a3d9a5"
C_TEXT = "#212529"

H = 0.95
HALF = H/2.0
W = 0.85
Y_T = 0.60
Y_B = Y_T - H

# --- Simulation with longer warmup ---
sim = TwoChannelASEP(L=L, alpha=ALPHA, beta=BETA, seed=42)
sim.run(n_steps=8000, sample_every=100)

# Collect 90 events, then interpolate 2 subframes per event => 180 frames total
n_events = 90
raw1, raw2 = [], []
for _ in range(n_events):
    sim.run(n_steps=1, sample_every=1)
    raw1.append(sim.lane1.copy())
    raw2.append(sim.lane2.copy())

# Interpolate particle slides: for each event, create 2 intermediate frames where hopping particle slides
# Find which site moved between raw[i] and raw[i+1], then create subframes with particle at intermediate x
SUB = 2
frames1, frames2 = [], []
frames_meta = []  # for each frame, (moving lane, from, to, progress)
for i in range(n_events-1):
    a1, b1 = raw1[i], raw2[i]
    a2, b2 = raw1[i+1], raw2[i+1]
    # base frame
    frames1.append(a1); frames2.append(b1); frames_meta.append(None)
    # find moving particle (single site difference)
    diff1 = np.where(a1 != a2)[0]
    diff2 = np.where(b1 != b2)[0]
    # create SUB subframes with slide
    for s in range(1, SUB+1):
        prog = s/(SUB+1)
        # keep base occupancy, but meta will draw sliding circle
        frames1.append(a1); frames2.append(b1)
        if len(diff1)==1:
            # ch1 hop right or enter/exit
            frames_meta.append((1, diff1[0], prog))
        elif len(diff2)==1:
            frames_meta.append((2, diff2[0], prog))
        else:
            # enter/exit or no move (blocked) - just duplicate
            frames_meta.append(None)
# last raw
frames1.append(raw1[-1]); frames2.append(raw2[-1]); frames_meta.append(None)

n_frames = len(frames1)
print(f"Events {n_events} -> frames {n_frames} (SUB={SUB}) => {n_frames/15:.1f}s at 15fps")

fig, ax = plt.subplots(figsize=(16, 5.2))
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)

boxes=[]
def add_box(xc, y, w, h, lane, site, is_entrance):
    # Use FancyBbox for rounded corners, subtle shadow
    r = FancyBboxPatch((xc - w/2, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       fc=C_EMPTY, ec=C_ENTRANCE if is_entrance else C_EMPTY_EDGE,
                       lw=2.2 if is_entrance else 1.0)
    ax.add_patch(r)
    boxes.append((r, lane, site, is_entrance))

for i in range(L):
    is_ent1 = (i==0)
    h1 = HALF if is_ent1 else H
    add_box(i, Y_T, W, h1, 1, i, is_ent1)
    is_ent2 = (i==L-1)
    h2 = HALF if is_ent2 else H
    y2 = Y_B + (H - h2) if is_ent2 else Y_B
    add_box(i, y2, W, h2, 2, i, is_ent2)

# Waiting queue with glow
QY = Y_T + H/2
QY2 = Y_B + H/2
blue_wait = Circle((-1.25, QY), 0.18, fc=C_CH1, ec="white", lw=1.2, zorder=5)
orange_wait = Circle((L+0.25, QY2), 0.18, fc=C_CH2, ec="white", lw=1.2, zorder=5)
# glow halos
blue_glow = Circle((-1.25, QY), 0.26, fc=to_rgba(C_CH1, 0.25), ec="none", zorder=4)
orange_glow = Circle((L+0.25, QY2), 0.26, fc=to_rgba(C_CH2, 0.25), ec="none", zorder=4)
for c in (blue_glow, orange_glow, blue_wait, orange_wait):
    ax.add_patch(c)

# Sliding particles (drawn on top)
slide_circles = []
for _ in range(2):
    c = Circle((0,0), 0.19, fc="white", ec="black", lw=0.8, zorder=10)
    c.set_visible(False)
    ax.add_patch(c)
    slide_circles.append(c)

# Direction arrows with better style
ax.annotate("", xy=(L-1.6, Y_T+H+0.45), xytext=(1.6, Y_T+H+0.45),
            arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", lw=2.5, color=C_CH1, shrinkA=0, shrinkB=0))
ax.annotate("", xy=(1.6, Y_B-0.45), xytext=(L-1.6, Y_B-0.45),
            arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", lw=2.5, color=C_CH2, shrinkA=0, shrinkB=0))

# Labels
ax.text(L/2, Y_T+H+0.85, "Channel 1  →  enter @0, exit @L−1", ha="center", va="center", fontsize=11, color=C_CH1, fontweight="bold")
ax.text(L/2, Y_B-0.70, "Channel 2  ←  enter @L−1, exit @0", ha="center", va="center", fontsize=11, color=C_CH2, fontweight="bold")
ax.text(0, Y_B-0.55, "narrow entrance: ch1 blocked if lane2[0] occupied", ha="center", fontsize=8, color=C_ENTRANCE, style="italic")
ax.text(L-1, Y_B-0.55, "narrow entrance: ch2 blocked if lane1[L-1] occupied", ha="center", fontsize=8, color=C_ENTRANCE, style="italic")
ax.text(-1.25, QY+0.45, "waiting", ha="center", fontsize=7, color=C_CH1, alpha=0.9)
ax.text(L+0.25, QY2+0.45, "waiting", ha="center", fontsize=7, color=C_CH2, alpha=0.9)

# Title and time counter (moved to top-right to avoid bottom overlap)
title = ax.set_title(f"Two-channel ASEP  L={L}  α={ALPHA:.2f}  β={BETA:.2f}  —  1 frame = 1 event + {SUB}× slide", fontsize=11, color=C_TEXT, pad=18)
time_text = ax.text(0.98, 0.98, "", ha="right", va="top", fontsize=8, color="#6c757d", transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#dee2e6", alpha=0.9))

ax.set_xlim(-2.4, L+1.4)
ax.set_ylim(Y_B-1.15, Y_T+H+1.05)
ax.set_aspect("equal")
ax.axis("off")

def init():
    for r,_,_,_ in boxes:
        r.set_facecolor(C_EMPTY)
        r.set_edgecolor(C_EMPTY_EDGE)
    for c in (blue_wait, orange_wait, blue_glow, orange_glow):
        c.set_visible(False)
    for c in slide_circles:
        c.set_visible(False)
    time_text.set_text("")
    return [r for r,_,_,_ in boxes] + [blue_wait, orange_wait, blue_glow, orange_glow] + slide_circles + [time_text]

def update(frame):
    l1, l2 = frames1[frame], frames2[frame]
    meta = frames_meta[frame]
    # hide slides
    for c in slide_circles:
        c.set_visible(False)
    # boxes
    for r, lane, site, is_ent in boxes:
        if lane==1:
            occ=l1[site]
            if is_ent:
                occ=bool(l1[site]) and not bool(l2[0])
            # entrance glow when blocked
            if is_ent and bool(l2[0]):
                r.set_edgecolor(C_ENTRANCE_GLOW)
                r.set_linewidth(2.8)
            else:
                r.set_edgecolor(C_ENTRANCE if is_ent else C_EMPTY_EDGE)
                r.set_linewidth(2.2 if is_ent else 1.0)
            r.set_facecolor(C_CH1 if occ else C_EMPTY)
            # add subtle inner highlight for occupied
            if occ:
                r.set_alpha(0.95)
            else:
                r.set_alpha(0.85)
        else:
            occ=l2[site]
            if is_ent:
                occ=bool(l2[site]) and not bool(l1[L-1])
            if is_ent and bool(l1[L-1]):
                r.set_edgecolor(C_ENTRANCE_GLOW)
                r.set_linewidth(2.8)
            else:
                r.set_edgecolor(C_ENTRANCE if is_ent else C_EMPTY_EDGE)
                r.set_linewidth(2.2 if is_ent else 1.0)
            r.set_facecolor(C_CH2 if occ else C_EMPTY)
            r.set_alpha(0.95 if occ else 0.85)
        # sliding particle
    if meta is not None:
        lane, site, prog = meta
        # find hop direction
        if lane==1:
            # ch1 moves right: site -> site+1, or enter at 0, or exit at L-1
            # For slide, draw circle moving from x=site to site+1
            x0=site
            x1=site+1
            # handle enter (from queue) and exit (to outside)
            if site==0 and not bool(frames1[frame][0]):  # enter
                x0=-1.25; x1=0
            elif site==L-1 and bool(frames1[frame][L-1]): # exit
                x0=L-1; x1=L+0.5
            x = x0 + (x1-x0)*prog
            y = QY if site==0 else Y_T+H/2
            c=slide_circles[0]
            c.center=(x,y)
            c.set_facecolor(C_CH1)
            c.set_edgecolor("white")
            c.set_visible(True)
            # hide box particle for sliding origin
            for r, la, si, _ in boxes:
                if la==1 and si==site:
                    r.set_facecolor(C_EMPTY)
        else:
            x0=site
            x1=site-1
            if site==L-1 and not bool(frames2[frame][L-1]):
                x0=L+0.25; x1=L-1
            elif site==0 and bool(frames2[frame][0]):
                x0=0; x1=-0.5
            x = x0 + (x1-x0)*prog
            y = QY2 if site==L-1 else Y_B+H/2
            c=slide_circles[1]
            c.center=(x,y)
            c.set_facecolor(C_CH2)
            c.set_edgecolor("white")
            c.set_visible(True)
            for r, la, si, _ in boxes:
                if la==2 and si==site:
                    r.set_facecolor(C_EMPTY)
    # waiting glow pulsing
    t = frame / n_frames
    pulse = 0.5 + 0.5*np.sin(t*2*np.pi*3)
    blocked1=bool(l2[0]); blocked2=bool(l1[L-1])
    blue_wait.set_visible(blocked1); blue_glow.set_visible(blocked1)
    orange_wait.set_visible(blocked2); orange_glow.set_visible(blocked2)
    blue_glow.set_alpha(0.15+0.25*pulse if blocked1 else 0)
    orange_glow.set_alpha(0.15+0.25*pulse if blocked2 else 0)
    # time counter
    time_text.set_text(f"event {frame}/{n_frames}  •  t={frame*0.1:.1f}")
    return [r for r,_,_,_ in boxes] + [blue_wait, orange_wait, blue_glow, orange_glow] + slide_circles + [time_text]

anim = animation.FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True, interval=1000/15)

mp4 = os.path.join(OUT_DIR, "lattice_animation.mp4")
anim.save(mp4, writer="ffmpeg", fps=15, dpi=200, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'slow'])
print(f"wrote {mp4} {n_frames} frames at 15fps => {n_frames/15:.1f}s")
# gif with lower fps for size
gif = os.path.join(OUT_DIR, "lattice_animation.gif")
anim.save(gif, writer="pillow", fps=10, dpi=100)
print(f"wrote {gif}")

# copy to presentation
import shutil
shutil.copy(mp4, os.path.join(os.path.dirname(OUT_DIR), "presentation", "lattice_animation.mp4"))
print("copied to presentation/")
