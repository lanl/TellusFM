#!/usr/bin/env python3
"""Redraw HOSS validation plots from the ``.npy`` files written by the DI_fractures ``hoss`` code.

File format (``logs/val_metrics/version_*/validation_samples/NNNN_validation_<mat>_<method>_<orientation>.npy``)
-------------------------------------------------------------------------------------------------------------
One array of shape (N, 5) per validation sample, one row per mesh element, in the order the network saw them
(sorted by x, then y)::

    column 0, 1 : x, y of the element midpoint (metres; swapped x<->y for "vertical" samples)
    column 2    : input value   (initial damage; 2.0 was mapped to 1.0)
    column 3    : target value  (final damage)
    column 4    : prediction    (relu of the network output)

The line endpoints of each element were NOT saved, so only midpoints can be drawn from the npy alone.

Two plot styles
---------------
branch : exactly what the original code produced - a scatter plot of the midpoints (inferno_r with white
         for the lowest colour, marker size 0.7, colour bars, 12x4 inch figure at 400 dpi).
paper  : the figure style of the manuscript - every element drawn as a line segment, "rocket_r" colours,
         only the cracks in the initial configuration, no colour bars. This needs the line endpoints, which are
         read from the raw simulation files ``frac_pull_z_<id>.h5``. The matching file is found automatically
         by comparing the midpoints, so no simulation id is needed; just point ``--source-dir`` at a folder
         that contains the raw files.

Examples
--------
python example/plot_hoss_npy.py path/to/validation_samples/0000_validation_shale_*_horizontal.npy --style branch
python example/plot_hoss_npy.py path/to/validation_samples --style paper --source-dir ../data/hoss --out plots
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

NAME = re.compile(r"^(\d+)_validation_([A-Za-z]+)_(.+)_(horizontal|vertical)\.npy$")

# seaborn "rocket" sampled every 17th entry of its 256-step table (dark -> light); used reversed
_ROCKET = [
    (0.01060815, 0.01808215, 0.10018654), (0.10132312, 0.06286782, 0.16604287),
    (0.19804623, 0.09465217, 0.23478512), (0.29977678, 0.11356089, 0.29254823),
    (0.40642019, 0.12151801, 0.33288464), (0.51728314, 0.11795580, 0.35453252),
    (0.63139686, 0.10067417, 0.35664819), (0.74297501, 0.08747196, 0.33329799),
    (0.83850183, 0.14025098, 0.28768780), (0.90848638, 0.24568473, 0.24598324),
    (0.94291042, 0.37549479, 0.26369821), (0.95681685, 0.49882008, 0.34555431),
    (0.96298491, 0.61262470, 0.45145074), (0.96544032, 0.72010124, 0.57640375),
    (0.97071849, 0.82151775, 0.71987163), (0.98137749, 0.92061729, 0.86536915),
]
ROCKET_R = mcolors.LinearSegmentedColormap.from_list("rocket_r", _ROCKET[::-1], N=256)
# in the paper a value of 1 (a crack) is the dark purple at ~74 % of the colormap (measured from the PNG)
PAPER_VMIN, PAPER_VMAX = 0.0, 1.0 / 0.74


def load_sample(path):
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 5:
        raise ValueError(f"{path}: expected an (N, 5) array [x, y, input, target, prediction], got {arr.shape}")
    return dict(mesh=arr[:, :2], inp=arr[:, 2], target=arr[:, 3], pred=arr[:, 4])


def parse_name(path):
    m = NAME.match(os.path.basename(path))
    if not m:
        raise ValueError(f"unexpected file name: {os.path.basename(path)}")
    index, mat, method, orientation = m.groups()
    return int(index), mat, method, orientation


# ---------------------------------------------------------------------------------------------- branch style
def branch_cmap():
    colors = plt.get_cmap("inferno_r")(np.linspace(0, 1, 256))
    colors[0] = [1.0, 1.0, 1.0, 1.0]            # white for the lowest colour
    return mcolors.ListedColormap(colors)


def plot_branch(sample, mat, orientation, out_png):
    cmap = branch_cmap()
    mesh = sample["mesh"]
    string = f"material:{mat}-direction:{7 if orientation == 'horizontal' else 8}"
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=400)
    panels = [("Input", sample["inp"], {}), ("Target", sample["target"], {"vmax": 1}),
              (f"Predicted {string}", sample["pred"], {})]
    for ax, (title, values, kw) in zip(axs, panels):
        sc = ax.scatter(mesh[:, 0], mesh[:, 1], c=values, s=0.7, cmap=cmap, **kw)
        fig.colorbar(sc, ax=ax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", "box")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- paper style
def _raw_geometry(h5_path, sim_id, orientation):
    """Midpoints (float32, in network order) and half-vectors of all elements of one raw simulation."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        data = f[f"model {sim_id} fracture initial"][:]
    x1, y1, x2, y2 = data[:, 0], data[:, 1], data[:, 3], data[:, 4]
    keep = (x1 > 0.001) & (x1 < 0.248) & (y1 > 0.001) & (y1 < 0.25) & (x2 > 0.001) & (x2 < 0.248) & (y2 > 0.001) & (y2 < 0.25)
    x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]
    mid = np.stack([(x1 + x2) / 2, (y1 + y2) / 2], axis=1).astype(np.float32)
    half = np.stack([(x2 - x1) / 2, (y2 - y1) / 2], axis=1)
    if orientation == "vertical":
        mid, half = mid[:, [1, 0]], half[:, [1, 0]]
    order = np.lexsort((mid[:, 1], mid[:, 0]))      # same ordering as the network input
    return mid[order], half[order]


def find_source(mesh, orientation, source_files, cache):
    """Find the raw simulation whose midpoints equal ``mesh`` (the npy file does not store the id)."""
    for path in source_files:
        sim_id = re.search(r"frac_pull_z_(\d+)\.h5$", path).group(1)
        key = (path, orientation)
        if key not in cache:
            try:
                cache[key] = _raw_geometry(path, sim_id, orientation)
            except KeyError:
                cache[key] = None
        geom = cache[key]
        if geom is not None and geom[0].shape == mesh.shape and np.allclose(geom[0], mesh, atol=1e-7):
            return path, geom[1]
    return None, None


def _draw_lines(ax, mesh, half, values, vmin, vmax, lo, hi, min_value=None, linewidth=0.6):
    if min_value is not None:
        keep = values >= min_value
        mesh, half, values = mesh[keep], half[keep], values[keep]
    segments = np.stack([mesh - half, mesh + half], axis=1)
    order = np.argsort(values, kind="stable")          # cracks (high values) drawn last, on top
    lc = LineCollection(segments[order], array=values[order], cmap=ROCKET_R, linewidths=linewidth)
    lc.set_clim(vmin, vmax)
    ax.add_collection(lc)
    pad = 0.05 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_aspect("equal", "box")
    ax.set_xticks([])
    ax.set_yticks([])
    return lc


def plot_paper(sample, half, out_png, linewidth=0.6, colorbar=False):
    """Three-panel figure laid out like the manuscript figure (panel size, spacing and title size measured
    from the published PNG: one group is ~1330 x 591 px with three 409 x 407 px panels)."""
    mesh = sample["mesh"].astype(float)
    ends = np.concatenate([mesh - half, mesh + half])
    lo, hi = ends.min(axis=0), ends.max(axis=0)
    width_px, height_px = 1330.0, 591.0
    fig = plt.figure(figsize=(width_px / 200, height_px / 200), dpi=400)   # 200 px per inch layout, saved at 2x
    panel_w, gap = 409.0, 18.0
    left0, bottom, panel_h = 31.0, 30.0, 407.0
    titles = ["Initial\nconfiguration", "Simulation", "Predicted"]
    fields = [(sample["inp"], 0.5), (sample["target"], None), (sample["pred"], None)]
    for k, (title, (values, min_value)) in enumerate(zip(titles, fields)):
        x0 = left0 + k * (panel_w + gap)
        ax = fig.add_axes([x0 / width_px, bottom / height_px, panel_w / width_px, panel_h / height_px])
        lc = _draw_lines(ax, mesh, half, values, PAPER_VMIN, PAPER_VMAX, lo, hi, min_value, linewidth)
        ax.set_aspect("auto")                      # the panel is square already; keeps the frame size fixed
        ax.set_title(title, fontsize=20, pad=8)
        if colorbar:
            fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_png)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="npy files and/or folders containing them")
    p.add_argument("--style", choices=["branch", "paper", "both"], default="both")
    p.add_argument("--source-dir", help="folder with raw frac_pull_z_<id>.h5 files (needed for the paper style)")
    p.add_argument("--out", default="hoss_npy_plots", help="output folder (default: hoss_npy_plots)")
    p.add_argument("--linewidth", type=float, default=0.6, help="line width for the paper style")
    p.add_argument("--colorbar", action="store_true", help="add colour bars to the paper style")
    args = p.parse_args()

    files = []
    for item in args.inputs:
        files += sorted(glob.glob(os.path.join(item, "*_validation_*.npy"))) if os.path.isdir(item) else [item]
    files = [f for f in files if NAME.match(os.path.basename(f))]
    if not files:
        sys.exit("no *_validation_*.npy files found")

    want_paper = args.style in ("paper", "both")
    source_files = []
    if want_paper:
        if not args.source_dir:
            if args.style == "paper":
                sys.exit("--source-dir is required for --style paper")
            print("note: no --source-dir given, drawing the branch style only")
            want_paper = False
        else:
            source_files = sorted(glob.glob(os.path.join(args.source_dir, "frac_pull_z_*.h5")))
            if not source_files:
                sys.exit(f"no frac_pull_z_<id>.h5 files in {args.source_dir}")

    os.makedirs(args.out, exist_ok=True)
    cache = {}
    for path in files:
        index, mat, method, orientation = parse_name(path)
        stem = os.path.splitext(os.path.basename(path))[0].replace(" ", "_")
        sample = load_sample(path)
        if args.style in ("branch", "both"):
            plot_branch(sample, mat, orientation, os.path.join(args.out, f"{stem}_branch.png"))
        if want_paper:
            src, half = find_source(sample["mesh"], orientation, source_files, cache)
            if src is None:
                print(f"{os.path.basename(path)}: no matching raw simulation in {args.source_dir}; paper style skipped")
                continue
            plot_paper(sample, half, os.path.join(args.out, f"{stem}_paper.png"), args.linewidth, args.colorbar)
            print(f"{os.path.basename(path)}: matched {os.path.basename(src)}")
    print(f"plots written to {args.out}")


if __name__ == "__main__":
    main()
