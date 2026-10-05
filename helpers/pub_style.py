"""
Final-artwork styling for journal figures.

Guide requirements this enforces:
  * width 88 mm (1 column) or 180 mm (2 columns); max height ~210 mm for a
    caption under 150 words;
  * all text sans-serif (Arial), 5-7 pt at final size;
  * vector PDF with editable text (fonttype 42); bitmapped parts (rasterised
    brain surfaces, video frames, word clouds) at >= 300 dpi at final size;
  * RGB.

Rather than re-plotting every figure by hand, the existing plotting functions
run unchanged inside `capture()`, which intercepts Figure.savefig so nothing
in results/ is written. `finalize()` then rescales each captured figure to its
final width, maps every font size linearly into [5, 7] pt (keeping the
original hierarchy), lowercases bold single-letter panel labels,
thins lines in proportion, and trims the figure so its tight bounding box is
the target width.
"""

import contextlib
import os
import re

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.collections import Collection

MM = 1 / 25.4
ONE_COL = 88 * MM
TWO_COL = 180 * MM
MAX_H = 210 * MM          # caption < 150 words
F_MIN, F_MAX = 5.0, 7.0
PANEL_SIZE = 7.0
DPI = 300

PAPER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'paper_figures')


def apply_rc():
    matplotlib.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Arial',
        'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold',
        'mathtext.sf': 'Arial',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    })


@contextlib.contextmanager
def capture():
    """Run plotting code without writing files; yields {basename: Figure}.

    Figure.savefig is intercepted (the first call per figure records it under
    the basename of the path it was given, minus extension) and plt.close is
    made a no-op so captured figures stay alive.
    """
    apply_rc()
    captured = {}
    orig_save, orig_close = Figure.savefig, plt.close

    def fake_save(self, fname, *a, **k):
        key = os.path.splitext(os.path.basename(str(fname)))[0]
        captured.setdefault(key, self)

    Figure.savefig = fake_save
    plt.close = lambda *a, **k: None
    try:
        yield captured
    finally:
        Figure.savefig, plt.close = orig_save, orig_close


def _is_panel_label(t):
    return (re.fullmatch(r'[A-H]', t.get_text().strip() or '') is not None
            and str(t.get_fontweight()) in ('bold', '700', 'heavy', 'black', '800', '900'))


def _visible_texts(fig):
    """Text that actually renders. Tick and axis labels of axes drawn with
    axis('off') (e.g. every 3D brain surface) keep visible=True but are never
    drawn, so they are excluded -- otherwise they skew the size mapping."""
    hidden = set()
    for ax in fig.axes:
        if not ax.axison:
            for axis in (ax.xaxis, ax.yaxis) + ((ax.zaxis,) if hasattr(ax, 'zaxis') else ()):
                hidden.update(id(l) for l in axis.get_ticklabels(which='both'))
                hidden.add(id(axis.label))
    return [t for t in fig.findobj(Text)
            if t.get_visible() and t.get_text().strip() and t.figure is fig
            and id(t) not in hidden]


def _rebuild_legends(fig, mapped):
    """Legend handle boxes and spacing are sized in font units when a legend is
    created, so shrinking its text alone leaves oversized swatches and gaps.
    Re-create each legend from its own handles at the new size. The text
    loop has already mapped the label sizes, so the original size is inverted
    from the recorded value."""
    legends = [(fig, l) for l in fig.legends] + \
              [(ax, ax.get_legend()) for ax in fig.axes if ax.get_legend() is not None]
    for parent, leg in legends:
        orig = getattr(leg, '_pub_orig_size', None)
        if orig is None:
            continue
        ns = mapped(orig)
        handles = list(leg.legend_handles)
        labels = [t.get_text() for t in leg.texts]
        title = leg.get_title().get_text()
        kw = dict(loc=leg._loc, ncol=leg._ncols, frameon=leg.get_frame_on(),
                  fontsize=ns, title=title or None, title_fontsize=ns,
                  handlelength=1.2, handleheight=0.8, handletextpad=0.4,
                  labelspacing=0.3, borderpad=0.3, columnspacing=0.8,
                  borderaxespad=0.3)
        if leg._bbox_to_anchor is not None:
            # re-express the anchor as fractions of the parent, so it follows resizes
            frame = parent.transAxes if parent is not fig else fig.transFigure
            b = leg.get_bbox_to_anchor().transformed(frame.inverted())
            kw['bbox_to_anchor'] = (b.x0, b.y0, b.width, b.height)
            kw['bbox_transform'] = frame
        title_colour = leg.get_title().get_color()
        leg.remove()
        new = parent.legend(handles=handles, labels=labels, **kw)
        new.get_title().set_color(title_colour)
        if hasattr(new, 'get_frame') and new.get_frame_on():
            new.get_frame().set_linewidth(0.4)


def finalize(fig, width=TWO_COL, max_height=MAX_H, f_min=F_MIN, f_max=F_MAX,
             size_override=None, lw_min=0.35):
    """Rescale a captured figure in place to final artwork size.

    size_override: optional callable(text_obj, mapped_size) -> size, for
    per-figure tweaks.
    """
    apply_rc()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    old_w, old_h = fig.get_size_inches()
    k = width / old_w
    if max_height and old_h * k > max_height:
        k = max_height / old_h

    # --- fonts: linear map of the original size range into [f_min, f_max] ---
    texts = _visible_texts(fig)
    for leg in list(fig.legends) + [ax.get_legend() for ax in fig.axes if ax.get_legend()]:
        if leg.texts:
            leg._pub_orig_size = leg.texts[0].get_fontsize()
    body =[t.get_fontsize() for t in texts if not _is_panel_label(t)]
    s_lo, s_hi = (min(body), max(body)) if body else (10, 10)

    def mapped(s):
        if s_hi - s_lo < 1e-6:
            return (f_min + f_max) / 2
        return f_min + (f_max - f_min) * (s - s_lo) / (s_hi - s_lo)

    # tick labels through tick_params so re-created ticks keep the new size
    tick_ids = {id(l) for ax in fig.axes for axis in (ax.xaxis, ax.yaxis)
                for l in axis.get_ticklabels(which='both')}
    texts = [t for t in texts if id(t) not in tick_ids]
    for ax in fig.axes:
        for axis, which in ((ax.xaxis, 'x'), (ax.yaxis, 'y')):
            labels = [l for l in axis.get_ticklabels() if l.get_text().strip()]
            if labels:
                ns = mapped(labels[0].get_fontsize())
                if size_override:
                    ns = size_override(labels[0], ns)
                ax.tick_params(axis=which, labelsize=ns)
        ax.tick_params(length=2.5, width=0.5, pad=1.5)
    for t in texts:
        t.set_fontfamily('sans-serif')
        if _is_panel_label(t):
            t.set_text(t.get_text().strip().lower())
            t.set_fontsize(PANEL_SIZE)
            t.set_fontweight('bold')
            continue
        ns = mapped(t.get_fontsize())
        if size_override:
            ns = size_override(t, ns)
        t.set_fontsize(ns)

    _rebuild_legends(fig, mapped)

    # --- lines and markers in proportion ---
    for ln in fig.findobj(Line2D):
        ln.set_linewidth(max(lw_min, ln.get_linewidth() * k))
        ln.set_markersize(max(1.0, ln.get_markersize() * k))
        mew = ln.get_markeredgewidth()
        ln.set_markeredgewidth(max(0.25, mew * k))
    for p in fig.findobj(Patch):
        if p.get_linewidth():
            p.set_linewidth(max(0.25, p.get_linewidth() * k))
    for c in fig.findobj(Collection):
        lws = np.atleast_1d(c.get_linewidths())
        if lws.size:
            c.set_linewidths(np.maximum(0.25, lws * k))
        if hasattr(c, 'get_sizes') and np.size(c.get_sizes()):
            c.set_sizes(np.maximum(1.0, np.asarray(c.get_sizes()) * k * k))

    fig.set_size_inches(old_w * k, old_h * k)

    # --- trim: make the tight bounding box the target width (text no longer
    # scales with the figure, so iterate a few times) ---
    target = min(width, old_w * k)
    for _ in range(4):
        fig.canvas.draw()
        tb = fig.get_tightbbox(fig.canvas.get_renderer())
        w, h = fig.get_size_inches()
        ratio = target / tb.width
        if abs(ratio - 1) < 0.005:
            break
        new_h = h * ratio
        if max_height and tb.height * ratio > max_height:
            ratio = max_height / tb.height
            new_h = h * ratio
        fig.set_size_inches(w * ratio, new_h)
    return fig


def save(fig, name, out_dir=PAPER_DIR, png=True):
    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(out_dir, name + '.pdf')
    fig.savefig(pdf, format='pdf', dpi=DPI, bbox_inches='tight', pad_inches=0.01)
    if png:
        fig.savefig(os.path.join(out_dir, name + '.png'), dpi=DPI,
                    bbox_inches='tight', pad_inches=0.01)
    return pdf


def report(pdf_path):
    """(width_mm, height_mm, min_pt, max_pt, fonts) of a saved PDF, from its
    text objects (pdffonts / pdftotext must be on PATH)."""
    import subprocess
    info = subprocess.run(['pdfinfo', pdf_path], capture_output=True, text=True).stdout
    m = re.search(r'Page size:\s+([\d.]+) x ([\d.]+)', info)
    w_pt, h_pt = float(m.group(1)), float(m.group(2))
    fonts = subprocess.run(['pdffonts', pdf_path], capture_output=True, text=True).stdout
    names = sorted({l.split()[0].split('+')[-1] for l in fonts.splitlines()[2:] if l.strip()})
    return w_pt / 72 * 25.4, h_pt / 72 * 25.4, names
