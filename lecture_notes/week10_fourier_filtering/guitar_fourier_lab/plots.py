import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from signals import envelope_indices, response

CLEAN = "#1d7060"
CLIP = "#a54728"
FILTER = "#364f7a"
OUTPUT = "#4c4792"
GRID = "#d5ddd9"
INK = "#21332f"
GREEN = "#eff7f1"
GRAY = "#f1f3f4"
CONFIG = {"displaylogo": False, "scrollZoom": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def base(fig, height, background):
    fig.update_layout(height=height, paper_bgcolor=background, plot_bgcolor=background,
                      font=dict(family="Arial, sans-serif", color=INK, size=12),
                      margin=dict(l=12, r=14, b=8, t=8), showlegend=False, dragmode="zoom")
    fig.update_xaxes(gridcolor=GRID, zerolinecolor="#abb8b4", automargin=True, nticks=6)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="#abb8b4", automargin=True, nticks=4)
    return fig


def waveform(samples, rate, bounds, amplitude, color, threshold=None):
    begin, end = bounds
    first = min(len(samples)-1, max(0, int(np.floor(begin * rate))))
    stop = min(len(samples), max(first + 1, int(np.ceil(end * rate)) + 1))
    visible = samples[first:stop]
    idx = envelope_indices(visible)
    fig = go.Figure(go.Scatter(x=(idx + first) / rate, y=visible[idx], mode="lines",
                               line=dict(color=color, width=1.3, simplify=False),
                               hovertemplate="%{x:.6f} s<br>Amplitude %{y:.4f}<extra></extra>"))
    base(fig, 180, GREEN)
    fig.update_xaxes(title_text="Time (s)", range=[begin, end])
    fig.update_yaxes(title_text="Amplitude", range=[-amplitude, amplitude])
    if threshold is not None and threshold > 0:
        for bound in (-threshold, threshold):
            fig.add_hline(y=bound, line=dict(color=CLIP, dash="dot", width=1))
    return fig


def continuous_steps(span, signed, bands, outside, nyquist):
    left = -span if signed else 0.0
    breaks = {left, span}
    for low, high, _ in bands:
        breaks.update(edge for edge in (-high, -low, low, high) if left < edge < span)
    edges = np.array(sorted(breaks))
    middle = (edges[:-1] + edges[1:]) / 2
    levels = response(middle, bands, outside, nyquist)
    # Draw vertical jumps exactly at the user-entered edges, including view endpoints.
    x = np.r_[left, np.column_stack((edges[:-1], edges[1:])).ravel(), span]
    y = np.r_[response(np.array([left]), bands, outside, nyquist),
              np.repeat(levels, 2), response(np.array([span]), bands, outside, nyquist)]
    return x, y


def mirror(x, y, signed):
    if not signed:
        return x, y
    positive = x > 0
    return np.r_[-x[positive][::-1], x], np.r_[y[positive][::-1], y]


def spectrum_figure(freq, mx, my, peaks, span, signed, scale, unit, bands, outside, nyquist):
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.055,
                        subplot_titles=["③   |X(ω)| · Clipped input", "④   H(ω) · Filter gain",
                                        "⑤   |Y(ω)| · Filtered output"])
    count = np.searchsorted(freq, span, side="right")
    # Include the sample just beyond the view so lines reach the visible boundary.
    drawn = min(count + 1, len(freq))
    for col, mag, color in [(1, mx, CLIP), (3, my, OUTPUT)]:
        idx = envelope_indices(mag[:drawn], budget=2600)
        x, y = mirror(freq[idx], mag[idx], signed)
        fig.add_trace(go.Scatter(x=x * scale, y=y, mode="lines",
                                 line=dict(color=color, width=1.5, simplify=False),
                                 hovertemplate="%{x:.4g} " + unit + "<br>Magnitude %{y:.5g}<extra></extra>"),
                      row=1, col=col)
    if peaks.size:
        ordered = np.sort(peaks)
        x, y = mirror(freq[ordered], mx[ordered], signed)
        fig.add_trace(go.Scatter(x=x * scale, y=y, mode="markers",
                                 marker=dict(size=6, color=CLIP, line=dict(color="#fff", width=1)),
                                 hovertemplate="Peak %{x:.5g} " + unit + "<br>|X| %{y:.5g}<extra></extra>"),
                      row=1, col=1)
    x, h = continuous_steps(span, signed, bands, outside, nyquist)
    fig.add_trace(go.Scatter(x=x * scale, y=h, mode="lines", line=dict(color=FILTER, width=2.2, simplify=False),
                             hovertemplate="%{x:.4g} " + unit + "<br>Gain %{y:g}<extra></extra>"),
                  row=1, col=2)
    base(fig, 265, GRAY)
    fig.update_layout(margin=dict(l=12, r=14, b=8, t=44))
    fig.update_annotations(font=dict(size=14, color=INK))
    xr = [(-span * scale if signed else 0), span * scale]
    symbol = "ω" if unit == "rad/s" else "f"
    fig.update_xaxes(title_text=f"{symbol} ({unit})", range=xr, nticks=5,
                     uirevision=f"{unit}-{signed}-{span}")
    for col in (2, 3):
        fig.update_xaxes(matches="x", row=1, col=col)
    ymax = float(np.max(mx[:max(1, count)])) * 1.08 or 1.0
    fig.update_yaxes(title_text="Magnitude", range=[0, ymax], row=1, col=1)
    fig.update_yaxes(range=[-0.1, 1.1], tickvals=[0, 0.5, 1], row=1, col=2)
    fig.update_yaxes(matches="y", range=[0, ymax], row=1, col=3)
    return fig
